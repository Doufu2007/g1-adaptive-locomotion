"""Joint target update dropout for command-channel robustness experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointTargetDropoutActionCfg(JointPositionActionCfg):
  """Position action with temporally correlated target-update loss."""

  enabled: bool = True
  eligible_joint_names: tuple[str, ...] = (
    "left_hip_.*_joint",
    "right_hip_.*_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_.*_joint",
    "right_ankle_.*_joint",
  )
  event_probability: float = 0.005
  dropout_probability_range: tuple[float, float] = (0.25, 0.75)
  duration_steps_range: tuple[int, int] = (5, 20)
  fault_joint_count_range: tuple[int, int] = (1, 2)
  curriculum_warmup_steps: int = 300 * 24
  curriculum_ramp_steps: int = 300 * 24
  forced_joint_names: tuple[str, ...] = ()
  forced_start_step: int = -1
  forced_end_step: int = -1
  forced_dropout_probability: float = -1.0

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 <= self.event_probability <= 1.0:
      raise ValueError("event_probability must be in [0, 1].")
    low, high = self.dropout_probability_range
    if not 0.0 <= low <= high <= 1.0:
      raise ValueError("dropout_probability_range must be ordered in [0, 1].")
    min_steps, max_steps = self.duration_steps_range
    if min_steps < 1 or min_steps > max_steps:
      raise ValueError("duration_steps_range must contain positive ordered steps.")
    min_joints, max_joints = self.fault_joint_count_range
    if min_joints < 1 or min_joints > max_joints:
      raise ValueError("fault_joint_count_range must contain positive ordered counts.")
    if self.curriculum_warmup_steps < 0 or self.curriculum_ramp_steps < 0:
      raise ValueError("curriculum steps must be non-negative.")
    if self.forced_dropout_probability < -1.0 or self.forced_dropout_probability > 1.0:
      raise ValueError("forced_dropout_probability must be -1 or in [0, 1].")
    if (self.forced_start_step < 0) != (self.forced_end_step < 0):
      raise ValueError("forced_start_step and forced_end_step must be set together.")
    if self.forced_start_step >= 0 and self.forced_start_step >= self.forced_end_step:
      raise ValueError("forced_end_step must be greater than forced_start_step.")

  def build(self, env: ManagerBasedRlEnv) -> JointTargetDropoutAction:
    return JointTargetDropoutAction(self, env)


def make_joint_target_dropout_action_cfg(
  base_cfg: JointPositionActionCfg,
  **overrides: object,
) -> JointTargetDropoutActionCfg:
  """Copy a position action config while changing only dropout parameters."""
  values: dict[str, object] = {
    "entity_name": base_cfg.entity_name,
    "actuator_names": base_cfg.actuator_names,
    "scale": base_cfg.scale,
    "offset": base_cfg.offset,
    "clip": base_cfg.clip,
    "preserve_order": base_cfg.preserve_order,
    "use_default_offset": base_cfg.use_default_offset,
  }
  values.update(overrides)
  return JointTargetDropoutActionCfg(**values)


class JointTargetDropoutAction(JointPositionAction):
  """Vectorized, bursty loss of selected joint target updates."""

  cfg: JointTargetDropoutActionCfg

  def __init__(self, cfg: JointTargetDropoutActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._eligible_indices = self._resolve_action_indices(cfg.eligible_joint_names)
    if not self._eligible_indices:
      raise ValueError(
        "eligible_joint_names did not match any controlled joint targets. "
        f"Available targets: {self.target_names}"
      )
    if cfg.fault_joint_count_range[1] > len(self._eligible_indices):
      raise ValueError(
        "fault_joint_count_range exceeds the number of eligible joint targets."
      )
    self._forced_indices = self._resolve_action_indices(cfg.forced_joint_names)
    if cfg.forced_joint_names and not self._forced_indices:
      raise ValueError(
        "forced_joint_names did not match any controlled joint targets. "
        f"Available targets: {self.target_names}"
      )

    default_targets = self._entity.data.default_joint_pos[:, self._target_ids]
    self._previous_applied_target = default_targets.clone()
    self._last_desired_target = default_targets.clone()
    self._fault_remaining = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    max_count = cfg.fault_joint_count_range[1]
    self._fault_indices = torch.full(
      (self.num_envs, max_count), -1, device=self.device, dtype=torch.long
    )
    self._fault_probability = torch.zeros(self.num_envs, device=self.device)
    self._dropout_mask = torch.zeros(
      (self.num_envs, self.action_dim), device=self.device, dtype=torch.bool
    )
    self._current_fault_mask = torch.zeros_like(self._dropout_mask)
    self._target_hold_error = torch.zeros(self.num_envs, device=self.device)

  def _resolve_action_indices(self, patterns: tuple[str, ...]) -> list[int]:
    if not patterns:
      return []
    return [
      index
      for index, name in enumerate(self.target_names)
      if any(re.fullmatch(pattern, name) for pattern in patterns)
    ]

  def _curriculum_factor(self) -> float:
    step = float(self._env.common_step_counter)
    warmup = float(self.cfg.curriculum_warmup_steps)
    ramp = float(self.cfg.curriculum_ramp_steps)
    if step <= warmup:
      factor = 0.0
    elif ramp == 0.0:
      factor = 1.0
    else:
      factor = min(1.0, (step - warmup) / ramp)
    return factor

  def _sample_fault_indices(self, start_mask: torch.Tensor, factor: float) -> None:
    max_count = self.cfg.fault_joint_count_range[1]
    min_count = self.cfg.fault_joint_count_range[0]
    max_count_now = min_count + int(round((max_count - min_count) * factor))
    max_count_now = max(min_count, min(max_count, max_count_now))
    scores = torch.rand(
      (self.num_envs, len(self._eligible_indices)), device=self.device
    )
    order = torch.topk(scores, k=max_count, dim=1).indices
    counts = torch.randint(
      min_count, max_count_now + 1, (self.num_envs,), device=self.device
    )
    valid = (
      torch.arange(max_count, device=self.device).unsqueeze(0)
      < counts.unsqueeze(1)
    )
    eligible = torch.tensor(
      self._eligible_indices, device=self.device, dtype=torch.long
    )
    selected = eligible[order]
    selected = torch.where(valid, selected, torch.full_like(selected, -1))
    self._fault_indices[start_mask] = selected[start_mask]

  def _sample_fault_probability(self, start_mask: torch.Tensor, factor: float) -> None:
    low, high = self.cfg.dropout_probability_range
    values = low + (high - low) * torch.rand(self.num_envs, device=self.device)
    values = low + (values - low) * factor
    self._fault_probability[start_mask] = values[start_mask]

  def _sample_fault_duration(self, start_mask: torch.Tensor, factor: float) -> None:
    min_steps, max_steps = self.cfg.duration_steps_range
    max_steps_now = min_steps + int(round((max_steps - min_steps) * factor))
    durations = torch.randint(
      min_steps, max_steps_now + 1, (self.num_envs,), device=self.device
    )
    self._fault_remaining[start_mask] = durations[start_mask] - 1

  def _selected_mask(self) -> torch.Tensor:
    selected = torch.zeros(
      (self.num_envs, self.action_dim), device=self.device, dtype=torch.bool
    )
    rows_all = torch.arange(self.num_envs, device=self.device)
    for slot in range(self._fault_indices.shape[1]):
      indices = self._fault_indices[:, slot]
      valid = indices >= 0
      selected[rows_all[valid], indices[valid]] = True
    return selected

  def _forced_fault_mask(self) -> torch.Tensor | None:
    if self.cfg.forced_start_step < 0:
      return None
    step = self._env.common_step_counter
    active = self.cfg.forced_start_step <= step < self.cfg.forced_end_step
    if not active:
      self._current_fault_mask.zero_()
      return torch.zeros_like(self._dropout_mask)
    selected = torch.zeros_like(self._dropout_mask)
    if self._forced_indices:
      selected[:, self._forced_indices] = True
    probability = self.cfg.forced_dropout_probability
    if probability < 0.0:
      probability = sum(self.cfg.dropout_probability_range) / 2.0
    self._current_fault_mask[:] = selected
    return selected & (torch.rand_like(selected.float()) < probability)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._last_desired_target[:] = self._processed_actions
    if not self.cfg.enabled:
      self._dropout_mask.zero_()
      self._current_fault_mask.zero_()
      self._target_hold_error.zero_()
      self._previous_applied_target[:] = self._processed_actions
      return

    forced_mask = self._forced_fault_mask()
    if forced_mask is not None:
      self._dropout_mask[:] = forced_mask
      self._fault_remaining.zero_()
    else:
      self._dropout_mask.zero_()
      self._current_fault_mask.zero_()
      factor = self._curriculum_factor()
      active_before = self._fault_remaining > 0
      selected = self._selected_mask()
      self._current_fault_mask[:] = active_before.unsqueeze(1) & selected
      random_drop = (
        torch.rand_like(selected.float()) < self._fault_probability.unsqueeze(1)
      )
      self._dropout_mask[:] = self._current_fault_mask & random_drop
      self._fault_remaining[active_before] -= 1

      command = self._env.command_manager.get_command("twist")
      walking = torch.linalg.norm(command[:, :2], dim=1) > 0.1
      start = (~active_before) & walking & (
        torch.rand(self.num_envs, device=self.device)
        < self.cfg.event_probability * factor
      )
      self._sample_fault_indices(start, factor)
      self._sample_fault_probability(start, factor)
      self._sample_fault_duration(start, factor)
      selected = self._selected_mask()
      random_drop = (
        torch.rand_like(selected.float()) < self._fault_probability.unsqueeze(1)
      )
      self._current_fault_mask[start] = selected[start]
      self._dropout_mask[start] = selected[start] & random_drop[start]

    self._processed_actions[:] = torch.where(
      self._dropout_mask,
      self._previous_applied_target,
      self._processed_actions,
    )
    self._target_hold_error[:] = torch.mean(
      torch.abs(self._last_desired_target - self._processed_actions), dim=1
    )
    self._previous_applied_target[:] = self._processed_actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    default_targets = self._entity.data.default_joint_pos[env_ids][:, self._target_ids]
    self._previous_applied_target[env_ids] = default_targets
    self._last_desired_target[env_ids] = self._previous_applied_target[env_ids]
    self._fault_remaining[env_ids] = 0
    self._fault_indices[env_ids] = -1
    self._fault_probability[env_ids] = 0.0
    self._dropout_mask[env_ids] = False
    self._current_fault_mask[env_ids] = False
    self._target_hold_error[env_ids] = 0.0

  @property
  def dropout_mask(self) -> torch.Tensor:
    return self._dropout_mask

  @property
  def selected_fault_mask(self) -> torch.Tensor:
    return self._current_fault_mask

  @property
  def target_hold_error(self) -> torch.Tensor:
    return self._target_hold_error


def _dropout_action_term(
  env: ManagerBasedRlEnv, action_term_name: str
) -> JointTargetDropoutAction:
  term = env.action_manager.get_term(action_term_name)
  if not isinstance(term, JointTargetDropoutAction):
    raise TypeError(f"Action term '{action_term_name}' is not dropout-enabled.")
  return term


def dropout_active_fraction(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  return _dropout_action_term(env, action_term_name).dropout_mask.float().mean(dim=1)


def dropout_event_active(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  mask = _dropout_action_term(env, action_term_name).selected_fault_mask
  return mask.any(dim=1).float()


def dropout_dropped_joint_count(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  mask = _dropout_action_term(env, action_term_name).dropout_mask
  return mask.float().sum(dim=1)


def dropout_selected_joint_count(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  return _dropout_action_term(env, action_term_name).selected_fault_mask.float().sum(dim=1)


def target_hold_error(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  return _dropout_action_term(env, action_term_name).target_hold_error
