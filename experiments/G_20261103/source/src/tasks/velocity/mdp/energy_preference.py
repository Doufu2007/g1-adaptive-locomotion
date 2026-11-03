from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


PreferenceMode = Literal["fixed", "unaware", "conditioned"]


class EnergyPreferenceCommand(CommandTerm):
  """Generate a performance-versus-efficiency preference per environment."""

  cfg: EnergyPreferenceCommandCfg

  def __init__(self, cfg: EnergyPreferenceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.preference = torch.full(
      (self.num_envs, 1), cfg.fixed_preference, device=self.device
    )
    self.target_preference = self.preference.clone()
    self.observed_preference = self.preference.clone()

  @property
  def command(self) -> torch.Tensor:
    return self.observed_preference

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.mode == "fixed":
      target = torch.full(
        (len(env_ids), 1), self.cfg.fixed_preference, device=self.device
      )
      self.time_left[env_ids] = self.cfg.disabled_resampling_time_s
    else:
      anchor_ids = torch.randint(
        len(self.cfg.anchors), (len(env_ids),), device=self.device
      )
      anchors = torch.tensor(self.cfg.anchors, device=self.device)
      target = anchors[anchor_ids].unsqueeze(-1)

      global_step = int(self._env.common_step_counter)
      if global_step < self.cfg.switch_start_step:
        self.time_left[env_ids] = self.cfg.disabled_resampling_time_s
      elif global_step < self.cfg.multi_switch_start_step:
        already_switched = self.command_counter[env_ids] >= 1
        self.time_left[env_ids[already_switched]] = self.cfg.disabled_resampling_time_s

    self.target_preference[env_ids] = target
    initial_sample = self.command_counter[env_ids] == 0
    if torch.any(initial_sample):
      initial_ids = env_ids[initial_sample]
      self.preference[initial_ids] = self.target_preference[initial_ids]
      if self.cfg.mode == "unaware":
        self.observed_preference[initial_ids] = self.cfg.unaware_observation
      else:
        self.observed_preference[initial_ids] = self.preference[initial_ids]

  def _update_command(self) -> None:
    if self.cfg.ramp_time_s <= 0.0:
      self.preference.copy_(self.target_preference)
    else:
      max_delta = self._env.step_dt / self.cfg.ramp_time_s
      delta = torch.clamp(
        self.target_preference - self.preference,
        min=-max_delta,
        max=max_delta,
      )
      self.preference.add_(delta)

    if self.cfg.mode == "unaware":
      self.observed_preference.fill_(self.cfg.unaware_observation)
    else:
      self.observed_preference.copy_(self.preference)


@dataclass(kw_only=True)
class EnergyPreferenceCommandCfg(CommandTermCfg):
  mode: PreferenceMode = "conditioned"
  fixed_preference: float = 0.5
  anchors: tuple[float, ...] = (0.0, 0.5, 1.0)
  ramp_time_s: float = 0.5
  switch_start_step: int = 500 * 24
  multi_switch_start_step: int = 1200 * 24
  unaware_observation: float = 0.5
  disabled_resampling_time_s: float = 1.0e9

  def build(self, env: ManagerBasedRlEnv) -> EnergyPreferenceCommand:
    if not 0.0 <= self.fixed_preference <= 1.0:
      raise ValueError("fixed_preference must be in [0, 1].")
    if not self.anchors or any(value < 0.0 or value > 1.0 for value in self.anchors):
      raise ValueError("anchors must contain values in [0, 1].")
    if self.switch_start_step > self.multi_switch_start_step:
      raise ValueError("switch_start_step must not exceed multi_switch_start_step.")
    return EnergyPreferenceCommand(self, env)
