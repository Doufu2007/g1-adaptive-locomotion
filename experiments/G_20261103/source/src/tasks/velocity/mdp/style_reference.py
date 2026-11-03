"""Robot gait references and dropout-conditioned imitation costs.

Fault information is used only by training rewards and diagnostics. Neither the
reference joint targets nor the dropout mask are added to policy observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .action_dropout import JointTargetDropoutAction
from .rewards import feet_gait

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


StyleMode = Literal["none", "fixed", "global", "selective"]
STYLE_MODES = ("none", "fixed", "global", "selective")


@dataclass(frozen=True)
class GaitReference:
  path: Path
  joint_names: tuple[str, ...]
  joint_pos: np.ndarray
  fps: float
  gait_period_s: float
  target_velocity: tuple[float, float, float]
  loop: bool
  provenance: dict
  sha256: str
  phase_offset_cycles: float = 0.0

  @property
  def duration_s(self) -> float:
    frames = len(self.joint_pos) if self.loop else len(self.joint_pos) - 1
    return frames / self.fps


def load_gait_reference(
  path: str | Path, *, allow_synthetic: bool = False
) -> GaitReference:
  """Load non-pickled NPZ plus a provenance JSON, failing on missing inputs."""
  path = Path(path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(
      f"Walking reference not found: {path}. Supply a reviewed G1 walking "
      "reference; dance files and software fixtures are not study evidence."
    )
  manifest_path = path.with_suffix(".json")
  if not manifest_path.is_file():
    raise FileNotFoundError(f"Reference provenance manifest missing: {manifest_path}")
  provenance = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
  if provenance.get("schema_version") != 1:
    raise ValueError("Reference schema_version must be 1.")
  kind = provenance.get("kind")
  if kind not in ("retargeted_human", "robot_reference", "synthetic_test"):
    raise ValueError("Reference kind must describe the actual source.")
  if kind == "synthetic_test" and not allow_synthetic:
    raise ValueError("Synthetic software fixtures cannot be used for study training.")
  for field in ("source_url", "license", "description"):
    value = provenance.get(field)
    if not isinstance(value, str) or not value.strip():
      raise ValueError(f"Reference provenance requires {field}.")
  if provenance.get("split") not in ("development", "test", "software_test"):
    raise ValueError("Reference split must be development, test, or software_test.")
  if kind == "synthetic_test" and provenance["split"] != "software_test":
    raise ValueError("Synthetic fixtures must have split=software_test.")
  if kind != "synthetic_test" and provenance["split"] == "software_test":
    raise ValueError("Real references must declare development or test split.")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if provenance.get("sha256") != digest:
    raise ValueError("Reference SHA-256 does not match its provenance manifest.")

  with np.load(path, allow_pickle=False) as data:
    required = {"joint_names", "joint_pos", "fps", "gait_period_s", "target_velocity", "loop"}
    if missing := required - set(data.files):
      raise ValueError(f"Reference is missing arrays: {sorted(missing)}")
    names_array = data["joint_names"]
    if names_array.ndim != 1 or names_array.dtype.kind not in "US":
      raise ValueError("joint_names must be a one-dimensional string array.")
    names = tuple(str(x) for x in names_array.astype(str))
    positions = np.asarray(data["joint_pos"], dtype=np.float32)
    fps = float(data["fps"].item())
    period = float(data["gait_period_s"].item())
    velocity = np.asarray(data["target_velocity"], dtype=np.float64)
    loop_array = data["loop"]
    if loop_array.shape != () or loop_array.dtype.kind != "b":
      raise ValueError("loop must be a scalar boolean.")
    loop = bool(loop_array.item())
    phase_offset = float(data["phase_offset_cycles"].item()) if "phase_offset_cycles" in data.files else 0.0
  if not names or len(set(names)) != len(names) or any(not n for n in names):
    raise ValueError("joint_names must be unique and nonempty.")
  if positions.ndim != 2 or positions.shape[1] != len(names) or len(positions) < 3:
    raise ValueError("joint_pos must be [frames >= 3, number of named joints].")
  if not np.isfinite(positions).all():
    raise ValueError("joint_pos contains NaN or infinity.")
  if not math.isfinite(fps) or not 10.0 <= fps <= 500.0:
    raise ValueError("fps must be finite and between 10 and 500 Hz.")
  if not math.isfinite(period) or period < 2.0 / fps:
    raise ValueError("gait_period_s must span at least two reference samples.")
  if not math.isfinite(phase_offset) or not 0.0 <= phase_offset < 1.0:
    raise ValueError("phase_offset_cycles must be finite and in [0, 1).")
  if velocity.shape != (3,) or not np.isfinite(velocity).all():
    raise ValueError("target_velocity must contain finite [vx, vy, yaw_rate].")
  if velocity[0] <= 0.0 or np.any(np.abs(velocity[1:]) > 1.0e-6):
    raise ValueError("The first protocol supports positive straight-ahead walking only.")
  if loop and np.max(np.abs(positions[-1] - positions[0])) > 0.2:
    raise ValueError("Loop seam exceeds 0.2 rad; use a validated cycle or a longer non-loop clip.")
  return GaitReference(
    path, names, positions, fps, period, tuple(float(x) for x in velocity),
    loop, provenance, digest, phase_offset,
  )


def update_loss_trace(
  trace: torch.Tensor, mask: torch.Tensor, dt: float,
  attack_tau_s: float, recovery_tau_s: float,
) -> torch.Tensor:
  """Causal per-channel trace; packet loss raises it and updates let it decay."""
  attack = math.exp(-dt / attack_tau_s)
  recovery = math.exp(-dt / recovery_tau_s)
  return torch.where(mask, attack * trace + (1.0 - attack), recovery * trace)


def imitation_weights(
  trace: torch.Tensor, base_weights: torch.Tensor, alpha: float, mode: StyleMode,
) -> torch.Tensor:
  """Global and selective modes have exactly the same weighted mass."""
  local = 1.0 - alpha * trace
  if mode == "none":
    return torch.zeros_like(trace)
  if mode == "fixed":
    return torch.ones_like(trace)
  if mode == "selective":
    return local
  if mode == "global":
    mass = (local * base_weights).sum(dim=-1, keepdim=True) / base_weights.sum()
    return mass.expand_as(trace)
  raise ValueError(f"Unknown style mode: {mode}")


class GaitReferenceCommand(CommandTerm):
  cfg: GaitReferenceCommandCfg

  def __init__(self, cfg: GaitReferenceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.reference = load_gait_reference(
      cfg.reference_file, allow_synthetic=cfg.allow_synthetic_reference
    )
    self.robot = env.scene[cfg.entity_name]
    self.joint_names = tuple(self.robot.joint_names)
    if set(self.reference.joint_names) != set(self.joint_names):
      raise ValueError("Reference must name every robot joint exactly once.")
    order = [self.reference.joint_names.index(n) for n in self.joint_names]
    self.poses = torch.as_tensor(
      self.reference.joint_pos[:, order], dtype=torch.float32, device=self.device
    )
    if not self.reference.loop and self.reference.duration_s < env.cfg.episode_length_s:
      raise ValueError("Non-loop reference is shorter than the configured episode.")
    if not np.allclose(self.reference.target_velocity, cfg.target_velocity, atol=1e-6, rtol=0):
      raise ValueError("Reference velocity and configured training command differ.")
    dimensions = len(self.joint_names)
    self.base_weights = torch.ones(dimensions, device=self.device)
    self.loss_trace = torch.zeros((self.num_envs, dimensions), device=self.device)
    self._weights = torch.ones_like(self.loss_trace)
    self._cost = torch.zeros(self.num_envs, device=self.device)
    self._unmasked_error = torch.zeros_like(self._cost)
    self._last_trace_step = -1
    self._action_joint_ids: list[int] | None = None

  @property
  def phase(self) -> torch.Tensor:
    return (self._env.episode_length_buf * self._env.step_dt / self.reference.gait_period_s
            + self.reference.phase_offset_cycles)

  @property
  def command(self) -> torch.Tensor:
    times = self._env.episode_length_buf * self._env.step_dt * self.reference.fps
    if self.reference.loop:
      positions = times.remainder(len(self.poses))
      low = positions.floor().long()
      high = (low + 1).remainder(len(self.poses))
    else:
      positions = times.clamp(max=len(self.poses) - 1)
      low = positions.floor().long()
      high = (low + 1).clamp(max=len(self.poses) - 1)
    fraction = (positions - low).unsqueeze(1)
    return torch.lerp(self.poses[low], self.poses[high], fraction)

  def diagnostics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rewards and metrics precede command_manager.compute(). Read the current
    # action mask here, once per control step, rather than using a stale mask.
    step = int(self._env.common_step_counter)
    if self._last_trace_step != step:
      term = self._env.action_manager.get_term(self.cfg.action_term_name)
      if not isinstance(term, JointTargetDropoutAction):
        raise TypeError("Style study requires a JointTargetDropoutAction.")
      if self._action_joint_ids is None:
        self._action_joint_ids = [self.joint_names.index(n) for n in term.target_names]
      mask = torch.zeros_like(self.loss_trace, dtype=torch.bool)
      mask[:, self._action_joint_ids] = term.dropout_mask
      self.loss_trace.copy_(update_loss_trace(
        self.loss_trace, mask, self._env.step_dt,
        self.cfg.attack_tau_s, self.cfg.recovery_tau_s,
      ))
      self._last_trace_step = step
    self._weights = imitation_weights(
      self.loss_trace, self.base_weights, self.cfg.alpha, self.cfg.mode
    )
    error = (self.robot.data.joint_pos - self.command) / self.cfg.joint_sigma_rad
    squared = error.square()
    self._cost = (squared * self._weights * self.base_weights).sum(-1) / self.base_weights.sum()
    # This evaluation diagnostic never masks faulted joints.
    self._unmasked_error = torch.sqrt((squared * self.base_weights).sum(-1) / self.base_weights.sum())
    mass = (self._weights * self.base_weights).sum(-1) / self.base_weights.sum()
    return self._cost, self._unmasked_error, mass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.loss_trace[env_ids] = 0
    self.time_left[env_ids] = 1.0e9

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class GaitReferenceCommandCfg(CommandTermCfg):
  reference_file: str = "data/style_dropout/walking_reference.npz"
  entity_name: str = "robot"
  action_term_name: str = "joint_pos"
  mode: StyleMode = "selective"
  target_velocity: tuple[float, float, float] = (0.7, 0.0, 0.0)
  joint_sigma_rad: float = 0.5
  alpha: float = 0.75
  attack_tau_s: float = 0.08
  recovery_tau_s: float = 0.25
  allow_synthetic_reference: bool = False

  def build(self, env: ManagerBasedRlEnv) -> GaitReferenceCommand:
    if self.mode not in STYLE_MODES:
      raise ValueError("Invalid style mode.")
    if not math.isfinite(self.alpha) or not 0.0 <= self.alpha < 1.0:
      raise ValueError("alpha must be finite and in [0, 1).")
    for name in ("joint_sigma_rad", "attack_tau_s", "recovery_tau_s"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return GaitReferenceCommand(self, env)


def reference_term(env: ManagerBasedRlEnv) -> GaitReferenceCommand:
  term = env.command_manager.get_term("gait_reference")
  if not isinstance(term, GaitReferenceCommand):
    raise TypeError("gait_reference command has the wrong type.")
  return term


def style_imitation_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  return reference_term(env).diagnostics()[0]


def style_unmasked_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  return reference_term(env).diagnostics()[1]


def style_weight_mass(env: ManagerBasedRlEnv) -> torch.Tensor:
  return reference_term(env).diagnostics()[2]


def reference_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  angle = 2.0 * math.pi * reference_term(env).phase
  return torch.stack((angle.sin(), angle.cos()), dim=-1)


def reference_feet_gait(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  reference = reference_term(env).reference
  kwargs["offset"] = [x + reference.phase_offset_cycles for x in kwargs["offset"]]
  return feet_gait(env, period=reference.gait_period_s, **kwargs)
