from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def zero_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Keep the phase observation dimensions while removing phase information."""
  return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)


def _sample_episode_vectors(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Create unit vectors that remain fixed for one episode per environment."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = int(env_ids.numel())
    raw = torch.randn(count, 2, device=env.device)
    vectors = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-8)
    return vectors


def reset_random_phase(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Reset the random-vector control condition independently per environment."""
  vectors = getattr(env, "_episode_random_phase", None)
  if vectors is None or vectors.shape != (env.num_envs, 2):
    env._episode_random_phase = _sample_episode_vectors(env)
    return
  if env_ids is None:
    env._episode_random_phase = _sample_episode_vectors(env)
  else:
    env._episode_random_phase[env_ids] = _sample_episode_vectors(env, env_ids)


def random_episode_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one random unit vector per environment, fixed until reset."""
  vectors = getattr(env, "_episode_random_phase", None)
  if vectors is None or vectors.shape != (env.num_envs, 2):
    reset_random_phase(env, None)
    vectors = env._episode_random_phase
  return vectors


# ── PHASOR: Learned phase ────────────────────────────────────────────

from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
from src.tasks.velocity.phasor.encoder import PhaseEncoder
from src.tasks.velocity.phasor.encoder_tcn import TCNEncoder
from src.tasks.velocity.phasor.runtime import PhaseHistoryState


@dataclass(frozen=True)
class _EncoderBundle:
  encoder: torch.nn.Module
  mean: torch.Tensor
  std: torch.Tensor
  window: int
  joint_dim: int
  joint_names: tuple[str, ...]
  identity: tuple[str, str, str]
  sha256: str


_ENCODER_CACHE: dict[tuple[str, str, str], _EncoderBundle] = {}


def _validated_encoder_stats(
  mean: object,
  std: object,
  *,
  joint_dim: int,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Convert checkpoint normalization data and reject unsafe contracts."""
  mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
  std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
  if mean_tensor.ndim == 2 and mean_tensor.shape[0] == 1:
    mean_tensor = mean_tensor.squeeze(0)
  if std_tensor.ndim == 2 and std_tensor.shape[0] == 1:
    std_tensor = std_tensor.squeeze(0)
  if mean_tensor.shape != (joint_dim,) or std_tensor.shape != (joint_dim,):
    raise ValueError(
      f"encoder normalization must have shape ({joint_dim},), got "
      f"mean={tuple(mean_tensor.shape)} std={tuple(std_tensor.shape)}"
    )
  if not torch.isfinite(mean_tensor).all().item() or not torch.isfinite(std_tensor).all().item():
    raise ValueError("encoder normalization values must be finite")
  if not (std_tensor > 0).all().item():
    raise ValueError("encoder standard deviations must be positive")
  return mean_tensor, std_tensor


def _joint_positions_in_order(
  asset: Entity,
  joint_positions: torch.Tensor,
  expected_joint_names: tuple[str, ...],
) -> torch.Tensor:
  """Select joint positions by the frozen checkpoint name order."""
  actual_names = tuple(getattr(asset, "joint_names", ()))
  if not actual_names or joint_positions.ndim != 2 or len(actual_names) != joint_positions.shape[1]:
    raise ValueError("joint names are missing or do not match joint position dimensions")
  if len(set(actual_names)) != len(actual_names) or len(set(expected_joint_names)) != len(expected_joint_names):
    raise ValueError("joint names must be unique")
  actual_indices = {name: index for index, name in enumerate(actual_names)}
  missing = [name for name in expected_joint_names if name not in actual_indices]
  if missing:
    raise ValueError(f"G1 model is missing checkpoint joints: {missing}")
  unexpected = [name for name in actual_names if name not in set(expected_joint_names)]
  if unexpected:
    raise ValueError(f"G1 model has unexpected encoder joints: {unexpected}")
  if len(expected_joint_names) != len(actual_names):
    raise ValueError("checkpoint joint count does not match the G1 model")
  indices = torch.as_tensor(
    [actual_indices[name] for name in expected_joint_names],
    dtype=torch.long,
    device=joint_positions.device,
  )
  return joint_positions.index_select(1, indices)


def _checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_phasor_encoder(env: ManagerBasedRlEnv, encoder_path: str) -> _EncoderBundle:
  """Load or reuse an encoder only for the exact file identity and device."""
  path = Path(encoder_path).expanduser().resolve(strict=True)
  device = str(torch.device(env.device))
  checkpoint_sha256 = _checkpoint_sha256(path)
  identity = (str(path), checkpoint_sha256, device)
  cached = _ENCODER_CACHE.get(identity)
  if cached is not None:
    expected_dt = getattr(cached.encoder, "runtime_control_dt_seconds", None)
    if expected_dt is not None and abs(float(env.step_dt) - expected_dt) > 1e-9:
      raise ValueError("experimental encoder sampling requires 50 Hz control_dt=0.02")
    return cached

  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, dict):
    raise ValueError("G1 PHASOR checkpoint must be a mapping")
  joint_dim = int(checkpoint.get("joint_dim", -1))
  window = int(checkpoint.get("window", -1))
  architecture = str(checkpoint.get("architecture", "phase_encoder"))
  contract = checkpoint.get("runtime_contract")
  experimental_context = (
    architecture == "tcn_long_context" and window == 20
    and isinstance(contract, dict)
    and contract.get("schema") == "g1.phase_runtime_checkpoint.experimental.v2"
  )
  experimental_long_rate = (
    architecture == "tcn_long_rate" and window == 100
    and isinstance(contract, dict)
    and contract.get("schema") == "g1.phase_runtime_checkpoint.experimental.v3"
  )
  experimental_tctst = (
    architecture == "tctst_repaired" and window == 100
    and isinstance(contract, dict)
    and contract.get("schema") in (
      "g1.phase_runtime_checkpoint.experimental.v3",
      "g1.phase_runtime_checkpoint.experimental.v4",
    )
  )
  experimental_long_history = experimental_context or experimental_long_rate or experimental_tctst
  if experimental_long_history and (
    abs(float(contract.get("control_dt_seconds", -1)) - 0.02) > 1e-9
    or abs(float(env.step_dt) - 0.02) > 1e-9
  ):
    raise ValueError("experimental long-history encoder sampling requires 50 Hz control_dt=0.02")
  if joint_dim != len(G1_JOINT_ORDER) or (window != 10 and not experimental_long_history):
    raise ValueError(
      f"G1 PHASOR checkpoint requires joint_dim={len(G1_JOINT_ORDER)} and window=10; "
      f"got joint_dim={joint_dim}, window={window}"
    )
  joint_names = tuple(checkpoint.get("joint_names", ()))
  if joint_names != G1_JOINT_ORDER:
    raise ValueError("checkpoint joint_names do not match the frozen G1 training order")
  if not isinstance(contract, dict):
    raise ValueError("checkpoint runtime contract is missing")
  required_hashes = (
    "source_checkpoint_sha256",
    "dataset_sha256",
    "metadata_sha256",
    "manifest_sha256",
    "reward_template_sha256",
  )
  valid_contract = (
    contract.get("schema") == (
      "g1.phase_runtime_checkpoint.experimental.v2" if experimental_context
      else "g1.phase_runtime_checkpoint.experimental.v3"
      if (experimental_long_rate or (experimental_tctst and
        contract.get("schema") == "g1.phase_runtime_checkpoint.experimental.v3"))
      else "g1.phase_runtime_checkpoint.experimental.v4"
      if experimental_tctst
      else "g1.phase_runtime_checkpoint.v1"
    )
    and tuple(contract.get("joint_order", ())) == G1_JOINT_ORDER
    and int(contract.get("joint_dim", -1)) == joint_dim
    and int(contract.get("window", -1)) == window
    and contract.get("phase_output_convention") == "(sin(phi), cos(phi))"
    and contract.get("dataset_sha256") == checkpoint.get("dataset_sha256")
    and contract.get("manifest_sha256") == checkpoint.get("data_manifest_sha256")
    and all(
      isinstance(contract.get(name), str)
      and len(contract[name]) == 64
      and all(character in "0123456789abcdef" for character in contract[name])
      for name in required_hashes
    )
  )
  if not valid_contract:
    raise ValueError("checkpoint runtime contract is invalid or inconsistent")
  if checkpoint.get("output_convention") != "(sin(phi), cos(phi))":
    raise ValueError("checkpoint output convention does not match the G1 runtime contract")
  mean, std = _validated_encoder_stats(
    checkpoint.get("mean"), checkpoint.get("std"), joint_dim=joint_dim, device=env.device
  )

  # Checkpoint construction must not change environment randomization or PPO
  # initialization merely by consuming random numbers for discarded weights.
  with torch.random.fork_rng(devices=[]):
    if experimental_context:
      from src.tasks.velocity.phasor.encoder_tcn_context import LongContextTCNEncoder
      encoder = LongContextTCNEncoder()
      encoder.runtime_control_dt_seconds = 0.02
    elif experimental_long_rate:
      from src.tasks.velocity.phasor.encoder_tcn_long_rate import LongRateTCNEncoder
      encoder = LongRateTCNEncoder()
    elif experimental_tctst:
      from src.tasks.velocity.phasor.encoder_tctst_runtime import RepairedTCTSTEncoder
      encoder = RepairedTCTSTEncoder(window=window)
    elif architecture == "tcn":
      encoder = TCNEncoder(
        joint_dim=joint_dim,
        hidden=int(checkpoint.get("tcn_hidden", 64)),
      )
    elif architecture == "phase_encoder":
      encoder = PhaseEncoder(
        joint_dim=joint_dim,
        window=window,
        preserve_time=bool(checkpoint.get("preserve_time", False)),
      )
    else:
      raise ValueError(f"unsupported G1 PHASOR encoder architecture: {architecture}")
  encoder = encoder.to(env.device)
  encoder.load_state_dict(checkpoint["state_dict"])
  encoder.eval()
  bundle = _EncoderBundle(
    encoder=encoder,
    mean=mean,
    std=std,
    window=window,
    joint_dim=joint_dim,
    joint_names=joint_names,
    identity=identity,
    sha256=checkpoint_sha256,
  )
  for cached_identity in list(_ENCODER_CACHE):
    if cached_identity[0] == str(path) and cached_identity[2] == device:
      del _ENCODER_CACHE[cached_identity]
  _ENCODER_CACHE[identity] = bundle
  return bundle


def _phase_history_for_bundle(
  env: ManagerBasedRlEnv,
  bundle: _EncoderBundle,
  current: torch.Tensor,
) -> PhaseHistoryState:
  state = getattr(env, "_phasor_phase_history", None)
  history_identity = getattr(env, "_phasor_phase_history_identity", None)
  if (
    state is None
    or history_identity != bundle.identity
    or state.buffer.shape != (env.num_envs, bundle.window, bundle.joint_dim)
  ):
    state = PhaseHistoryState(
      num_envs=env.num_envs,
      window=bundle.window,
      joint_dim=bundle.joint_dim,
      device=env.device,
    )
    normalized = (current - bundle.mean) / bundle.std
    state.reset(fill_value=normalized)
    env._phasor_phase_history = state
    env._phasor_phase_history_identity = bundle.identity
  return state


def reset_learned_phase_history(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  encoder_path: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset selected histories and seed them with the current mapped pose."""
  bundle = _load_phasor_encoder(env, encoder_path)
  asset: Entity = env.scene[asset_cfg.name]
  current = _joint_positions_in_order(asset, asset.data.joint_pos, bundle.joint_names)
  state = getattr(env, "_phasor_phase_history", None)
  history_identity = getattr(env, "_phasor_phase_history_identity", None)
  if state is None or history_identity != bundle.identity:
    _phase_history_for_bundle(env, bundle, current)
    return
  normalized = (current - bundle.mean) / bundle.std
  state.reset(env_ids, fill_value=normalized)


@torch.no_grad()
def learned_phase(
  env: ManagerBasedRlEnv,
  encoder_path: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Infer canonical ``(sin(phi), cos(phi))`` from ordered G1 joint history."""
  bundle = _load_phasor_encoder(env, encoder_path)
  asset: Entity = env.scene[asset_cfg.name]
  current = _joint_positions_in_order(asset, asset.data.joint_pos, bundle.joint_names)
  state = _phase_history_for_bundle(env, bundle, current)
  phase = state.update(
    current,
    control_step=int(env.common_step_counter),
    encoder=bundle.encoder,
    mean=bundle.mean,
    std=bundle.std,
  )

  command = env.command_manager.get_command("twist")
  if command is not None:
    stand_mask = torch.linalg.norm(command[:, :2], dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
  return phase
