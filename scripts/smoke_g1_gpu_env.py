"""Run one CUDA reset/step smoke test for the G1 flat velocity environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any


if not os.environ.get("CUDA_VISIBLE_DEVICES"):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# mjlab 1.2.0 accesses ``warp.context`` while Warp 1.13 exposes it through
# ``warp._src.context``. Match the compatibility setup used by train.py.
try:
  import warp as _warp

  if not hasattr(_warp, "context"):
    import warp._src.context as _warp_context

    _warp.context = _warp_context
except ImportError:
  pass

from src.runtime_compat import (
  configure_windows_ascii_temp_dir,
  configure_warp_ascii_paths,
)

WARP_TEMP_DIR = configure_windows_ascii_temp_dir()
WARP_PATHS = configure_warp_ascii_paths(_warp)

import torch
import tempfile
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401  Register the project task configurations.
from src.assets.robots.unitree_g1.g1_constants import G1_XML
from src.tasks.velocity.phasor.mujoco_io import (
  install_mjlab_scene_xml_unicode_compat,
)


def _tensor_leaves(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
  if isinstance(value, torch.Tensor):
    return [(prefix or "root", value)]
  if isinstance(value, Mapping):
    leaves: list[tuple[str, torch.Tensor]] = []
    for key, item in value.items():
      leaves.extend(_tensor_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    return leaves
  if isinstance(value, (tuple, list)):
    leaves = []
    for index, item in enumerate(value):
      leaves.extend(_tensor_leaves(item, f"{prefix}[{index}]"))
    return leaves
  return []


def _require_finite(label: str, value: Any) -> list[dict[str, Any]]:
  leaves = _tensor_leaves(value)
  if not leaves:
    raise AssertionError(f"{label} contains no torch tensors")
  summary = []
  for name, tensor in leaves:
    if not torch.isfinite(tensor).all().item():
      raise AssertionError(f"{label} tensor {name} contains NaN or infinity")
    summary.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
  return summary


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable to PyTorch")
  device = "cuda:0"
  if torch.cuda.device_count() < 1:
    raise RuntimeError("No visible CUDA device is available")

  install_mjlab_scene_xml_unicode_compat()
  configure_torch_backends()

  cfg = load_env_cfg("Unitree-G1-Flat")
  if not hasattr(cfg.scene, "num_envs"):
    raise RuntimeError("Environment config has no scene.num_envs field")
  cfg.scene.num_envs = 1
  cfg.seed = 12345

  env = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    reset_result = env.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    reset_summary = _require_finite("reset observation", reset_obs)

    action_manager = env.action_manager
    action_dim = getattr(action_manager, "action_dim", None)
    if action_dim is None:
      action_dim = getattr(action_manager, "total_action_dim", None)
    if action_dim is None:
      action = getattr(action_manager, "action", None)
      if isinstance(action, torch.Tensor) and action.ndim >= 2:
        action_dim = int(action.shape[-1])
    if not isinstance(action_dim, int) or action_dim <= 0:
      available = [name for name in dir(action_manager) if "dim" in name or name == "action"]
      raise RuntimeError(f"Cannot determine action dimension; available fields: {available}")

    actions = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
    step_summaries = []
    for step_index in range(3):
      step_result = env.step(actions)
      if not isinstance(step_result, tuple) or len(step_result) < 2:
        raise RuntimeError(f"Unexpected env.step result: {type(step_result)!r}")
      step_obs, reward = step_result[0], step_result[1]
      step_summaries.append(
        {
          "step": step_index + 1,
          "observations": _require_finite(f"step {step_index + 1} observation", step_obs),
          "reward": _require_finite(f"step {step_index + 1} reward", reward),
        }
      )
    torch.cuda.synchronize()

    mjlab_spec = importlib.util.find_spec("mjlab")
    if mjlab_spec is None or mjlab_spec.origin is None:
      raise RuntimeError("Cannot locate installed mjlab package")
    scene_xml = Path(mjlab_spec.origin).parent / "scene" / "scene.xml"
    report = {
      "status": "pass",
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "task": "Unitree-G1-Flat",
      "device": device,
      "gpu_name": torch.cuda.get_device_name(0),
      "torch_version": torch.__version__,
      "num_envs": 1,
      "step_count": len(step_summaries),
      "warp_temp_dir": WARP_TEMP_DIR,
      "warp_paths": WARP_PATHS,
      "python_temp_dir": tempfile.gettempdir(),
      "action_shape": list(actions.shape),
      "reset_observations": reset_summary,
      "steps": step_summaries,
      "xml": {
        "mjlab_scene": str(scene_xml),
        "mjlab_scene_sha256": _sha256(scene_xml),
        "g1_model": str(G1_XML),
        "g1_model_sha256": _sha256(G1_XML),
        "g1_mesh_asset_count": len(cfg.scene.entities["robot"].spec_fn().assets),
      },
      "smoke_script_sha256": _sha256(Path(__file__).resolve()),
      "runtime_compat_sha256": _sha256(Path("src/runtime_compat.py")),
    }
    report_dir = Path("results/g1_phase_gate_v2")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"gpu_smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
  finally:
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
