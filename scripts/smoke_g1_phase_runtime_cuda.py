"""Smoke the learned G1 phase actor/reward path on one CUDA environment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.smoke_g1_gpu_env import _require_finite, _sha256
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401 - register the project task configurations.
from src.tasks.velocity.mdp.observations import _load_phasor_encoder
from src.tasks.velocity.phasor.mujoco_io import install_mjlab_scene_xml_unicode_compat


TASK_ID = "Unitree-G1-Flat-Phase-Joint"
CHECKPOINT = Path("results/g1_phase_dataset_20261019/phase_encoder_runtime_v1.pt")
TEMPLATE = Path("results/g1_phase_dataset_20261019/g1_reward_template_v1.json")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device cuda:0 is required for the learned-phase smoke")
    install_mjlab_scene_xml_unicode_compat()
    configure_torch_backends()

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 1
    cfg.seed = 20261019
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        bundle = _load_phasor_encoder(env, str(CHECKPOINT.resolve(strict=True)))
        forward_calls = 0

        def count_forward(_module, _inputs, _output):
            nonlocal forward_calls
            forward_calls += 1

        hook = bundle.encoder.register_forward_hook(count_forward)
        try:
            reset_result = env.reset()
            reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            reset_summary = _require_finite("reset observation", reset_obs)
            history = getattr(env, "_phasor_phase_history", None)
            if history is None or tuple(history.buffer.shape) != (1, 10, 29):
                raise AssertionError("learned-phase reset did not initialize a [1, 10, 29] history")

            action_manager = env.action_manager
            action_dim = getattr(action_manager, "action_dim", None)
            if action_dim is None:
                action_dim = getattr(action_manager, "total_action_dim", None)
            if action_dim is None:
                action_dim = int(action_manager.action.shape[-1])
            actions = torch.zeros((1, action_dim), dtype=torch.float32, device="cuda:0")

            steps = []
            for index in range(3):
                result = env.step(actions)
                observation, reward = result[0], result[1]
                steps.append({
                    "step": index + 1,
                    "observation": _require_finite(f"step {index + 1} observation", observation),
                    "reward": _require_finite(f"step {index + 1} reward", reward),
                    "phase_state": _require_finite(f"step {index + 1} phase", history.phase),
                })
            torch.cuda.synchronize()
            if forward_calls < 1:
                raise AssertionError("learned-phase encoder was not executed")
        finally:
            hook.remove()

        report = {
            "schema": "g1.phase_runtime_cuda_smoke.v1",
            "status": "pass",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": TASK_ID,
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "num_envs": 1,
            "step_count": len(steps),
            "encoder_forward_calls": forward_calls,
            "history_shape": list(history.buffer.shape),
            "phase_shape": list(history.phase.shape),
            "checkpoint_path": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "reward_template_path": str(TEMPLATE.resolve()),
            "reward_template_sha256": _sha256(TEMPLATE),
            "reset_observations": reset_summary,
            "steps": steps,
            "smoke_script_sha256": _sha256(Path(__file__).resolve()),
        }
        report_dir = Path("results/g1_phase_gate_v2")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"phase_runtime_cuda_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
