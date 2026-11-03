"""Closed-loop evaluation for experiment B (secondary outcome).

Same protocol as the 2026-10-16 pilot, applied to the rough-terrain tasks: one
environment, a fixed forward command, a fixed 20 s horizon, fixed episode seeds,
and a terminated episode scored as zero forward velocity for the rest of the
horizon.  Reusing the pilot's protocol is deliberate — it keeps the secondary
outcome comparable across the two studies instead of inventing a third scale.

This is the SECONDARY outcome.  The primary outcome (fall rate) is read from
TensorBoard by ``scripts/analyze_g1_rough_equalbudget.py`` and needs no rollout.

Usage:
    python scripts/evaluate_g1_rough_equalbudget.py --runs runs.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT = ROOT / "results" / "phasor_rough_equalbudget_20261020"
EPISODE_SEEDS = (20261101, 20261102, 20261103, 20261104)
FORWARD_COMMAND_MPS = 0.5
HORIZON_SECONDS = 20.0

TASKS = {"baseline": "Unitree-G1-Rough", "phasor": "Unitree-G1-Rough-PHASOR"}


def final_checkpoint(log_dir: Path) -> Path:
    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {log_dir}")
    return max(checkpoints, key=lambda path: int(path.stem.split("_")[1]))


def import_runtime():
    """Import numpy and torch with the project's tasks registered.

    On Windows this defers to the 2026-10-16 pilot harness unchanged: its
    ASCII-path bootstrap is required there because a non-ASCII user profile
    breaks NVRTC, and the harness owns the DOS device aliases that fix it.

    On POSIX the harness raises by design
    ("The recorded MuJoCo/Warp path workaround requires Windows"), which made
    this script unrunnable on any Linux host — including the rented box that
    experiment B trains on. What is actually needed off Windows is the rest of
    what ``import_project()`` does, so it is done here directly:

    * install the ``warp.context`` alias, because mjlab 1.2.0 reads
      ``warp.context.runtime`` and warp-lang 1.13 only exposes
      ``warp._src.context``;
    * call ``configure_torch_backends()``, as ``scripts/train.py`` does;
    * import ``src.tasks``, which is what registers the ``Unitree-G1-*`` ids —
      without it ``load_env_cfg`` raises ``KeyError: 'Unitree-G1-Rough'``.

    Verification status: the Windows branch is the code that ran before this
    change, unchanged. The POSIX branch is those same calls lifted from
    ``results/phasor_remote_pilot_20261020/rerun_bjb1/`` — the pilot driver that
    ran both arms through ``evaluate_one`` on the rented Linux box on 2026-10-20
    — but this function has **not** been executed as part of this script on
    POSIX. It compiles; treat the first Linux run as a test.

    ``MUJOCO_GL`` is set *after* ``import mujoco`` on purpose. Setting it first
    makes ``mujoco/gl_context.py`` pull in ``mujoco.egl``, which needs a generic
    ``libEGL.so.1``; images that ship only ``libEGL_nvidia.so.0`` then fail
    inside PyOpenGL before any simulator exists. ``scripts/train.py`` gets this
    ordering right by importing mjlab at module scope and setting the variable
    later inside ``launch_training``.
    """
    import os

    import numpy
    import torch

    if os.name == "nt":
        import importlib.util

        harness_path = ROOT / "results" / "phasor_pilot_20261016" / "run_pilot.py"
        spec = importlib.util.spec_from_file_location("_phasor_pilot_harness", harness_path)
        harness = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(harness)
        return harness.import_project()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import warp

    if not hasattr(warp, "context"):
        import warp._src.context as warp_context

        warp.context = warp_context
    from src.runtime_compat import configure_warp_ascii_paths

    configure_warp_ascii_paths(warp)

    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    import src.tasks  # noqa: F401  (registers the Unitree-G1-* task ids)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get("LOCAL_RANK", "0")

    return numpy, torch


def evaluate_one(condition: str, seed: int, log_dir: Path, np, torch) -> dict:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    task_id = TASKS[condition]
    checkpoint = final_checkpoint(log_dir)
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = HORIZON_SECONDS
    command_cfg = env_cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (HORIZON_SECONDS + 1.0, HORIZON_SECONDS + 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.seed = seed

    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.seed = seed
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    dt = float(raw_env.step_dt)
    horizon_steps = int(round(env_cfg.episode_length_s / dt))
    episodes = []
    try:
        for episode_seed in EPISODE_SEEDS:
            command_cfg.ranges.lin_vel_x = (FORWARD_COMMAND_MPS, FORWARD_COMMAND_MPS)
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
            raw_env.reset(seed=episode_seed)
            robot = raw_env.scene["robot"]
            errors: list[float] = []
            fell = False
            fall_step = None
            for step in range(horizon_steps):
                command = raw_env.command_manager.get_command("twist")
                current_command = float(command[0, 0].item())
                current_vx = float(robot.data.root_link_lin_vel_b[0, 0].item())
                errors.append(abs(current_vx - current_command))
                obs = wrapped_env.get_observations()
                with torch.inference_mode():
                    action = policy(obs)
                _, _, terminated, _, _ = raw_env.step(action)
                if bool(terminated[0].item()):
                    fell = True
                    fall_step = step + 1
                    errors.extend([abs(FORWARD_COMMAND_MPS)] * (horizon_steps - len(errors)))
                    break
            errors_np = np.asarray(errors, dtype=np.float64)
            if errors_np.size != horizon_steps:
                raise RuntimeError(f"episode {episode_seed} covered {errors_np.size} steps")
            episodes.append(
                {
                    "episode_seed": episode_seed,
                    "mae_forward_velocity_mps": float(errors_np.mean()),
                    "rmse_forward_velocity_mps": float(np.sqrt(np.mean(errors_np**2))),
                    "fell": fell,
                    "fall_step": fall_step,
                }
            )
    finally:
        wrapped_env.close()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return {
        "schema": "g1.rough_equalbudget_evaluation.v1",
        "condition": condition,
        "ppo_seed": seed,
        "task_id": task_id,
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "mae_forward_velocity_mps": float(np.mean([e["mae_forward_velocity_mps"] for e in episodes])),
        "rmse_forward_velocity_mps": float(np.mean([e["rmse_forward_velocity_mps"] for e in episodes])),
        "fall_fraction": float(np.mean([e["fell"] for e in episodes])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve runs and checkpoints, then stop without building a simulator",
    )
    args = parser.parse_args()

    if args.dry_run:
        runs = json.loads(Path(args.runs).read_text(encoding="utf-8"))
        missing = 0
        for condition in ("baseline", "phasor"):
            for seed_str, log_dir in runs.get(condition, {}).items():
                checkpoint = final_checkpoint(Path(log_dir))
                print(f"{condition} seed {seed_str}: {checkpoint}", flush=True)
                missing += 0 if checkpoint.is_file() else 1
        print(f"dry run: {missing} missing checkpoint(s)")
        return 0 if missing == 0 else 1

    np, torch = import_runtime()
    runs = json.loads(Path(args.runs).read_text(encoding="utf-8"))

    summaries = []
    for condition in ("baseline", "phasor"):
        for seed_str, log_dir in runs[condition].items():
            output_dir = EXPERIMENT / "evaluations" / f"{condition}_seed_{seed_str}"
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "summary.json"
            if summary_path.is_file():
                print(f"cached {summary_path.relative_to(ROOT)}", flush=True)
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            summary = evaluate_one(condition, int(seed_str), Path(log_dir), np, torch)
            summary_path.write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                f"{condition} seed {seed_str}: MAE {summary['mae_forward_velocity_mps']:.4f} "
                f"falls {summary['fall_fraction']:.2f} -> {summary_path.relative_to(ROOT)}",
                flush=True,
            )
            summaries.append(summary)

    aggregate = EXPERIMENT / "evaluation_summary.json"
    aggregate.write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {aggregate.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
