"""Check whether a frozen PHASOR encoder follows contacts during G1 gait.

This is a pre-training validity diagnostic, not a policy-performance evaluation.
It runs a frozen fixed-clock baseline and compares its contact trace with both
the fixed gait clock and the PHASOR latent on the same simulated states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_mujoco_path_loading() -> None:
    """Read XML through Python when MuJoCo's C loader cannot open Unicode paths."""
    import mujoco

    mujoco.MjSpec.from_file = lambda path: mujoco.MjSpec.from_string(
        Path(path).read_text(encoding="utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=PROJECT_ROOT / "src/tasks/velocity/phasor/phasor_encoder_v2.pt",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--vx-min", type=float, default=0.5)
    parser.add_argument("--vx-max", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/phasor_gait_contact_validation_2026-10-15",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    encoder_path = args.encoder.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not encoder_path.is_file():
        raise FileNotFoundError(encoder_path)
    if args.num_envs < 1 or args.steps < 2:
        raise ValueError("num-envs must be positive and steps must be at least 2")
    if not 0.0 < args.vx_min <= args.vx_max:
        raise ValueError("the validation command range must be strictly forward")

    safe_mujoco_path_loading()
    import warp

    if not hasattr(warp, "context"):
        warp.context = warp._src.context

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from src.tasks.velocity.mdp.observations import (
        learned_phase,
        reset_learned_phase_history,
    )
    from src.tasks.velocity.phasor.diagnostics import phase_contact_summary

    task_id = "Unitree-G1-Flat"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    command_cfg = env_cfg.commands["twist"]
    command_cfg.ranges.lin_vel_x = (args.vx_min, args.vx_max)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)

    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.seed = args.seed
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cpu")
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")
    wrapped_env.reset()

    phases: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    clocks: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    episode_id = torch.zeros(args.num_envs, dtype=torch.int64)
    contact_sensor = raw_env.scene["feet_ground_contact"]

    try:
        for _ in range(args.steps):
            observation = wrapped_env.get_observations()
            phase = learned_phase(raw_env, str(encoder_path))
            contact_time = contact_sensor.data.current_contact_time
            if contact_time is None:
                raise RuntimeError("feet_ground_contact has no contact-time data")
            contact = contact_time > 0.0

            elapsed = raw_env.episode_length_buf.to(torch.float64) * raw_env.step_dt
            clock_cycle = torch.remainder(elapsed / 0.6, 1.0)
            clock = torch.stack(
                (torch.cos(clock_cycle * (2.0 * torch.pi)),
                 torch.sin(clock_cycle * (2.0 * torch.pi))),
                dim=-1,
            )
            command = raw_env.command_manager.get_command("twist")
            if command is None:
                raise RuntimeError("twist command is unavailable")

            phases.append(phase.cpu().numpy().copy())
            contacts.append(contact.cpu().numpy().copy())
            clocks.append(clock.cpu().numpy().copy())
            commands.append(command.cpu().numpy().copy())
            episode_ids.append(episode_id.numpy().copy())

            with torch.inference_mode():
                action = policy(observation)
            _, _, terminated, truncated, _ = raw_env.step(action)
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel():
                reset_learned_phase_history(raw_env, done_ids)
                episode_id[done_ids.cpu()] += 1
    finally:
        wrapped_env.close()

    phase_array = np.stack(phases)
    contact_array = np.stack(contacts).astype(bool)
    clock_array = np.stack(clocks)
    command_array = np.stack(commands)
    episode_array = np.stack(episode_ids)
    summaries: list[dict[str, object]] = []

    for env_id in range(args.num_envs):
        for current_episode in np.unique(episode_array[:, env_id]):
            mask = episode_array[:, env_id] == current_episode
            if mask.sum() < 2:
                continue
            phase_seq = phase_array[mask, env_id : env_id + 1]
            contact_seq = contact_array[mask, env_id : env_id + 1]
            clock_seq = clock_array[mask, env_id : env_id + 1]
            command_seq = command_array[mask, env_id]
            summaries.append(
                {
                    "env_id": env_id,
                    "episode_id": int(current_episode),
                    "steps": int(mask.sum()),
                    "mean_vx_command_mps": float(command_seq[:, 0].mean()),
                    "min_vx_command_mps": float(command_seq[:, 0].min()),
                    "max_vx_command_mps": float(command_seq[:, 0].max()),
                    "learned_phase": phase_contact_summary(
                        phase_seq, contact_seq, offsets=(0.0, 0.5), threshold=0.56
                    ),
                    "fixed_clock": phase_contact_summary(
                        clock_seq, contact_seq, offsets=(0.0, 0.5), threshold=0.56
                    ),
                }
            )

    if not summaries:
        raise RuntimeError("no episode contained enough samples for analysis")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = args.output_dir / "trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        phase=phase_array,
        fixed_clock=clock_array,
        contacts=contact_array,
        commands=command_array,
        episode_ids=episode_array,
    )
    result = {
        "task_id": task_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "encoder": str(encoder_path),
        "encoder_sha256": sha256(encoder_path),
        "seed": args.seed,
        "device": "cpu",
        "num_envs": args.num_envs,
        "steps_per_env": args.steps,
        "control_dt_s": float(raw_env.step_dt),
        "command_range_vx_mps": [args.vx_min, args.vx_max],
        "command_range_vy_mps": [0.0, 0.0],
        "command_range_yaw_rate_radps": [0.0, 0.0],
        "comparison": "same frozen fixed-clock policy states; learned latent vs 0.6-s clock",
        "threshold": 0.56,
        "offsets": [0.0, 0.5],
        "episodes": summaries,
        "trajectory_file": trajectory_path.name,
        "trajectory_sha256": sha256(trajectory_path),
        "interpretation_limit": (
            "One frozen baseline checkpoint and a finite set of simulated episodes; "
            "this is a phase/contact validity diagnostic, not evidence of policy efficacy."
        ),
    }
    result_path = args.output_dir / "summary.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
