"""Collect G1 contact-labelled phase data, train, and evaluate a frozen encoder.

The locked test seed is intentionally unreadable until the validation summary
passes the user-approved phase/contact gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RUN_ROOT = PROJECT_ROOT / "results/phasor_contact_supervised_v3/run_20261016"

TRAIN_SEEDS = (20261021, 20261022, 20261023)
VALIDATION_SEED = 20261024
LOCKED_TEST_SEED = 20261025
ALL_SEEDS = (*TRAIN_SEEDS, VALIDATION_SEED, LOCKED_TEST_SEED)
FROZEN_POLICY = PROJECT_ROOT / (
    "logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt"
)
WINDOW = 10
JOINT_DIM = 29
CONTROL_HZ = 50.0
CONTACT_THRESHOLD = 0.56
CONTACT_OFFSETS = (0.0, 0.5)
CONTACT_GATE = 0.90
MIN_SPEED = 0.1
EXPECTED_POLICY_SHA256 = (
    "949fae33632b9a3592e6346a6155eea059ee2fd39487207a0a0cea5d35edd8b4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def git_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty}


def environment_provenance(device: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "device": device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "experiment_script_sha256": sha256_file(Path(__file__)),
        **git_provenance(),
    }


def safe_mujoco_path_loading() -> None:
    """Read MuJoCo XML through Python when its C loader rejects Unicode paths."""
    import mujoco

    mujoco.MjSpec.from_file = lambda path: mujoco.MjSpec.from_string(
        Path(path).read_text(encoding="utf-8")
    )


def unlock_test(validation_summary_path: Path) -> dict[str, Any]:
    summary_path = validation_summary_path.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evaluation_split") != "validation":
        raise RuntimeError("The supplied summary is not a validation result")
    if summary.get("gate_pass") is not True:
        raise RuntimeError("Validation failed; the locked test seed stays unopened")
    return {
        "validation_summary": str(summary_path),
        "validation_summary_sha256": sha256_file(summary_path),
        "validation_checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def collect(args: argparse.Namespace) -> None:
    if args.seed not in ALL_SEEDS:
        raise ValueError(f"Seed must be one of {ALL_SEEDS}")
    unlock: dict[str, Any] | None = None
    if args.seed == LOCKED_TEST_SEED:
        if args.validation_summary is None:
            raise RuntimeError("Collecting the locked test requires a passing validation summary")
        unlock = unlock_test(args.validation_summary)

    checkpoint = args.policy_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy_hash = sha256_file(checkpoint)
    if policy_hash != EXPECTED_POLICY_SHA256:
        raise RuntimeError(
            f"Frozen policy hash mismatch: expected {EXPECTED_POLICY_SHA256}, got {policy_hash}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    safe_mujoco_path_loading()
    import warp

    if not hasattr(warp, "context"):
        warp.context = warp._src.context
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg("Unitree-G1-Flat", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    command_cfg = env_cfg.commands["twist"]
    command_cfg.ranges.lin_vel_x = (0.5, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)

    agent_cfg = load_rl_cfg("Unitree-G1-Flat")
    agent_cfg.seed = args.seed
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls("Unitree-G1-Flat") or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cpu")
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")
    wrapped_env.reset()

    contact_sensor = raw_env.scene["feet_ground_contact"]
    slot_names = list(dict.fromkeys(slot.primary_name for slot in contact_sensor._slots))
    foot_index: dict[str, int] = {}
    for index, name in enumerate(slot_names):
        lowered = name.lower()
        if "left" in lowered and "foot" not in foot_index:
            foot_index["left"] = index
        elif "right" in lowered and "right" not in foot_index:
            foot_index["right"] = index
    if set(foot_index) != {"left", "right"}:
        raise RuntimeError(f"Could not identify left/right contact sensor slots: {slot_names}")
    if contact_sensor.data.current_contact_time is None:
        raise RuntimeError("feet_ground_contact is missing contact-time tracking")

    joints: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    current_episode_id = torch.zeros(args.num_envs, dtype=torch.int64)
    robot = raw_env.scene["robot"]
    try:
        for _ in range(args.steps):
            qpos = robot.data.joint_pos
            current_contact_time = contact_sensor.data.current_contact_time
            command = raw_env.command_manager.get_command("twist")
            if command is None:
                raise RuntimeError("twist command is unavailable")
            if qpos.shape[1] != JOINT_DIM:
                raise RuntimeError(f"Expected {JOINT_DIM} joint positions, got {qpos.shape[1]}")
            joints.append(qpos.detach().cpu().numpy().copy())
            contacts.append(
                np.stack(
                    (
                        (current_contact_time[:, foot_index["left"]] > 0.0).cpu().numpy(),
                        (current_contact_time[:, foot_index["right"]] > 0.0).cpu().numpy(),
                    ),
                    axis=-1,
                )
            )
            commands.append(command.detach().cpu().numpy().copy())
            episode_ids.append(current_episode_id.numpy().copy())
            with torch.inference_mode():
                action = policy(wrapped_env.get_observations())
            _, _, terminated, truncated, _ = raw_env.step(action)
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel():
                current_episode_id[done_ids.cpu()] += 1
    finally:
        wrapped_env.close()

    joint_array = np.stack(joints).astype(np.float32, copy=False)
    contact_array = np.stack(contacts).astype(bool, copy=False)
    command_array = np.stack(commands).astype(np.float32, copy=False)
    episode_array = np.stack(episode_ids).astype(np.int64, copy=False)
    control_dt = float(raw_env.step_dt)
    if not np.isclose(control_dt, 1.0 / CONTROL_HZ, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"Expected 50 Hz control, got dt={control_dt}")

    trajectory_path = output_dir / "trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        joint_pos=joint_array,
        contacts=contact_array,
        commands=command_array,
        episode_ids=episode_array,
        foot_order=np.asarray(("left", "right")),
    )
    summary = {
        "schema": "phasor.contact_trajectory.v1",
        "seed": args.seed,
        "split": "train"
        if args.seed in TRAIN_SEEDS
        else "validation"
        if args.seed == VALIDATION_SEED
        else "test",
        "task_id": "Unitree-G1-Flat",
        "policy_checkpoint": str(checkpoint),
        "policy_checkpoint_sha256": policy_hash,
        "simulator_seed": args.seed,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps,
        "control_dt_s": control_dt,
        "control_hz": 1.0 / control_dt,
        "command_ranges": {"vx_mps": [0.5, 1.0], "vy_mps": [0.0, 0.0], "yaw_rate_radps": [0.0, 0.0]},
        "joint_shape": list(joint_array.shape),
        "contact_shape": list(contact_array.shape),
        "episode_id_counts_by_env": [
            int(len(np.unique(episode_array[:, env_id]))) for env_id in range(args.num_envs)
        ],
        "foot_order": ["left", "right"],
        "sensor_primary_names": slot_names,
        "mean_command_vx_by_env": command_array[:, :, 0].mean(axis=0).tolist(),
        "moving_fraction_by_env": (
            np.linalg.norm(command_array[:, :, :2], axis=-1) >= MIN_SPEED
        ).mean(axis=0).tolist(),
        "trajectory_file": trajectory_path.name,
        "trajectory_sha256": sha256_file(trajectory_path),
        "test_unlock": unlock,
        "environment": environment_provenance("cpu"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))


@dataclass
class TrajectoryExamples:
    seed: int
    env_id: int
    episode_id: int
    windows: np.ndarray
    target_cycles: np.ndarray
    contacts: np.ndarray
    steps: np.ndarray
    touchdown_steps: np.ndarray
    touchdown_feet: np.ndarray
    valid_frame_count: int
    valid_cycle_count: int
    invalid_reason: str | None


def phase_labels(
    contacts: np.ndarray, commands: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, Any]]:
    """Interpolate a forward phase only across valid alternating touchdowns."""
    contact = np.asarray(contacts, dtype=bool)
    command = np.asarray(commands, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError("contacts must have shape [time, left/right]")
    if command.ndim != 2 or command.shape[0] != contact.shape[0] or command.shape[1] < 2:
        raise ValueError("commands must have shape [time, >=2]")
    moving = np.linalg.norm(command[:, :2], axis=-1) >= MIN_SPEED
    previous = np.vstack((np.zeros((1, 2), dtype=bool), contact[:-1]))
    touchdown = contact & ~previous
    events: list[tuple[int, int] | None] = []
    for step, row in enumerate(touchdown):
        feet = np.flatnonzero(row)
        if len(feet) > 1:
            events.append(None)
        elif len(feet) == 1:
            foot = int(feet[0])
            events.append((step, foot) if moving[step] else None)

    runs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def finish_run() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for event in events:
        if event is None:
            finish_run()
            continue
        step, foot = event
        if current:
            previous_step, previous_foot = current[-1]
            interval_moving = bool(moving[previous_step : step + 1].all())
            if foot == previous_foot or not interval_moving:
                finish_run()
        current.append(event)
    finish_run()

    target = np.full(contact.shape[0], np.nan, dtype=np.float32)
    valid = np.zeros(contact.shape[0], dtype=bool)
    valid_events: list[tuple[int, int]] = []
    cycle_count = 0
    for run in runs:
        if len(run) < 3:
            continue
        valid_events.extend(run)
        cycle_count += (len(run) - 1) // 2
        first_foot = run[0][1]
        base = 0.0 if first_foot == 0 else 0.5
        for event_index, ((start, _), (end, _)) in enumerate(zip(run, run[1:])):
            start_phase = base + event_index * 0.5
            count = end - start
            if count <= 0:
                continue
            target[start : end + 1] = np.linspace(
                start_phase, start_phase + 0.5, count + 1, dtype=np.float32
            )
            valid[start : end + 1] = True

    invalid_reason = None if cycle_count > 0 else "no_complete_alternating_touchdown_cycle"
    info = {
        "touchdown_count": int(touchdown.sum()),
        "touchdown_counts_by_foot": touchdown.sum(axis=0).astype(int).tolist(),
        "alternating_run_lengths": [len(run) for run in runs],
        "valid_cycle_count": int(cycle_count),
        "valid_phase_frame_count": int(valid.sum()),
        "moving_fraction": float(moving.mean()),
        "invalid_reason": invalid_reason,
    }
    return target, valid, valid_events, info


def read_trajectory(root: Path, seed: int, expected_split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = root / f"seed_{seed}"
    data_path = directory / "trajectories.npz"
    summary_path = directory / "summary.json"
    if not data_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing collected seed artifact under {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("seed") != seed or summary.get("split") != expected_split:
        raise RuntimeError(f"Seed/split mismatch in {summary_path}")
    if summary.get("trajectory_sha256") != sha256_file(data_path):
        raise RuntimeError(f"Trajectory hash mismatch: {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, summary


def make_examples(
    arrays: dict[str, np.ndarray], seed: int, window: int = WINDOW
) -> tuple[list[TrajectoryExamples], list[dict[str, Any]], list[np.ndarray]]:
    qpos = np.asarray(arrays["joint_pos"], dtype=np.float32)
    contacts = np.asarray(arrays["contacts"], dtype=bool)
    commands = np.asarray(arrays["commands"], dtype=np.float32)
    if qpos.ndim != 3 or qpos.shape[-1] != JOINT_DIM:
        raise ValueError(f"joint_pos must have shape [time, env, {JOINT_DIM}]")
    if contacts.shape != (*qpos.shape[:2], 2) or commands.shape[:2] != qpos.shape[:2]:
        raise ValueError("contact/command arrays do not match joint trajectory shape")
    episode_ids = np.asarray(
        arrays.get("episode_ids", np.zeros(qpos.shape[:2], dtype=np.int64)),
        dtype=np.int64,
    )
    if episode_ids.shape != qpos.shape[:2]:
        raise ValueError("episode_ids must have shape [time, env]")

    examples: list[TrajectoryExamples] = []
    records: list[dict[str, Any]] = []
    norm_frames: list[np.ndarray] = []
    for env_id in range(qpos.shape[1]):
        for episode_id in np.unique(episode_ids[:, env_id]):
            frame_indices = np.flatnonzero(episode_ids[:, env_id] == episode_id)
            if len(frame_indices) == 0:
                continue
            if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
                raise ValueError("episode frames are not a contiguous interval")
            frame_start, frame_end = int(frame_indices[0]), int(frame_indices[-1]) + 1
            episode_qpos = qpos[frame_start:frame_end, env_id]
            episode_contacts = contacts[frame_start:frame_end, env_id]
            episode_commands = commands[frame_start:frame_end, env_id]
            target, valid, events, info = phase_labels(episode_contacts, episode_commands)
            moving = np.linalg.norm(episode_commands[:, :2], axis=-1) >= MIN_SPEED
            step_ids: list[int] = []
            windows: list[np.ndarray] = []
            phase_targets: list[float] = []
            sample_contacts: list[np.ndarray] = []
            for step in range(window - 1, len(episode_qpos)):
                start = step - window + 1
                if not valid[step] or not moving[start : step + 1].all():
                    continue
                windows.append(episode_qpos[start : step + 1])
                phase_targets.append(float(target[step]))
                sample_contacts.append(episode_contacts[step])
                step_ids.append(step)
            invalid_reason = info["invalid_reason"]
            if not windows and invalid_reason is None:
                invalid_reason = "no_valid_input_window"
            event_steps = np.asarray([event[0] for event in events], dtype=np.int64)
            event_feet = np.asarray([event[1] for event in events], dtype=np.int64)
            examples.append(
                TrajectoryExamples(
                    seed=seed,
                    env_id=env_id,
                    episode_id=int(episode_id),
                    windows=np.asarray(windows, dtype=np.float32).reshape(-1, window, JOINT_DIM),
                    target_cycles=np.asarray(phase_targets, dtype=np.float32),
                    contacts=np.asarray(sample_contacts, dtype=bool).reshape(-1, 2),
                    steps=np.asarray(step_ids, dtype=np.int64),
                    touchdown_steps=event_steps,
                    touchdown_feet=event_feet,
                    valid_frame_count=int(valid.sum()),
                    valid_cycle_count=int(info["valid_cycle_count"]),
                    invalid_reason=invalid_reason,
                )
            )
            records.append(
                {
                    "seed": seed,
                    "env_id": env_id,
                    "episode_id": int(episode_id),
                    **info,
                    "example_count": len(windows),
                    "invalid_reason": invalid_reason,
                }
            )
            if valid.any():
                norm_frames.append(episode_qpos[valid])
    return examples, records, norm_frames


def target_vectors(cycles: torch.Tensor) -> torch.Tensor:
    angles = cycles * (2.0 * torch.pi)
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)


def contiguous_segments(steps: np.ndarray) -> list[np.ndarray]:
    if len(steps) == 0:
        return []
    breaks = np.flatnonzero(np.diff(steps) != 1) + 1
    return [segment for segment in np.split(np.arange(len(steps)), breaks) if len(segment)]


def trajectory_metrics(example: TrajectoryExamples, prediction: np.ndarray) -> dict[str, Any]:
    from src.tasks.velocity.phasor.diagnostics import circular_coverage_rad, phase_contact_summary

    if len(prediction) != len(example.steps):
        raise ValueError("prediction/sample length mismatch")
    segments = contiguous_segments(example.steps)
    longest = max(segments, key=len) if segments else np.empty(0, dtype=np.int64)
    angle = np.arctan2(prediction[:, 0], prediction[:, 1])
    unwrapped = np.unwrap(angle[longest]) if len(longest) else np.empty(0)
    net_delta = float(unwrapped[-1] - unwrapped[0]) if len(unwrapped) else 0.0
    phase_coverage = circular_coverage_rad(angle) if len(angle) else 0.0
    summary = phase_contact_summary(
        prediction[:, None, :],
        example.contacts[:, None, :],
        offsets=CONTACT_OFFSETS,
        threshold=CONTACT_THRESHOLD,
    ) if len(prediction) else {"contact_agreement": 0.0}

    sample_index = {int(step): i for i, step in enumerate(example.steps)}
    unwrapped_by_step = {
        int(example.steps[index]): float(phase)
        for index, phase in zip(longest, unwrapped)
    }
    segment_start = int(example.steps[longest[0]]) if len(longest) else -1
    segment_end = int(example.steps[longest[-1]]) if len(longest) else -1
    valid_touches = [
        (int(step), int(foot))
        for step, foot in zip(example.touchdown_steps, example.touchdown_feet)
        if int(step) in sample_index and segment_start <= int(step) <= segment_end
    ]
    # Unwrap the dense control-step sequence before sampling touchdowns. Sparse
    # alternating-foot events are separated by approximately pi radians, where
    # np.unwrap cannot distinguish forward from backward progression.
    touch_unwrapped = np.asarray(
        [unwrapped_by_step[step] for step, _ in valid_touches], dtype=np.float64
    )
    touchdown_order_preserved = bool(
        len(touch_unwrapped) >= 3 and np.all(np.diff(touch_unwrapped) > 0.0)
    )
    cycle_pass = net_delta >= 2.0 * np.pi and touchdown_order_preserved
    return {
        "seed": example.seed,
        "env_id": example.env_id,
        "episode_id": example.episode_id,
        "valid_phase_frames": int(len(prediction)),
        "valid_cycles": example.valid_cycle_count,
        "invalid_reason": example.invalid_reason,
        "contact_agreement": float(summary["contact_agreement"]),
        "contact_agreement_by_foot": summary.get("contact_agreement_by_foot"),
        "circular_coverage_rad": float(phase_coverage),
        "longest_contiguous_segment_steps": int(len(longest)),
        "positive_net_phase_advance_rad": net_delta,
        "positive_net_windings": float(net_delta / (2.0 * np.pi)),
        "touchdown_count_with_predictions": int(len(valid_touches)),
        "touchdown_counts_by_foot_with_predictions": [
            int(sum(foot == foot_index for _, foot in valid_touches))
            for foot_index in range(2)
        ],
        "touchdown_order_preserved": touchdown_order_preserved,
        "full_time_ordered_cycle": bool(cycle_pass),
        "mean_net_to_total_angular_travel": float(summary.get("mean_net_to_total_angular_travel", 0.0)),
    }


def evaluate_examples(
    model: torch.nn.Module,
    trajectories: list[TrajectoryExamples],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    model.eval()
    all_prediction: list[np.ndarray] = []
    per_trajectory: list[dict[str, Any]] = []
    phase_losses: list[float] = []
    reconstruction_sse = 0.0
    reconstruction_count = 0
    with torch.inference_mode():
        for example in trajectories:
            windows = torch.as_tensor(example.windows, dtype=torch.float32, device=device)
            if len(windows) == 0:
                prediction = np.empty((0, 2), dtype=np.float32)
                all_prediction.append(prediction)
                per_trajectory.append(trajectory_metrics(example, prediction))
                continue
            normalized = (windows - mean) / std
            outputs: list[torch.Tensor] = []
            for start in range(0, len(normalized), batch_size):
                batch = normalized[start : start + batch_size]
                phase = model.encoder(batch)
                decoded = model.decoder(phase)
                targets = target_vectors(
                    torch.as_tensor(
                        example.target_cycles[start : start + len(batch)],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                phase_losses.append(float((1.0 - (phase * targets).sum(dim=-1)).mean().item()))
                reconstruction_sse += float(F.mse_loss(decoded, batch, reduction="sum").item())
                reconstruction_count += int(batch.numel())
                outputs.append(phase.cpu())
            prediction = torch.cat(outputs, dim=0).numpy()
            all_prediction.append(prediction)
            per_trajectory.append(trajectory_metrics(example, prediction))
    agreements = [item["contact_agreement"] for item in per_trajectory if item["valid_phase_frames"]]
    full_cycles = [item["full_time_ordered_cycle"] for item in per_trajectory if item["valid_phase_frames"]]
    valid_all = bool(per_trajectory) and all(
        trajectory.invalid_reason is None for trajectory in trajectories
    )
    contact_macro = float(np.mean(agreements)) if agreements else 0.0
    full_cycle_all = bool(full_cycles) and all(full_cycles)
    metrics = {
        "contact_agreement_macro_by_trajectory": contact_macro,
        "contact_agreement_by_trajectory": agreements,
        "all_trajectories_have_full_time_ordered_cycle": full_cycle_all,
        "all_trajectories_have_valid_contact_cycles": valid_all,
        "phase_loss": float(np.mean(phase_losses)) if phase_losses else None,
        "validation_reconstruction_mse": reconstruction_sse / reconstruction_count if reconstruction_count else None,
        "trajectories": per_trajectory,
        "gate_pass": bool(valid_all and contact_macro >= CONTACT_GATE and full_cycle_all),
    }
    return metrics, all_prediction


def load_split_examples(
    root: Path,
) -> tuple[list[TrajectoryExamples], list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    all_examples: list[TrajectoryExamples] = []
    records: list[dict[str, Any]] = []
    norm_frames: list[np.ndarray] = []
    data_lineage: list[dict[str, Any]] = []
    for seed in (*TRAIN_SEEDS, VALIDATION_SEED):
        expected_split = "train" if seed in TRAIN_SEEDS else "validation"
        arrays, summary = read_trajectory(root, seed, expected_split)
        if summary.get("policy_checkpoint_sha256") != EXPECTED_POLICY_SHA256:
            raise RuntimeError(f"Unexpected policy checkpoint for seed {seed}")
        data_lineage.append(
            {
                "seed": seed,
                "split": expected_split,
                "trajectory_sha256": summary["trajectory_sha256"],
                "summary_sha256": sha256_file(root / f"seed_{seed}" / "summary.json"),
            }
        )
        examples, current_records, frames = make_examples(arrays, seed)
        all_examples.extend(examples)
        records.extend(current_records)
        if seed in TRAIN_SEEDS:
            norm_frames.extend(frames)
    return all_examples, records, norm_frames, data_lineage


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    examples, acquisition_records, train_norm_frames, data_lineage = load_split_examples(args.data_root.resolve())
    write_json(output_dir / "phase_label_acquisition.json", acquisition_records)
    failures = [record for record in acquisition_records if record["invalid_reason"] is not None]
    if failures:
        write_json(output_dir / "acquisition_failure.json", {"failed_trajectories": failures})
        raise RuntimeError(f"Phase-label acquisition failed for {len(failures)} trajectory/trajectories")

    train_examples = [example for example in examples if example.seed in TRAIN_SEEDS]
    validation_examples = [example for example in examples if example.seed == VALIDATION_SEED]
    train_windows = [example.windows for example in train_examples if len(example.windows)]
    train_cycles = [example.target_cycles for example in train_examples if len(example.target_cycles)]
    if not train_windows or not validation_examples:
        raise RuntimeError("Training or validation split is empty")
    x_train = np.concatenate(train_windows, axis=0)
    phase_train = np.concatenate(train_cycles, axis=0)
    norm_data = np.concatenate(train_norm_frames, axis=0)
    mean_np = norm_data.mean(axis=0, keepdims=True).astype(np.float32)
    std_np = norm_data.std(axis=0, keepdims=True).astype(np.float32)
    std_np = np.maximum(std_np, 1e-8)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    mean = torch.as_tensor(mean_np, device=device)
    std = torch.as_tensor(std_np, device=device)
    windows = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    phase_targets = torch.as_tensor(phase_train, dtype=torch.float32, device=device)
    normalized_train = (windows - mean) / std

    from src.tasks.velocity.phasor.encoder import PhaseAutoencoder

    model = PhaseAutoencoder(joint_dim=JOINT_DIM, window=WINDOW).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_key: tuple[int, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(normalized_train), generator=rng, device=device)
        epoch_phase = 0.0
        epoch_recon = 0.0
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = normalized_train[indices]
            target = target_vectors(phase_targets[indices])
            phase, reconstruction = model(batch)
            loss_phase = (1.0 - (phase * target).sum(dim=-1)).mean()
            loss_recon = F.mse_loss(reconstruction, batch)
            loss = loss_phase + 0.1 * loss_recon
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_phase += float(loss_phase.detach().item())
            epoch_recon += float(loss_recon.detach().item())
            batches += 1

        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            val_metrics, _ = evaluate_examples(model, validation_examples, mean, std, device)
            val_contact = float(val_metrics["contact_agreement_macro_by_trajectory"])
            val_loss = float(val_metrics["phase_loss"] or float("inf"))
            passes = int(bool(val_metrics["gate_pass"]))
            key = (passes, val_contact, -val_loss)
            row = {
                "epoch": epoch + 1,
                "train_phase_loss": epoch_phase / max(batches, 1),
                "train_reconstruction_mse": epoch_recon / max(batches, 1),
                "validation": val_metrics,
            }
            history.append(row)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                checkpoint = {
                    "state_dict": {k.removeprefix("encoder."): v for k, v in best_state.items() if k.startswith("encoder.")},
                    "decoder_state_dict": {k.removeprefix("decoder."): v for k, v in best_state.items() if k.startswith("decoder.")},
                    "joint_dim": JOINT_DIM,
                    "window": WINDOW,
                    "mean": torch.as_tensor(mean_np),
                    "std": torch.as_tensor(std_np),
                    "source_fps": CONTROL_HZ,
                    "policy_fps": CONTROL_HZ,
                    "pretraining_seed": seed,
                    "selected_epoch": best_epoch,
                    "training_seeds": list(TRAIN_SEEDS),
                    "validation_seed": VALIDATION_SEED,
                    "policy_checkpoint_sha256": EXPECTED_POLICY_SHA256,
                }
                torch.save(checkpoint, output_dir / "phasor_encoder_g1_contact_v3.pt")
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"epoch={epoch + 1} train_phase={row['train_phase_loss']:.5f} "
                    f"train_recon={row['train_reconstruction_mse']:.5f} "
                    f"val_contact={val_contact:.4f} val_phase={val_loss:.5f} "
                    f"val_gate={bool(passes)}"
                )

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    final_metrics, _ = evaluate_examples(model, validation_examples, mean, std, device)
    checkpoint_path = output_dir / "phasor_encoder_g1_contact_v3.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    summary = {
        "schema": "phasor.phase_gate.v1",
        "evaluation_split": "validation",
        "gate_threshold_contact_agreement": CONTACT_GATE,
        "gate_coverage_rule": "each valid trajectory has positive time-ordered net phase advance >= 2*pi and touchdown order is preserved",
        "gate_pass": bool(final_metrics["gate_pass"]),
        "selected_epoch": best_epoch,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "training_seed": seed,
        "training_simulator_seeds": list(TRAIN_SEEDS),
        "validation_simulator_seed": VALIDATION_SEED,
        "locked_test_simulator_seed": LOCKED_TEST_SEED,
        "train_window_count": int(len(x_train)),
        "train_normalization_frame_count": int(len(norm_data)),
        "normalization_fit_seeds": list(TRAIN_SEEDS),
        "data_lineage": data_lineage,
        "normalization_mean": mean_np.reshape(-1).tolist(),
        "normalization_std": std_np.reshape(-1).tolist(),
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": args.batch_size,
            "epoch_ceiling": args.epochs,
            "loss": "circular_phase_loss + 0.1 * normalized_window_reconstruction_mse",
            "device": str(device),
            "deterministic_algorithms_warn_only": True,
        },
        "metrics": final_metrics,
        "interpretation_limit": "Results are conditional on one frozen policy checkpoint and one encoder training seed; simulator seeds do not count as independent policy-training or encoder-training repetitions.",
        "environment": environment_provenance(str(device)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation_summary.json", summary)
    write_json(output_dir / "training_history.json", history)
    print(json.dumps(summary, indent=2, default=json_default))


def evaluate_locked_test(args: argparse.Namespace) -> None:
    unlock = unlock_test(args.validation_summary)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if unlock["validation_checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Validation passed for a different encoder checkpoint")
    arrays, data_summary = read_trajectory(args.data_root.resolve(), LOCKED_TEST_SEED, "test")
    if data_summary.get("test_unlock", {}).get("validation_summary_sha256") != unlock["validation_summary_sha256"]:
        raise RuntimeError("Test data was not collected after this passing validation")
    examples, records, _ = make_examples(arrays, LOCKED_TEST_SEED)
    failures = [record for record in records if record["invalid_reason"] is not None]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from src.tasks.velocity.phasor.encoder import PhaseAutoencoder

    model = PhaseAutoencoder(joint_dim=checkpoint["joint_dim"], window=checkpoint["window"]).to(device)
    model.encoder.load_state_dict(checkpoint["state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    mean = checkpoint["mean"].to(device)
    std = checkpoint["std"].to(device)
    metrics, _ = evaluate_examples(model, examples, mean, std, device)
    metrics["evaluation_split"] = "locked_test"
    metrics["gate_pass"] = bool(metrics["gate_pass"] and not failures)
    summary = {
        "schema": "phasor.phase_gate.v1",
        "evaluation_split": "locked_test",
        "gate_threshold_contact_agreement": CONTACT_GATE,
        "gate_pass": metrics["gate_pass"],
        "validation_summary_sha256": unlock["validation_summary_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "test_data_sha256": data_summary["trajectory_sha256"],
        "simulator_seed": LOCKED_TEST_SEED,
        "phase_label_acquisition": records,
        "invalid_trajectories": failures,
        "metrics": metrics,
        "environment": environment_provenance(str(device)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2, default=json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--num-envs", type=int, default=4)
    collect_parser.add_argument("--steps", type=int, default=1250)
    collect_parser.add_argument("--policy-checkpoint", type=Path, default=FROZEN_POLICY)
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    collect_parser.add_argument("--validation-summary", type=Path)
    collect_parser.set_defaults(func=collect)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--data-root", type=Path, default=RUN_ROOT
    )
    train_parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--evaluate-every", type=int, default=10)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser("evaluate-test")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--validation-summary", type=Path, required=True)
    evaluate_parser.add_argument(
        "--data-root", type=Path, default=RUN_ROOT
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--cpu", action="store_true")
    evaluate_parser.set_defaults(func=evaluate_locked_test)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
