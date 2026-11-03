"""Collect a fresh fixed-clock G1 policy rollout for encoder confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from scripts.smoke_g1_gpu_env import _require_finite
from scripts.smoke_g1_gpu_env import configure_torch_backends

import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.velocity.mdp.observations import _joint_positions_in_order
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
from src.tasks.velocity.phasor.confirmation import (
    CONDITION_NAMES,
    DISTURBANCE_STEPS,
    advance_episode_segments,
    command_for_step,
    onnx_batch_one_actions,
    quaternion_wxyz_to_rpy,
)
from src.tasks.velocity.phasor.mujoco_io import install_mjlab_scene_xml_unicode_compat
import src.tasks  # noqa: F401 - register the project task configurations.


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261041
STEPS_PER_EPISODE = 750
EPISODE_COUNT = 2
NUM_ENVS = 4
TASK_ID = "Unitree-G1-Flat"
PILOT_MANIFEST = ROOT / "results" / "phasor_pilot_20261016" / "runs" / "pilot_baseline_seed_20261033_attempt_03" / "manifest.json"
DEFAULT_RUN_DIR = ROOT / "results" / "g1_phase_confirmation_20261019" / f"fixed_clock_seed_{SEED}"

CONDITIONS = [
    {"id": index, "name": name}
    for index, name in enumerate(CONDITION_NAMES)
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def _obs_from_result(result: Any) -> dict[str, torch.Tensor]:
    observations = result[0] if isinstance(result, tuple) else result
    if not isinstance(observations, dict) or "actor" not in observations:
        raise RuntimeError("G1 environment did not return the actor observation group")
    return observations


def collect(
    run_dir: Path,
    seed: int = SEED,
    *,
    pilot_manifest_path: Path = PILOT_MANIFEST,
    policy_source_path: Path | None = None,
    run_kind: str = "confirmation",
) -> dict[str, Any]:
    if run_kind not in {"confirmation", "controller_feasibility_pilot"}:
        raise ValueError(f"unsupported G1 rollout kind: {run_kind}")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if policy_source_path is None:
        pilot_manifest_path = pilot_manifest_path.resolve(strict=True)
        pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
        if (
            pilot_manifest.get("status") != "completed"
            or pilot_manifest.get("condition") != "baseline"
            or pilot_manifest.get("task_id") != TASK_ID
        ):
            raise RuntimeError("the selected fixed-clock pilot policy lacks a completed matching manifest")
        policy_manifest = pilot_manifest
        log_dir = Path(pilot_manifest["log_dir"])
        policy_path = log_dir / "policy.onnx"
        policy_checkpoint_path = log_dir / "model_1000.pt"
    else:
        pilot_manifest_path = policy_source_path.resolve(strict=True)
        policy_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
        if (
            policy_manifest.get("schema") != "g1.controller_policy_source.v1"
            or policy_manifest.get("status") != "completed"
            or policy_manifest.get("task_id") != TASK_ID
            or policy_manifest.get("condition") != "baseline"
        ):
            raise RuntimeError("the selected historical policy source manifest is invalid")
        policy_path = Path(policy_manifest["policy_onnx_path"]).resolve(strict=True)
        policy_checkpoint_path = Path(policy_manifest["policy_checkpoint_path"]).resolve(strict=True)
        if sha256_file(policy_path) != policy_manifest.get("policy_onnx_sha256"):
            raise RuntimeError("the historical policy ONNX hash does not match its source manifest")
        if sha256_file(policy_checkpoint_path) != policy_manifest.get("policy_checkpoint_sha256"):
            raise RuntimeError("the historical policy checkpoint hash does not match its source manifest")
    for path in (policy_path, policy_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect the G1 confirmation rollout")
    device = "cuda:0"
    install_mjlab_scene_xml_unicode_compat()
    configure_torch_backends()

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    policy = ort.InferenceSession(
        str(policy_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    input_info = policy.get_inputs()
    output_info = policy.get_outputs()
    if len(input_info) != 1 or input_info[0].name != "obs" or input_info[0].shape[-1] != 98:
        raise RuntimeError(f"unexpected fixed-clock ONNX input contract: {[(i.name, i.shape) for i in input_info]}")
    if len(output_info) != 1 or output_info[0].name != "actions" or output_info[0].shape[-1] != 29:
        raise RuntimeError(f"unexpected fixed-clock ONNX output contract: {[(o.name, o.shape) for o in output_info]}")

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = seed
    push_cfg = cfg.events.pop("push_robot", None)
    if push_cfg is None or push_cfg.mode != "interval":
        raise RuntimeError("fixed-clock training config is missing its interval push event")
    push_func = push_cfg.func
    push_params = dict(push_cfg.params)

    env = None
    rows_joint: list[np.ndarray] = []
    rows_contact: list[np.ndarray] = []
    rows_command: list[np.ndarray] = []
    rows_episode: list[np.ndarray] = []
    rows_condition: list[np.ndarray] = []
    rows_step: list[np.ndarray] = []
    rows_terminated: list[np.ndarray] = []
    rows_truncated: list[np.ndarray] = []
    rows_root_pos_w: list[np.ndarray] = []
    rows_root_quat_wxyz: list[np.ndarray] = []
    rows_root_velocity_w: list[np.ndarray] = []
    rows_root_rpy_rad: list[np.ndarray] = []
    rows_policy_actions: list[np.ndarray] = []
    rows_applied_action_values: list[np.ndarray] = []
    rows_feet_found: list[np.ndarray] = []
    rows_feet_force: list[np.ndarray] = []
    rows_termination_terms: list[np.ndarray] = []
    termination_term_names: list[str] = []
    disturbance_rows: list[dict[str, Any]] = []
    termination_rows: list[dict[str, Any]] = []
    telemetry_context: dict[str, Any] = {}
    applied_action_source = "policy_action_fallback"
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        reset_obs = _obs_from_result(env.reset())
        if tuple(reset_obs["actor"].shape) != (NUM_ENVS, 98):
            raise RuntimeError(f"unexpected actor observation shape: {tuple(reset_obs['actor'].shape)}")

        command = env.command_manager.get_command("twist")
        env_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=device)
        asset = env.scene["robot"]
        contact_sensor = env.scene["feet_ground_contact"]
        if tuple(asset.data.joint_pos.shape) != (NUM_ENVS, len(G1_JOINT_ORDER)):
            raise RuntimeError(f"unexpected G1 joint shape: {tuple(asset.data.joint_pos.shape)}")

        termination_term_names[:] = env.termination_manager.active_terms
        original_compute = env.termination_manager.compute

        def compute_and_capture_terminal_state() -> torch.Tensor:
            reset_mask = original_compute()
            root_pos = asset.data.root_link_pos_w.detach().cpu().numpy().copy()
            root_quat = asset.data.root_link_quat_w.detach().cpu().numpy().copy()
            root_velocity = asset.data.root_link_vel_w.detach().cpu().numpy().copy()
            root_rpy = quaternion_wxyz_to_rpy(root_quat)
            feet_found = contact_sensor.data.found
            feet_force = contact_sensor.data.force
            if feet_found is None or feet_force is None:
                raise RuntimeError("foot contact sensor must expose found and force telemetry")
            feet_found_array = feet_found.detach().cpu().numpy().copy()
            feet_force_array = feet_force.detach().cpu().numpy().copy()
            term_values = [
                env.termination_manager.get_term(name).detach().cpu().numpy().copy()
                for name in termination_term_names
            ]
            term_mask = np.stack(term_values, axis=-1) if term_values else np.zeros((NUM_ENVS, 0), dtype=bool)
            policy_action = np.asarray(telemetry_context["policy_action"], dtype=np.float32).copy()
            applied_action = getattr(asset.data, "joint_pos_target", None)
            nonlocal applied_action_source
            if applied_action is not None:
                applied_action_source = "asset.data.joint_pos_target"
                applied_action_array = applied_action.detach().cpu().numpy().copy()
            else:
                applied_action = getattr(env.action_manager, "action", None)
                if applied_action is not None:
                    applied_action_source = "action_manager.action"
                    applied_action_array = applied_action.detach().cpu().numpy().copy()
                else:
                    applied_action_array = policy_action.copy()

            rows_root_pos_w.append(root_pos)
            rows_root_quat_wxyz.append(root_quat)
            rows_root_velocity_w.append(root_velocity)
            rows_root_rpy_rad.append(root_rpy)
            rows_policy_actions.append(policy_action)
            rows_applied_action_values.append(applied_action_array)
            rows_feet_found.append(feet_found_array)
            rows_feet_force.append(feet_force_array)
            rows_termination_terms.append(term_mask)

            done_values = reset_mask.detach().cpu().numpy().astype(bool, copy=False)
            for env_index in np.flatnonzero(done_values):
                reasons = [
                    name for column, name in enumerate(termination_term_names)
                    if bool(term_mask[env_index, column])
                ]
                context = telemetry_context
                segment_before = int(context["segment_counts"][env_index])
                termination_rows.append({
                    "block_id": int(context["block_id"]),
                    "condition_id": int(env_index),
                    "condition_name": CONDITION_NAMES[env_index],
                    "step_index": int(context["step_index"]),
                    "sim_time_s": float(context["step_index"]) * float(env.step_dt),
                    "terminated": bool(env.termination_manager.terminated[env_index].item()),
                    "truncated": bool(env.termination_manager.time_outs[env_index].item()),
                    "termination_reasons": reasons or ["unattributed_done"],
                    "episode_segment_before_reset": int(context["block_id"] * (STEPS_PER_EPISODE + 1) + segment_before),
                    "episode_segment_after_reset": None,
                    "root_position_w": root_pos[env_index].tolist(),
                    "root_quaternion_wxyz": root_quat[env_index].tolist(),
                    "root_rpy_rad": root_rpy[env_index].tolist(),
                    "root_velocity_w": root_velocity[env_index].tolist(),
                    "command": np.asarray(context["command"])[env_index].tolist(),
                    "policy_action": policy_action[env_index].tolist(),
                    "applied_action_value": applied_action_array[env_index].tolist(),
                    "applied_action_source": applied_action_source,
                    "feet_found": feet_found_array[env_index].tolist(),
                    "feet_force": feet_force_array[env_index].tolist(),
                })
            return reset_mask

        env.termination_manager.compute = compute_and_capture_terminal_state

        for episode_id in range(EPISODE_COUNT):
            if episode_id > 0:
                reset_obs = _obs_from_result(env.reset())
            segment_counts = np.zeros(NUM_ENVS, dtype=np.int64)
            for step_index in range(STEPS_PER_EPISODE):
                speeds = torch.tensor(
                    [command_for_step(condition_id, step_index) for condition_id in range(NUM_ENVS)],
                    dtype=torch.float32,
                    device=device,
                )
                command[:, 0].copy_(speeds)
                command[:, 1].zero_()
                command[:, 2].zero_()

                if step_index in DISTURBANCE_STEPS:
                    before = asset.data.root_link_vel_w[env_ids].clone()
                    push_func(env, env_ids, **push_params)
                    env.sim.forward()
                    after = asset.data.root_link_vel_w[env_ids].clone()
                    actual_delta = after - before
                    disturbance_rows.append({
                        "episode_id": episode_id,
                        "step_index": step_index,
                        "sim_time_s": step_index * float(env.step_dt),
                        "event": "push_by_setting_velocity",
                        "velocity_range": push_params["velocity_range"],
                        "applied_velocity_delta": actual_delta.detach().cpu().tolist(),
                        "root_velocity_before": before.detach().cpu().tolist(),
                        "root_velocity_after": after.detach().cpu().tolist(),
                        "post_write_cache_refreshed": True,
                    })

                command_snapshot = command.clone()
                observations = env.observation_manager.compute()
                actor_obs = observations["actor"]
                if actor_obs.shape != (NUM_ENVS, 98) or not torch.isfinite(actor_obs).all().item():
                    raise RuntimeError("actor observation is malformed or non-finite")
                action_array = onnx_batch_one_actions(
                    policy,
                    actor_obs.detach().cpu().numpy(),
                    input_name=input_info[0].name,
                    output_name=output_info[0].name,
                    action_dim=29,
                )
                if action_array.shape != (NUM_ENVS, 29) or not np.isfinite(action_array).all():
                    raise RuntimeError(f"fixed-clock policy returned invalid actions: {action_array.shape}")
                actions = torch.as_tensor(action_array, dtype=torch.float32, device=device)
                telemetry_context.update({
                    "block_id": episode_id,
                    "step_index": step_index,
                    "segment_counts": segment_counts.copy(),
                    "command": command_snapshot.detach().cpu().numpy().copy(),
                    "policy_action": action_array.copy(),
                })
                termination_start = len(termination_rows)
                result = env.step(actions)
                if not isinstance(result, tuple) or len(result) < 4:
                    raise RuntimeError("unexpected ManagerBasedRlEnv.step result")
                terminated = torch.as_tensor(result[2], device=device, dtype=torch.bool).reshape(NUM_ENVS)
                truncated = torch.as_tensor(result[3], device=device, dtype=torch.bool).reshape(NUM_ENVS)
                terminated_array = terminated.detach().cpu().numpy().copy()
                truncated_array = truncated.detach().cpu().numpy().copy()
                segment_counts, episode_frame_ids = advance_episode_segments(
                    block_id=episode_id,
                    reset_counts=segment_counts,
                    terminated=terminated_array,
                    truncated=truncated_array,
                    block_span=STEPS_PER_EPISODE + 1,
                )
                for event in termination_rows[termination_start:]:
                    event["episode_segment_after_reset"] = int(episode_frame_ids[event["condition_id"]])

                joint = _joint_positions_in_order(asset, asset.data.joint_pos, G1_JOINT_ORDER)
                contacts = contact_sensor.data.current_contact_time > 0.0
                _require_finite("confirmation joint positions", joint)
                if not torch.isfinite(command_snapshot).all().item():
                    raise RuntimeError("command contains NaN or infinity")
                rows_joint.append(joint.detach().cpu().numpy().astype(np.float32, copy=True))
                rows_contact.append(contacts.detach().cpu().numpy().astype(bool, copy=True))
                rows_command.append(command_snapshot.detach().cpu().numpy().astype(np.float32, copy=True))
                rows_episode.append(episode_frame_ids.astype(np.int32, copy=True))
                rows_condition.append(np.arange(NUM_ENVS, dtype=np.int8))
                rows_step.append(np.full(NUM_ENVS, step_index, dtype=np.int32))
                rows_terminated.append(terminated.detach().cpu().numpy().copy())
                rows_truncated.append(truncated.detach().cpu().numpy().copy())

        torch.cuda.synchronize()
    except Exception as error:
        if rows_joint:
            np.savez_compressed(
                run_dir / "trajectories_partial.npz",
                joint_pos=np.stack(rows_joint),
                contacts=np.stack(rows_contact),
                commands=np.stack(rows_command),
                episode_ids=np.stack(rows_episode),
                condition_ids=np.stack(rows_condition),
                step_indices=np.stack(rows_step),
                terminated=np.stack(rows_terminated),
                truncated=np.stack(rows_truncated),
                root_position_w=np.stack(rows_root_pos_w),
                root_quaternion_wxyz=np.stack(rows_root_quat_wxyz),
                root_velocity_w=np.stack(rows_root_velocity_w),
                root_rpy_rad=np.stack(rows_root_rpy_rad),
                policy_actions=np.stack(rows_policy_actions),
                applied_action_values=np.stack(rows_applied_action_values),
                feet_found=np.stack(rows_feet_found),
                feet_force=np.stack(rows_feet_force),
                termination_terms=np.stack(rows_termination_terms),
                termination_term_names=np.asarray(termination_term_names),
                foot_order=np.asarray(["left", "right"]),
            )
        failure = {
            "status": "failed",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "traceback": __import__("traceback").format_exc(),
            "partial_steps": len(rows_joint),
        }
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()

    rollout_path = run_dir / "trajectories.npz"
    np.savez_compressed(
        rollout_path,
        joint_pos=np.stack(rows_joint),
        contacts=np.stack(rows_contact),
        commands=np.stack(rows_command),
        episode_ids=np.stack(rows_episode),
        condition_ids=np.stack(rows_condition),
        step_indices=np.stack(rows_step),
        terminated=np.stack(rows_terminated),
        truncated=np.stack(rows_truncated),
        root_position_w=np.stack(rows_root_pos_w),
        root_quaternion_wxyz=np.stack(rows_root_quat_wxyz),
        root_velocity_w=np.stack(rows_root_velocity_w),
        root_rpy_rad=np.stack(rows_root_rpy_rad),
        policy_actions=np.stack(rows_policy_actions),
        applied_action_values=np.stack(rows_applied_action_values),
        feet_found=np.stack(rows_feet_found),
        feet_force=np.stack(rows_feet_force),
        termination_terms=np.stack(rows_termination_terms),
        termination_term_names=np.asarray(termination_term_names),
        foot_order=np.asarray(["left", "right"]),
    )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "onnxruntime": ort.__version__,
        "mjlab": _version("mjlab"),
        "mujoco": _version("mujoco"),
        "warp": _version("warp-lang"),
        "device": device,
        "onnx_provider": "CPUExecutionProvider",
        "onnx_intra_op_threads": 1,
        "determinism": "seeded; CUDA/Warp kernels are not claimed bitwise deterministic",
    }
    config = {
        "run_kind": run_kind,
        "task_id": TASK_ID,
        "seed": seed,
        "num_envs": NUM_ENVS,
        "control_frequency_hz": 1.0 / float(env.step_dt),
        "steps_per_episode": STEPS_PER_EPISODE,
        "episode_count": EPISODE_COUNT,
        "conditions": CONDITIONS,
        "command_schedule": {
            "condition_0_1_2": [0.4, 0.7, 0.9],
            "transition_condition_boundaries": {"0": 0.0, "100": 0.6, "250": 0.0, "350": 0.6},
            "command_components": ["vx", "vy=0", "yaw_rate=0"],
        },
        "disturbances_per_episode": list(DISTURBANCE_STEPS),
        "disturbance_event": "same push_by_setting_velocity function and velocity range as training; scheduled at 5s and 10s",
        "contact_use": "saved only for offline evaluation; policy ONNX input contains actor observation only",
        "policy_seed": policy_manifest["seed"],
        "policy_manifest_path": str(pilot_manifest_path),
        "policy_training_run_status": policy_manifest["status"],
        "applied_action_telemetry_source": applied_action_source,
        "terminal_state_capture": "after termination_manager.compute and before automatic reset",
        "root_velocity_telemetry": "world-frame root link linear and angular velocity at the termination check",
        "disturbance_delta_telemetry": "root velocity refreshed by sim.forward immediately after write",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    code_paths = {
        "collector": Path(__file__).resolve(),
        "confirmation_helpers": ROOT / "src" / "tasks" / "velocity" / "phasor" / "confirmation.py",
        "env_cfgs": ROOT / "src" / "tasks" / "velocity" / "config" / "g1" / "env_cfgs.py",
        "observations": ROOT / "src" / "tasks" / "velocity" / "mdp" / "observations.py",
        "preregistration": ROOT / "docs" / "PHASE_CONTROL_PREREGISTRATION.md",
        "data_provenance": ROOT / "docs" / "data" / "g1_phase_data_provenance.json",
    }
    ended_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "g1.phase_confirmation_rollout.v2",
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "seed": seed,
        "status": "completed",
        "started_at": started_at,
        "ended_at": ended_at,
        "argv": sys.argv,
        "git_revision": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_branch": __import__("subprocess").check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(__import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "pilot_manifest_path": str(pilot_manifest_path),
        "pilot_manifest_sha256": sha256_file(pilot_manifest_path),
        "policy_onnx_path": str(policy_path),
        "policy_onnx_sha256": sha256_file(policy_path),
        "policy_checkpoint_path": str(policy_checkpoint_path),
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint_path),
        "source_sha256": {key: sha256_file(path) for key, path in code_paths.items()},
        "disturbance_events": disturbance_rows,
        "termination_events": termination_rows,
        "termination_term_names": termination_term_names,
        "diagnostic_telemetry_fields": [
            "root_position_w", "root_quaternion_wxyz", "root_velocity_w", "root_rpy_rad",
            "policy_actions", "applied_action_values", "feet_found", "feet_force", "termination_terms",
        ],
        "termination_count": sum(bool(row["terminated"]) for row in termination_rows),
        "truncation_count": sum(bool(row["truncated"]) for row in termination_rows),
        "trajectory_path": str(rollout_path),
        "trajectory_sha256": sha256_file(rollout_path),
        "trajectory_shape": list(np.load(rollout_path, allow_pickle=False)["joint_pos"].shape),
        "formal_ppo_started": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy-manifest", type=Path, default=PILOT_MANIFEST)
    parser.add_argument("--run-kind", choices=("confirmation", "controller_feasibility_pilot"), default="confirmation")
    args = parser.parse_args()
    run_dir = args.output_dir or (
        ROOT / "results" / "g1_phase_confirmation_20261019" / f"fixed_clock_seed_{args.seed}"
    )
    print(json.dumps(
        collect(
            run_dir,
            seed=args.seed,
            pilot_manifest_path=args.policy_manifest,
            run_kind=args.run_kind,
        ),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
