"""Evaluate a fresh G1 rollout under the frozen validation-only phase gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.evaluate_g1_onpolicy_phase_gate import reference_phase
from scripts.evaluate_phase_study import evaluate_phase_predictions
from src.tasks.velocity.phasor.encoder import PhaseEncoder


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
DEFAULT_CHECKPOINT = ARTIFACT_DIR / "phase_encoder_runtime_v1.pt"
DEFAULT_CALIBRATION = ROOT / "results" / "g1_phase_gate_v3" / "rollout_gate_calibration.json"
DEFAULT_HELDOUT_REPORT = ROOT / "results" / "g1_phase_gate_v3" / "phase_encoder_report_macro.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_touchdown_cycles(
    predicted_phase: np.ndarray,
    prediction_frames: np.ndarray,
    touchdown: np.ndarray,
    episode_ids: np.ndarray,
    condition_ids: np.ndarray,
    *,
    tolerance_rad: float,
    maximum_forward_step_rad: float,
    maximum_reverse_step_rad: float,
    history_frames: int = 10,
) -> dict[str, Any]:
    """Score consecutive left touchdowns using predictions at both endpoints."""
    vectors = np.asarray(predicted_phase, dtype=np.float64)
    frames = np.asarray(prediction_frames, dtype=np.int64)
    events = np.asarray(touchdown, dtype=bool)
    episodes = np.asarray(episode_ids)
    conditions = np.asarray(condition_ids)
    if vectors.shape != (len(frames), 2) or len(np.unique(frames)) != len(frames):
        raise ValueError("predictions must have one unique frame index per phase vector")
    if events.ndim != 1 or episodes.shape != events.shape or conditions.shape != events.shape:
        raise ValueError("touchdowns, episodes, and conditions must be one-dimensional and aligned")
    if frames.size and (frames.min() < 0 or frames.max() >= len(events)):
        raise ValueError("prediction frame index is outside the rollout")
    if history_frames < 1:
        raise ValueError("history_frames must be positive")

    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    prediction_at = {int(frame): index for index, frame in enumerate(frames)}
    starts: list[int] = []
    ends: list[int] = []
    cycle_conditions: list[int] = []
    cycle_episodes: list[Any] = []
    net_progression: list[float] = []
    max_forward_steps: list[float] = []
    max_reverse_steps: list[float] = []
    completed: list[bool] = []
    insufficient_history = 0
    unclosed = 0
    mixed_condition = 0

    segment_starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    segment_ends = np.r_[segment_starts[1:], len(episodes)]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        event_frames = np.flatnonzero(events[segment_start:segment_end]) + segment_start
        if event_frames.size:
            unclosed += 1
        for start, end in zip(event_frames[:-1], event_frames[1:]):
            if start not in prediction_at:
                insufficient_history += 1
                continue
            if end not in prediction_at:
                continue
            interval_conditions = np.unique(conditions[start : end + 1])
            if len(interval_conditions) != 1:
                mixed_condition += 1
                continue

            start_pos = prediction_at[int(start)]
            end_pos = prediction_at[int(end)]
            interval_positions = np.arange(start_pos, end_pos + 1)
            interval_frames = frames[interval_positions]
            if not np.array_equal(interval_frames, np.arange(start, end + 1)):
                # A gap means the phase trace is not continuous over this period.
                continue
            unwrapped = np.unwrap(angles[interval_positions])
            deltas = np.diff(unwrapped)
            net = float(unwrapped[-1] - unwrapped[0])
            forward = float(deltas.max(initial=0.0))
            reverse = float(np.maximum(-deltas, 0.0).max(initial=0.0))
            one_turn = (
                net > 0.0
                and abs(net - 2.0 * np.pi) <= tolerance_rad
                and forward <= maximum_forward_step_rad + 1e-9
                and reverse <= maximum_reverse_step_rad + 1e-9
            )
            starts.append(int(start))
            ends.append(int(end))
            cycle_conditions.append(int(conditions[start]))
            cycle_episodes.append(episodes[start].item() if hasattr(episodes[start], "item") else episodes[start])
            net_progression.append(net)
            max_forward_steps.append(forward)
            max_reverse_steps.append(reverse)
            completed.append(bool(one_turn))

    return {
        "cycle_start_frame_indices": starts,
        "cycle_end_frame_indices": ends,
        "cycle_condition_ids": cycle_conditions,
        "cycle_episode_ids": cycle_episodes,
        "cycle_net_progression_rad": net_progression,
        "cycle_max_forward_step_rad": max_forward_steps,
        "cycle_max_reverse_step_rad": max_reverse_steps,
        "cycle_completed_one_turn": completed,
        "complete_cycle_count": len(completed),
        "unclosed_cycle_count": unclosed,
        "insufficient_history_cycle_count": insufficient_history,
        "mixed_condition_cycle_count": mixed_condition,
        "cycle_completed_one_turn_fraction": float(np.mean(completed)) if completed else None,
    }


def evaluate_rollout_arrays(
    joint_pos: np.ndarray,
    contacts: np.ndarray,
    episode_ids: np.ndarray,
    model: torch.nn.Module,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    calibration: dict[str, Any],
    point_threshold_rad: float,
    condition_ids: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate every environment without bridging windows or cycles over reset."""
    joint = np.asarray(joint_pos, dtype=np.float32)
    contact = np.asarray(contacts, dtype=bool)
    episodes = np.asarray(episode_ids)
    if joint.ndim != 3 or joint.shape[2] != 29:
        raise ValueError("joint_pos must have shape [time, env, 29]")
    if contact.shape != (joint.shape[0], joint.shape[1], 2):
        raise ValueError("contacts must have shape [time, env, 2]")
    if episodes.shape != joint.shape[:2]:
        raise ValueError("episode_ids must have shape [time, env]")
    if not np.isfinite(joint).all():
        raise ValueError("joint_pos contains NaN or infinity")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (29,) or std.shape != (29,):
        raise ValueError("normalization statistics must have shape (29,)")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("normalization statistics must be finite with positive standard deviations")
    if condition_ids is not None and np.asarray(condition_ids).shape != joint.shape[:2]:
        raise ValueError("condition_ids must have shape [time, env]")
    termination = (
        np.zeros(joint.shape[:2], dtype=bool)
        if terminated is None else np.asarray(terminated, dtype=bool)
    )
    truncation = (
        np.zeros(joint.shape[:2], dtype=bool)
        if truncated is None else np.asarray(truncated, dtype=bool)
    )
    if termination.shape != joint.shape[:2] or truncation.shape != joint.shape[:2]:
        raise ValueError("terminated and truncated must have shape [time, env]")

    cycle_tolerance = float(calibration["tolerance_rad"])
    max_forward = float(calibration["maximum_forward_step_rad"])
    max_reverse = float(calibration["maximum_reverse_step_rad"])
    required_fraction = float(calibration["macro_completion_fraction_required"])
    if cycle_tolerance < 0 or max_forward <= 0 or max_reverse < 0 or not 0 < required_fraction <= 1:
        raise ValueError("invalid frozen cycle progression calibration")

    per_environment: list[dict[str, Any]] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_target_valid: list[np.ndarray] = []
    all_left_touchdown: list[np.ndarray] = []
    all_right_touchdown: list[np.ndarray] = []
    all_cycle_ids: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []
    all_env_ids: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []
    all_condition_ids: list[np.ndarray] = []
    total_episode_count = 0
    expected_conditions: set[int] = set()

    for env_index in range(joint.shape[1]):
        env_episode_ids = episodes[:, env_index]
        env_termination_count = int(termination[:, env_index].sum())
        env_truncation_count = int(truncation[:, env_index].sum())
        left_target, left_touchdown, cycle_ids = reference_phase(
            contact[:, env_index, 0], episode_ids=env_episode_ids
        )
        _, right_touchdown, _ = reference_phase(
            contact[:, env_index, 1], episode_ids=env_episode_ids
        )
        unique_episodes = list(dict.fromkeys(env_episode_ids.tolist()))
        total_episode_count += len(unique_episodes)
        windows: list[np.ndarray] = []
        target_indices: list[int] = []
        for frame in range(9, joint.shape[0]):
            episode_window = env_episode_ids[frame - 9 : frame + 1]
            if not np.all(episode_window == env_episode_ids[frame]):
                continue
            window = (joint[frame - 9 : frame + 1, env_index] - mean) / std
            windows.append(window)
            target_indices.append(frame)

        if not windows:
            per_environment.append({
                "env": env_index,
                "evaluable": False,
                "reason": "no complete contact-reference frames have a within-episode 10-frame window",
                "episode_count": len(unique_episodes),
                "valid_count": 0,
                "prediction_count": 0,
                "complete_cycle_count": 0,
                "unclosed_cycle_count": 0,
                "cycle_coverage_status": "insufficient_coverage",
                "termination_count": env_termination_count,
                "truncation_count": env_truncation_count,
            })
            continue

        with torch.inference_mode():
            prediction = model(torch.from_numpy(np.asarray(windows, dtype=np.float32)))
        predicted = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
        if predicted.shape != (len(windows), 2) or not np.isfinite(predicted).all():
            raise ValueError(f"encoder returned invalid phase vectors for env {env_index}")
        indices = np.asarray(target_indices, dtype=np.int64)
        target_valid = np.isfinite(left_target[indices])
        point_report = evaluate_phase_predictions(
            predicted[target_valid],
            left_target[indices][target_valid],
            left_touchdown=left_touchdown[indices][target_valid],
            right_touchdown=right_touchdown[indices][target_valid],
            episode_ids=env_episode_ids[indices][target_valid],
            threshold_rad=point_threshold_rad,
        ) if target_valid.any() else {
            "circular_phase_error_rad": None,
            "circular_phase_error_median_rad": None,
            "sample_count": 0,
            "left_touchdown_phase_error_rad": None,
            "left_touchdown_count": 0,
            "right_touchdown_phase_cycles": [],
            "right_touchdown_count": 0,
        }
        if condition_ids is None:
            env_conditions = np.full(joint.shape[0], env_index, dtype=np.int16)
        else:
            env_conditions = np.asarray(condition_ids)[:, env_index]
        env_condition_values = np.unique(env_conditions).astype(int).tolist()
        expected_conditions.update(env_condition_values)
        cycle_report = _score_touchdown_cycles(
            predicted,
            indices,
            left_touchdown,
            env_episode_ids,
            env_conditions,
            tolerance_rad=cycle_tolerance,
            maximum_forward_step_rad=max_forward,
            maximum_reverse_step_rad=max_reverse,
        )
        report = {**point_report, **cycle_report}
        report.update({
            "env": env_index,
            "evaluable": True,
            "episode_count": len(unique_episodes),
            "valid_count": int(target_valid.sum()),
            "prediction_count": int(len(indices)),
            "termination_count": env_termination_count,
            "truncation_count": env_truncation_count,
            "condition_ids": env_condition_values,
        })
        per_environment.append(report)

        all_predictions.append(predicted)
        all_targets.append(left_target[indices].astype(np.float32))
        all_target_valid.append(target_valid)
        all_left_touchdown.append(left_touchdown[indices])
        all_right_touchdown.append(right_touchdown[indices])
        all_cycle_ids.append(cycle_ids[indices])
        all_episode_ids.append(env_episode_ids[indices])
        all_env_ids.append(np.full(len(indices), env_index, dtype=np.int16))
        all_frame_indices.append(indices)
        if condition_ids is not None:
            all_condition_ids.append(np.asarray(condition_ids)[indices, env_index])

    valid_reports = [row for row in per_environment if row.get("evaluable") and row.get("circular_phase_error_rad") is not None]
    macro_error = (
        float(np.mean([row["circular_phase_error_rad"] for row in valid_reports]))
        if valid_reports else None
    )
    minimum_cycles = int(calibration.get("minimum_cycles_per_condition", 1))
    minimum_episodes = int(calibration.get("minimum_episodes_per_condition", 1))
    if minimum_cycles < 1 or minimum_episodes < 1:
        raise ValueError("minimum coverage requirements must be positive")
    cycle_rows_by_condition: dict[int, list[tuple[int, Any, bool]]] = {
        condition: [] for condition in sorted(expected_conditions)
    }
    for row in per_environment:
        for condition, episode, completion in zip(
            row.get("cycle_condition_ids", []),
            row.get("cycle_episode_ids", []),
            row.get("cycle_completed_one_turn", []),
        ):
            cycle_rows_by_condition.setdefault(int(condition), []).append(
                (int(row["env"]), episode, bool(completion))
            )
    condition_coverage = {}
    covered_fractions = []
    for condition, cycles in sorted(cycle_rows_by_condition.items()):
        episode_keys = {(env, episode) for env, episode, _ in cycles}
        fraction = float(np.mean([passed for _, _, passed in cycles])) if cycles else None
        enough = len(cycles) >= minimum_cycles and len(episode_keys) >= minimum_episodes
        condition_coverage[str(condition)] = {
            "complete_cycle_count": len(cycles),
            "episode_count_with_valid_cycles": len(episode_keys),
            "cycle_completed_one_turn_fraction": fraction,
            "minimum_cycles_required": minimum_cycles,
            "minimum_episodes_required": minimum_episodes,
            "coverage_status": "adequate" if enough else "insufficient_coverage",
        }
        if enough and fraction is not None:
            covered_fractions.append(fraction)
    for row in per_environment:
        row_conditions = row.get("condition_ids", [])
        row_ok = bool(row_conditions) and all(
            condition_coverage.get(str(condition), {}).get("coverage_status") == "adequate"
            for condition in row_conditions
        )
        row["cycle_coverage_status"] = "adequate" if row_ok else "insufficient_coverage"
    coverage_ok = bool(condition_coverage) and all(
        row["coverage_status"] == "adequate" for row in condition_coverage.values()
    )
    macro_cycle = float(np.mean(covered_fractions)) if coverage_ok else None
    pointwise_pass = None if macro_error is None else bool(macro_error <= point_threshold_rad)
    cycle_pass = None if not coverage_ok else bool(macro_cycle >= required_fraction)
    gate_pass = None if not coverage_ok else bool(pointwise_pass and cycle_pass)
    report = {
        "per_environment": per_environment,
        "episode_count": total_episode_count,
        "termination_count": int(termination.sum()),
        "truncation_count": int(truncation.sum()),
        "eligible_episode_count": int(sum(value["episode_count_with_valid_cycles"] for value in condition_coverage.values())),
        "evaluable_environment_count": len(valid_reports),
        "macro_aggregation": "equal-weight mean across evaluable environments; each environment carries its scheduled conditions",
        "macro_circular_phase_error_rad": macro_error,
        "pointwise_phase_threshold_rad": float(point_threshold_rad),
        "onpolicy_pointwise_phase_pass": pointwise_pass,
        "condition_coverage": condition_coverage,
        "minimum_cycles_per_condition": minimum_cycles,
        "minimum_episodes_per_condition": minimum_episodes,
        "cycle_coverage_status": "adequate" if coverage_ok else "insufficient_coverage",
        "macro_cycle_completed_one_turn_fraction": macro_cycle,
        "cycle_progression_tolerance_rad": cycle_tolerance,
        "maximum_forward_step_rad": max_forward,
        "maximum_reverse_step_rad": max_reverse,
        "macro_completion_fraction_required": required_fraction,
        "cycle_progression_pass": cycle_pass,
        "phase_rollout_gate_pass": gate_pass,
        "phase_gate_status": "insufficient_coverage" if not coverage_ok else ("pass" if gate_pass else "fail"),
    }
    predictions = {
        "predicted_phase": np.concatenate(all_predictions) if all_predictions else np.empty((0, 2), dtype=np.float64),
        "target_phase": np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32),
        "target_valid_mask": np.concatenate(all_target_valid) if all_target_valid else np.empty(0, dtype=bool),
        "left_touchdown": np.concatenate(all_left_touchdown) if all_left_touchdown else np.empty(0, dtype=bool),
        "right_touchdown": np.concatenate(all_right_touchdown) if all_right_touchdown else np.empty(0, dtype=bool),
        "cycle_ids": np.concatenate(all_cycle_ids) if all_cycle_ids else np.empty(0, dtype=np.int64),
        "episode_ids": np.concatenate(all_episode_ids) if all_episode_ids else np.empty(0, dtype=np.int64),
        "env_ids": np.concatenate(all_env_ids) if all_env_ids else np.empty(0, dtype=np.int16),
        "frame_indices": np.concatenate(all_frame_indices) if all_frame_indices else np.empty(0, dtype=np.int64),
    }
    if condition_ids is not None:
        predictions["condition_ids"] = (
            np.concatenate(all_condition_ids) if all_condition_ids else np.empty(0, dtype=np.int8)
        )
    return report, predictions


def evaluate_confirmation(
    rollout_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    heldout_report_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    checkpoint_path = checkpoint_path.resolve(strict=True)
    calibration_path = calibration_path.resolve(strict=True)
    heldout_report_path = heldout_report_path.resolve(strict=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("source_split") != "validation_only":
        raise ValueError("rollout calibration must have been frozen on validation only")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("runtime_contract", {})
    if contract.get("source_checkpoint_sha256") != calibration.get("encoder_checkpoint_sha256"):
        raise ValueError("runtime checkpoint weights do not match the calibration source checkpoint")
    if contract.get("dataset_sha256") != calibration.get("dataset_sha256"):
        raise ValueError("runtime checkpoint dataset identity does not match the frozen calibration")
    model = PhaseEncoder(
        joint_dim=29,
        window=10,
        preserve_time=bool(checkpoint["preserve_time"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    heldout = json.loads(heldout_report_path.read_text(encoding="utf-8"))
    point_threshold = float(heldout["phase_gate_threshold_rad"])

    with np.load(rollout_path, allow_pickle=False) as arrays:
        condition_ids = arrays["condition_ids"] if "condition_ids" in arrays.files else None
        report, prediction_arrays = evaluate_rollout_arrays(
            arrays["joint_pos"],
            arrays["contacts"],
            arrays["episode_ids"],
            model,
            mean=np.asarray(checkpoint["mean"], dtype=np.float32),
            std=np.asarray(checkpoint["std"], dtype=np.float32),
            calibration=calibration,
            point_threshold_rad=point_threshold,
            condition_ids=condition_ids,
            terminated=arrays["terminated"] if "terminated" in arrays.files else None,
            truncated=arrays["truncated"] if "truncated" in arrays.files else None,
        )
    heldout_error = float(heldout["test_report"]["macro_circular_phase_error_rad"])
    heldout_pass = heldout_error <= point_threshold
    source_sha256 = {
        "confirmation_evaluator": sha256_file(Path(__file__).resolve()),
        "phase_metric_evaluator": sha256_file(ROOT / "scripts" / "evaluate_phase_study.py"),
        "contact_reference": sha256_file(ROOT / "scripts" / "evaluate_g1_onpolicy_phase_gate.py"),
    }
    report["heldout_circular_phase_error_rad"] = heldout_error
    report["heldout_phase_pass"] = bool(heldout_pass)
    if report["cycle_coverage_status"] == "adequate":
        report["phase_rollout_gate_pass"] = bool(heldout_pass and report["phase_rollout_gate_pass"])
        report["phase_gate_status"] = "pass" if report["phase_rollout_gate_pass"] else "fail"
    else:
        report["phase_rollout_gate_pass"] = None
        report["phase_gate_status"] = "insufficient_coverage"
    report["heldout_point_threshold_rad"] = point_threshold
    report["evaluation_source_sha256"] = source_sha256
    return report, prediction_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--heldout-report", type=Path, default=DEFAULT_HELDOUT_REPORT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    rollout_path = args.rollout.resolve(strict=True)
    run_manifest_path = (args.run_manifest or rollout_path.parent / "manifest.json").resolve(strict=True)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        raise ValueError("confirmation evaluation requires a completed rollout bundle")
    if run_manifest.get("trajectory_sha256") != sha256_file(rollout_path):
        raise ValueError("rollout data hash does not match its collection manifest")
    output = (args.output or rollout_path.parent / "phase_gate.json").resolve()
    prediction_path = output.parent / "phase_predictions.npz"
    evaluation_manifest_path = output.parent / "evaluation_manifest.json"
    if any(path.exists() for path in (output, prediction_path, evaluation_manifest_path)):
        raise FileExistsError("refusing to replace existing confirmation evaluation artifacts")

    report, predictions = evaluate_confirmation(
        rollout_path,
        args.checkpoint,
        args.calibration,
        args.heldout_report,
    )
    np.savez_compressed(prediction_path, **predictions)
    report.update({
        "schema": "g1.phase_confirmation_gate.v1",
        "status": report["phase_gate_status"],
        "decision": (
            "PASS_PHASE_CONFIRMATION" if report["phase_gate_status"] == "pass" else
            "INCONCLUSIVE_CYCLE_COVERAGE" if report["phase_gate_status"] == "insufficient_coverage" else
            "FAIL_PHASE_CONFIRMATION"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_path": str(rollout_path),
        "rollout_sha256": sha256_file(rollout_path),
        "rollout_manifest_sha256": sha256_file(run_manifest_path),
        "evaluation_source_sha256": report["evaluation_source_sha256"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_source_sha256": torch.load(args.checkpoint, map_location="cpu", weights_only=False)["runtime_contract"]["source_checkpoint_sha256"],
        "calibration_path": str(args.calibration.resolve()),
        "calibration_sha256": sha256_file(args.calibration.resolve()),
        "heldout_report_sha256": sha256_file(args.heldout_report.resolve()),
        "input_contract": "joint positions only; contact arrays are used only after inference to build offline references",
        "formal_ppo_started": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    evaluation_manifest = {
        "schema": "g1.phase_confirmation_evaluation.v1",
        "status": "completed",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "report_path": str(output),
        "report_sha256": sha256_file(output),
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "evaluation_source_sha256": report["evaluation_source_sha256"],
    }
    evaluation_manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
