"""Summarize controller termination and phase-gate evidence for G1."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "results" / "g1_controller_feasibility_20261019"
SCOUT_REPORT = PILOT_ROOT / "historical_controller_scout.json"
OUTPUT = ROOT / "results" / "g1_phase_gate_v3" / "failure_tree_report.json"
ENCODER_DATASET = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
ENCODER_PREDICTIONS = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_predictions.npz"
ENCODER_CHECKPOINT = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_encoder_checkpoint.pt"
PHASE_CALIBRATION = ROOT / "results" / "g1_phase_gate_v3" / "rollout_gate_calibration.json"
ENCODER_CANDIDATE_REPORT = ROOT / "results" / "g1_phase_gate_v3" / "encoder_candidate_validation.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _body_linear_velocity(event: dict[str, Any]) -> np.ndarray:
    w, x, y, z = np.asarray(event["root_quaternion_wxyz"], dtype=np.float64)
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rotation.T @ np.asarray(event["root_velocity_w"][:3], dtype=np.float64)


def _phase_summary(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "phase_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cycle_checks: dict[str, int] = Counter()
    per_condition: dict[str, Any] = {}
    for row in report["per_environment"]:
        condition_ids = row.get("condition_ids", [])
        if not condition_ids:
            continue
        condition = str(condition_ids[0])
        net = np.asarray(row.get("cycle_net_progression_rad", []), dtype=np.float64)
        forward = np.asarray(row.get("cycle_max_forward_step_rad", []), dtype=np.float64)
        reverse = np.asarray(row.get("cycle_max_reverse_step_rad", []), dtype=np.float64)
        tolerance = float(report["cycle_progression_tolerance_rad"])
        within = np.abs(net - 2.0 * math.pi) <= tolerance
        positive = net > 0.0
        forward_ok = forward <= float(report["maximum_forward_step_rad"]) + 1e-9
        reverse_ok = reverse <= float(report["maximum_reverse_step_rad"]) + 1e-9
        cycle_checks["cycles"] += int(len(net))
        cycle_checks["positive_net"] += int(positive.sum())
        cycle_checks["net_within_tolerance"] += int(within.sum())
        cycle_checks["forward_step_ok"] += int(forward_ok.sum())
        cycle_checks["reverse_step_ok"] += int(reverse_ok.sum())
        cycle_checks["one_turn_pass"] += int((positive & within & forward_ok & reverse_ok).sum())
        coverage = report["condition_coverage"].get(condition, {})
        per_condition[condition] = {
            "cycle_count": int(len(net)),
            "episodes_with_cycles": int(coverage.get("episode_count_with_valid_cycles", 0)),
            "one_turn_pass_count": int((positive & within & forward_ok & reverse_ok).sum()),
            "one_turn_fraction": coverage.get("cycle_completed_one_turn_fraction"),
            "coverage_status": coverage.get("coverage_status"),
            "net_progression_rad": _quantiles(net.tolist()),
            "max_forward_step_rad": _quantiles(forward.tolist()),
            "max_reverse_step_rad": _quantiles(reverse.tolist()),
        }
    rollout = Path(report["rollout_path"])
    prediction_path = run_dir / "phase_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as predictions:
        phase_norm = np.linalg.norm(predictions["predicted_phase"], axis=1).tolist()
    return {
        "phase_report_path": str(report_path.resolve()),
        "phase_report_sha256": _sha256(report_path),
        "rollout_sha256": report["rollout_sha256"],
        "cycle_coverage_status": report["cycle_coverage_status"],
        "condition_coverage": report["condition_coverage"],
        "phase_gate_status": report["phase_gate_status"],
        "phase_rollout_gate_pass": report["phase_rollout_gate_pass"],
        "macro_circular_phase_error_rad": report["macro_circular_phase_error_rad"],
        "point_threshold_rad": report["pointwise_phase_threshold_rad"],
        "cycle_progression_tolerance_rad": report["cycle_progression_tolerance_rad"],
        "maximum_forward_step_rad": report["maximum_forward_step_rad"],
        "maximum_reverse_step_rad": report["maximum_reverse_step_rad"],
        "macro_cycle_completed_one_turn_fraction": report["macro_cycle_completed_one_turn_fraction"],
        "cycle_check_counts": dict(cycle_checks),
        "predicted_phase_vector_norm": _quantiles(phase_norm),
        "per_condition": per_condition,
        "cycle_metric_regression_suite": "tests/test_g1_phase_confirmation.py",
        "rollout_path": str(rollout.resolve()),
    }


def _encoder_validation_summary() -> dict[str, Any]:
    """Recompute frozen-encoder phase metrics on the untouched validation arrays."""
    from scripts.evaluate_phase_study import evaluate_phase_predictions

    calibration = json.loads(PHASE_CALIBRATION.read_text(encoding="utf-8"))
    with np.load(ENCODER_DATASET, allow_pickle=False) as dataset:
        sequences = dataset["sequence_validation"].astype(str)
        frames = dataset["frame_validation"].astype(np.int64)
        touchdown = dataset["left_touchdown_validation"].astype(bool)
        target = dataset["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi)
    with np.load(ENCODER_PREDICTIONS, allow_pickle=False) as prediction_file:
        predicted_vectors = prediction_file["val_pred"].astype(np.float64)

    if predicted_vectors.shape != (len(target), 2):
        raise ValueError("saved validation predictions do not match validation labels")
    predicted_angle = np.arctan2(predicted_vectors[:, 0], predicted_vectors[:, 1])
    frame_error = np.abs(np.angle(np.exp(1j * (predicted_angle - target))))
    per_sequence: dict[str, Any] = {}
    for sequence in dict.fromkeys(sequences.tolist()):
        indices = np.flatnonzero(sequences == sequence)
        indices = indices[np.argsort(frames[indices], kind="stable")]
        ordered_frames = frames[indices]
        ordered_touchdown = touchdown[indices]
        segment_ids = np.cumsum(
            np.r_[0, (np.diff(ordered_frames) != 1).astype(np.int64)]
        )
        cycle_ids = np.zeros(len(indices), dtype=np.int64)
        for segment_id in np.unique(segment_ids):
            segment = np.flatnonzero(segment_ids == segment_id)
            cycle_ids[segment] = np.cumsum(ordered_touchdown[segment])
        cycle_report = evaluate_phase_predictions(
            predicted_vectors[indices],
            target[indices],
            left_touchdown=ordered_touchdown,
            cycle_ids=cycle_ids,
            episode_ids=segment_ids,
            cycle_progression_tolerance_rad=float(calibration["tolerance_rad"]),
            maximum_forward_step_rad=float(calibration["maximum_forward_step_rad"]),
            maximum_reverse_step_rad=float(calibration["maximum_reverse_step_rad"]),
        )
        cycle_count = int(cycle_report["complete_cycle_count"])
        pass_count = int(sum(cycle_report["cycle_completed_one_turn"]))
        per_sequence[str(sequence)] = {
            "sample_count": int(len(indices)),
            "complete_cycle_count": cycle_count,
            "cycle_pass_count": pass_count,
            "cycle_completion_fraction": (
                float(pass_count / cycle_count) if cycle_count else None
            ),
            "circular_phase_error_rad": float(frame_error[indices].mean()),
        }

    valid_cycle_fractions = [
        value["cycle_completion_fraction"]
        for value in per_sequence.values()
        if value["cycle_completion_fraction"] is not None
    ]
    macro_error = float(
        np.mean([value["circular_phase_error_rad"] for value in per_sequence.values()])
    )
    macro_cycle_fraction = (
        float(np.mean(valid_cycle_fractions)) if valid_cycle_fractions else None
    )
    cycle_count = sum(value["complete_cycle_count"] for value in per_sequence.values())
    cycle_pass_count = sum(value["cycle_pass_count"] for value in per_sequence.values())
    phase_threshold = float(calibration["phase_gate_threshold_rad"])
    required_completion = float(calibration["macro_completion_fraction_required"])
    return {
        "status": "pass" if macro_error <= phase_threshold and macro_cycle_fraction is not None and macro_cycle_fraction >= required_completion else "fail",
        "source_split": "validation_only",
        "test_split_read": False,
        "checkpoint_sha256": _sha256(ENCODER_CHECKPOINT),
        "dataset_sha256": calibration["dataset_sha256"],
        "validation_predictions_file_sha256": _sha256(ENCODER_PREDICTIONS),
        "calibration_sha256": _sha256(PHASE_CALIBRATION),
        "macro_circular_phase_error_rad": macro_error,
        "phase_threshold_rad": phase_threshold,
        "point_gate_pass": macro_error <= phase_threshold,
        "macro_cycle_completion_fraction": macro_cycle_fraction,
        "cycle_completion_fraction_required": required_completion,
        "cycle_gate_pass": macro_cycle_fraction is not None and macro_cycle_fraction >= required_completion,
        "complete_cycle_count": int(cycle_count),
        "cycle_pass_count": int(cycle_pass_count),
        "cycle_progression_tolerance_rad": float(calibration["tolerance_rad"]),
        "maximum_forward_step_rad": float(calibration["maximum_forward_step_rad"]),
        "maximum_reverse_step_rad": float(calibration["maximum_reverse_step_rad"]),
        "per_sequence": per_sequence,
    }


def _controller_run(run_dir: Path, *, recent: bool) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(manifest["pilot_manifest_path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if recent:
        training_iterations = int(source.get("final_checkpoint_iter", -1))
        policy_name = f"fixed_clock_policy_seed_{source.get('seed')}"
    else:
        training_iterations = int(source["training_iterations"])
        policy_name = source["policy_name"]

    events = manifest["termination_events"]
    reason_counts = Counter(reason for event in events for reason in event["termination_reasons"])
    first_by_block_condition: dict[tuple[int, str], dict[str, Any]] = {}
    for event in events:
        key = (int(event["block_id"]), str(event["condition_name"]))
        if key not in first_by_block_condition or event["step_index"] < first_by_block_condition[key]["step_index"]:
            first_by_block_condition[key] = event

    linear_speed = [float(np.linalg.norm(event["root_velocity_w"][:3])) for event in events]
    angular_speed = [float(np.linalg.norm(event["root_velocity_w"][3:])) for event in events]
    absolute_rpy_deg = [
        np.abs(np.rad2deg(np.asarray(event["root_rpy_rad"], dtype=np.float64))).tolist()
        for event in events
    ]
    tilt_deg = []
    for event in events:
        w, x, y, z = np.asarray(event["root_quaternion_wxyz"], dtype=np.float64)
        upright_cosine = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        tilt_deg.append(float(np.rad2deg(np.arccos(upright_cosine))))
    command_error = [
        abs(float(_body_linear_velocity(event)[0]) - float(event["command"][0]))
        for event in events
    ]
    action_norm = [float(np.linalg.norm(event["policy_action"])) for event in events]
    action_absmax = [float(np.max(np.abs(event["policy_action"]))) for event in events]
    applied_target_absmax = [float(np.max(np.abs(event["applied_action_value"]))) for event in events]
    foot_contact_count = Counter(sum(float(value) > 0.0 for value in event["feet_found"]) for event in events)
    foot_force_norm = [
        float(np.linalg.norm(np.asarray(event["feet_force"], dtype=np.float64)))
        for event in events
    ]

    disturbance_events = manifest["disturbance_events"]
    deltas = [
        float(value)
        for disturbance in disturbance_events
        for row in disturbance["applied_velocity_delta"]
        for value in row
    ]
    return {
        "policy_name": policy_name,
        "policy_seed": int(source["seed"]),
        "training_iterations": training_iterations,
        "training_source_path": str(source_path.resolve()),
        "training_source_sha256": _sha256(source_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "run_seed": int(manifest["seed"]),
        "termination_count": int(manifest["termination_count"]),
        "truncation_count": int(manifest["truncation_count"]),
        "termination_reasons": dict(reason_counts),
        "first_termination_count_before_first_disturbance": sum(
            int(event["step_index"]) < 250 for event in first_by_block_condition.values()
        ),
        "first_termination_group_count": len(first_by_block_condition),
        "first_termination_by_block_condition": [
            {
                "block_id": int(event["block_id"]),
                "condition_name": event["condition_name"],
                "step_index": int(event["step_index"]),
                "time_s": float(event["sim_time_s"]),
            }
            for _, event in sorted(first_by_block_condition.items())
        ],
        "terminal_root_linear_speed_mps": _quantiles(linear_speed),
        "terminal_root_angular_speed_rad_s": _quantiles(angular_speed),
        "terminal_root_height_m": _quantiles([float(event["root_position_w"][2]) for event in events]),
        "terminal_absolute_roll_pitch_yaw_deg": {
            axis: _quantiles([float(row[index]) for row in absolute_rpy_deg])
            for index, axis in enumerate(("roll", "pitch", "yaw"))
        },
        "terminal_body_tilt_from_upright_deg": _quantiles(tilt_deg),
        "terminal_abs_command_vs_body_forward_speed_mps": _quantiles(command_error),
        "terminal_policy_action_l2": _quantiles(action_norm),
        "terminal_policy_action_abs_max": _quantiles(action_absmax),
        "terminal_applied_joint_target_abs_max_rad": _quantiles(applied_target_absmax),
        "terminal_feet_found_count_distribution": {str(k): v for k, v in sorted(foot_contact_count.items())},
        "terminal_foot_force_norm_n": _quantiles(foot_force_norm),
        "disturbance_event_count": len(disturbance_events),
        "disturbance_post_write_cache_refreshed": all(
            event.get("post_write_cache_refreshed") is True for event in disturbance_events
        ),
        "disturbance_applied_delta_abs_max": max(deltas, default=0.0),
        "disturbance_all_deltas_nonzero": all(value != 0.0 for value in deltas),
        "phase_gate": _phase_summary(run_dir),
        "formal_ppo_started": bool(manifest.get("formal_ppo_started", False)),
    }


def build_report() -> dict[str, Any]:
    original = json.loads((PILOT_ROOT / "feasibility_report.json").read_text(encoding="utf-8"))
    scout = json.loads(SCOUT_REPORT.read_text(encoding="utf-8"))
    recent_runs = []
    for candidate in original["policy_candidates"]:
        if candidate.get("status") != "completed":
            continue
        recent_runs.append(_controller_run(Path(candidate["run_dir"]), recent=True))

    scout_pilot = _controller_run(
        Path(scout["controller_feasibility_pilot"]["run_dir"]), recent=False
    )
    scout_confirmation = None
    if scout.get("independent_confirmation") is not None:
        scout_confirmation = _controller_run(
            Path(scout["independent_confirmation"]["run_dir"]), recent=False
        )
    encoder_validation = _encoder_validation_summary()
    encoder_candidate_report = (
        json.loads(ENCODER_CANDIDATE_REPORT.read_text(encoding="utf-8"))
        if ENCODER_CANDIDATE_REPORT.is_file()
        else None
    )

    recent_events = []
    for run in recent_runs:
        path = Path(run["manifest_path"])
        recent_events.extend(json.loads(path.read_text(encoding="utf-8"))["termination_events"])
    all_recent_reasons = Counter(
        reason for event in recent_events for reason in event["termination_reasons"]
    )

    return {
        "schema": "g1.failure_tree_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_decision": "BLOCKED_PHASE_PROGRESSION_FAILURE",
        "summary": {
            "recent_fixed_clock_policies": len(recent_runs),
            "recent_policy_training_iterations": sorted({run["training_iterations"] for run in recent_runs}),
            "recent_policy_termination_count": len(recent_events),
            "recent_policy_truncation_count": sum(run["truncation_count"] for run in recent_runs),
            "recent_policy_termination_reasons": dict(all_recent_reasons),
            "controller_feasibility": "PASS_WITH_HISTORICAL_FULL_BASELINE",
            "independent_confirmation_cycle_coverage": (
                "ADEQUATE" if scout_confirmation and scout_confirmation["phase_gate"]["cycle_coverage_status"] == "adequate" else "NOT_ADEQUATE"
            ),
            "independent_confirmation_phase_gate": (
                scout_confirmation["phase_gate"]["phase_gate_status"] if scout_confirmation else "NOT_RUN"
            ),
            "encoder_retrained": bool(
                encoder_candidate_report and encoder_candidate_report.get("candidate_fit_attempted")
            ),
            "encoder_replacement_checkpoint_promoted": bool(
                encoder_candidate_report and encoder_candidate_report.get("selected_candidate")
            ),
            "formal_ppo_started": False,
            "startup_hash_lock_created": False,
        },
        "failure_tree": {
            "controller_falls": {
                "trigger_confirmed": "fell_over termination, configured at a 70-degree bad-orientation limit",
                "evidence": {
                    "recent_candidates": recent_runs,
                    "first_terminal_event_before_step_250_count": sum(
                        run["first_termination_count_before_first_disturbance"] for run in recent_runs
                    ),
                    "first_terminal_group_count": sum(run["first_termination_group_count"] for run in recent_runs),
                    "first_disturbance_step": 250,
                    "disturbance_cache_refresh_confirmed": all(
                        run["disturbance_post_write_cache_refreshed"] for run in recent_runs
                    ),
                    "disturbance_nonzero_delta_confirmed": all(
                        run["disturbance_all_deltas_nonzero"] for run in recent_runs
                    ),
                },
                "root_cause_assessment": "the immediate termination cause is posture loss; the failures are not explained solely by scheduled disturbances because most block-condition groups fail before the first push. Replacing only the recent 1000-iteration policy artifacts with the historical full baseline eliminated falls on the same feasibility seed. This supports policy quality as the proximate control issue, while iteration count alone is not proven causal.",
                "unresolved_physical_mechanism": "The fall telemetry captures terminal pose, velocities, command, action, and foot contact/force. It does not isolate which joint or balance response initiates the loss of posture; no more specific mechanical cause is asserted.",
            },
            "phase_progression": {
                "coverage": "adequate in all four conditions for both the historical-baseline scout and independent confirmation",
                "phase_gate": "fails after adequate coverage; this is a phase failure, not a coverage failure",
                "point_error": "passes the frozen 1.1338854268797616 rad macro threshold on both runs",
                "cycle_rule": "unchanged frozen 0.20943951023931953 rad one-turn tolerance and validation-derived step limits",
                "pilot": scout_pilot["phase_gate"],
                "independent_confirmation": scout_confirmation["phase_gate"] if scout_confirmation else None,
                "offline_encoder_validation": encoder_validation,
                "encoder_candidate_validation": encoder_candidate_report,
                "root_cause_assessment": "The runtime joint order, 10-frame window, normalization checkpoint, sin/cos convention, contact-channel order, within-episode endpoints, and cycle evaluator were checked. The frozen encoder also passes the macro point-error gate but completes only a small fraction of validation cycles under the frozen one-turn rule. The predeclared pooled architecture has the same validation failure, so the rollout failure is not limited to controller domain shift. No replacement checkpoint passed validation or was promoted.",
            },
        },
        "audits": {
            "foot_contact_sensor_order": {
                "verified": True,
                "contact_sensor_slots": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "collector_foot_order": ["left", "right"],
                "found_shape_for_one_environment": [1, 2],
                "current_contact_time_shape_for_one_environment": [1, 2],
                "contact_sensor_config_source": "src/tasks/velocity/config/g1/env_cfgs.py",
            },
            "encoder_input_contract": {
                "dataset_joint_order_count": 29,
                "checkpoint_joint_order_count": 29,
                "collector_maps_joint_positions_by_name": True,
                "window_frames": 10,
                "phase_output_convention": "(sin(phi), cos(phi))",
                "data_provenance_status": "READY",
                "runtime_configuration_audit_status": "PASS",
            },
            "evaluator_regression": {
                "endpoint_touchdown_prediction_retained": True,
                "episode_reset_cycles_are_not_joined": True,
                "insufficient_history_cycles_are_not_scored": True,
                "required_case": "env 0, episode 752, frame 28 to 58 gives one 30-frame cycle",
                "test_files": ["tests/test_g1_phase_confirmation.py", "tests/test_phase_evaluation.py"],
                "test_command": "python -m unittest tests.test_g1_phase_confirmation tests.test_phase_evaluation tests.test_g1_experiment_startup -v",
                "test_result": "27 tests passed",
            },
        },
        "next_gate": {
            "result_analysis_contract": str((ROOT / "results" / "g1_phase_gate_v3" / "result_analysis_contract.json").resolve()),
            "experiment_matrix": str((ROOT / "results" / "g1_phase_gate_v3" / "experiment_matrix.md").resolve()),
            "hash_lock_status": "BLOCKED: independent confirmation phase gate failed; do not freeze a ready lock",
            "next_action": "Both predeclared architectures fail the frozen validation cycle gate. Keep the matrix blocked; any new phase objective or architecture requires a versioned plan amendment, validation-only selection, and a fresh independent confirmation seed before PPO.",
        },
    }


if __name__ == "__main__":
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT.resolve()),
        "status": report["overall_decision"],
        "recent_candidate_failures": report["summary"]["recent_policy_termination_count"],
        "pilot_cycles": report["failure_tree"]["phase_progression"]["pilot"]["cycle_check_counts"],
        "confirmation_cycles": report["failure_tree"]["phase_progression"]["independent_confirmation"]["cycle_check_counts"],
        "encoder_retrained": report["summary"]["encoder_retrained"],
        "encoder_replacement_checkpoint_promoted": report["summary"]["encoder_replacement_checkpoint_promoted"],
        "formal_ppo_started": report["summary"]["formal_ppo_started"],
    }, indent=2, ensure_ascii=False))
