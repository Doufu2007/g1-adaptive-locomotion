"""Versioned cycle-level evaluation contract for the G1 phase study.

This module is intentionally separate from v3. It scores every consecutive
touchdown pair and makes coverage failures visible instead of silently
dropping cycles with gaps, reset boundaries, condition changes, or insufficient
history.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np


PHASE_GATE_VERSION = "g1-phase-gate-v4.1"
NET_TOLERANCE_RAD = float(np.pi / 6.0)
MAX_ABS_STEP_RAD = float(np.pi / 2.0)
MIN_MACRO_COMPLETION = 0.90
MIN_CONDITION_COMPLETION = 0.80
MIN_LENGTH_STRATUM_COMPLETION = 0.80
PERIOD_STRATA = ("short", "medium", "long")


def _item(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _key_text(value: Any) -> str:
    return str(_item(value))


def _phase_angles(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("phase vectors must have shape [samples, 2]")
    if not np.isfinite(values).all():
        raise ValueError("phase vectors must be finite")
    return np.arctan2(values[:, 0], values[:, 1])


def _aligned_inputs(
    encoder_phase_vectors: np.ndarray,
    reference_phase_rad: np.ndarray,
    sequence_ids: np.ndarray,
    frame_ids: np.ndarray,
    left_touchdown: np.ndarray,
    condition_ids: np.ndarray | None,
    episode_ids: np.ndarray | None,
    history_valid: np.ndarray | None,
    reference_valid: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    angles = _phase_angles(encoder_phase_vectors)
    reference = np.asarray(reference_phase_rad, dtype=np.float64)
    sequences = np.asarray(sequence_ids)
    frames = np.asarray(frame_ids)
    touchdowns = np.asarray(left_touchdown, dtype=bool)
    conditions = sequences if condition_ids is None else np.asarray(condition_ids)
    episodes = np.zeros(len(angles), dtype=np.int64) if episode_ids is None else np.asarray(episode_ids)
    history = np.ones(len(angles), dtype=bool) if history_valid is None else np.asarray(history_valid, dtype=bool)
    ref_valid = np.isfinite(reference) if reference_valid is None else np.asarray(reference_valid, dtype=bool)
    arrays = (reference, sequences, frames, touchdowns, conditions, episodes, history, ref_valid)
    if any(array.shape != (len(angles),) for array in arrays):
        raise ValueError("all phase, ID, frame, touchdown, and history arrays must align")
    if len(angles) == 0:
        raise ValueError("phase evaluation requires at least one sample")
    if not np.isfinite(frames).all() or not np.equal(frames, np.floor(frames)).all():
        raise ValueError("frame IDs must be finite integers")
    if not np.isfinite(reference[ref_valid]).all():
        raise ValueError("reference phase must be finite wherever reference_valid is true")
    frames = frames.astype(np.int64)
    return angles, reference, ref_valid, sequences, frames, touchdowns, conditions, episodes, history


def _period_pairs(
    sequences: np.ndarray,
    frames: np.ndarray,
    touchdowns: np.ndarray,
    conditions: np.ndarray,
    episodes: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for index, (sequence, episode) in enumerate(zip(sequences, episodes)):
        groups[(_item(sequence), _item(episode))].append(index)

    pairs: list[dict[str, Any]] = []
    unclosed_segments = 0
    seen_frames: set[tuple[Any, Any, int]] = set()
    for (sequence, episode), indices in groups.items():
        order = sorted(indices, key=lambda index: (int(frames[index]), index))
        for index in order:
            key = (sequence, episode, int(frames[index]))
            if key in seen_frames:
                raise ValueError("duplicate frame IDs within a sequence and episode")
            seen_frames.add(key)
        events = [index for index in order if touchdowns[index]]
        if len(events) < 2:
            unclosed_segments += 1
            continue
        frame_to_index = {int(frames[index]): index for index in order}
        for start_index, end_index in zip(events[:-1], events[1:]):
            start_frame = int(frames[start_index])
            end_frame = int(frames[end_index])
            period = end_frame - start_frame
            expected_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
            interval_indices = [frame_to_index.get(int(frame)) for frame in expected_frames]
            present = [index for index in interval_indices if index is not None]
            condition_values = {_item(conditions[index]) for index in present}
            condition = _item(conditions[start_index])
            reasons: list[str] = []
            if period <= 0:
                reasons.append("nonpositive_period")
            if len(present) != len(expected_frames):
                reasons.append("frame_gap")
            if len(condition_values) != 1:
                reasons.append("condition_boundary")
            pairs.append({
                "sequence_id": sequence,
                "episode_id": episode,
                "condition_id": condition,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "period_frames": period,
                "indices": present,
                "complete_frame_interval": len(present) == len(expected_frames),
                "failure_reasons": reasons,
            })
    return pairs, unclosed_segments


def _classify_period(period: int, cuts: tuple[float, float] | None) -> str:
    if cuts is None:
        return "unstratified"
    lower, upper = cuts
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("period cuts must be finite and strictly increasing")
    if period < lower:
        return "short"
    if period < upper:
        return "medium"
    return "long"


def _rate(records: list[dict[str, Any]], key: str, selected: list[int] | None = None) -> float:
    ids = range(len(records)) if selected is None else selected
    values = [bool(records[index][key]) for index in ids if records[index]["evaluable"]]
    return float(np.mean(values)) if values else 0.0


def _group_summary(
    records: list[dict[str, Any]], group_key: str, values: list[Any],
    *, min_cycles: int, completion_key: str,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for value in values:
        label = _key_text(value)
        indexes = [i for i, row in enumerate(records) if _key_text(row[group_key]) == label]
        evaluable = [i for i in indexes if records[i]["evaluable"]]
        episode_count = len({
            (_key_text(records[i]["sequence_id"]), _key_text(records[i]["episode_id"]))
            for i in evaluable
        })
        summary[label] = {
            "observed_cycle_count": len(indexes),
            "evaluable_cycle_count": len(evaluable),
            "completed_cycle_count": sum(bool(records[i][completion_key]) for i in evaluable),
            "completion_fraction": _rate(records, completion_key, indexes),
            "episode_segment_count": episode_count,
            "coverage_pass": len(evaluable) >= min_cycles,
        }
    return summary


def score_phase_cycles_v4(
    encoder_phase_vectors: np.ndarray,
    reference_phase_rad: np.ndarray,
    sequence_ids: np.ndarray,
    frame_ids: np.ndarray,
    left_touchdown: np.ndarray,
    *,
    condition_ids: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
    history_valid: np.ndarray | None = None,
    reference_valid: np.ndarray | None = None,
    period_cuts_frames: tuple[float, float] | None = None,
    required_conditions: list[Any] | None = None,
    minimum_cycles_per_condition: int = 10,
    minimum_cycles_per_length_stratum: int = 10,
    minimum_episode_segments_per_condition: int = 1,
) -> dict[str, Any]:
    """Score every consecutive touchdown interval without crossing boundaries."""
    if minimum_cycles_per_condition < 1 or minimum_cycles_per_length_stratum < 1:
        raise ValueError("cycle coverage requirements must be positive")
    if minimum_episode_segments_per_condition < 1:
        raise ValueError("episode coverage requirement must be positive")
    (angles, reference, ref_valid, sequences, frames, touchdowns, conditions,
     episodes, history) = _aligned_inputs(
        encoder_phase_vectors, reference_phase_rad, sequence_ids, frame_ids,
        left_touchdown, condition_ids, episode_ids, history_valid, reference_valid,
    )
    pairs, unclosed_segments = _period_pairs(sequences, frames, touchdowns, conditions, episodes)
    records: list[dict[str, Any]] = []
    point_error = np.full(len(angles), np.nan, dtype=np.float64)
    point_error[ref_valid] = np.abs(np.angle(np.exp(1j * (angles[ref_valid] - reference[ref_valid]))))
    condition_names = sorted({_key_text(value) for value in conditions})
    if required_conditions is not None:
        condition_names = sorted({_key_text(value) for value in required_conditions})

    for pair in pairs:
        period = int(pair["period_frames"])
        stratum = _classify_period(period, period_cuts_frames)
        indexes = pair["indices"]
        reasons = list(pair["failure_reasons"])
        if not indexes:
            reasons.append("no_interval_samples")
        if indexes and not bool(np.all(history[indexes])):
            reasons.append("insufficient_history")
        if indexes and not bool(np.all(ref_valid[indexes])):
            reasons.append("reference_phase_invalid")
        evaluable = not reasons and pair["complete_frame_interval"] and period > 0
        encoder_net = 0.0
        encoder_max_abs_step = 0.0
        reference_net = 0.0
        reference_max_abs_step = 0.0
        encoder_completed = False
        reference_completed = False
        cycle_error = None
        if evaluable:
            interval_angles = np.unwrap(angles[indexes])
            interval_reference = np.unwrap(reference[indexes])
            encoder_deltas = np.diff(interval_angles)
            reference_deltas = np.diff(interval_reference)
            encoder_net = float(interval_angles[-1] - interval_angles[0])
            reference_net = float(interval_reference[-1] - interval_reference[0])
            encoder_max_abs_step = float(np.max(np.abs(encoder_deltas), initial=0.0))
            reference_max_abs_step = float(np.max(np.abs(reference_deltas), initial=0.0))
            encoder_completed = bool(
                encoder_net > 0.0
                and abs(encoder_net - 2.0 * np.pi) <= NET_TOLERANCE_RAD + 1e-12
                and encoder_max_abs_step < MAX_ABS_STEP_RAD
            )
            reference_completed = bool(
                reference_net > 0.0
                and abs(reference_net - 2.0 * np.pi) <= NET_TOLERANCE_RAD + 1e-12
                and reference_max_abs_step < MAX_ABS_STEP_RAD
            )
            cycle_error = float(np.mean(point_error[indexes]))
        records.append({
            "sequence_id": _item(pair["sequence_id"]),
            "episode_id": _item(pair["episode_id"]),
            "condition_id": _item(pair["condition_id"]),
            "start_frame": int(pair["start_frame"]),
            "end_frame": int(pair["end_frame"]),
            "period_frames": period,
            "period_length_stratum": stratum,
            "evaluable": bool(evaluable),
            "failure_reasons": sorted(set(reasons)),
            "encoder_net_progression_rad": encoder_net,
            "encoder_max_abs_step_rad": encoder_max_abs_step,
            "encoder_completed_one_turn": encoder_completed,
            "reference_net_progression_rad": reference_net,
            "reference_max_abs_step_rad": reference_max_abs_step,
            "reference_completed_one_turn": reference_completed,
            "point_error_rad": cycle_error,
        })

    observed_conditions = sorted({_key_text(row["condition_id"]) for row in records})
    if required_conditions is not None:
        observed_conditions = condition_names
    condition_rows = _group_summary(
        records, "condition_id", observed_conditions,
        min_cycles=minimum_cycles_per_condition,
        completion_key="encoder_completed_one_turn",
    )
    reference_condition_rows = _group_summary(
        records, "condition_id", observed_conditions,
        min_cycles=minimum_cycles_per_condition,
        completion_key="reference_completed_one_turn",
    )
    length_rows = _group_summary(
        records, "period_length_stratum", list(PERIOD_STRATA) if period_cuts_frames is not None else ["unstratified"],
        min_cycles=minimum_cycles_per_length_stratum,
        completion_key="encoder_completed_one_turn",
    )
    reference_length_rows = _group_summary(
        records, "period_length_stratum", list(PERIOD_STRATA) if period_cuts_frames is not None else ["unstratified"],
        min_cycles=minimum_cycles_per_length_stratum,
        completion_key="reference_completed_one_turn",
    )
    condition_rates = [condition_rows[name]["completion_fraction"] for name in observed_conditions]
    condition_reference_rates = [reference_condition_rows[name]["completion_fraction"] for name in observed_conditions]
    length_rates = [row["completion_fraction"] for row in length_rows.values()]
    length_reference_rates = [row["completion_fraction"] for row in reference_length_rows.values()]
    evaluable_records = [row for row in records if row["evaluable"]]
    condition_point_errors: dict[str, float] = {}
    for name in condition_names:
        mask = np.array([_key_text(value) == name for value in conditions], dtype=bool)
        valid_mask = mask & ref_valid
        condition_point_errors[name] = float(np.mean(point_error[valid_mask])) if valid_mask.any() else float("inf")
    point_macro = float(np.mean(list(condition_point_errors.values()))) if condition_point_errors else float("inf")
    condition_coverage_pass = bool(condition_rows) and all(
        row["coverage_pass"] and row["episode_segment_count"] >= minimum_episode_segments_per_condition
        for row in condition_rows.values()
    )
    length_coverage_pass = bool(length_rows) and all(row["coverage_pass"] for row in length_rows.values())
    encoder_condition_pass = bool(condition_rows) and all(
        row["completion_fraction"] >= MIN_CONDITION_COMPLETION for row in condition_rows.values()
    )
    encoder_length_pass = bool(length_rows) and all(
        row["completion_fraction"] >= MIN_LENGTH_STRATUM_COMPLETION for row in length_rows.values()
    )
    reference_condition_pass = bool(reference_condition_rows) and all(
        row["completion_fraction"] >= MIN_CONDITION_COMPLETION for row in reference_condition_rows.values()
    )
    reference_length_pass = bool(reference_length_rows) and all(
        row["completion_fraction"] >= MIN_LENGTH_STRATUM_COMPLETION for row in reference_length_rows.values()
    )
    macro_completion = float(np.mean(condition_rates)) if condition_rates else 0.0
    reference_macro = float(np.mean(condition_reference_rates)) if condition_reference_rates else 0.0
    reference_pass = bool(
        reference_macro >= MIN_MACRO_COMPLETION and reference_condition_pass and reference_length_pass
    )
    coverage_pass = bool(
        condition_coverage_pass and length_coverage_pass
        and (required_conditions is None or set(observed_conditions) == {_key_text(x) for x in required_conditions})
    )
    return {
        "schema": "g1.phase_gate_report.v4",
        "gate_version": PHASE_GATE_VERSION,
        "cycle_rule": {
            "net_tolerance_rad": NET_TOLERANCE_RAD,
            "max_abs_step_rad_strictly_below": MAX_ABS_STEP_RAD,
            "minimum_macro_completion_fraction": MIN_MACRO_COMPLETION,
        },
        "observed_cycle_count": len(records),
        "evaluable_cycle_count": len(evaluable_records),
        "unscorable_cycle_count": len(records) - len(evaluable_records),
        "unclosed_episode_segment_count": unclosed_segments,
        "insufficient_history_cycle_count": sum("insufficient_history" in row["failure_reasons"] for row in records),
        "frame_gap_cycle_count": sum("frame_gap" in row["failure_reasons"] for row in records),
        "condition_boundary_cycle_count": sum("condition_boundary" in row["failure_reasons"] for row in records),
        "encoder_completed_cycle_count": sum(row["encoder_completed_one_turn"] for row in evaluable_records),
        "encoder_micro_cycle_completion_fraction": float(np.mean([row["encoder_completed_one_turn"] for row in evaluable_records])) if evaluable_records else 0.0,
        "encoder_completion_fraction": macro_completion,
        "macro_cycle_completion_fraction": macro_completion,
        "reference_clock_completed_cycle_count": sum(row["reference_completed_one_turn"] for row in evaluable_records),
        "reference_clock_micro_completion_fraction": float(np.mean([row["reference_completed_one_turn"] for row in evaluable_records])) if evaluable_records else 0.0,
        "reference_clock_completion_fraction": reference_macro,
        "condition_completion": condition_rows,
        "reference_condition_completion": reference_condition_rows,
        "period_length_strata": length_rows,
        "reference_period_length_strata": reference_length_rows,
        "condition_macro_point_error_rad": point_macro,
        "condition_point_error_rad": condition_point_errors,
        "coverage_pass": coverage_pass,
        "reference_clock_pass": reference_pass,
        "condition_completion_pass": encoder_condition_pass,
        "period_length_completion_pass": encoder_length_pass,
        "macro_completion_pass": macro_completion >= MIN_MACRO_COMPLETION,
        "encoder_freeze_eligible": bool(
            coverage_pass and reference_pass and encoder_condition_pass
            and encoder_length_pass and macro_completion >= MIN_MACRO_COMPLETION
        ),
        "cycle_records": records,
    }


def calibrate_phase_gate_v4(
    encoder_phase_vectors: np.ndarray,
    reference_phase_rad: np.ndarray,
    sequence_ids: np.ndarray,
    frame_ids: np.ndarray,
    left_touchdown: np.ndarray,
    *,
    condition_ids: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
    history_valid: np.ndarray | None = None,
    reference_valid: np.ndarray | None = None,
    required_conditions: list[Any] | None = None,
    minimum_cycles_per_condition: int = 10,
    minimum_cycles_per_length_stratum: int = 10,
    minimum_episode_segments_per_condition: int = 1,
) -> dict[str, Any]:
    """Freeze period strata from reference touchdown intervals and score both clocks."""
    (angles, reference, ref_valid, sequences, frames, touchdowns, conditions,
     episodes, history) = _aligned_inputs(
        encoder_phase_vectors, reference_phase_rad, sequence_ids, frame_ids,
        left_touchdown, condition_ids, episode_ids, history_valid, reference_valid,
    )
    pairs, _ = _period_pairs(sequences, frames, touchdowns, conditions, episodes)
    periods = np.asarray(
        [pair["period_frames"] for pair in pairs if pair["period_frames"] > 0 and pair["complete_frame_interval"] and not pair["failure_reasons"]],
        dtype=np.float64,
    )
    if len(periods) < 3:
        raise ValueError("calibration requires at least three contiguous reference cycles")
    cuts_array = np.quantile(periods, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    cuts = (float(cuts_array[0]), float(cuts_array[1]))
    if cuts[0] >= cuts[1]:
        raise ValueError("validation period lengths do not support three distinct strata")
    report = score_phase_cycles_v4(
        np.stack((np.sin(angles), np.cos(angles)), axis=-1),
        reference,
        sequences,
        frames,
        touchdowns,
        condition_ids=conditions,
        episode_ids=episodes,
        history_valid=history,
        reference_valid=ref_valid,
        period_cuts_frames=cuts,
        required_conditions=required_conditions,
        minimum_cycles_per_condition=minimum_cycles_per_condition,
        minimum_cycles_per_length_stratum=minimum_cycles_per_length_stratum,
        minimum_episode_segments_per_condition=minimum_episode_segments_per_condition,
    )
    encoder_report = score_phase_cycles_v4(
        encoder_phase_vectors,
        reference,
        sequences,
        frames,
        touchdowns,
        condition_ids=conditions,
        episode_ids=episodes,
        history_valid=history,
        reference_valid=ref_valid,
        period_cuts_frames=cuts,
        required_conditions=required_conditions,
        minimum_cycles_per_condition=minimum_cycles_per_condition,
        minimum_cycles_per_length_stratum=minimum_cycles_per_length_stratum,
        minimum_episode_segments_per_condition=minimum_episode_segments_per_condition,
    )
    return {
        "schema": "g1.phase_gate_calibration.v4",
        "gate_version": PHASE_GATE_VERSION,
        "source_split": "validation_only",
        "net_tolerance_rad": NET_TOLERANCE_RAD,
        "net_tolerance_source": "pre-registered period-level task resolution pi/6",
        "max_abs_step_rad": MAX_ABS_STEP_RAD,
        "max_abs_step_source": "pre-registered per-period maximum absolute adjacent increment; strict bound",
        "minimum_macro_completion_fraction": MIN_MACRO_COMPLETION,
        "minimum_condition_completion_fraction": MIN_CONDITION_COMPLETION,
        "minimum_length_stratum_completion_fraction": MIN_LENGTH_STRATUM_COMPLETION,
        "period_length_cut_points_frames": list(cuts),
        "period_length_cut_source": "1/3 and 2/3 quantiles of contiguous validation reference touchdown intervals",
        "reference_clock": {
            "completion_fraction": report["reference_clock_completion_fraction"],
            "micro_completion_fraction": report["reference_clock_micro_completion_fraction"],
            "condition_completion": report["reference_condition_completion"],
            "period_length_strata": report["reference_period_length_strata"],
            "pass": report["reference_clock_pass"],
        },
        "encoder_calibration": {
            "completion_fraction": encoder_report["encoder_completion_fraction"],
            "micro_completion_fraction": encoder_report["encoder_micro_cycle_completion_fraction"],
            "condition_completion": encoder_report["condition_completion"],
            "period_length_strata": encoder_report["period_length_strata"],
            "freeze_eligible": encoder_report["encoder_freeze_eligible"],
        },
        "reference_clock_completion_fraction": report["reference_clock_completion_fraction"],
        "reference_clock_pass": report["reference_clock_pass"],
        "coverage_pass": report["coverage_pass"],
        "encoder_completion_fraction": encoder_report["encoder_completion_fraction"],
        "encoder_freeze_eligible": encoder_report["encoder_freeze_eligible"],
        "reference_report": report,
        "encoder_report": encoder_report,
        "thresholds_valid": bool(report["reference_clock_pass"] and report["coverage_pass"]),
        "calibration_status": (
            "READY_REFERENCE_CLOCK_ENC_V1_REJECTED"
            if report["reference_clock_pass"] and report["coverage_pass"] and not encoder_report["encoder_freeze_eligible"]
            else "READY"
            if report["reference_clock_pass"] and report["coverage_pass"] and encoder_report["encoder_freeze_eligible"]
            else "BLOCKED_REFERENCE_OR_COVERAGE_FAILURE"
        ),
    }


def select_v4_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Choose by completion first, point error second, and short context last."""
    if not candidates:
        raise ValueError("at least one validation candidate is required")
    for name, report in candidates.items():
        for key in ("macro_cycle_completion_fraction", "condition_macro_point_error_rad", "context_frames"):
            if key not in report:
                raise ValueError(f"candidate {name!r} is missing {key}")
        values = [float(report[key]) for key in ("macro_cycle_completion_fraction", "condition_macro_point_error_rad", "context_frames")]
        epoch = float(report.get("epoch", 0))
        if not np.isfinite([*values, epoch]).all() or values[0] < 0.0 or values[2] < 1.0 or epoch < 0.0:
            raise ValueError(f"candidate {name!r} has invalid selection values")
    return min(
        candidates,
        key=lambda name: (
            -float(candidates[name]["macro_cycle_completion_fraction"]),
            float(candidates[name]["condition_macro_point_error_rad"]),
            float(candidates[name]["context_frames"]),
            float(candidates[name].get("epoch", 0)),
            name,
        ),
    )
