"""Evaluation contracts for the contact-event phase and control study.

These helpers consume frozen predictions and rollout summaries.  They never
turn training termination scalars into test fall probabilities and never use
test metrics to select an encoder.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np


class RolloutGateError(RuntimeError):
    """Raised when held-out phase success does not transfer to policy rollout."""


def estimate_seed_count(
    paired_improvements: Iterable[float],
    *,
    minimum_effect: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Estimate paired training seeds with a normal approximation.

    The estimate is based only on engineering-trial seed differences.  The
    returned count is a planning value, not evidence from the final test set.
    """
    values = np.asarray(list(paired_improvements), dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two engineering-trial seed differences are required")
    if minimum_effect <= 0.0 or not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("invalid effect, alpha, or power")
    standard_deviation = float(values.std(ddof=1))
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(target_power)
    raw = ((z_alpha + z_power) * standard_deviation / minimum_effect) ** 2
    return max(2, int(np.ceil(raw)))


def _phase_angles(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 2:
        raise ValueError("phase vectors must have final dimension 2")
    if not np.isfinite(array).all():
        raise ValueError("phase vectors must be finite")
    return np.arctan2(array[..., 0], array[..., 1])


def _target_angles(target: np.ndarray) -> np.ndarray:
    array = np.asarray(target)
    if array.shape[-1:] == (2,):
        return _phase_angles(array)
    if not np.isfinite(array).all():
        raise ValueError("target phase must be finite")
    return np.asarray(array, dtype=np.float64)


def _circular_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * values)))


def evaluate_phase_predictions(
    predicted_phase: np.ndarray,
    target_phase: np.ndarray,
    *,
    left_touchdown: np.ndarray | None = None,
    right_touchdown: np.ndarray | None = None,
    fixed_clock_phase: np.ndarray | None = None,
    cycle_ids: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
    cycle_progression_tolerance_rad: float = 0.0,
    maximum_forward_step_rad: float | None = None,
    maximum_reverse_step_rad: float | None = None,
    threshold_rad: float | None = None,
) -> dict[str, Any]:
    """Report held-out circular error and contact-event progression metrics."""
    predicted = _phase_angles(predicted_phase)
    target = _target_angles(target_phase)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target phase shapes must match")
    if cycle_progression_tolerance_rad < 0.0:
        raise ValueError("cycle_progression_tolerance_rad must be non-negative")
    if maximum_forward_step_rad is not None and maximum_forward_step_rad <= 0.0:
        raise ValueError("maximum_forward_step_rad must be positive")
    if maximum_reverse_step_rad is not None and maximum_reverse_step_rad < 0.0:
        raise ValueError("maximum_reverse_step_rad must be non-negative")
    error = _circular_abs(predicted - target)
    report: dict[str, Any] = {
        "circular_phase_error_rad": float(error.mean()),
        "circular_phase_error_median_rad": float(np.median(error)),
        "sample_count": int(error.size),
    }

    if left_touchdown is not None:
        mask = np.asarray(left_touchdown, dtype=bool)
        if mask.shape != predicted.shape:
            raise ValueError("left_touchdown must match phase shape")
        values = _circular_abs(predicted[mask])
        report["left_touchdown_phase_error_rad"] = (
            float(values.mean()) if values.size else None
        )
        report["left_touchdown_count"] = int(values.size)

    if right_touchdown is not None:
        mask = np.asarray(right_touchdown, dtype=bool)
        if mask.shape != predicted.shape:
            raise ValueError("right_touchdown must match phase shape")
        values = np.mod(predicted[mask], 2.0 * np.pi) / (2.0 * np.pi)
        report["right_touchdown_phase_cycles"] = values.tolist()
        report["right_touchdown_count"] = int(values.size)

    if cycle_ids is not None:
        ids = np.asarray(cycle_ids)
        if ids.shape != predicted.shape:
            raise ValueError("cycle_ids must match phase shape")
        if episode_ids is None:
            episodes = np.zeros(predicted.shape, dtype=np.int64)
        else:
            episodes = np.asarray(episode_ids)
            if episodes.shape != predicted.shape:
                raise ValueError("episode_ids must match phase shape")
        touchdown_mask = None
        if left_touchdown is not None:
            touchdown_mask = np.asarray(left_touchdown, dtype=bool)
            if touchdown_mask.shape != predicted.shape:
                raise ValueError("left_touchdown must match phase shape")
        per_cycle: list[float] = []
        per_cycle_pass: list[bool] = []
        per_cycle_max_forward_step: list[float] = []
        per_cycle_max_reverse_step: list[float] = []
        closed_cycle_count = 0
        unclosed_cycle_count = 0
        finite_ids = np.isfinite(ids)
        segment_starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
        segment_ends = np.r_[segment_starts[1:], len(episodes)]
        for segment_start, segment_end in zip(segment_starts, segment_ends):
            segment_indices = np.arange(segment_start, segment_end)
            for cycle in np.unique(ids[segment_indices[finite_ids[segment_indices]]]):
                cycle_indices = segment_indices[finite_ids[segment_indices] & (ids[segment_indices] == cycle)]
                if cycle_indices.size < 2:
                    unclosed_cycle_count += 1
                    continue

                # cycle_ids label half-open touchdown intervals. Include the
                # next touchdown only when it belongs to the same episode.
                start_index = int(cycle_indices[0])
                last_index = int(cycle_indices[-1])
                has_start = touchdown_mask is None or bool(touchdown_mask[start_index])
                if touchdown_mask is None:
                    candidates = segment_indices[
                        finite_ids[segment_indices]
                        & (segment_indices > last_index)
                        & (ids[segment_indices] != cycle)
                    ]
                else:
                    candidates = segment_indices[
                        finite_ids[segment_indices]
                        & touchdown_mask[segment_indices]
                        & (segment_indices > last_index)
                    ]
                has_endpoint = bool(has_start and candidates.size)
                values = predicted[cycle_indices]
                if has_endpoint:
                    end_index = int(candidates[0])
                    values = np.concatenate((values, predicted[end_index:end_index + 1]))
                    closed_cycle_count += 1
                else:
                    unclosed_cycle_count += 1
                if values.size > 1 and has_endpoint:
                    unwrapped = np.unwrap(values)
                    step_delta = np.diff(unwrapped)
                    net_progression = float(unwrapped[-1] - unwrapped[0])
                    per_cycle.append(net_progression)
                    max_forward = float(np.max(step_delta))
                    max_reverse = float(max(0.0, -float(np.min(step_delta))))
                    per_cycle_max_forward_step.append(max_forward)
                    per_cycle_max_reverse_step.append(max_reverse)
                    if cycle_progression_tolerance_rad == 0.0:
                        one_turn = net_progression >= 2.0 * np.pi
                    else:
                        one_turn = abs(net_progression - 2.0 * np.pi) <= cycle_progression_tolerance_rad
                    if maximum_forward_step_rad is not None:
                        one_turn = one_turn and max_forward <= maximum_forward_step_rad + 1e-9
                    if maximum_reverse_step_rad is not None:
                        one_turn = one_turn and max_reverse <= maximum_reverse_step_rad + 1e-9
                    per_cycle_pass.append(bool(one_turn))
        report["cycle_net_progression_rad"] = per_cycle
        report["complete_cycle_count"] = int(closed_cycle_count)
        report["unclosed_cycle_count"] = int(unclosed_cycle_count)
        report["cycle_progression_tolerance_rad"] = float(cycle_progression_tolerance_rad)
        report["cycle_completed_one_turn"] = per_cycle_pass
        report["cycle_max_forward_step_rad"] = per_cycle_max_forward_step
        report["cycle_max_reverse_step_rad"] = per_cycle_max_reverse_step
        report["cycle_completed_one_turn_fraction"] = (
            float(np.mean(per_cycle_pass))
            if per_cycle_pass
            else None
        )
    else:
        unwrapped = np.unwrap(predicted, axis=0)
        net = unwrapped[-1] - unwrapped[0] if unwrapped.shape[0] else np.zeros(())
        report["net_progression_rad"] = float(np.asarray(net).mean()) if np.asarray(net).size else 0.0
        report["completed_one_turn"] = bool(np.all(np.abs(net) >= 2.0 * np.pi))

    if fixed_clock_phase is not None:
        baseline = _target_angles(np.asarray(fixed_clock_phase))
        if baseline.shape != target.shape:
            raise ValueError("fixed_clock_phase must match target phase shape")
        baseline_error = _circular_abs(baseline - target)
        report["fixed_clock_circular_phase_error_rad"] = float(baseline_error.mean())
        report["phase_error_improvement_over_fixed_clock_rad"] = float(
            baseline_error.mean() - error.mean()
        )

    if threshold_rad is not None:
        if threshold_rad <= 0.0:
            raise ValueError("threshold_rad must be positive")
        report["phase_threshold_rad"] = float(threshold_rad)
        report["phase_pass"] = bool(float(error.mean()) <= threshold_rad)
    return report


def calibrate_cycle_progression(
    validation_predicted_phase: np.ndarray,
    validation_sequence_ids: np.ndarray,
    validation_frame_indices: np.ndarray,
    validation_left_touchdown: np.ndarray,
    *,
    task_resolution_rad: float,
    quantile: float = 0.95,
    validation_minimum_completion_fraction: float = 0.90,
) -> dict[str, Any]:
    """Freeze rollout-cycle tolerances using validation only.

    The cycle-end tolerance is the selected quantile of the one-frame phase
    resolution (2π / touchdown interval in frames). It must fit inside the
    pre-registered task resolution; model errors are never used to widen it.
    Forward and backward step limits describe the frozen validation encoder.
    """
    predicted = np.asarray(validation_predicted_phase, dtype=np.float64)
    sequences = np.asarray(validation_sequence_ids)
    frames = np.asarray(validation_frame_indices)
    touchdown = np.asarray(validation_left_touchdown, dtype=bool)
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("validation predictions must have shape [samples, 2]")
    if any(values.shape != (len(predicted),) for values in (sequences, frames, touchdown)):
        raise ValueError("validation sequence, frame, and touchdown arrays must match predictions")
    if len(predicted) == 0 or not np.isfinite(predicted).all():
        raise ValueError("validation predictions must be non-empty and finite")
    if not np.isfinite(frames).all() or not np.equal(frames, np.floor(frames)).all():
        raise ValueError("validation frame indices must be finite integers")
    if task_resolution_rad <= 0.0 or not 0.0 < quantile < 1.0:
        raise ValueError("task resolution must be positive and quantile must be in (0, 1)")
    if not 0.0 < validation_minimum_completion_fraction <= 1.0:
        raise ValueError("validation minimum completion fraction must be in (0, 1]")

    angles = _phase_angles(predicted)
    periods: list[int] = []
    cycle_angle_series: list[np.ndarray] = []
    seen_frames: set[tuple[str, int]] = set()
    for sequence in dict.fromkeys(sequences.tolist()):
        indices = np.flatnonzero(sequences == sequence)
        order = indices[np.argsort(frames[indices], kind="stable")]
        ordered_frames = frames[order].astype(np.int64)
        keys = [(str(sequence), int(frame)) for frame in ordered_frames]
        if any(key in seen_frames for key in keys):
            raise ValueError("validation has duplicate frame indices within a sequence")
        seen_frames.update(keys)
        breaks = np.flatnonzero(np.diff(ordered_frames) != 1) + 1
        for segment in np.split(order, breaks):
            starts = np.flatnonzero(touchdown[segment])
            for first, last in zip(starts[:-1], starts[1:]):
                cycle_indices = segment[first:last + 1]
                period = int(frames[cycle_indices[-1]] - frames[cycle_indices[0]])
                if period <= 0:
                    continue
                periods.append(period)
                cycle_angles = np.unwrap(angles[cycle_indices])
                cycle_angle_series.append(cycle_angles)

    if not periods or not cycle_angle_series:
        raise ValueError("validation contains no contiguous complete touchdown cycles")
    tolerance = float(np.quantile(2.0 * np.pi / np.asarray(periods, dtype=np.float64), quantile))
    if tolerance > task_resolution_rad:
        raise ValueError(
            "validation sample resolution cannot support the pre-registered task resolution"
        )
    deltas = np.concatenate([np.diff(values) for values in cycle_angle_series])
    maximum_forward_step = float(np.quantile(deltas, 0.99))
    maximum_reverse_step = float(np.quantile(np.maximum(-deltas, 0.0), 0.99))
    if maximum_forward_step <= 0.0:
        raise ValueError("validation phase has no forward progression")
    maximum_forward_step += 1e-6
    maximum_reverse_step += 1e-6
    completion: list[bool] = []
    for cycle_angles in cycle_angle_series:
        deltas = np.diff(cycle_angles)
        net = float(cycle_angles[-1] - cycle_angles[0])
        max_forward = float(deltas.max(initial=0.0))
        max_reverse = float(np.maximum(-deltas, 0.0).max(initial=0.0))
        completion.append(
            abs(net - 2.0 * np.pi) <= tolerance
            and max_forward <= maximum_forward_step
            and max_reverse <= maximum_reverse_step
        )
    completion_fraction = float(np.mean(completion)) if completion else 0.0
    return {
        "tolerance_rad": tolerance,
        "tolerance_source": f"{quantile:.3f} quantile of one-frame phase resolution from validation touchdown intervals",
        "task_resolution_rad": float(task_resolution_rad),
        "maximum_forward_step_rad": maximum_forward_step,
        "maximum_reverse_step_rad": maximum_reverse_step,
        "step_limit_source": "validation encoder predictions on contiguous complete cycles",
        "validation_cycle_count": len(periods),
        "validation_period_frames_q05_q50_q95": [
            float(value) for value in np.quantile(periods, [0.05, 0.5, 0.95])
        ],
        "validation_cycle_completion_fraction_diagnostic": completion_fraction,
        "validation_minimum_completion_fraction_reference": float(validation_minimum_completion_fraction),
        "resolution_calibration_pass": True,
    }


def select_macro_phase_threshold(
    validation_predicted_phase: np.ndarray,
    validation_target_phase: np.ndarray,
    validation_sequence_ids: np.ndarray,
    *,
    quantile: float = 0.95,
    task_resolution_rad: float | None = None,
) -> dict[str, Any]:
    """Freeze a rollout macro-error threshold from sequence-macro validation errors."""
    predicted = _phase_angles(validation_predicted_phase)
    target = _target_angles(validation_target_phase)
    sequences = np.asarray(validation_sequence_ids)
    if predicted.shape != target.shape or sequences.shape != (len(predicted),):
        raise ValueError("validation predictions, targets, and sequence IDs must align")
    if not np.isfinite(target).all():
        raise ValueError("validation target phase must be finite")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    if task_resolution_rad is not None and task_resolution_rad <= 0.0:
        raise ValueError("task_resolution_rad must be positive")

    frame_errors = _circular_abs(predicted - target)
    sequence_errors = {
        str(sequence): float(frame_errors[sequences == sequence].mean())
        for sequence in dict.fromkeys(sequences.tolist())
    }
    if len(sequence_errors) < 2:
        raise ValueError("macro phase calibration requires at least two validation sequences")
    macro_values = np.asarray(list(sequence_errors.values()), dtype=np.float64)
    threshold = float(np.quantile(macro_values, quantile))
    if task_resolution_rad is not None:
        threshold = max(threshold, float(task_resolution_rad))
    return {
        "phase_gate_threshold_rad": threshold,
        "threshold_quantile": float(quantile),
        "threshold_source": (
            f"{quantile:.3f} quantile of equal-weight sequence-macro validation circular errors"
        ),
        "validation_sequence_count": len(sequence_errors),
        "validation_sequence_macro_errors_rad": sequence_errors,
        "validation_macro_error_rad": float(macro_values.mean()),
        "task_resolution_rad": None if task_resolution_rad is None else float(task_resolution_rad),
    }


def select_phase_threshold(
    validation_predicted_phase: np.ndarray,
    validation_target_phase: np.ndarray,
    *,
    quantile: float = 0.95,
    task_resolution_rad: float | None = None,
) -> float:
    """Freeze a phase gate from validation errors before test evaluation."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    errors = _circular_abs(
        _phase_angles(validation_predicted_phase)
        - _target_angles(validation_target_phase)
    )
    threshold = float(np.quantile(errors, quantile))
    if task_resolution_rad is not None:
        if task_resolution_rad <= 0.0:
            raise ValueError("task_resolution_rad must be positive")
        threshold = max(threshold, float(task_resolution_rad))
    return threshold


def evaluate_rollout_gate(
    heldout_report: Mapping[str, Any],
    onpolicy_report: Mapping[str, Any],
    *,
    threshold_rad: float,
) -> dict[str, Any]:
    """Require phase success on both held-out motion and policy rollouts."""
    heldout_error = float(heldout_report["circular_phase_error_rad"])
    onpolicy_error = float(onpolicy_report["circular_phase_error_rad"])
    heldout_pass = heldout_error <= threshold_rad
    onpolicy_pass = onpolicy_error <= threshold_rad
    result = {
        "heldout_phase_pass": heldout_pass,
        "onpolicy_phase_pass": onpolicy_pass,
        "threshold_rad": float(threshold_rad),
    }
    if heldout_pass and not onpolicy_pass:
        raise RolloutGateError(
            "held-out motion passes the phase gate but on-policy rollout does not"
        )
    result["phase_rollout_gate_pass"] = bool(heldout_pass and onpolicy_pass)
    return result


def control_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_group: str = "joint",
    baseline_group: str = "fixed_clock",
    minimum_mean_improvement: float = 0.0,
) -> dict[str, Any]:
    """Aggregate one independent observation per training seed and group."""
    by_group: dict[str, dict[Any, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        group = str(record["group"])
        seed = record["seed"]
        if seed in by_group[group]:
            raise ValueError(f"duplicate training seed {seed!r} in group {group!r}")
        if "xy_speed_error" not in record:
            raise ValueError("control records require the pre-registered xy_speed_error")
        by_group[group][seed] = record

    summaries: dict[str, Any] = {}
    for group, group_records in by_group.items():
        values = np.asarray([float(item["xy_speed_error"]) for item in group_records.values()])
        summaries[group] = {
            "seed_count": int(values.size),
            "xy_speed_error_mean": float(values.mean()),
            "xy_speed_error_by_seed": {
                str(seed): float(item["xy_speed_error"])
                for seed, item in group_records.items()
            },
        }
        # Preserve an explicitly evaluated test probability; do not infer one
        # from a training termination scalar.
        if all("test_fall_probability" in item for item in group_records.values()):
            summaries[group]["test_fall_probability_by_seed"] = {
                str(seed): float(item["test_fall_probability"])
                for seed, item in group_records.items()
            }

    if primary_group not in summaries or baseline_group not in summaries:
        raise ValueError("both primary and fixed-clock groups are required")
    improvement = summaries[baseline_group]["xy_speed_error_mean"] - summaries[primary_group]["xy_speed_error_mean"]
    return {
        "groups": summaries,
        "primary_comparison": {
            "primary_group": primary_group,
            "baseline_group": baseline_group,
            "mean_xy_speed_error_improvement": float(improvement),
            "meets_minimum_practical_improvement": bool(
                improvement >= minimum_mean_improvement
            ),
        },
        "statistical_unit": "training_seed",
    }
