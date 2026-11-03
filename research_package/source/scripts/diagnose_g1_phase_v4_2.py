"""Step-1 diagnostics for the v2.1 redesign, computed under the v4.2 cycle rule.

Answers three questions the v4.1 sweep report could not:

  1. Where inside the cycle does progress stall, if it stalls at all?
  2. What fraction of frame-to-frame steps move forward?
  3. Is low cycle completion explained by absolute phase error, or independent
     of it?

Everything here is validation-only scoring of train-only fits. The test split
is not opened. These probes are diagnostic: they are not candidates and cannot
be frozen.

Usage:
    python scripts/diagnose_g1_phase_v4_2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2


DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
PREDICTIONS_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_predictions.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"

PHASE_BINS = 12
RIDGE_LAMBDA = 1e4


def angles_of(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    return np.arctan2(values[:, 0], values[:, 1])


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray) -> np.ndarray:
    """Closed-form ridge on the flattened 10-frame window, target sin/cos."""
    x_tr = x_train.reshape(len(x_train), -1)
    x_va = x_validation.reshape(len(x_validation), -1)
    mean, scale = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-6
    x_tr = np.clip((x_tr - mean) / scale, -10.0, 10.0)
    x_va = np.clip((x_va - mean) / scale, -10.0, 10.0)
    design = np.concatenate([x_tr, np.ones((len(x_tr), 1))], axis=1)
    target = np.stack((np.sin(y_train), np.cos(y_train)), axis=-1)
    gram = design.T @ design + RIDGE_LAMBDA * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)
    design_va = np.concatenate([x_va, np.ones((len(x_va), 1))], axis=1)
    prediction = design_va @ weights
    return prediction / np.linalg.norm(prediction, axis=1, keepdims=True)


def cycle_frames(record: dict[str, Any], frame_lookup: dict[str, dict[int, int]]) -> np.ndarray:
    sequence = str(record["sequence_id"])
    frames = frame_lookup[sequence]
    span = np.arange(record["start_frame"], record["end_frame"] + 1, dtype=np.int64)
    missing = [int(f) for f in span if int(f) not in frames]
    if missing:
        raise ValueError(f"cycle in {sequence} has missing frames: {missing[:5]}")
    return np.asarray([frames[int(f)] for f in span], dtype=np.int64)


def stall_profile(
    report: dict[str, Any], angles: np.ndarray, reference: np.ndarray,
    frame_lookup: dict[str, dict[int, int]],
) -> dict[str, Any]:
    """Median (encoder delta / reference delta) per reference-phase bin.

    The ratio is ~1 where the clock tracks the cycle and ~0 where it stalls.
    Bins are over the reference angle at the start of each frame pair, so the
    profile says *where* in the cycle the stall happens.
    """
    ratios: list[np.ndarray] = []
    for record in report["cycle_records"]:
        if not record["evaluable"]:
            continue
        index = cycle_frames(record, frame_lookup)[:-1]  # half-open, v4.2
        if len(index) < 3:
            continue
        enc = np.unwrap(angles[index])
        ref = np.unwrap(reference[index])
        d_enc, d_ref = np.diff(enc), np.diff(ref)
        moving = d_ref > 1e-6
        if not moving.any():
            continue
        phase_at = np.mod(ref[:-1][moving], 2.0 * np.pi)
        ratios.append(np.stack((phase_at, d_enc[moving] / d_ref[moving]), axis=-1))
    if not ratios:
        return {"bins": [], "note": "no evaluable cycles"}
    stacked = np.concatenate(ratios, axis=0)
    edges = np.linspace(0.0, 2.0 * np.pi, PHASE_BINS + 1)
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (stacked[:, 0] >= low) & (stacked[:, 0] < high)
        values = stacked[mask, 1]
        bins.append({
            "phase_low_rad": float(low),
            "phase_high_rad": float(high),
            "sample_count": int(mask.sum()),
            "median_ratio": float(np.median(values)) if len(values) else None,
            "forward_fraction": float(np.mean(values > 0.0)) if len(values) else None,
        })
    medians = [row["median_ratio"] for row in bins if row["median_ratio"] is not None]
    return {
        "bins": bins,
        "stalled_bin_count": int(sum(1 for value in medians if value < 0.5)),
        "worst_bin_median_ratio": float(min(medians)) if medians else None,
        "overall_median_ratio": float(np.median(stacked[:, 1])),
    }


def forward_ratio(
    report: dict[str, Any], angles: np.ndarray, frame_lookup: dict[str, dict[int, int]],
) -> float:
    forwards: list[bool] = []
    for record in report["cycle_records"]:
        if not record["evaluable"]:
            continue
        index = cycle_frames(record, frame_lookup)[:-1]
        if len(index) < 2:
            continue
        forwards.extend((np.diff(np.unwrap(angles[index])) > 0.0).tolist())
    return float(np.mean(forwards)) if forwards else 0.0


def point_error_split(report: dict[str, Any]) -> dict[str, Any]:
    done = [r["point_error_rad"] for r in report["cycle_records"] if r["evaluable"] and r["encoder_completed_one_turn"]]
    failed = [r["point_error_rad"] for r in report["cycle_records"] if r["evaluable"] and not r["encoder_completed_one_turn"]]
    return {
        "completed_cycle_count": len(done),
        "failed_cycle_count": len(failed),
        "median_point_error_completed_rad": float(np.median(done)) if done else None,
        "median_point_error_failed_rad": float(np.median(failed)) if failed else None,
        "mean_point_error_completed_rad": float(np.mean(done)) if done else None,
        "mean_point_error_failed_rad": float(np.mean(failed)) if failed else None,
    }


def grouped(report: dict[str, Any], angles: np.ndarray, reference: np.ndarray,
            frame_lookup: dict[str, dict[int, int]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = sorted({str(record[key]) for record in report["cycle_records"]})
    for label in labels:
        subset = [r for r in report["cycle_records"] if str(r[key]) == label]
        sub_report = {**report, "cycle_records": subset}
        evaluable = [r for r in subset if r["evaluable"]]
        nets = [r["encoder_net_progression_rad"] for r in evaluable]
        steps = [r["encoder_max_abs_step_rad"] for r in evaluable]
        completion = float(np.mean([r["encoder_completed_one_turn"] for r in evaluable])) if evaluable else 0.0
        result[label] = {
            "evaluable_cycle_count": len(evaluable),
            "completion_fraction": completion,
            "median_net_rad": float(np.median(nets)) if nets else None,
            "median_max_abs_step_rad": float(np.median(steps)) if steps else None,
            "median_point_error_rad": float(np.median([r["point_error_rad"] for r in evaluable])) if evaluable else None,
            "forward_ratio": forward_ratio(sub_report, angles, frame_lookup),
            "stall": stall_profile(sub_report, angles, reference, frame_lookup),
            "point_error_split": point_error_split(sub_report),
        }
    return result


def main() -> int:
    dataset = np.load(DATA_PATH, allow_pickle=False)
    predictions = np.load(PREDICTIONS_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    sequence_ids = dataset["sequence_validation"]
    frame_ids = dataset["frame_validation"]
    reference = dataset["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi)
    touchdowns = dataset["left_touchdown_validation"].astype(bool)
    conditions = sorted(np.unique(sequence_ids).tolist())

    frame_lookup: dict[str, dict[int, int]] = {}
    for index, (sequence, frame) in enumerate(zip(sequence_ids, frame_ids)):
        frame_lookup.setdefault(str(sequence), {})[int(frame)] = index

    y_train = dataset["y_train_cycles"].astype(np.float64) * (2.0 * np.pi)
    ridge_prediction = fit_ridge(dataset["x_train"], y_train, dataset["x_validation"])

    estimators = {
        "reference_clock": np.stack((np.sin(reference), np.cos(reference)), axis=-1),
        "ridge_lambda_1e4": ridge_prediction,
        "encoder_v1": predictions["val_pred"],
    }

    payload: dict[str, Any] = {
        "schema": "g1.phase_v4_2_diagnostic.v1",
        "source_split": "validation_only",
        "test_split_read": False,
        "purpose": "step-1 diagnostics for the v2.1 objective redesign; diagnostic only, not candidates",
        "period_length_cut_points_frames": list(period_cuts),
        "definitions": {
            "stall_ratio": "median per-frame (encoder delta / reference delta) binned by reference phase",
            "forward_ratio": "fraction of frame pairs where the encoder phase strictly advances",
            "point_error_split": "mean/median per-cycle point error for completed vs failed cycles",
        },
        "estimators": {},
    }

    for name, vectors in estimators.items():
        report = score_phase_cycles_v4_2(
            vectors, reference, sequence_ids, frame_ids, touchdowns,
            condition_ids=sequence_ids, episode_ids=sequence_ids,
            history_valid=np.ones(len(frame_ids), dtype=bool),
            period_cuts_frames=period_cuts, required_conditions=conditions,
            minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=1,
        )
        angles = angles_of(vectors)
        payload["estimators"][name] = {
            "macro_completion_fraction": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "freeze_eligible": report["encoder_freeze_eligible"],
            "forward_ratio": forward_ratio(report, angles, frame_lookup),
            "point_error_split": point_error_split(report),
            "overall_stall": stall_profile(report, angles, reference, frame_lookup),
            "by_condition": grouped(report, angles, reference, frame_lookup, "condition_id"),
            "by_period_stratum": grouped(report, angles, reference, frame_lookup, "period_length_stratum"),
        }
        print(f"{name:20s} macro {report['macro_cycle_completion_fraction']:.4f} "
              f"point {report['condition_macro_point_error_rad']:.3f} "
              f"forward {payload['estimators'][name]['forward_ratio']:.3f} "
              f"stalled_bins {payload['estimators'][name]['overall_stall']['stalled_bin_count']}/{PHASE_BINS}",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diagnostic_step1_v4_2.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("wrote", (OUT_DIR / "diagnostic_step1_v4_2.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
