"""Is the ceiling a per-frame estimator problem or a temporal-integration one?

Every arm in this project is a memoryless map from a window of joint angles to
a phase. The gate does not score that map. It scores the *net progression*
between two ground-truth touchdowns, which is a sum of the per-frame increments
over 45-122 frames, against a tolerance of pi/6.

If the per-frame estimate has scatter sigma, a memoryless map's accumulated
error behaves like sigma * sqrt(N). With sigma about 0.22 rad here (measured
independently: the median spread of the true phase over each frame's nearest
neighbours in joint space, within and across subjects) and N between 45 and
122, that is 1.5-2.5 rad against a 0.524 rad tolerance. If that arithmetic is
the cause, then a purely temporal constraint applied to the *existing*
predictions should move the gate, and no amount of retraining a memoryless map
will.

This probe applies several label-free filters to the preserved epoch-1 v2.1b
predictions and re-scores each with the unmodified gate. Nothing is trained.
The filters are causal and use frame order only -- no touchdown, no reference
phase, no label of any kind enters the filter.

  raw            control; must reproduce 0.6962 or the harness is wrong
  ema_rate       smooth the per-frame increment, then re-integrate
  monotone       clamp negative increments to zero (no backtracking)
  monotone_ema   both

Validation split only. The test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    V4_CALIBRATION_PATH,
    build_context_windows,
    split_arrays,
)
from scripts.summarise_phase_arm_epochs import summarise  # noqa: E402
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
PREDICTIONS = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_epoch01_predictions_preserved.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "temporal_filter_probe.json"


def angles_from_vectors(vectors: np.ndarray) -> np.ndarray:
    # Stored convention is (sin, cos); arctan2(raw[:, 1], raw[:, 0]) reflects it.
    return np.arctan2(vectors[:, 0], vectors[:, 1]).astype(np.float64)


def vectors_from_angles(angles: np.ndarray) -> np.ndarray:
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1).astype(np.float32)


def filter_sequence(angle: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    """Rebuild a phase track from its increments under a label-free constraint."""
    unwrapped = np.unwrap(angle)
    delta = np.diff(unwrapped, prepend=unwrapped[0])
    if mode in ("monotone", "monotone_ema"):
        delta = np.maximum(delta, 0.0)
    if mode in ("ema_rate", "monotone_ema"):
        smoothed = np.empty_like(delta)
        running = delta[0]
        for index, value in enumerate(delta):
            running = (1.0 - alpha) * running + alpha * value
            smoothed[index] = running
        delta = smoothed
    return unwrapped[0] + np.cumsum(delta)


def apply_filter(vectors: np.ndarray, sequence: np.ndarray, frame: np.ndarray,
                 mode: str, alpha: float) -> np.ndarray:
    angles = angles_from_vectors(vectors)
    out = np.empty_like(angles)
    for name in np.unique(sequence):
        rows = np.flatnonzero(sequence == name)
        rows = rows[np.argsort(frame[rows])]
        out[rows] = filter_sequence(angles[rows], mode, alpha)
    return vectors_from_angles(out)


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")
    _, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)
    cuts = tuple(float(x) for x in json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})

    stored = np.load(PREDICTIONS)
    raw = stored[sorted(stored.files)[0]]

    arms = [("raw", "none", 0.0)]
    arms += [(f"ema_rate_{a}", "ema_rate", a) for a in (0.5, 0.2, 0.05)]
    arms += [("monotone", "monotone", 0.0)]
    arms += [(f"monotone_ema_{a}", "monotone_ema", a) for a in (0.2, 0.05)]

    results = {}
    for label, mode, alpha in arms:
        filtered = raw if mode == "none" else apply_filter(raw, validation["sequence"], validation["frame"], mode, alpha)
        report = score_phase_cycles_v4_2(
            filtered, reference, validation["sequence"], validation["frame"],
            validation["left_touchdown"].astype(bool),
            condition_ids=validation["sequence"], episode_ids=validation["sequence"],
            history_valid=history_valid,
            period_cuts_frames=cuts, required_conditions=conditions,
            minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=1,
        )
        row = summarise(report)
        results[label] = row
        print("%-18s macro %.4f point %.4f net %+.2f sd %.2f slip %3d long %+.2f"
              % (label, row["macro_completion"], row["point_error_rad"], row["mean_net_error"],
                 row["sd_net_error"], row["whole_turn_slip_count"], row.get("mean_net_error_long", 0.0)),
              flush=True)

    payload = {
        "schema": "g1.temporal_filter_probe.v1",
        "split": "validation",
        "test_split_read": False,
        "source_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "gate": "g1-phase-gate-v4.2, unmodified",
        "filters_use": "frame order only; no touchdown, reference or label",
        "arms": results,
        "control_reproduces_baseline": abs(results["raw"]["macro_completion"] - 0.6962) < 5e-4,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
