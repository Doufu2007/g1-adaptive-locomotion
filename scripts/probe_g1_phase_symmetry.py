"""Is the phase identifiable only up to the gait's left/right symmetry?

Probe (b) found the point error is bimodal: 54.7% of frames within 0.5 rad and
20.3% within 0.1 rad of pi, with no constant offset and no phase or subject
localisation. That shape has a specific meaning for a cyclic bipedal gait. If
the left and right halves of the cycle present near-identical joint
configurations -- which a symmetric walk nearly does -- then the joint angles
determine the phase only modulo pi. The estimator is then right half the time
and half a cycle out the other half, which is exactly what the histogram shows.

That claim is testable without any new data, by changing nothing but what is
being predicted:

    theta   : the reference phase, in [0, 2*pi)
    2*theta : the doubled phase, which identifies a point in the STRIDE rather
              than in the step and is invariant to the left/right swap

Both are predicted the same way: circular mean of the same k neighbours found
by the same estimator under the same temporal exclusion. If the doubled phase
is recovered much better, then the input does carry cycle position and what it
lacks is the left/right bit specifically -- the target, not the input, is asking
for something the observations do not contain. If the doubled phase is no
better, the information is simply not there and the symmetry reading is wrong.

Note the naive floor: a half-cycle error in theta is a zero error in 2*theta, so
the doubled phase is not harder to predict, it is strictly coarser. The
comparison is therefore between two quantities on the same scale only after
halving the doubled error back into theta units, which is what is reported.

Validation only; the test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_g1_phase_identifiability import (
    NEIGHBOURS,
    PCA_COMPONENTS,
    TEMPORAL_EXCLUSION,
    circular_mean,
    neighbour_indices,
    reduce_contexts,
)
from scripts.train_g1_phase_encoder_v2_1b import CONTEXT_FRAMES, build_context_windows, split_arrays

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "phase_symmetry_probe.json"

BASELINE_MACRO_EXPECTED = 0.5051
BASELINE_POINT_EXPECTED = 0.7996
CONTROL_TOLERANCE = 5e-3


def angular_error(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (predicted - reference))))


def summarise(error: np.ndarray) -> dict[str, float]:
    return {
        "mean_rad": float(error.mean()),
        "median_rad": float(np.median(error)),
        "share_below_half_rad": float((error < 0.5).mean()),
        "share_above_pi_half_rad": float((error > 1.5).mean()),
    }


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)
    sequence = validation["sequence"]
    frame = validation["frame"].astype(np.int64)

    windows, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    reduced = reduce_contexts(windows)
    neighbours, distances = neighbour_indices(reduced, sequence, frame, NEIGHBOURS, TEMPORAL_EXCLUSION)

    baseline_prediction = circular_mean(reference[neighbours], axis=1)
    baseline_error = angular_error(baseline_prediction, reference)

    doubled_reference = 2.0 * reference
    doubled_prediction = circular_mean(doubled_reference[neighbours], axis=1)
    # Halve back into theta units so the two errors are on the same scale.
    doubled_error = 0.5 * angular_error(doubled_prediction, doubled_reference)

    mask = history_valid
    baseline = summarise(baseline_error[mask])
    doubled = summarise(doubled_error[mask])

    # The symmetry reading predicts the error concentrates near pi. Count how
    # many baseline errors sit in the half-cycle band, and how many of those the
    # doubled target turns into small errors.
    near_half_cycle = mask & (baseline_error > 0.5 * np.pi)
    fixed_by_doubling = near_half_cycle & (doubled_error < 0.5)

    control_ok = (
        abs(baseline["mean_rad"] - BASELINE_POINT_EXPECTED) < 0.02
    )
    payload = {
        "schema": "g1.phase_symmetry_probe.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "estimator": "k-NN circular mean, no fitting; only the predicted quantity varies",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_components": PCA_COMPONENTS,
            "frames_evaluated": int(mask.sum()),
        },
        "control": {
            "expected_point_error_rad": BASELINE_POINT_EXPECTED,
            "expected_macro_completion": BASELINE_MACRO_EXPECTED,
            "note": "the theta row must reproduce the identifiability probe's point error; macro is not recomputed here",
            "reproduced": bool(control_ok),
        },
        "theta": baseline,
        "doubled_theta": doubled,
        "reduction_rad": float(baseline["mean_rad"] - doubled["mean_rad"]),
        "half_cycle_band": {
            "frames_with_error_above_pi_over_2": int(near_half_cycle.sum()),
            "share_of_frames": float(near_half_cycle.sum() / max(int(mask.sum()), 1)),
            "share_of_those_below_half_rad_once_doubled": float(fixed_by_doubling.sum() / max(int(near_half_cycle.sum()), 1)),
        },
        "median_neighbour_distance": float(np.median(distances[:, 0])),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"control (theta) reproduced within tolerance: {control_ok}")
    print()
    print(f"{'target':12s} {'mean':>8s} {'median':>8s} {'<0.5rad':>9s} {'>1.5rad':>9s}")
    for name, row in (("theta", baseline), ("2*theta", doubled)):
        print(f"{name:12s} {row['mean_rad']:8.4f} {row['median_rad']:8.4f} "
              f"{row['share_below_half_rad']:9.1%} {row['share_above_pi_half_rad']:9.1%}")
    print()
    print(f"reduction from doubling: {payload['reduction_rad']:.4f} rad")
    band = payload["half_cycle_band"]
    print(f"frames with theta error > pi/2: {band['frames_with_error_above_pi_over_2']} "
          f"({band['share_of_frames']:.1%}); of those, "
          f"{band['share_of_those_below_half_rad_once_doubled']:.1%} fall below 0.5 rad once doubled")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
