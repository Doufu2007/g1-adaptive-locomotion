"""Re-score the k-NN phase estimator after a fixed causal smoothing step.

`probe_g1_knn_integral_gate.py` scored the raw k-NN and got macro 0.0016, which
is not a statement about the input. The raw prediction is locally excellent
(median per-frame error 0.074 rad) but heavy-tailed (sd 1.301, p95 2.76), and a
single wild frame inside a cycle breaks the unwrap that the gate integrates.
That is a property of an unconstrained nearest-neighbour average, not of the
information in the window.

This arm applies one pre-declared stabiliser and reports whatever comes out:

    causal median over 5 frames on the unit vector, then renormalise

Five frames is 100 ms at the dataset's 50 Hz, shorter than the shortest cycle
(47 frames) and shorter than the shortest stride, so it does not move phase
across a cycle boundary. No width sweep, no selection: 5 is the only value run.

If the smoothed estimator clears the 0.90 bar, the information survives
integration and the trained arms are the problem. If it lands near the trained
arms, the integral is the binding limit for per-frame estimators on this input.

Read-only. No training, no threshold change, validation queries against a
training pool, test split never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.probe_g1_phase_identifiability as probe  # noqa: E402
from scripts.probe_g1_knn_integral_gate import (  # noqa: E402
    NEIGHBOURS,
    PCA_COMPONENTS,
    cross_neighbours,
    fit_reduce,
    summarise,
)
from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    V4_CALIBRATION_PATH,
    build_context_windows,
    split_arrays,
)
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "knn_smoothed_integral_gate.json"

MEDIAN_WIDTH = 5


def causal_median(phase: np.ndarray, sequence: np.ndarray, width: int) -> np.ndarray:
    """Median over the last `width` frames of the same sequence, on the unit circle."""
    cos = np.cos(phase)
    sin = np.sin(phase)
    out_cos = np.empty_like(cos)
    out_sin = np.empty_like(sin)
    for start in range(0, len(phase), 4096):
        stop = min(start + 4096, len(phase))
        for i in range(start, stop):
            lo = i
            while lo > 0 and i - lo + 1 < width and sequence[lo - 1] == sequence[i]:
                lo -= 1
            out_cos[i] = np.median(cos[lo:i + 1])
            out_sin[i] = np.median(sin[lo:i + 1])
    return np.arctan2(out_sin, out_cos)


def main() -> int:
    probe.PCA_COMPONENTS = PCA_COMPONENTS
    archive = np.load(DATA_PATH, allow_pickle=False)
    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_joints, train_history = build_context_windows(train, CONTEXT_FRAMES)
    valid_joints, valid_history = build_context_windows(validation, CONTEXT_FRAMES)

    reduced_train, reduced_valid = fit_reduce(train_joints[train_history],
                                              valid_joints[valid_history])
    phase_train = train["y"].astype(np.float64)[train_history] * (2.0 * np.pi)

    print("retrieving training neighbours", flush=True)
    idx, _ = cross_neighbours(reduced_valid, reduced_train)
    predicted = probe.circular_mean(phase_train[idx], axis=1)

    sequence_valid = validation["sequence"][valid_history]
    print(f"smoothing with causal median width {MEDIAN_WIDTH}", flush=True)
    smoothed = causal_median(predicted, sequence_valid, MEDIAN_WIDTH)

    truth = validation["y"].astype(np.float64)[valid_history] * (2.0 * np.pi)
    raw_err = np.angle(np.exp(1j * (predicted - truth)))
    smooth_err = np.angle(np.exp(1j * (smoothed - truth)))
    print(f"per-frame |err| median raw {np.median(np.abs(raw_err)):.4f} "
          f"smoothed {np.median(np.abs(smooth_err)):.4f}", flush=True)
    print(f"per-frame err sd raw {raw_err.std():.3f} smoothed {smooth_err.std():.3f}", flush=True)

    cuts = tuple(float(x) for x in json.loads(
        V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})

    results = {}
    for label, values in (("raw", predicted), ("smoothed", smoothed)):
        n_rows = len(validation["frame"])
        val_pred = np.zeros((n_rows, 2), dtype=np.float64)
        val_pred[:, 0] = 1.0
        val_pred[valid_history, 0] = np.cos(values)
        val_pred[valid_history, 1] = np.sin(values)
        report = score_phase_cycles_v4_2(
            val_pred,
            validation["y"].astype(np.float64) * (2.0 * np.pi),
            validation["sequence"], validation["frame"],
            validation["left_touchdown"].astype(bool),
            condition_ids=validation["sequence"], episode_ids=validation["sequence"],
            history_valid=valid_history,
            period_cuts_frames=cuts, required_conditions=conditions,
            minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=1,
        )
        results[label] = summarise(report)
        print(f"{label:9s} macro {results[label]['macro_completion']:.4f} "
              f"point {results[label]['point_error_rad']:.4f} "
              f"net {results[label]['mean_net_error']:+.2f} "
              f"sd {results[label]['sd_net_error']:.2f}", flush=True)

    payload = {
        "schema": "g1.knn_smoothed_integral_gate.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": {"pool": "train", "query": "validation"},
        "test_split_read": False,
        "purpose": ("stabilise the k-NN estimator with one pre-declared causal filter, so that the "
                    "raw arm's heavy-tailed frames do not decide the question"),
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "pca_components": PCA_COMPONENTS,
            "median_width_frames": MEDIAN_WIDTH,
            "median_width_ms": 100.0,
            "median_justification": "shorter than the shortest cycle (47 frames); one value, no sweep",
        },
        "per_frame_error": {
            "raw_median_abs_rad": float(np.median(np.abs(raw_err))),
            "raw_sd_rad": float(raw_err.std()),
            "smoothed_median_abs_rad": float(np.median(np.abs(smooth_err))),
            "smoothed_sd_rad": float(smooth_err.std()),
        },
        "result_raw": results["raw"],
        "result_smoothed": results["smoothed"],
        "reference_points": {
            "oracle": 0.9984177215189873,
            "best_over_grid_fixed_clock": 0.21866427203253935,
            "walking_encoder": 0.41280424052165965,
            "v2_1b_epoch1": 0.6962456093142473,
            "gate_bar": 0.9,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
