"""Is the reference phase identifiable from the 30-frame proprioceptive context?

Every learned estimator in this study lands between 0.60 and 0.70 macro and never
gets closer than 0.77 rad to the reference phase. That admits two readings and
they call for opposite next moves:

  A. the information is in the input and the model is not extracting it
     -> change the model (context, capacity, architecture);
  B. the information is not in the input
     -> no loss or architecture change can help; change the features or the
        target definition.

This script decides between them without training anything. It is a nearest
neighbour estimator: predict a frame's phase as the circular mean of the phases
of its k nearest neighbours in context space. No fitting, no gradient, no split
of train and validation. Any error it achieves is an upper bound on what a
learned model could do with the same inputs, because the neighbours are drawn
from the same distribution.

The one thing this must not get wrong is temporal leakage. Adjacent frames are
near-duplicates in context *and* in phase, so a naive nearest neighbour trivially
scores near zero and proves nothing. Every neighbour is therefore required to be
at least TEMPORAL_EXCLUSION frames away within its own sequence. The same
exclusion is what makes the answer meaningful: it asks whether the *phase of a
cycle* is recoverable from the pose, not whether frame i is close to frame i+1.

Reads validation only. The test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import build_context_windows, split_arrays
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "identifiability_probe.json"

CONTEXT_FRAMES = 30
NEIGHBOURS = 10
TEMPORAL_EXCLUSION = 200          # frames; typical period is 45-70, so ~3-4 cycles
PCA_COMPONENTS = 32
QUERY_CHUNK = 500
REFERENCE_POINT_ERROR = 0.7716    # encoder v2.1b epoch 1, same validation frames
REFERENCE_MACRO = 0.6962


def circular_mean(angles: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.arctan2(np.sin(angles).mean(axis=axis), np.cos(angles).mean(axis=axis))


def circular_std(angles: np.ndarray, axis: int = -1) -> np.ndarray:
    resultant = np.sqrt(np.sin(angles).mean(axis=axis) ** 2 + np.cos(angles).mean(axis=axis) ** 2)
    return np.sqrt(-2.0 * np.log(np.clip(resultant, 1e-12, 1.0)))


def reduce_contexts(windows: np.ndarray) -> np.ndarray:
    """Standardise then PCA the flattened context. Deterministic, no labels used."""
    from sklearn.decomposition import PCA

    flat = windows.reshape(len(windows), -1).astype(np.float64)
    flat -= flat.mean(axis=0, keepdims=True)
    scale = flat.std(axis=0, keepdims=True)
    flat /= np.where(scale < 1e-8, 1.0, scale)
    components = min(PCA_COMPONENTS, flat.shape[0], flat.shape[1])
    reduced = PCA(n_components=components, svd_solver="randomized", random_state=0).fit_transform(flat)
    return reduced.astype(np.float32)


def neighbour_indices(reduced: np.ndarray, sequence: np.ndarray, frame: np.ndarray,
                      k: int, exclusion: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the k nearest rows that are far enough away in time.

    Rows from a different sequence are always eligible. Rows from the same
    sequence must differ by more than `exclusion` frames, which is what stops
    the estimator from looking up the adjacent frame.
    """
    count = len(reduced)
    squared_norm = np.einsum("ij,ij->i", reduced, reduced)
    indices_out = np.empty((count, k), dtype=np.int64)
    distance_out = np.empty((count, k), dtype=np.float32)
    # Integer codes: comparing the string array inside the chunk loop costs on
    # the order of 2e9 string comparisons per call, against ~65 GFLOP for the
    # matmuls. The partition is unchanged, so results are identical; this is
    # speed only.
    _, sequence_codes = np.unique(np.asarray(sequence), return_inverse=True)
    frame_int = frame.astype(np.int32)

    for start in range(0, count, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, count)
        block = reduced[start:stop]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        distances = (
            squared_norm[start:stop, None]
            + squared_norm[None, :]
            - 2.0 * (block @ reduced.T)
        )
        same_sequence = sequence_codes[start:stop, None] == sequence_codes[None, :]
        frame_gap = np.abs(frame_int[start:stop, None] - frame_int[None, :])
        forbidden = same_sequence & (frame_gap <= exclusion)
        forbidden |= np.arange(start, stop)[:, None] == np.arange(count)[None, :]
        distances[forbidden] = np.inf
        part = np.argpartition(distances, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        chosen = np.take_along_axis(distances, part, axis=1)
        order = np.argsort(chosen, axis=1)
        indices_out[start:stop] = np.take_along_axis(part, order, axis=1)
        distance_out[start:stop] = np.sqrt(np.take_along_axis(chosen, order, axis=1))
    return indices_out, distance_out


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    validation = split_arrays(archive, "validation")
    windows, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)
    sequence = validation["sequence"]
    frame = validation["frame"].astype(np.int64)

    reduced = reduce_contexts(windows)
    neighbours, distances = neighbour_indices(reduced, sequence, frame, NEIGHBOURS, TEMPORAL_EXCLUSION)

    predicted = circular_mean(reference[neighbours], axis=1)
    vectors = np.stack((np.sin(predicted), np.cos(predicted)), axis=-1)

    base = {
        "sequence": sequence,
        "frame": validation["frame"],
        "touchdowns": validation["left_touchdown"].astype(bool),
        "history_valid": history_valid,
        "conditions": sorted(np.unique(sequence).tolist()),
    }
    report = score_phase_cycles_v4_2(
        vectors, reference, base["sequence"], base["frame"], base["touchdowns"],
        condition_ids=base["sequence"], episode_ids=base["sequence"],
        history_valid=base["history_valid"],
        period_cuts_frames=period_cuts, required_conditions=base["conditions"],
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )

    neighbour_spread = circular_std(reference[neighbours], axis=1)
    # Null: pairs chosen at random among eligible rows, to show what no-information looks like.
    rng = np.random.default_rng(20261020)
    random_partner = rng.integers(0, len(reference), size=len(reference))
    eligible = (sequence != sequence[random_partner]) | (np.abs(frame - frame[random_partner]) > TEMPORAL_EXCLUSION)
    random_gap = np.abs(np.angle(np.exp(1j * (reference - reference[random_partner]))))[eligible]

    own_error = np.abs(np.angle(np.exp(1j * (predicted - reference))))

    payload = {
        "schema": "g1.phase_identifiability_probe.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "estimator": "k-NN circular mean over 30-frame context, no fitting",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_components": PCA_COMPONENTS,
            "frames": int(len(reference)),
        },
        "result": {
            "macro_completion": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
            "reference_clock_completion": report["reference_clock_completion_fraction"],
            "coverage_pass": report["coverage_pass"],
        },
        "ambiguity": {
            "mean_neighbour_distance": float(distances[:, 0].mean()),
            "median_neighbour_distance": float(np.median(distances[:, 0])),
            "mean_circular_spread_among_neighbours_rad": float(neighbour_spread.mean()),
            "median_circular_spread_among_neighbours_rad": float(np.median(neighbour_spread)),
            "mean_own_point_error_rad": float(own_error.mean()),
            "mean_point_error_random_partner_rad": float(random_gap.mean()),
            "median_point_error_random_partner_rad": float(np.median(random_gap)),
        },
        "comparison": {
            "encoder_v2_1b_epoch1_point_error_rad": REFERENCE_POINT_ERROR,
            "encoder_v2_1b_epoch1_macro_completion": REFERENCE_MACRO,
            "ridge_point_error_rad": 0.831,
            "ridge_macro_completion": 0.6239,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"k-NN macro completion      : {payload['result']['macro_completion']:.4f}")
    print(f"k-NN point error           : {payload['result']['condition_macro_point_error_rad']:.4f} rad")
    print(f"  per condition            : {payload['result']['condition_completion']}")
    print(f"  per stratum              : {payload['result']['period_length_strata']}")
    print()
    print(f"median nearest-neighbour distance : {payload['ambiguity']['median_neighbour_distance']:.4f}"
          f" (PCA space, {PCA_COMPONENTS} dims)")
    print(f"median phase spread among the {NEIGHBOURS} neighbours : "
          f"{payload['ambiguity']['median_circular_spread_among_neighbours_rad']:.4f} rad")
    print(f"mean point error  k-NN {payload['ambiguity']['mean_own_point_error_rad']:.4f} rad  |  "
          f"random partner {payload['ambiguity']['mean_point_error_random_partner_rad']:.4f} rad  |  "
          f"v2.1b epoch1 {REFERENCE_POINT_ERROR:.4f} rad")
    print()
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
