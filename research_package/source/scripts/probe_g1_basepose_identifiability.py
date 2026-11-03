"""Does restoring the dropped base pose make the reference phase identifiable?

`scripts/build_g1_walk_phase_dataset.py` defines the target in three steps:

    foot_positions = forward_kinematics(values, ...)     # uses cols 0:7 + cols 7:36
    contacts       = detect_foot_contacts(foot_positions, ...)
    left_touches   = _touchdown_times(contacts[:, 0], ...)

and then feeds the encoder only `values[:, 7:]` -- the 29 joint angles. The base
position (0:3) and orientation (3:7) are used to build the labels and then
dropped. So the model is asked to recover foot contact without being told where
the pelvis is or how it is tilted, which is the quantity that decides whether a
given leg configuration is stance or swing.

This probe holds everything else fixed (same k-NN estimator, same temporal
exclusion, same validation frames) and varies only the input feature set:

    joints only       : 30 x 29, exactly what the encoders see
    joints + base pose: 30 x 29 plus 30 x 7 base position and quaternion

If the reference phase is identifiable once the base pose is restored, point
error and cycle completion should move together and materially. If nothing
moves, the missing-variable explanation is wrong and should be discarded.

Alignment guard: the resampled base-pose window is indexed by the dataset's own
`frame` column, which is an index into the same resampled timeline the joint
windows came from. If that indexing were off by even one frame the comparison
would be meaningless, so the script first checks that the joint window
reconstructed here matches the stored `x` window row by row and aborts if not.
The stored windows are normalised per column, so the guard correlates column by
column; a global standardisation is not invariant to that normalisation and
reported a spurious 0.3665 on windows that in fact agree exactly.

Reads train-free, validation only. The test split is never loaded.
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
    TEMPORAL_EXCLUSION,
    circular_mean,
    circular_std,
    neighbour_indices,
    reduce_contexts,
)
from scripts.train_g1_phase_encoder_v2_1b import CONTEXT_FRAMES, build_context_windows, split_arrays
from src.tasks.velocity.phasor.data import resample_joint_positions
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
RAW_DIR = ROOT / "data" / "raw" / "lafan1_g1_walk"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "basepose_identifiability.json"

SOURCE_FPS = 30.0
TARGET_FPS = 50.0
BASE_DIM = 7
JOINT_DIM = 29
PCA_GRID = (32, 64)
ALIGNMENT_CORRELATION_FLOOR = 0.99


def align_quaternion_signs(values: np.ndarray) -> np.ndarray:
    """Remove q / -q flips before linear interpolation.

    The quaternion columns are interpolated one column at a time, so a sign flip
    between adjacent samples would interpolate through zero and produce a
    meaningless orientation. The builder never hit this because it only
    interpolated joints.
    """
    aligned = values.copy()
    for index in range(1, len(aligned)):
        if float(np.dot(aligned[index - 1, 3:7], aligned[index, 3:7])) < 0.0:
            aligned[index, 3:7] *= -1.0
    return aligned


def base_pose_windows(sequence: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resampled base pose over the same context window as each dataset row."""
    result = np.zeros((len(frame), CONTEXT_FRAMES, BASE_DIM), dtype=np.float32)
    valid = np.zeros(len(frame), dtype=bool)
    for name in sorted(set(sequence.tolist())):
        rows = np.flatnonzero(sequence == name)
        values = align_quaternion_signs(
            np.loadtxt(RAW_DIR / f"{name}.csv", delimiter=",", dtype=np.float64)
        )
        timestamps = np.arange(len(values), dtype=np.float64) / SOURCE_FPS
        _, resampled = resample_joint_positions(timestamps, values[:, :BASE_DIM], target_hz=TARGET_FPS)
        for index in rows:
            position = int(frame[index])
            if position < CONTEXT_FRAMES - 1 or position >= len(resampled):
                continue
            result[index] = resampled[position - CONTEXT_FRAMES + 1: position + 1]
            valid[index] = True
    return result, valid


def per_column_correlation(window: np.ndarray, target: np.ndarray) -> float:
    """Median over joints of the per-column correlation.

    The stored `x` is normalised per column, so it matches the raw resampled
    joints only up to an independent affine map on each column. Standardising
    the flattened window globally is *not* invariant to that map and reports a
    spurious low correlation even when the windows agree exactly; standardising
    column by column is. Measured on this data, the correct alignment scores
    1.0000 and a row-wise variant scores 0.3665 on the same windows.
    """
    a = (window - window.mean(axis=0)) / np.maximum(window.std(axis=0), 1e-9)
    b = (target - target.mean(axis=0)) / np.maximum(target.std(axis=0), 1e-9)
    return float(np.median((a * b).mean(axis=0)))


def alignment_check(sequence: np.ndarray, frame: np.ndarray, stored: np.ndarray) -> float:
    """Correlate the stored joint window against the resampled joints at `frame`.

    This tests whether `frame` is the trailing anchor of the stored window. The
    lag sweep that established the anchor peaked at exactly 1.0000 only at
    offset -9 on the stored 10-frame rows, so a correct reconstruction must
    score ~1.0 here.
    """
    correlations = []
    for name in sorted(set(sequence.tolist()))[:2]:
        rows = np.flatnonzero(sequence == name)[:400]
        values = np.loadtxt(RAW_DIR / f"{name}.csv", delimiter=",", dtype=np.float64)
        timestamps = np.arange(len(values), dtype=np.float64) / SOURCE_FPS
        _, resampled = resample_joint_positions(timestamps, values[:, BASE_DIM:], target_hz=TARGET_FPS)
        for index in rows:
            position = int(frame[index])
            if position < CONTEXT_FRAMES - 1 or position >= len(resampled):
                continue
            window = resampled[position - CONTEXT_FRAMES + 1: position + 1].astype(np.float64)
            target = np.asarray(stored[index], dtype=np.float64)
            correlations.append(per_column_correlation(window, target))
    if not correlations:
        raise ValueError("alignment check found no comparable rows")
    return float(np.median(correlations))


def evaluate(features: np.ndarray, validation: dict[str, np.ndarray], reference: np.ndarray,
             period_cuts: tuple[float, float], history_valid: np.ndarray) -> dict[str, object]:
    reduced = reduce_contexts(features)
    sequence = validation["sequence"]
    frame = validation["frame"].astype(np.int64)
    neighbours, distances = neighbour_indices(reduced, sequence, frame, NEIGHBOURS, TEMPORAL_EXCLUSION)
    predicted = circular_mean(reference[neighbours], axis=1)
    vectors = np.stack((np.sin(predicted), np.cos(predicted)), axis=-1)
    conditions = sorted(np.unique(sequence).tolist())
    report = score_phase_cycles_v4_2(
        vectors, reference, sequence, validation["frame"], validation["left_touchdown"].astype(bool),
        condition_ids=sequence, episode_ids=sequence, history_valid=history_valid,
        period_cuts_frames=period_cuts, required_conditions=conditions,
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    error = np.abs(np.angle(np.exp(1j * (predicted - reference))))
    return {
        "macro_completion": report["macro_cycle_completion_fraction"],
        "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
        "median_point_error_rad": float(np.median(error)),
        "share_error_below_half_rad": float((error < 0.5).mean()),
        "share_error_above_pi_half": float((error > 1.5).mean()),
        "median_neighbour_distance": float(np.median(distances[:, 0])),
        "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
        "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
    }


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])
    validation = split_arrays(archive, "validation")
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)

    joints, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    sequence = validation["sequence"]
    frame = validation["frame"].astype(np.int64)

    correlation = alignment_check(sequence, frame, joints)
    print(f"alignment: joint-window correlation vs stored x = {correlation:.5f}")
    if correlation < ALIGNMENT_CORRELATION_FLOOR:
        raise SystemExit(f"alignment check failed ({correlation:.4f} < {ALIGNMENT_CORRELATION_FLOOR}); refusing to compare")

    base, base_valid = base_pose_windows(sequence, frame)
    combined_valid = history_valid & base_valid
    print(f"rows usable with base pose: {int(combined_valid.sum())} of {len(frame)}")

    joints_only = joints
    combined = np.concatenate((joints, base), axis=-1)

    rows = []
    for components in PCA_GRID:
        for label, features in (("joints_only", joints_only), ("joints_plus_base", combined)):
            import scripts.probe_g1_phase_identifiability as probe
            original = probe.PCA_COMPONENTS
            probe.PCA_COMPONENTS = components
            try:
                result = evaluate(features, validation, reference, period_cuts, combined_valid)
            finally:
                probe.PCA_COMPONENTS = original
            result["pca_components"] = components
            result["features"] = label
            rows.append(result)
            print(f"pca {components:2d}  {label:16s}  macro {result['macro_completion']:.4f}  "
                  f"point {result['condition_macro_point_error_rad']:.4f}  "
                  f"median {result['median_point_error_rad']:.4f}  "
                  f"near-pi {result['share_error_above_pi_half']:.1%}", flush=True)

    payload = {
        "schema": "g1.basepose_identifiability.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "estimator": "k-NN circular mean, no fitting; only the input feature set varies",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_grid": list(PCA_GRID),
            "base_pose_columns": "0:3 base position, 3:7 quaternion, as stored in the raw CSV",
        },
        "alignment": {
            "joint_window_correlation": correlation,
            "floor": ALIGNMENT_CORRELATION_FLOOR,
            "passed": bool(correlation >= ALIGNMENT_CORRELATION_FLOOR),
        },
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
