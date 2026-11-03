"""Does a context spanning a full cycle make the reference phase identifiable?

The first identifiability probe used 30 frames -- the same context the models use
-- so it could only show that 30 frames is not enough. It cannot separate:

  A. the information is present but the model is not extracting it
  B. the information is not present in proprioception at all

Typical cycle length on this data is 45-70 frames, so 30 frames is 0.4-0.7 of a
cycle. No context covering a whole cycle has ever been evaluated in this study,
in any form. This sweep holds the estimator fixed (k-NN circular mean, no
fitting, validation only) and varies only the context length.

Reading it:
  * if point error falls materially as context grows past one cycle, the
    information is there and the next model change is context, not loss;
  * if it flattens at ~0.77-0.80 rad, proprioception at these scales does not
    pin the absolute phase, and the next decision is about the contract, not
    about training.

The 30-frame row is the control: it must reproduce the earlier probe
(macro 0.5051, point error 0.7996) or the sweep is not comparable.

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

from scripts.probe_g1_phase_identifiability import (
    NEIGHBOURS,
    PCA_COMPONENTS,
    TEMPORAL_EXCLUSION,
    circular_mean,
    circular_std,
    neighbour_indices,
    reduce_contexts,
)
from scripts.train_g1_phase_encoder_v2_1b import split_arrays
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "context_identifiability_sweep.json"

CONTEXTS = (30, 60, 90)
CONTROL_CONTEXT = 30
CONTROL_EXPECTED_MACRO = 0.5051
CONTROL_EXPECTED_POINT = 0.7996
FEATURE_DIM = 29


def build_windows(split: dict[str, np.ndarray], context_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Generic version of the trainer's builder; that one is pinned at 30 frames."""
    count = len(split["frame"])
    result = np.empty((count, context_frames, FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    groups: dict[str, list[int]] = {}
    for index, sequence in enumerate(split["sequence"]):
        groups.setdefault(str(sequence), []).append(index)
    for indexes in groups.values():
        order = sorted(indexes, key=lambda index: int(split["frame"][index]))
        frames = np.asarray([split["frame"][index] for index in order], dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            segment = order[int(start):int(end)]
            for position, row_index in enumerate(segment):
                if position == 0:
                    stored = np.asarray(split["x"][row_index], dtype=np.float32)
                    pad = context_frames - stored.shape[0]
                    window = np.concatenate((np.repeat(stored[0][None, :], pad, axis=0), stored), axis=0)
                else:
                    previous = result[segment[position - 1]]
                    latest = np.asarray(split["x"][row_index, -1], dtype=np.float32)
                    window = np.concatenate((previous[1:], latest[None, :]), axis=0)
                result[row_index] = window
                valid[row_index] = position + 10 >= context_frames
    return result, valid


def run(context_frames: int, validation: dict[str, np.ndarray], period_cuts: tuple[float, float],
        reference: np.ndarray) -> dict[str, object]:
    windows, history_valid = build_windows(validation, context_frames)
    reduced = reduce_contexts(windows)
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
    spread = circular_std(reference[neighbours], axis=1)
    return {
        "context_frames": context_frames,
        "frames_usable": int(history_valid.sum()),
        "macro_completion": report["macro_cycle_completion_fraction"],
        "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
        "mean_point_error_rad": float(error.mean()),
        "median_point_error_rad": float(np.median(error)),
        "share_error_below_half_rad": float((error < 0.5).mean()),
        "share_error_above_pi_half": float((error > 1.5).mean()),
        "median_neighbour_distance": float(np.median(distances[:, 0])),
        "median_neighbour_phase_spread_rad": float(np.median(spread)),
        "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
        "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
    }


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])
    validation = split_arrays(archive, "validation")
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)

    rows = []
    for context_frames in CONTEXTS:
        row = run(context_frames, validation, period_cuts, reference)
        rows.append(row)
        print(f"context {context_frames:3d}  macro {row['macro_completion']:.4f}  "
              f"point {row['condition_macro_point_error_rad']:.4f}  "
              f"median err {row['median_point_error_rad']:.4f}  "
              f"<0.5rad {row['share_error_below_half_rad']:.1%}  "
              f"near-pi {row['share_error_above_pi_half']:.1%}  "
              f"nframes {row['frames_usable']}", flush=True)

    control = next(row for row in rows if row["context_frames"] == CONTROL_CONTEXT)
    control_ok = (
        abs(control["macro_completion"] - CONTROL_EXPECTED_MACRO) < 5e-3
        and abs(control["condition_macro_point_error_rad"] - CONTROL_EXPECTED_POINT) < 5e-3
    )

    payload = {
        "schema": "g1.context_identifiability_sweep.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "estimator": "k-NN circular mean, no fitting; only the context length varies",
        "settings": {
            "contexts": list(CONTEXTS),
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_components": PCA_COMPONENTS,
        },
        "control": {
            "context_frames": CONTROL_CONTEXT,
            "expected_macro": CONTROL_EXPECTED_MACRO,
            "expected_point_error_rad": CONTROL_EXPECTED_POINT,
            "reproduced": bool(control_ok),
        },
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print()
    print(f"control (context {CONTROL_CONTEXT}) reproduced: {control_ok}")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
