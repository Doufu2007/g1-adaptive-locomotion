"""Which part of the base pose would carry the phase, if any?

The joints-vs-joints+base-pose comparison has a hole in it that must be closed
before its result can be believed. The base position in the raw CSV is a world
position, and these are straight walking sequences, so its x and y increase
monotonically along the trajectory. A k-NN estimator that is handed x and y can
match a query to a neighbour at a similar point along the walk and recover the
phase from trajectory position rather than from body state. That is not the
missing-variable explanation -- it is an index into the recording, and no
deployed policy has it.

So the base pose is split and each part is added to the joints separately:

    joints                       the input every encoder has seen
    joints + horizontal position cols 0:2 -- the suspected leak
    joints + posture             col 2 (height) and cols 3:7 (quaternion)

Reading it:

  * gain only in the horizontal-position arm -> the missing-variable story is
    wrong; what the estimator was using is where it is in the recording;
  * gain in the posture arm -> height and trunk tilt really are information the
    joint angles do not carry, and the missing-variable story survives.

The posture arm still contains absolute pelvis height, which is partly
recoverable from the joint angles plus a contact assumption; it is reported
separately from the position arm so that the two cannot be confused.

Alignment: reuses the verified guard from the parent probe and aborts below the
same floor. Validation only; the test split is never loaded.

Run at PCA 32 only. The parent probe already showed the sign of the base-pose
effect is consistent at 32 and 64 components, so a second dimension here would
add runtime without adding a branch. What this run can add that the parent
cannot is the cancellation case: position could help while posture hurts, and
the summed arm would be null for that reason. Separating them is the point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_g1_basepose_identifiability import (
    ALIGNMENT_CORRELATION_FLOOR,
    DATA_PATH,
    OUT_PATH as PARENT_OUT,
    V4_CALIBRATION_PATH,
    alignment_check,
    base_pose_windows,
    evaluate,
)
from scripts.train_g1_phase_encoder_v2_1b import CONTEXT_FRAMES, build_context_windows, split_arrays

OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "basepose_ablation.json"

PCA_GRID = (32,)
ARMS = {
    "joints": (),
    "joints_plus_horizontal_position": (0, 1),
    "joints_plus_posture": (2, 3, 4, 5, 6),
    "joints_plus_base_all": (0, 1, 2, 3, 4, 5, 6),
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
    print(f"this run does not reproduce the standalone probe's control exactly: the "
          f"{int((history_valid & ~base_valid).sum())} rows without a base-pose window are dropped from "
          f"history_valid in every arm, so the joints-only row is the control for this table only.")

    rows = []
    for components in PCA_GRID:
        for label, columns in ARMS.items():
            features = joints if not columns else np.concatenate(
                (joints, base[..., list(columns)]), axis=-1
            )
            import scripts.probe_g1_phase_identifiability as probe
            original = probe.PCA_COMPONENTS
            probe.PCA_COMPONENTS = components
            try:
                result = evaluate(features, validation, reference, period_cuts, combined_valid)
            finally:
                probe.PCA_COMPONENTS = original
            result["pca_components"] = components
            result["features"] = label
            result["base_columns"] = list(columns)
            rows.append(result)
            print(f"pca {components:2d}  {label:32s}  macro {result['macro_completion']:.4f}  "
                  f"point {result['condition_macro_point_error_rad']:.4f}  "
                  f"median {result['median_point_error_rad']:.4f}  "
                  f"near-pi {result['share_error_above_pi_half']:.1%}", flush=True)

    payload = {
        "schema": "g1.basepose_ablation.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "parent_probe": str(PARENT_OUT.relative_to(ROOT)),
        "reason": "base position is world position on straight walks; x,y index the recording and must be separated from posture",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "pca_grid": list(PCA_GRID),
            "arms": {k: list(v) for k, v in ARMS.items()},
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
