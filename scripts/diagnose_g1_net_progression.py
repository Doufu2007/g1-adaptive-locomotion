"""Why do cycles miss the net-progression band?  Read-only diagnostic.

Lever 2 was originally framed as "make the output monotone so it cannot
backtrack".  The step-1 diagnostic refutes that target: under v4.2 the step
bound already passes 505/603 cycles (0.837) while net progression passes only
276/603 (0.458).  This script asks what the net failures actually look like
before anything is designed to fix them.

Validation split only.  The test split is not read.  No threshold is changed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import CONTEXT_FRAMES, build_context_windows, split_arrays
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
PREDICTIONS_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_epoch_predictions.npz"
CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "net_progression_diagnosis.json"

EPOCH_KEY = "epoch_01"


def main() -> int:
    archive = np.load(DATA_PATH)
    validation = split_arrays(archive, "validation")
    _, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    cuts = tuple(float(x) for x in json.loads(CALIBRATION_PATH.read_text())["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})

    prediction = np.load(PREDICTIONS_PATH)[EPOCH_KEY]
    # The archive stores cycle fractions in [0, 1); the gate scores radians.
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)

    report = score_phase_cycles_v4_2(
        prediction, reference, validation["sequence"], validation["frame"],
        validation["left_touchdown"],
        condition_ids=validation["sequence"], episode_ids=validation["sequence"],
        history_valid=history_valid,
        period_cuts_frames=cuts, required_conditions=conditions,
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    print("control: macro=%r point=%r ref=%r" % (
        report.get("macro_cycle_completion_fraction"),
        report.get("condition_macro_point_error_rad"),
        report.get("reference_clock_completion_fraction")))

    cycles = report["cycle_records"]
    print(f"\nscored cycles: {len(cycles)}")
    print("record keys:", sorted(cycles[0].keys()))

    rows = []
    for c in cycles:
        rows.append({
            "condition": str(c.get("condition_id")),
            "stratum": str(c.get("period_length_stratum")),
            "period": int(c.get("period_frames", 0)),
            "net": float(c.get("encoder_net_progression_rad", 0.0)),
            "ref_net": float(c.get("reference_net_progression_rad", 0.0)),
            "points": int(c.get("metric_sample_count", 0)),
            "step": float(c.get("encoder_max_abs_step_rad", 0.0)),
            "completed": bool(c.get("encoder_completed_one_turn")),
            "evaluable": bool(c.get("evaluable", True)),
        })

    ev = [r for r in rows if r["evaluable"]]
    net_err = np.array([r["net"] - 2 * np.pi for r in ev])
    print(f"\nevaluable {len(ev)}  completed {sum(r['completed'] for r in ev)}")
    print("net-2pi: mean %+.3f  median %+.3f  sd %.3f  IQR [%+.3f, %+.3f]"
          % (net_err.mean(), np.median(net_err), net_err.std(),
             *np.percentile(net_err, [25, 75])))
    tol = np.pi / 6
    print("  |net-2pi| <= tol : %d / %d = %.3f" % ((np.abs(net_err) <= tol).sum(), len(ev),
                                                   (np.abs(net_err) <= tol).mean()))
    print("  net too small    : %d" % (net_err < -tol).sum())
    print("  net too large    : %d" % (net_err > tol).sum())

    # Does net error scale with cycle length?  A constant rate error gives a
    # proportional error; a constant offset gives none.
    period = np.array([r["period"] for r in ev], dtype=float)
    rate_err = net_err / np.maximum(period, 1)
    print("\nnet error / period (rad per frame): mean %+.5f  sd %.5f" % (rate_err.mean(), rate_err.std()))
    corr = np.corrcoef(period, net_err)[0, 1] if len(ev) > 2 else float("nan")
    print("corr(period, net error) = %+.3f" % corr)

    print("\nby condition:")
    by_cond = defaultdict(list)
    for r in ev:
        by_cond[r["condition"]].append(r)
    for name, rs in sorted(by_cond.items()):
        e = np.array([r["net"] - 2 * np.pi for r in rs])
        print("  %-16s n=%3d  pass=%5.3f  mean net-2pi=%+.3f  sd=%.3f"
              % (name, len(rs), np.mean([r["completed"] for r in rs]), e.mean(), e.std()))

    print("\nby stratum:")
    by_str = defaultdict(list)
    for r in ev:
        by_str[r["stratum"]].append(r)
    for name, rs in sorted(by_str.items()):
        e = np.array([r["net"] - 2 * np.pi for r in rs])
        print("  %-8s n=%3d  pass=%5.3f  mean net-2pi=%+.3f  sd=%.3f"
              % (name, len(rs), np.mean([r["completed"] for r in rs]), e.mean(), e.std()))

    # How much of the failure is net versus step, among the ones that fail?
    failed = [r for r in ev if not r["completed"]]
    net_fail = sum(1 for r in failed if abs(r["net"] - 2 * np.pi) > tol)
    step_fail = sum(1 for r in failed if r["step"] >= np.pi / 2)
    print(f"\nof {len(failed)} failed cycles: net-only {net_fail}, step-only {step_fail}, "
          f"both {sum(1 for r in failed if abs(r['net']-2*np.pi) > tol and r['step'] >= np.pi/2)}")

    payload = {
        "schema": "g1.net_progression_diagnosis.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "epoch_key": EPOCH_KEY,
        "n_cycles": len(ev),
        "net_minus_two_pi": {
            "mean": float(net_err.mean()), "median": float(np.median(net_err)),
            "sd": float(net_err.std()),
            "q25": float(np.percentile(net_err, 25)), "q75": float(np.percentile(net_err, 75)),
            "share_within_tolerance": float((np.abs(net_err) <= tol).mean()),
            "count_too_small": int((net_err < -tol).sum()),
            "count_too_large": int((net_err > tol).sum()),
        },
        "rate_error_rad_per_frame": {"mean": float(rate_err.mean()), "sd": float(rate_err.std())},
        "corr_period_net_error": float(corr),
        "by_condition": {k: {"n": len(v), "pass": float(np.mean([r["completed"] for r in v])),
                             "mean_net_error": float(np.mean([r["net"] - 2 * np.pi for r in v]))}
                         for k, v in by_cond.items()},
        "by_stratum": {k: {"n": len(v), "pass": float(np.mean([r["completed"] for r in v])),
                           "mean_net_error": float(np.mean([r["net"] - 2 * np.pi for r in v]))}
                       for k, v in by_str.items()},
        "gate_summary": {k: v for k, v in report.items() if k != "cycle_records"},
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
