"""How much of the period modulation does the encoder actually reproduce?

The stratified net error is short -0.56, medium -0.34, long +1.37 rad with
corr(period, net error) = 0.64. That sign pattern is what a clock produces:
a fixed rate over-advances long cycles and under-advances short ones. But the
same pattern is also what a *partially* correct tracker produces, so the
pattern alone does not say how much of the modulation is missing.

This script fits the one number that does:

    net_error_i  ~  a + b * (period_i - mean_period)

A perfect estimator has b = 0. A pure clock advancing at 2*pi / mean_period
has b = 2*pi / mean_period. The ratio |b| / (2*pi / mean_period) is the share
of the clock's period error the encoder still carries, i.e. 1 - (share of the
modulation it has captured).

Read-only: re-scores stored predictions with the unmodified gate. No training,
no threshold change, validation split only, test split never loaded.
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
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
PRED_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21e_control_epoch_predictions.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "clock_residual.json"


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")
    _, history_valid = build_context_windows(validation, CONTEXT_FRAMES)

    stored = np.load(PRED_PATH)
    epoch1 = stored[sorted(stored.files)[0]]

    cuts = tuple(float(x) for x in json.loads(
        V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})
    report = score_phase_cycles_v4_2(
        epoch1,
        validation["y"].astype(np.float64) * (2.0 * np.pi),
        validation["sequence"], validation["frame"],
        validation["left_touchdown"].astype(bool),
        condition_ids=validation["sequence"], episode_ids=validation["sequence"],
        history_valid=history_valid,
        period_cuts_frames=cuts, required_conditions=conditions,
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )

    records = [row for row in report["cycle_records"] if row["evaluable"]]
    period = np.asarray([row["period_frames"] for row in records], dtype=float)
    net = np.asarray([row["encoder_net_progression_rad"] for row in records], dtype=float)
    error = net - 2.0 * np.pi

    mean_period = float(period.mean())
    slope, intercept = np.polyfit(period - mean_period, error, 1)
    clock_slope = 2.0 * np.pi / mean_period

    # Residual after removing the *fitted* period trend. The clock model is not used here:
    # it is 1/0.411 times steeper than the fitted trend, so subtracting it overshoots and
    # would make the "explained" share negative by construction, which says nothing.
    fitted = intercept + slope * (period - mean_period)
    raw_sd = float(error.std())
    trend_residual_sd = float((error - fitted).std())
    r = float(np.corrcoef(period, error)[0, 1])

    payload = {
        "schema": "g1.clock_residual.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": "validation",
        "test_split_read": False,
        "source_predictions": str(PRED_PATH.relative_to(ROOT)),
        "epoch_key": sorted(stored.files)[0],
        "n_cycles": int(len(records)),
        "mean_period_frames": mean_period,
        "macro_completion": report["macro_cycle_completion_fraction"],
        "fit": {
            "model": "net_error_rad = intercept + slope * (period_frames - mean_period)",
            "intercept_rad": float(intercept),
            "slope_rad_per_frame": float(slope),
            "corr_period_net_error": float(np.corrcoef(period, error)[0, 1]),
        },
        "clock_model": {
            "definition": "fixed rate 2*pi/mean_period over each cycle's own frame count",
            "slope_rad_per_frame": float(clock_slope),
            "share_of_clock_period_error_retained": float(slope / clock_slope),
            "share_of_period_modulation_captured": float(1.0 - slope / clock_slope),
        },
        "period_trend": {
            "net_error_sd_rad": raw_sd,
            "sd_after_removing_fitted_trend_rad": trend_residual_sd,
            "variance_explained_by_period": float(r * r),
        },
        "caveat": (
            "A single linear fit over all valid cycles. It describes the dominant trend, not the "
            "per-cycle mechanism; the tail of the distribution may have a different cause."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"slope {slope:.5f} rad/frame vs clock {clock_slope:.5f} "
          f"(retains {slope / clock_slope:.3f})")
    print(f"corr {payload['fit']['corr_period_net_error']:.3f}  "
          f"period explains {payload['period_trend']['variance_explained_by_period']:.3f} of variance")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
