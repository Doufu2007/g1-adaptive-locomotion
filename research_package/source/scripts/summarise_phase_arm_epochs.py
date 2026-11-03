"""Per-epoch net-progression anatomy for one stored prediction archive.

Reads only the validation split and an existing `*_epoch_predictions.npz`, so it
adds no experiment: it re-scores predictions that were already produced and
reports the distribution behind each epoch's macro completion.

Usage:
    python scripts/summarise_phase_arm_epochs.py <predictions.npz> <output.json>
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

from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    V4_CALIBRATION_PATH,
    build_context_windows,
    split_arrays,
)
from src.tasks.velocity.phasor.encoder_v2_1 import NET_TOLERANCE_RAD  # noqa: E402
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    records = [row for row in report["cycle_records"] if row["evaluable"]]
    error = np.asarray([row["encoder_net_progression_rad"] - 2.0 * np.pi for row in records])
    periods = np.asarray([row["period_frames"] for row in records], dtype=float)
    slip = np.abs(np.abs(error) - 2.0 * np.pi) < 1.0
    summary = {
        "macro_completion": report["macro_cycle_completion_fraction"],
        "point_error_rad": report["condition_macro_point_error_rad"],
        "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
        "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
        "n_cycles": int(len(records)),
        "mean_net_error": float(error.mean()),
        "sd_net_error": float(error.std()),
        "median_net_error": float(np.median(error)),
        "share_within_tolerance": float(np.mean(np.abs(error) <= NET_TOLERANCE_RAD)),
        "whole_turn_slip_count": int(slip.sum()),
        "mean_period_of_slip": float(periods[slip].mean()) if slip.any() else 0.0,
    }
    strata = np.asarray([str(row["period_length_stratum"]) for row in records])
    for name in ("short", "medium", "long"):
        mask = strata == name
        if mask.any():
            summary[f"mean_net_error_{name}"] = float(error[mask].mean())
            summary[f"completion_{name}"] = float(np.mean([row["encoder_completed_one_turn"] for row, keep in zip(records, mask) if keep]))
    return summary


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    predictions_path = Path(argv[1]).resolve()
    output_path = Path(argv[2]).resolve()

    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")
    _, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)
    cuts = tuple(float(x) for x in json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})

    stored = np.load(predictions_path)
    epochs: dict[str, Any] = {}
    for key in sorted(stored.files):
        report = score_phase_cycles_v4_2(
            stored[key], reference, validation["sequence"], validation["frame"],
            validation["left_touchdown"].astype(bool),
            condition_ids=validation["sequence"], episode_ids=validation["sequence"],
            history_valid=history_valid,
            period_cuts_frames=cuts, required_conditions=conditions,
            minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=1,
        )
        epochs[key] = summarise(report)
        row = epochs[key]
        print("%s macro %.4f point %.4f net %+.2f sd %.2f slip %2d long %+.2f"
              % (key, row["macro_completion"], row["point_error_rad"], row["mean_net_error"],
                 row["sd_net_error"], row["whole_turn_slip_count"], row.get("mean_net_error_long", 0.0)),
              flush=True)

    output_path.write_text(json.dumps({
        "schema": "g1.phase_arm_epoch_summary.v1",
        "source_predictions": str(predictions_path.relative_to(ROOT)),
        "split": "validation",
        "test_split_read": False,
        "epochs": epochs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
