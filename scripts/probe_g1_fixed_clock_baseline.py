"""Score the thing the learned encoder is supposed to replace.

The study's claim is comparative: a learned phase replaces the fixed gait clock
and adapts to the actual cycle instead of assuming one period. Every gate table
so far has scored the oracle reference clock (which is the label itself, 0.9984)
and the encoders (0.41-0.70). It has never scored **the fixed clock** -- the
free-running oscillator at `velocity_env_cfg.py` period 0.6 s that the whole
comparison is against.

Without that row the tables say "the encoder is not accurate enough in absolute
terms". They cannot say "the encoder is or is not better than what it replaces",
which is what the paper's claim actually needs.

The fixed clock is a free-running sinusoid, so it is defined up to a period and
a phase offset. Both are swept, and the reported number is the **maximum over
that grid** -- the fixed clock is given every advantage, including the two
parameters an analyst tuning it would have chosen. A number obtained this way
is generous to the baseline by construction, so a learned encoder that beats it
beats it under the least favourable accounting.

At 50 Hz the configured period of 0.6 s is 30 frames. Reference touchdown
intervals on this data run 45-122 frames (typical 45-70).

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
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "fixed_clock_baseline.json"

DATA_FPS = 50.0
CONFIGURED_PERIOD_S = 0.6
CONFIGURED_PERIOD_FRAMES = CONFIGURED_PERIOD_S * DATA_FPS  # 30.0

PERIOD_GRID = tuple(float(p) for p in range(25, 131, 5))
OFFSET_GRID = tuple(float(o) for o in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")
    _, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    reference = validation["y"].astype(np.float64) * (2.0 * np.pi)
    cuts = tuple(float(x) for x in json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})
    frame = validation["frame"].astype(np.float64)

    def score(period: float, offset: float) -> float:
        angle = (2.0 * np.pi * frame / period + offset) % (2.0 * np.pi)
        vectors = np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)
        report = score_phase_cycles_v4_2(
            vectors, reference, validation["sequence"], validation["frame"],
            validation["left_touchdown"].astype(bool),
            condition_ids=validation["sequence"], episode_ids=validation["sequence"],
            history_valid=history_valid,
            period_cuts_frames=cuts, required_conditions=conditions,
            minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=1,
        )
        return float(report["macro_cycle_completion_fraction"])

    grid = []
    best = (-1.0, None, None)
    for period in PERIOD_GRID:
        row_best = (-1.0, None)
        for offset in OFFSET_GRID:
            macro = score(period, offset)
            grid.append({"period_frames": period, "period_seconds": period / DATA_FPS,
                         "offset_rad": offset, "macro_completion": macro})
            if macro > row_best[0]:
                row_best = (macro, offset)
            if macro > best[0]:
                best = (macro, period, offset)
        print(f"period {period:5.1f} frames  best macro {row_best[0]:.4f} at offset {row_best[1]:.3f}", flush=True)

    configured = max(
        (g["macro_completion"] for g in grid
         if g["period_frames"] == CONFIGURED_PERIOD_FRAMES),
        default=None,
    )
    print(f"\nconfigured period ({CONFIGURED_PERIOD_S}s = {CONFIGURED_PERIOD_FRAMES:.0f} frames): "
          f"best macro {configured}", flush=True)
    print(f"best over the whole grid: macro {best[0]:.4f} at period {best[1]:.1f} frames "
          f"({best[1] / DATA_FPS:.2f}s), offset {best[2]:.3f} rad", flush=True)

    payload = {
        "schema": "g1.fixed_clock_baseline.v1",
        "split": "validation",
        "test_split_read": False,
        "gate": "g1-phase-gate-v4.2, unmodified",
        "purpose": "score the fixed gait clock that the learned encoder is meant to replace",
        "settings": {
            "data_fps": DATA_FPS,
            "configured_period_seconds": CONFIGURED_PERIOD_S,
            "configured_period_frames": CONFIGURED_PERIOD_FRAMES,
            "period_grid_frames": list(PERIOD_GRID),
            "offset_grid_rad": list(OFFSET_GRID),
            "reported_statistic": "maximum over the (period, offset) grid",
        },
        "configured_period_best_macro": configured,
        "best_over_grid": {
            "macro_completion": best[0],
            "period_frames": best[1],
            "period_seconds": None if best[1] is None else best[1] / DATA_FPS,
            "offset_rad": best[2],
        },
        "comparison": {
            "oracle_reference_clock": 0.9984177215189873,
            "learned_encoder_v2_1b_epoch1": 0.6962,
            "learned_encoder_v2_1c_epoch1": 0.7032,
        },
        "grid": grid,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
