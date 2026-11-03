"""Split the v2.1b point error into a constant offset and a variable part.

The point cap (pi/6) treats every radian of absolute phase error the same, but
the two components are not the same thing downstream:

  * a constant offset is invisible to the cycle gate -- net progression and max
    step are both differences -- and a policy that consumes the phase clock
    can absorb it as a fixed shift;
  * a variable error changes the shape of the clock and cannot be absorbed.

So "point error = 0.77 rad" is ambiguous on its own. This script decomposes it
per sequence into a mean signed offset (circular mean of the signed error) and
a residual spread (circular standard deviation about that offset). If the spread
is small the cap is purely an alignment convention; if the spread is large the
error corrupts the clock shape and the cap is doing real work.

Reads validation only. The test split is never loaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
PREDICTIONS_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_epoch_predictions.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21b_point_error_decomposition.json"

EPOCH_CANDIDATES = (1, 2, 3, 4, 5)


def circular_error(prediction_rad: np.ndarray, reference_rad: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(prediction_rad - reference_rad), np.cos(prediction_rad - reference_rad))


def decompose(error: np.ndarray) -> dict[str, float]:
    offset = float(np.arctan2(np.sin(error).mean(), np.cos(error).mean()))
    residual = circular_error(error, np.full_like(error, offset))
    return {
        "mean_signed_offset_rad": offset,
        "mean_absolute_offset_rad": float(np.abs(error).mean()),
        "circular_std_about_offset_rad": float(np.sqrt((residual ** 2).mean())),
        "fraction_of_error_that_is_constant": float(
            1.0 - (residual ** 2).mean() / max((error ** 2).mean(), 1e-12)
        ),
        "frames": int(len(error)),
    }


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = np.load(PREDICTIONS_PATH, allow_pickle=False)

    sequence = archive["sequence_validation"]
    reference = archive["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi)

    available = [int(key.split("_")[-1]) for key in validation.keys()]
    epochs = [epoch for epoch in EPOCH_CANDIDATES if epoch in available]
    if not epochs:
        raise ValueError(f"no requested epoch present in {PREDICTIONS_PATH.name}")

    report: dict[str, object] = {
        "schema": "g1.v21b_point_error_decomposition.v1",
        "test_split_read": False,
        "note": (
            "offset = circular mean of the signed error per sequence; residual = circular "
            "error about that offset. A small residual means the error is a constant shift."
        ),
        "epochs": {},
    }

    for epoch in epochs:
        vectors = validation[f"epoch_{epoch:02d}"].astype(np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 2:
            raise ValueError(f"epoch {epoch} predictions must be [samples, 2], got {vectors.shape}")
        prediction = np.arctan2(vectors[:, 0], vectors[:, 1])
        if len(prediction) != len(reference):
            raise ValueError(f"epoch {epoch} prediction length {len(prediction)} != reference {len(reference)}")
        error = circular_error(prediction, reference)
        per_sequence = {}
        for name in sorted(set(sequence.tolist())):
            mask = sequence == name
            per_sequence[name] = decompose(error[mask])
        pooled = decompose(error)
        report["epochs"][f"epoch_{epoch:02d}"] = {
            "pooled": pooled,
            "per_sequence": per_sequence,
        }

    OUT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for epoch in epochs:
        block = report["epochs"][f"epoch_{epoch:02d}"]  # type: ignore[index]
        pooled = block["pooled"]  # type: ignore[index]
        print(f"epoch {epoch:2d}  offset {pooled['mean_signed_offset_rad']:+.3f} rad  "
              f"residual sd {pooled['circular_std_about_offset_rad']:.3f} rad  "
              f"constant share {pooled['fraction_of_error_that_is_constant']:.1%}")
        for name, row in block["per_sequence"].items():  # type: ignore[union-attr]
            print(f"          {name:18s} offset {row['mean_signed_offset_rad']:+.3f}  "
                  f"residual sd {row['circular_std_about_offset_rad']:.3f}  "
                  f"constant {row['fraction_of_error_that_is_constant']:5.1%}")
    print(f"\n-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
