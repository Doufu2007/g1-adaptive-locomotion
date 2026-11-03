"""Measure how much of the v2.1b objective supervises the quantity the gate grades.

Why this exists
---------------
Gate v4.2 credits a cycle only when the unwrapped predicted phase advances one
full turn inside it. On the best v2.1b epoch, 69.3% of the 603 validation
cycles already satisfy that, the shortfall splits into 113 cycles that advance
too little and 72 that advance too much, and the net-turn error correlates with
the cycle period at r = 0.64. The failure therefore lives in the *cycle
integral*, not in the pointwise phase.

The v2.1b objective contains exactly one row per complete cycle that supervises
that integral: the KIND_CYCLE anchor/target pair. Every other row comes from a
fixed lag of 1/4/8/16/32 frames, and the median training cycle is 59 frames, so
no fixed lag spans a cycle. This script measures what share of the training
weight the cycle row actually receives. It trains nothing and changes nothing.

The frozen weight itself is not the question; the question is whether 0.70% is
enough, and that is what v2.1e varies. This file records the arithmetic that
motivates the sweep so the rationale is auditable rather than asserted.

Reads the train split and, for context only, the already-published validation
diagnosis. The test split is never loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import split_arrays  # noqa: E402
from src.tasks.velocity.phasor.encoder_v2_1 import (  # noqa: E402
    CYCLE_WEIGHT,
    KIND_CYCLE,
    KIND_LAG1,
    KIND_SCALE,
    MULTI_SCALE_LAGS,
    build_multiscale_pairs,
)

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
NET_DIAGNOSIS_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "net_progression_diagnosis.json"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "v21e_cycle_weight_decomposition.json"

#: Frozen expectations; a mismatch means the builder drifted and the shares below
#: no longer describe the objective that produced the published numbers.
EXPECTED_PAIRS = 301551
EXPECTED_CYCLE_ROWS = 948
EXPECTED_LAG1_ROWS = 70708
EXPECTED_SCALE_ROWS = 229895

CANDIDATE_WEIGHTS = (20.0, 60.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def share_at(cycle_weight: float, cycle_base: float, other: float) -> float:
    """Share of the increment-term row weight the cycle rows receive.

    ``_weighted`` divides by the batch weight sum, so only ratios matter; scaling
    the cycle numerator by ``cycle_weight / CYCLE_WEIGHT`` is exact.
    """
    numerator = cycle_weight * cycle_base
    return float(numerator / (other + numerator))


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    train = split_arrays(archive, "train")
    anchor, _target, _lag, kind, weight = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool)
    )

    if len(anchor) != EXPECTED_PAIRS:
        raise ValueError(f"pair count {len(anchor)} != frozen {EXPECTED_PAIRS}")
    counts = {
        "lag1": int((kind == KIND_LAG1).sum()),
        "scale": int((kind == KIND_SCALE).sum()),
        "cycle": int((kind == KIND_CYCLE).sum()),
    }
    if counts != {
        "lag1": EXPECTED_LAG1_ROWS,
        "scale": EXPECTED_SCALE_ROWS,
        "cycle": EXPECTED_CYCLE_ROWS,
    }:
        raise ValueError(f"pair-kind counts drifted from the frozen recipe: {counts}")

    total = float(weight.sum())
    cycle_base = float(weight[kind == KIND_CYCLE].sum()) / CYCLE_WEIGHT
    other = total - float(weight[kind == KIND_CYCLE].sum())

    cycle_frames = (_lag[kind == KIND_CYCLE] + 1).astype(np.int64)
    percentiles = {
        f"p{q:02d}": float(np.percentile(cycle_frames, q))
        for q in (0, 5, 25, 50, 75, 90, 95, 99, 100)
    }

    projection = {
        f"{weight_value:g}": {
            "cycle_weight": weight_value,
            "full_cycle_row_weight_share": share_at(weight_value, cycle_base, other),
        }
        for weight_value in (CYCLE_WEIGHT, *CANDIDATE_WEIGHTS)
    }

    validation_context: dict[str, object] = {}
    if NET_DIAGNOSIS_PATH.exists():
        published = json.loads(NET_DIAGNOSIS_PATH.read_text(encoding="utf-8"))
        validation_context = {
            "source": str(NET_DIAGNOSIS_PATH.relative_to(ROOT)),
            "n_cycles": published["n_cycles"],
            "share_within_tolerance": published["net_minus_two_pi"]["share_within_tolerance"],
            "count_too_small": published["net_minus_two_pi"]["count_too_small"],
            "count_too_large": published["net_minus_two_pi"]["count_too_large"],
            "net_error_sd": published["net_minus_two_pi"]["sd"],
            "net_error_median": published["net_minus_two_pi"]["median"],
            "net_error_q25": published["net_minus_two_pi"]["q25"],
            "net_error_q75": published["net_minus_two_pi"]["q75"],
            "corr_period_net_error": published["corr_period_net_error"],
            "by_stratum_mean_net_error": {
                key: value["mean_net_error"]
                for key, value in published["by_stratum"].items()
            },
        }

    payload = {
        "schema": "g1.v21e_cycle_weight_decomposition.v1",
        "test_split_read": False,
        "purpose": (
            "Measure the share of the v2.1b training objective that supervises the one-turn-per-cycle "
            "integral, which is the only quantity gate v4.2 credits."
        ),
        "dataset_sha256": sha256_file(DATA_PATH),
        "train_pairs": len(anchor),
        "pair_kind_rows": counts,
        "frozen_weights": {
            "cycle_weight": CYCLE_WEIGHT,
            "multi_scale_lags": list(MULTI_SCALE_LAGS),
        },
        "weight_shares_at_frozen_settings": {
            "lag1": float(weight[kind == KIND_LAG1].sum()) / total,
            "scale": float(weight[kind == KIND_SCALE].sum()) / total,
            "cycle": float(weight[kind == KIND_CYCLE].sum()) / total,
        },
        "cycle_row_weight_share_projection": projection,
        "train_cycle_length_frames": {
            "count": int(len(cycle_frames)),
            "percentiles": percentiles,
            "share_above_120_frames": float((cycle_frames > 120).mean()),
            "note": (
                "A fixed lag never exceeds 32 frames, so for the median 59-frame cycle no fixed-lag row "
                "spans a cycle. The single KIND_CYCLE row per cycle is the only full-cycle supervision."
            ),
        },
        "point_term_invariance": (
            "The point term is weighted by base_weights (row weight divided by its kind multiplier), which "
            "is independent of the multiplier, so changing CYCLE_WEIGHT leaves the point term unchanged and "
            "redistributes only the increment term."
        ),
        "validation_context_from_published_diagnosis": validation_context,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"pairs {len(anchor)}  cycle rows {counts['cycle']}")
    for key, block in projection.items():
        print(f"  cycle weight {key:>5s} -> full-cycle share {block['full_cycle_row_weight_share'] * 100:5.2f}%")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
