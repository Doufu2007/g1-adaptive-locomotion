"""How much phase ambiguity does the input itself carry?

Every arm in this project -- k-NN, ridge, MLP, three CNNs -- lands in
0.50-0.70 macro completion against a 0.90 bar. That clustering is consistent
with two very different stories:

  (a) the estimators are all leaving skill on the table; or
  (b) frames that are close in joint space can sit at genuinely different
      phases, so no frame-wise function of the joints can be right on them.

This probe measures (b) directly, without training anything.

Two statistics, both label-using but estimator-free:

  ambiguity   circular spread of the true phase over each row's k nearest
              neighbours in joint space, with a temporal exclusion so the
              neighbouring frame cannot stand in for the query. This is what
              a local estimator is forced to average over. If a large share of
              neighbourhoods span more than pi/2, no k-NN can resolve them.

  height split
              the label is built by forward kinematics *with base pose* to
              detect foot contact, then the encoder is fed joints only. At a
              fixed joint configuration, stance and swing differ mainly in how
              high the pelvis sits. So within each neighbourhood, split the k
              neighbours into the half closest and the half farthest in base
              height, and compare their phase spreads. Both halves have the
              same size, so there is no subset-size confound. A drop in the
              close half is direct evidence that the dropped column carries
              the disambiguating information.

              Control: the same split with the height column shuffled across
              rows. Shuffling preserves the marginal distribution and destroys
              any relation to phase, so any apparent gain under the control is
              the split procedure itself, not height.

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

import scripts.probe_g1_phase_identifiability as probe  # noqa: E402
from scripts.probe_g1_basepose_identifiability import base_pose_windows  # noqa: E402
from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    build_context_windows,
    split_arrays,
)

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "input_ambiguity_probe.json"

NEIGHBOURS = 10
TEMPORAL_EXCLUSION = 200
PCA_COMPONENTS = 32
SEED = 20261020


def paired_delta(a: np.ndarray, b: np.ndarray) -> dict:
    delta = a - b
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta": float(delta.mean()),
        "sd_delta": float(delta.std(ddof=1)),
        "stderr": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "share_a_below_b": float(np.mean(a < b)),
    }


def split_by_height(neighbours: np.ndarray, height: np.ndarray, phase: np.ndarray,
                    half: int) -> tuple[np.ndarray, np.ndarray]:
    """Split each neighbourhood into the half closest / farthest in height."""
    query = height[:, None]
    gap = np.abs(height[neighbours] - query)
    order = np.argsort(gap, axis=1)
    close = np.take_along_axis(neighbours, order[:, :half], axis=1)
    far = np.take_along_axis(neighbours, order[:, -half:], axis=1)
    return probe.circular_std(phase[close], axis=1), probe.circular_std(phase[far], axis=1)


def main() -> int:
    probe.PCA_COMPONENTS = PCA_COMPONENTS
    archive = np.load(DATA_PATH, allow_pickle=False)
    validation = split_arrays(archive, "validation")

    joints, history_valid = build_context_windows(validation, CONTEXT_FRAMES)
    base, base_valid = base_pose_windows(validation["sequence"], validation["frame"])
    usable = history_valid & base_valid

    reduced = probe.reduce_contexts(joints[usable])
    sequence = validation["sequence"][usable]
    frame = validation["frame"][usable].astype(np.int64)
    phase = validation["y"].astype(np.float64)[usable] * (2.0 * np.pi)
    height = base[usable][:, -1, 2].astype(np.float64)

    print(f"rows usable: {len(phase)} of {len(validation['frame'])}", flush=True)
    neighbours, distances = probe.neighbour_indices(
        reduced, sequence, frame, NEIGHBOURS, TEMPORAL_EXCLUSION
    )
    complete = np.isfinite(distances).all(axis=1)
    print(f"rows with all {NEIGHBOURS} neighbours finite: {int(complete.sum())}", flush=True)

    neighbours = neighbours[complete]
    phase = phase[complete]
    height = height[complete]

    spread = probe.circular_std(phase[neighbours], axis=1)

    half = NEIGHBOURS // 2
    close, far = split_by_height(neighbours, height, phase, half)

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(height)
    close_ctl, far_ctl = split_by_height(neighbours, shuffled, phase, half)

    ambiguous = spread > (np.pi / 2.0)
    print(f"neighbourhood phase spread: median {np.median(spread):.3f} rad, "
          f"share above pi/2 {ambiguous.mean():.1%}", flush=True)
    print(f"height split (all rows): close {close.mean():.3f} vs far {far.mean():.3f} rad", flush=True)
    print(f"height split (control):  close {close_ctl.mean():.3f} vs far {far_ctl.mean():.3f} rad", flush=True)
    print(f"height split (ambiguous rows only): "
          f"close {close[ambiguous].mean():.3f} vs far {far[ambiguous].mean():.3f} rad", flush=True)

    payload = {
        "schema": "g1.input_ambiguity_probe.v1",
        "split": "validation",
        "test_split_read": False,
        "purpose": "measure how much of the 0.50-0.70 ceiling is input ambiguity rather than estimator weakness",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_components": PCA_COMPONENTS,
            "seed": SEED,
            "height_column": "base pose column 2 (pelvis height) at the window anchor frame",
        },
        "rows": {
            "usable": int(len(phase)),
            "total": int(len(validation["frame"])),
        },
        "ambiguity": {
            "mean_spread_rad": float(spread.mean()),
            "median_spread_rad": float(np.median(spread)),
            "share_spread_above_pi_over_2": float(ambiguous.mean()),
            "share_spread_above_pi_over_4": float((spread > np.pi / 4.0).mean()),
        },
        "height_split_all_rows": paired_delta(close, far),
        "height_split_control": paired_delta(close_ctl, far_ctl),
        "height_split_ambiguous_rows": {
            **paired_delta(close[ambiguous], far[ambiguous]),
            "n_rows": int(ambiguous.sum()),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
