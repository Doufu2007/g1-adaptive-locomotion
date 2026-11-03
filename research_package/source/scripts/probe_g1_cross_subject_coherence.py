"""Does the same joint configuration mean the same phase on another subject?

`probe_g1_phase_ambiguity.py` measured phase coherence among neighbours drawn
from the validation subject alone and found a median spread of 0.217 rad with
only 1.7% of neighbourhoods spanning more than pi/2. So within a subject the
joints almost determine the phase locally.

Every trained estimator, however, learns on three other subjects and is scored
on this one, and they all land in 0.50-0.70 macro. If the joint-to-phase map
shifts between subjects, that is where the ceiling comes from; if it does not,
then the transfer story is wrong too and the cause is something else again.

The comparison is made without training. The same nearest-neighbour retrieval
is run twice over the same queries:

  within     neighbours drawn from the validation subject (temporal exclusion
             200 frames, as in the ambiguity probe)
  cross      neighbours drawn from the training subjects only (3/4/5)

Both use the same standardisation and PCA, fit on the training subjects and
applied to both splits, so the space is identical and only the candidate pool
changes.

One bias runs in the cross-subject arm's favour: it draws from ~73k candidates
against ~45k, so its neighbours are geometrically closer. A larger cross-subject
spread is therefore not an artefact of a thinner pool.

Validation and train splits only. The test split is never loaded.
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
from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    build_context_windows,
    split_arrays,
)

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "cross_subject_coherence.json"

NEIGHBOURS = 10
TEMPORAL_EXCLUSION = 200
PCA_COMPONENTS = 32
QUERY_CHUNK = 500


def fit_reduce(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise and PCA on the training windows, then apply to both."""
    from sklearn.decomposition import PCA

    flat_train = train.reshape(len(train), -1).astype(np.float64)
    flat_other = other.reshape(len(other), -1).astype(np.float64)
    mean = flat_train.mean(axis=0, keepdims=True)
    scale = flat_train.std(axis=0, keepdims=True)
    flat_train = (flat_train - mean) / np.where(scale < 1e-8, 1.0, scale)
    flat_other = (flat_other - mean) / np.where(scale < 1e-8, 1.0, scale)
    components = min(PCA_COMPONENTS, flat_train.shape[0], flat_train.shape[1])
    model = PCA(n_components=components, svd_solver="randomized", random_state=0).fit(flat_train)
    return (model.transform(flat_train).astype(np.float32),
            model.transform(flat_other).astype(np.float32))


def cross_neighbours(query: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pool_norm = np.einsum("ij,ij->i", pool, pool)
    indices = np.empty((len(query), NEIGHBOURS), dtype=np.int64)
    distances = np.empty((len(query), NEIGHBOURS), dtype=np.float32)
    for start in range(0, len(query), QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, len(query))
        block = query[start:stop]
        d = (np.einsum("ij,ij->i", block, block)[:, None] + pool_norm[None, :]
             - 2.0 * (block @ pool.T))
        part = np.argpartition(d, NEIGHBOURS - 1, axis=1)[:, :NEIGHBOURS]
        chosen = np.take_along_axis(d, part, axis=1)
        order = np.argsort(chosen, axis=1)
        indices[start:stop] = np.take_along_axis(part, order, axis=1)
        distances[start:stop] = np.sqrt(np.take_along_axis(chosen, order, axis=1))
    return indices, distances


def describe(spread: np.ndarray) -> dict:
    return {
        "mean_spread_rad": float(spread.mean()),
        "median_spread_rad": float(np.median(spread)),
        "share_above_pi_over_2": float((spread > np.pi / 2.0).mean()),
        "share_above_pi_over_4": float((spread > np.pi / 4.0).mean()),
    }


def main() -> int:
    probe.PCA_COMPONENTS = PCA_COMPONENTS
    archive = np.load(DATA_PATH, allow_pickle=False)
    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_joints, train_history = build_context_windows(train, CONTEXT_FRAMES)
    valid_joints, valid_history = build_context_windows(validation, CONTEXT_FRAMES)

    train_keep = train_history
    valid_keep = valid_history
    print(f"train rows {int(train_keep.sum())}, validation rows {int(valid_keep.sum())}", flush=True)

    reduced_train, reduced_valid = fit_reduce(train_joints[train_keep], valid_joints[valid_keep])

    phase_train = train["y"].astype(np.float64)[train_keep] * (2.0 * np.pi)
    phase_valid = validation["y"].astype(np.float64)[valid_keep] * (2.0 * np.pi)
    sequence_valid = validation["sequence"][valid_keep]
    frame_valid = validation["frame"][valid_keep].astype(np.int64)

    within_idx, within_dist = probe.neighbour_indices(
        reduced_valid, sequence_valid, frame_valid, NEIGHBOURS, TEMPORAL_EXCLUSION
    )
    within = probe.circular_std(phase_valid[within_idx], axis=1)

    cross_idx, cross_dist = cross_neighbours(reduced_valid, reduced_train)
    cross = probe.circular_std(phase_train[cross_idx], axis=1)

    print(f"within-subject  median spread {np.median(within):.3f} rad   "
          f"above pi/2 {describe(within)['share_above_pi_over_2']:.1%}", flush=True)
    print(f"cross-subject   median spread {np.median(cross):.3f} rad   "
          f"above pi/2 {describe(cross)['share_above_pi_over_2']:.1%}", flush=True)

    payload = {
        "schema": "g1.cross_subject_coherence.v1",
        "splits": {"pool": "train", "query": "validation"},
        "test_split_read": False,
        "purpose": "measure whether the joint-to-phase relation shifts between subjects, without training",
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "temporal_exclusion_frames": TEMPORAL_EXCLUSION,
            "pca_components": PCA_COMPONENTS,
            "pca_fit_on": "train",
        },
        "rows": {"train_pool": int(train_keep.sum()), "validation_queries": int(valid_keep.sum())},
        "within_subject": {
            **describe(within),
            "nearest_pool_distance_median": float(np.median(within_dist[:, 0])),
        },
        "cross_subject": {
            **describe(cross),
            "nearest_pool_distance_median": float(np.median(cross_dist[:, 0])),
        },
        "interpretation": (
            "Both arms share the same feature space and the same k. The cross arm draws "
            "from a larger pool, which makes its neighbours geometrically closer; a larger "
            "cross-subject spread therefore is not a thin-pool artefact."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
