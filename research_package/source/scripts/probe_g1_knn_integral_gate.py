"""What does the best label-free local estimator reach on the cycle gate?

Every trained arm lands in 0.63-0.70 macro against a 0.90 bar, and four
different levers failed to move that. Two stories remain and they have opposite
consequences:

  (a) the information is present in a 30-frame joint window, and every learner
      so far has failed to extract it  -> the paper needs a better learner;
  (b) the per-frame phase is locally recoverable (the ambiguity probe found a
      median neighbour spread of 0.217 rad) but the *cycle integral* is not
      determined to within the gate's pi/6 tolerance -> no learner can pass on
      this input contract, and the paper states that as its limit.

The ambiguity probe measured (b) at the level of a single frame. It never
measured (b) at the level the gate actually grades, which is the integral of
the predicted phase over a whole cycle. This probe closes that gap without
training anything.

The estimator is a k-NN over the training subjects: standardise and PCA on the
training windows, retrieve the 10 nearest training rows for each validation row,
and predict the circular mean of their phases. It is nonparametric, it never
sees a validation label, and it is the strongest evidence available about what
the input carries.

  if macro >= 0.90   the information survives integration; the learners are the
                     problem, and the encoder line should be reopened
  if macro ~= 0.70   the ceiling is a property of the input contract and of what
                     a per-frame regression can express, not of one architecture

Read-only. No training, no threshold change, validation queries against a
training pool, test split never loaded.
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

import scripts.probe_g1_phase_identifiability as probe  # noqa: E402
from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    V4_CALIBRATION_PATH,
    build_context_windows,
    split_arrays,
)
from src.tasks.velocity.phasor.encoder_v2_1 import NET_TOLERANCE_RAD  # noqa: E402
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "knn_integral_gate.json"

NEIGHBOURS = 10
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
        if (start // QUERY_CHUNK) % 20 == 0:
            print(f"  neighbours {stop}/{len(query)}", flush=True)
    return indices, distances


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    records = [row for row in report["cycle_records"] if row["evaluable"]]
    error = np.asarray([row["encoder_net_progression_rad"] - 2.0 * np.pi for row in records])
    periods = np.asarray([row["period_frames"] for row in records], dtype=float)
    slip = np.abs(np.abs(error) - 2.0 * np.pi) < 1.0
    strata = np.asarray([str(row["period_length_stratum"]) for row in records])
    summary = {
        "macro_completion": report["macro_cycle_completion_fraction"],
        "point_error_rad": report["condition_macro_point_error_rad"],
        "condition_completion": {k: v["completion_fraction"]
                                 for k, v in report["condition_completion"].items()},
        "period_length_strata": {k: v["completion_fraction"]
                                 for k, v in report["period_length_strata"].items()},
        "n_cycles": int(len(records)),
        "mean_net_error": float(error.mean()),
        "sd_net_error": float(error.std()),
        "median_net_error": float(np.median(error)),
        "share_within_tolerance": float(np.mean(np.abs(error) <= NET_TOLERANCE_RAD)),
        "count_too_small": int((error < -NET_TOLERANCE_RAD).sum()),
        "count_too_large": int((error > NET_TOLERANCE_RAD).sum()),
        "whole_turn_slip_count": int(slip.sum()),
        "corr_period_net_error": float(np.corrcoef(periods, error)[0, 1]) if len(error) > 2 else None,
    }
    for name in ("short", "medium", "long"):
        mask = strata == name
        if mask.any():
            summary[f"mean_net_error_{name}"] = float(error[mask].mean())
    return summary


def main() -> int:
    probe.PCA_COMPONENTS = PCA_COMPONENTS
    archive = np.load(DATA_PATH, allow_pickle=False)
    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_joints, train_history = build_context_windows(train, CONTEXT_FRAMES)
    valid_joints, valid_history = build_context_windows(validation, CONTEXT_FRAMES)
    print(f"train rows {int(train_history.sum())}, validation rows {int(valid_history.sum())}",
          flush=True)

    reduced_train, reduced_valid = fit_reduce(train_joints[train_history],
                                              valid_joints[valid_history])
    phase_train = train["y"].astype(np.float64)[train_history] * (2.0 * np.pi)

    print("retrieving training neighbours", flush=True)
    idx, dist = cross_neighbours(reduced_valid, reduced_train)
    predicted = probe.circular_mean(phase_train[idx], axis=1)

    n_rows = len(validation["frame"])
    val_pred = np.zeros((n_rows, 2), dtype=np.float64)
    val_pred[:, 0] = 1.0
    val_pred[valid_history, 0] = np.cos(predicted)
    val_pred[valid_history, 1] = np.sin(predicted)

    cuts = tuple(float(x) for x in json.loads(
        V4_CALIBRATION_PATH.read_text(encoding="utf-8"))["period_length_cut_points_frames"])
    conditions = sorted({str(x) for x in validation["sequence"]})
    report = score_phase_cycles_v4_2(
        val_pred,
        validation["y"].astype(np.float64) * (2.0 * np.pi),
        validation["sequence"], validation["frame"],
        validation["left_touchdown"].astype(bool),
        condition_ids=validation["sequence"], episode_ids=validation["sequence"],
        history_valid=valid_history,
        period_cuts_frames=cuts, required_conditions=conditions,
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    summary = summarise(report)
    print(f"knn macro {summary['macro_completion']:.4f} point {summary['point_error_rad']:.4f} "
          f"net {summary['mean_net_error']:+.2f} sd {summary['sd_net_error']:.2f} "
          f"slip {summary['whole_turn_slip_count']}", flush=True)

    payload = {
        "schema": "g1.knn_integral_gate.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "split": {"pool": "train", "query": "validation"},
        "test_split_read": False,
        "purpose": ("score a nonparametric k-NN phase estimator on the cycle gate, to test whether "
                    "the cycle integral is recoverable from the input at all"),
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "neighbours": NEIGHBOURS,
            "pca_components": PCA_COMPONENTS,
            "pca_fit_on": "train",
            "prediction": "circular mean of neighbour true phases (train labels only)",
            "nearest_train_distance_median": float(np.median(dist[:, 0])),
        },
        "result": summary,
        "reference_points": {
            "oracle": 0.9984177215189873,
            "best_over_grid_fixed_clock": 0.21866427203253935,
            "walking_encoder": 0.41280424052165965,
            "v2_1b_epoch1": 0.6962456093142473,
            "v2_1c_epoch1": 0.703175,
            "gate_bar": 0.9,
        },
        "reading": (
            "If macro is near the trained arms, the ceiling is a property of the input "
            "contract rather than of one architecture. If it clears 0.90, the information "
            "survives integration and the learners are the problem."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
