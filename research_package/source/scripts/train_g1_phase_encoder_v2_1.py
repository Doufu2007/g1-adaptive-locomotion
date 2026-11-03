"""Train the single pre-registered v2.1 configuration and select with gate v4.2.

Pre-registered before the first run (no result-informed changes):

    context 30 frames, seed 20261020, 30 epochs, batch 512, lr 1e-3, hidden 64
    objective: multi-scale lags 1/4/8/16/32 + full-cycle integral, cycle-total weight 1
    selection: validation only, v4.2 freeze eligibility AND point error <= pi/6
    ranking:   macro completion desc, then point error asc, then epoch asc

Train reads only the train split; selection reads only validation. The test
split is never loaded. A checkpoint is written only if something passes.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.encoder_v2 import PhaseEncoderV2, predict_phase_v2
from src.tasks.velocity.phasor.encoder_v2_1 import (
    CYCLE_WEIGHT,
    JUMP_WEIGHT,
    LAG1_RATE_WEIGHT,
    MULTI_SCALE_LAGS,
    MULTISCALE_WEIGHT,
    NET_TOLERANCE_RAD,
    POINT_WEIGHT,
    REVERSE_WEIGHT,
    build_multiscale_pairs,
    phase_v21_loss,
)
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2


DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"
REPORT_PATH = OUT_DIR / "v21_epoch_sweep.json"
PREDICTIONS_PATH = OUT_DIR / "v21_epoch_predictions.npz"

SEED = 20261020
CONTEXT_FRAMES = 30
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
HIDDEN = 64

POINT_ERROR_CAP_RAD = NET_TOLERANCE_RAD
#: Pre-declared stop rule. If the best configuration cannot clear this after the
#: multi-scale objective, the bar is unattainable under this data and target
#: definition and the next move is data or target definition, not another loss.
RIDGE_FLOOR_MACRO = 0.6239
STOP_THRESHOLD_MACRO = 0.65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_context_windows(split: dict[str, np.ndarray], context_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous pose context from the stored 10-frame windows.

    Copied unchanged from the frozen v2 recipe so the two runs share inputs.
    """
    if context_frames != CONTEXT_FRAMES:
        raise ValueError(f"the v2.1 recipe is fixed at {CONTEXT_FRAMES} context frames")
    count = len(split["frame"])
    result = np.empty((count, context_frames, 29), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    groups: dict[str, list[int]] = {}
    for index, sequence in enumerate(split["sequence"]):
        groups.setdefault(str(sequence), []).append(index)
    for indexes in groups.values():
        order = sorted(indexes, key=lambda index: int(split["frame"][index]))
        frames = np.asarray([split["frame"][index] for index in order], dtype=np.int64)
        segment_starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
        segment_ends = np.r_[segment_starts[1:], len(order)]
        for start, end in zip(segment_starts, segment_ends):
            segment_order = order[int(start):int(end)]
            for position, row_index in enumerate(segment_order):
                if position == 0:
                    stored = np.asarray(split["x"][row_index], dtype=np.float32)
                    pad = context_frames - stored.shape[0]
                    window = np.concatenate((np.repeat(stored[0][None, :], pad, axis=0), stored), axis=0)
                else:
                    previous = result[segment_order[position - 1]]
                    latest = np.asarray(split["x"][row_index, -1], dtype=np.float32)
                    window = np.concatenate((previous[1:], latest[None, :]), axis=0)
                result[row_index] = window
                valid[row_index] = position + 10 >= context_frames
    return result, valid


def split_arrays(archive: Any, split: str) -> dict[str, np.ndarray]:
    return {
        "x": archive[f"x_{split}"],
        "y": archive[f"y_{split}_cycles"],
        "sequence": archive[f"sequence_{split}"],
        "frame": archive[f"frame_{split}"],
        "left_touchdown": archive[f"left_touchdown_{split}"],
    }


class PairDataset(Dataset):
    """Index-only dataset: the pose windows stay in one base tensor."""

    def __init__(self, anchor: np.ndarray, target: np.ndarray, weight: np.ndarray,
                 base_weight: np.ndarray, kind: np.ndarray) -> None:
        self.anchor = torch.from_numpy(anchor)
        self.target = torch.from_numpy(target)
        self.weight = torch.from_numpy(weight)
        self.base_weight = torch.from_numpy(base_weight)
        self.kind = torch.from_numpy(kind)

    def __len__(self) -> int:
        return len(self.anchor)

    def __getitem__(self, index: int):
        return (self.anchor[index], self.target[index], self.weight[index],
                self.base_weight[index], self.kind[index])


def evaluate(model: PhaseEncoderV2, windows: torch.Tensor, reference: np.ndarray,
             base: dict[str, Any], period_cuts: tuple[float, float]) -> dict[str, Any]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), 2048):
            predictions.append(predict_phase_v2(model, windows[start:start + 2048]).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    report = score_phase_cycles_v4_2(
        prediction, reference, base["sequence"], base["frame"], base["touchdowns"],
        condition_ids=base["sequence"], episode_ids=base["sequence"],
        history_valid=base["history_valid"],
        period_cuts_frames=period_cuts, required_conditions=base["conditions"],
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    return prediction, report


def main() -> int:
    if REPORT_PATH.exists():
        raise FileExistsError(f"{REPORT_PATH.relative_to(ROOT)} exists; preserve it and use a new run directory")

    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_windows_np, _ = build_context_windows(train, CONTEXT_FRAMES)
    validation_windows_np, validation_history = build_context_windows(validation, CONTEXT_FRAMES)
    train_windows = torch.from_numpy(train_windows_np).to(DEVICE)
    validation_windows = torch.from_numpy(validation_windows_np).to(DEVICE)

    anchor, target, lag, kind, weights = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool),
    )
    multiplier = np.where(kind == 0, LAG1_RATE_WEIGHT, np.where(kind == 2, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
    base_weights = weights / multiplier
    train_phase = (train["y"].astype(np.float64) * (2.0 * np.pi)).astype(np.float32)
    phase_tensor = torch.from_numpy(train_phase).to(DEVICE)
    dataset = PairDataset(anchor, target, weights, base_weights, kind)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)

    reference_validation = validation["y"].astype(np.float64) * (2.0 * np.pi)
    base = {
        "sequence": validation["sequence"],
        "frame": validation["frame"],
        "touchdowns": validation["left_touchdown"].astype(bool),
        "history_valid": validation_history,
        "conditions": sorted(np.unique(validation["sequence"]).tolist()),
    }

    set_seed(SEED)
    model = PhaseEncoderV2(context_frames=CONTEXT_FRAMES, hidden=HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs: list[dict[str, Any]] = []
    predictions_store: dict[str, np.ndarray] = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        accumulators: dict[str, list[float]] = {key: [] for key in
                                                ("loss", "point_loss", "increment_loss",
                                                 "negative_progress_penalty", "jump_penalty")}
        for index_anchor, index_target, weight, base_weight, batch_kind in loader:
            index_anchor = index_anchor.to(DEVICE, non_blocking=True)
            index_target = index_target.to(DEVICE, non_blocking=True)
            predicted_anchor = predict_phase_v2(model, train_windows[index_anchor])
            predicted_target = predict_phase_v2(model, train_windows[index_target])
            losses = phase_v21_loss(
                predicted_anchor, predicted_target,
                phase_tensor[index_anchor], phase_tensor[index_target],
                row_weights=weight.to(DEVICE), base_weights=base_weight.to(DEVICE),
                kind=batch_kind.to(DEVICE),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            for key in accumulators:
                accumulators[key].append(float(losses[key].detach().cpu()))

        prediction, report = evaluate(model, validation_windows, reference_validation, base, period_cuts)
        predictions_store[f"epoch_{epoch:02d}"] = prediction.astype(np.float32)
        record = {
            "epoch": epoch,
            "train": {key: float(np.mean(values)) for key, values in accumulators.items()},
            "macro_completion": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
            "freeze_eligible": report["encoder_freeze_eligible"],
            "point_error_within_cap": bool(report["condition_macro_point_error_rad"] <= POINT_ERROR_CAP_RAD),
        }
        epochs.append(record)
        print(f"ep {epoch:2d} loss {record['train']['loss']:.4f} point {record['train']['point_loss']:.4f} "
              f"inc {record['train']['increment_loss']:.4f} | val macro {record['macro_completion']:.4f} "
              f"pt {record['condition_macro_point_error_rad']:.3f} "
              f"cond {' '.join(f'{v:.2f}' for v in record['condition_completion'].values())} "
              f"strata {' '.join(f'{v:.2f}' for v in record['period_length_strata'].values())}", flush=True)

    eligible = [r for r in epochs if r["freeze_eligible"] and r["point_error_within_cap"]]
    ranked = sorted(eligible, key=lambda r: (-r["macro_completion"], r["condition_macro_point_error_rad"], r["epoch"]))
    best_by_macro = max(epochs, key=lambda r: r["macro_completion"])
    selected = ranked[0] if ranked else None

    if selected is not None:
        status = "READY_SELECTED_CANDIDATE"
    elif best_by_macro["macro_completion"] <= STOP_THRESHOLD_MACRO:
        status = "STOP_BAR_UNATTAINABLE_ON_THIS_DATA_AND_TARGET"
    else:
        status = "BLOCKED_NO_EPOCH_PASSED_V4_2_WITH_POINT_CAP"

    payload = {
        "schema": "g1.phase_encoder_v21_validation.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "source_split": "train_only_fit_validation_only_selection",
        "test_split_read": False,
        "preregistered": {
            "context_frames": CONTEXT_FRAMES, "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "hidden": HIDDEN,
            "multi_scale_lags": list(MULTI_SCALE_LAGS),
            "weights": {
                "point": POINT_WEIGHT, "lag1_rate": LAG1_RATE_WEIGHT,
                "multiscale": MULTISCALE_WEIGHT, "cycle": CYCLE_WEIGHT,
                "reverse": REVERSE_WEIGHT, "jump": JUMP_WEIGHT,
            },
            "point_error_cap_rad": POINT_ERROR_CAP_RAD,
            "selection": "freeze_eligible AND point_error <= cap; rank macro desc, point asc, epoch asc",
            "stop_rule": f"best macro <= {STOP_THRESHOLD_MACRO} => bar unattainable on this data/target",
            "ridge_floor_macro_v4_2": RIDGE_FLOOR_MACRO,
        },
        "dataset_sha256": sha256_file(DATA_PATH),
        "pair_count": int(len(anchor)),
        "cycle_row_count": int((kind == 2).sum()),
        "status": status,
        "selected_epoch": selected["epoch"] if selected else None,
        "best_epoch_by_macro": best_by_macro["epoch"],
        "best_macro_completion": best_by_macro["macro_completion"],
        "beats_ridge_floor": bool(best_by_macro["macro_completion"] > RIDGE_FLOOR_MACRO),
        "checkpoint_frozen": False,
        "ppo_blocked": True,
        "epochs": epochs,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(PREDICTIONS_PATH, **predictions_store)
    print(json.dumps({
        "status": status,
        "selected_epoch": payload["selected_epoch"],
        "best_epoch_by_macro": payload["best_epoch_by_macro"],
        "best_macro_completion": payload["best_macro_completion"],
        "beats_ridge_floor": payload["beats_ridge_floor"],
        "outputs": [str(REPORT_PATH.relative_to(ROOT)), str(PREDICTIONS_PATH.relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
