"""Bidirectional +-1 cycle upper bound on phase identifiability.

See `docs/superpowers/plans/g1-phase-upper-bound-preregistration.md`, written
before this run. The short version: probe (c) concluded that a longer context
does not help, but it measured that with a k-NN over a fixed PCA-32 projection,
so growing the input from 870 to 2610 dimensions confounds information loss with
lack of information. This run removes that confound by giving a parametric model
one full cycle in both directions with no projection bottleneck.

Everything except the window and the architecture is imported unchanged from the
v2.1b run, so exactly one thing varies.

This is a diagnostic upper bound, not a candidate. Selection is
best-epoch-on-validation, which peeks at the selection metric; the number is
optimistically biased and cannot be used to freeze a checkpoint or unblock PPO.
The test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.data import resample_joint_positions
from src.tasks.velocity.phasor.encoder_v2 import FEATURE_DIM, normalize_phase, predict_phase_v2
from src.tasks.velocity.phasor.encoder_v2_1 import (
    CYCLE_WEIGHT,
    LAG1_RATE_WEIGHT,
    MULTISCALE_WEIGHT,
    NET_TOLERANCE_RAD,
    build_multiscale_pairs,
)
from src.tasks.velocity.phasor.encoder_v2_1b import DROPOUT, GRAD_CLIP, LEARNING_RATE, WEIGHT_DECAY, phase_v21b_loss
from scripts.train_g1_phase_encoder_v2_1b import (
    BATCH_SIZE,
    EPOCHS,
    PairDataset,
    build_context_windows,
    evaluate,
    set_seed,
    sha256_file,
    split_arrays,
)

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
RAW_DIR = ROOT / "data" / "raw" / "lafan1_g1_walk"
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"
REPORT_PATH = OUT_DIR / "upper_bound_epoch_sweep.json"

SEED = 20261020
HALF_WINDOW = 75
CONTEXT_FRAMES = 2 * HALF_WINDOW + 1
DILATIONS = (1, 2, 4, 8, 16)
HIDDEN = 64
SOURCE_FPS = 30.0
TARGET_FPS = 50.0
JOINT_DIM = 29

WIN_AT_OR_BELOW = 0.60
CLOSED_AT_OR_ABOVE = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BidirectionalPhaseEncoder(nn.Module):
    """Dilated same-padding CNN read out at the centre frame.

    Every layer is centred, so the centre tap sees HALF_WINDOW frames on both
    sides; the dilations give a receptive field of 1 + 4*(1+2+4+8+16) = 125
    frames, about 0.83 of the window and well over one gait cycle at this data's
    45-70 frame periods.
    """

    def __init__(self, context_frames: int = CONTEXT_FRAMES, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.context_frames = int(context_frames)
        self.centre = self.context_frames // 2
        layers: list[nn.Module] = []
        channels = FEATURE_DIM
        for dilation in DILATIONS:
            layers.append(nn.Conv1d(channels, hidden, kernel_size=5, padding=2 * dilation, dilation=dilation))
            layers.append(nn.GELU())
            channels = hidden
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.context_frames, FEATURE_DIM):
            raise ValueError(
                f"expected [batch, {self.context_frames}, {FEATURE_DIM}] feature input, got {tuple(features.shape)}"
            )
        encoded = self.conv(features.permute(0, 2, 1))
        return normalize_phase(self.head(self.dropout(encoded[:, :, self.centre])))


def build_windows(split: dict[str, np.ndarray], half: int, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, int]:
    """+-`half` resampled joint frames around each row's anchor frame.

    The stored `x` is `(raw - train_mean) / train_std` (verified exactly on the
    stored rows), so the same map is applied here; skipping it would feed the
    model a different scale than every previous run.
    """
    width = 2 * half + 1
    count = len(split["frame"])
    result = np.empty((count, width, JOINT_DIM), dtype=np.float32)
    padded_rows = 0
    for name in sorted(set(split["sequence"].tolist())):
        rows = np.flatnonzero(split["sequence"] == name)
        values = np.loadtxt(RAW_DIR / f"{name}.csv", delimiter=",", dtype=np.float64)
        timestamps = np.arange(len(values), dtype=np.float64) / SOURCE_FPS
        _, resampled = resample_joint_positions(timestamps, values[:, 7:], target_hz=TARGET_FPS)
        normalised = (resampled - mean) / scale
        for index in rows:
            anchor = int(split["frame"][index])
            low, high = anchor - half, anchor + half + 1
            segment = normalised[max(low, 0):min(high, len(normalised))]
            if low < 0:
                segment = np.concatenate((np.repeat(normalised[0][None, :], -low, axis=0), segment), axis=0)
                padded_rows += 1
            if high > len(normalised):
                segment = np.concatenate((segment, np.repeat(normalised[-1][None, :], high - len(normalised), axis=0)), axis=0)
                padded_rows += 1
            if segment.shape[0] != width:
                raise ValueError(f"window for {name} frame {anchor} has length {segment.shape[0]}, expected {width}")
            result[index] = segment
    return result, padded_rows


def main() -> int:
    if REPORT_PATH.exists():
        raise FileExistsError(f"{REPORT_PATH.relative_to(ROOT)} exists; preserve it and use a new run")

    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    mean = archive["train_mean"].astype(np.float64)
    scale = archive["train_std"].astype(np.float64)
    if np.any(scale < 1e-8):
        raise ValueError("train_std has a zero entry; the normalisation would divide by zero")

    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_windows_np, train_padded = build_windows(train, HALF_WINDOW, mean, scale)
    validation_windows_np, validation_padded = build_windows(validation, HALF_WINDOW, mean, scale)
    # Reuse the 30-frame builder only to recover the row mask every previous run
    # was scored on, so the evaluated frames are identical to v2.1b's.
    _, validation_history = build_context_windows(validation, 30)

    train_windows = torch.from_numpy(train_windows_np).to(DEVICE)
    validation_windows = torch.from_numpy(validation_windows_np).to(DEVICE)
    print(f"windows {CONTEXT_FRAMES} frames; edge-padded rows: train {train_padded}, validation {validation_padded}",
          flush=True)

    anchor, target, lag, kind, weights = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool),
    )
    multiplier = np.where(kind == 0, LAG1_RATE_WEIGHT, np.where(kind == 2, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
    base_weights = weights / multiplier
    train_phase = (train["y"].astype(np.float64) * (2.0 * np.pi)).astype(np.float32)
    phase_tensor = torch.from_numpy(train_phase).to(DEVICE)

    loader = DataLoader(
        PairDataset(anchor, target, weights, base_weights, kind), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )

    reference_validation = validation["y"].astype(np.float64) * (2.0 * np.pi)
    base = {
        "sequence": validation["sequence"],
        "frame": validation["frame"],
        "touchdowns": validation["left_touchdown"].astype(bool),
        "history_valid": validation_history,
        "conditions": sorted(np.unique(validation["sequence"]).tolist()),
    }

    set_seed(SEED)
    model = BidirectionalPhaseEncoder().to(DEVICE)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {parameters}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    epochs: list[dict[str, object]] = []
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
            losses = phase_v21b_loss(
                predicted_anchor, predicted_target,
                phase_tensor[index_anchor], phase_tensor[index_target],
                row_weights=weight.to(DEVICE), base_weights=base_weight.to(DEVICE),
                kind=batch_kind.to(DEVICE),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            for key in accumulators:
                accumulators[key].append(float(losses[key].detach().cpu()))

        _, report = evaluate(model, validation_windows, reference_validation, base, period_cuts)
        record = {
            "epoch": epoch,
            "train": {key: float(np.mean(values)) for key, values in accumulators.items()},
            "macro_completion": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
            "freeze_eligible": report["encoder_freeze_eligible"],
        }
        epochs.append(record)
        print(f"ep {epoch:2d} loss {record['train']['loss']:.4f} point {record['train']['point_loss']:.4f} | "
              f"val macro {record['macro_completion']:.4f} pt {record['condition_macro_point_error_rad']:.3f} "
              f"cond {' '.join(f'{v:.2f}' for v in record['condition_completion'].values())} "
              f"strata {' '.join(f'{v:.2f}' for v in record['period_length_strata'].values())}", flush=True)

    best = min(epochs, key=lambda r: r["condition_macro_point_error_rad"])
    best_macro = max(epochs, key=lambda r: r["macro_completion"])
    point = float(best["condition_macro_point_error_rad"])
    if point <= WIN_AT_OR_BELOW:
        verdict = "CONTEXT_CONFOUND_REAL_GO_BACK_TO_CONTEXT_AND_CAPACITY"
    elif point >= CLOSED_AT_OR_ABOVE:
        verdict = "INPUT_SIDE_CLOSED"
    else:
        verdict = "INCONCLUSIVE_DO_NOT_DECIDE"

    payload = {
        "schema": "g1.phase_upper_bound.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "source_split": "train_only_fit_validation_only_selection",
        "test_split_read": False,
        "preregistration": "docs/superpowers/plans/g1-phase-upper-bound-preregistration.md",
        "diagnostic_only": {
            "checkpoint_frozen": False,
            "ppo_blocked": True,
            "reason": "best-epoch-on-validation selection peeks at the selection metric; optimistically biased, not a validated candidate",
        },
        "preregistered": {
            "half_window_frames": HALF_WINDOW,
            "context_frames": CONTEXT_FRAMES,
            "dilations": list(DILATIONS),
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "parameters": int(parameters),
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "objective": "phase_v21b_loss, imported unchanged",
            "normalisation": "(raw - train_mean) / train_std, verified exact on stored rows",
            "point_error_cap_rad": NET_TOLERANCE_RAD,
            "decision_rule": {
                "win_at_or_below_rad": WIN_AT_OR_BELOW,
                "closed_at_or_above_rad": CLOSED_AT_OR_ABOVE,
                "measured_on": "condition_macro_point_error_rad",
            },
        },
        "dataset_sha256": sha256_file(DATA_PATH),
        "edge_padded_rows": {"train": int(train_padded), "validation": int(validation_padded)},
        "pair_count": int(len(anchor)),
        "verdict": verdict,
        "best_point_error_epoch": best["epoch"],
        "best_point_error_rad": point,
        "best_macro_completion": float(best_macro["macro_completion"]),
        "best_macro_epoch": best_macro["epoch"],
        "epochs": epochs,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print()
    print(f"best point error {point:.4f} rad at epoch {best['epoch']} "
          f"(thresholds: <= {WIN_AT_OR_BELOW} win, >= {CLOSED_AT_OR_ABOVE} closed)")
    print(f"best macro completion {best_macro['macro_completion']:.4f} at epoch {best_macro['epoch']}")
    print(f"verdict: {verdict}")
    print(f"-> {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
