"""Run the frozen G1 contact-filter, architecture, and phase-head comparisons.

Only train and validation keys are read. The already viewed test split is not
loaded. See results/g1_phase_dataset_20261019/preregister_g1_phase_followup_v1.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
ORIGINAL_PATH = DATA_DIR / "phase_dataset.npz"
FILTERED_PATH = DATA_DIR / "contact_consistency_filter_v1" / "phase_dataset_train_validation.npz"
MANUAL_PATH = DATA_DIR / "contact_consistency_filter_v1" / "touchdown_review_v1" / "manual_touchdown_audit_v1" / "manual_review_queue_B001-B144_v1.csv"
PROTOCOL_PATH = DATA_DIR / "preregister_g1_phase_followup_v1.md"
OUT_DIR = ROOT / "results" / "g1_contact_followup_20261021" / "run_v1_replay_optimized"

SEEDS = (20261021, 20261022, 20261023)
EPOCHS = 8
BATCH_SIZE = 512
HIDDEN = 64
LEARNING_RATE = 1e-3
WINDOW = 10
JOINT_DIM = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    """Read only the named train or validation arrays from an archive."""
    suffixes = {
        "x": f"x_{split}",
        "y": f"y_{split}_cycles",
        "sequence": f"sequence_{split}",
        "frame": f"frame_{split}",
        "left_touchdown": f"left_touchdown_{split}",
        "right_touchdown": f"right_touchdown_{split}",
    }
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[key] for name, key in suffixes.items()}
        result["mean"] = archive["train_mean"]
        result["std"] = archive["train_std"]
    return result


def unit_phase(raw: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
    return raw / norm


def phase_targets(cycles: torch.Tensor) -> torch.Tensor:
    radians = cycles * (2.0 * np.pi)
    return torch.stack((torch.sin(radians), torch.cos(radians)), dim=-1)


def circular_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (predicted * target).sum(dim=-1)).mean()


class CNNEncoder(nn.Module):
    """Two-layer temporal CNN with mean pooling and a final phase vector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(JOINT_DIM, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(HIDDEN, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).mean(dim=-1)
        return unit_phase(self.head(h))


class LSTMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(JOINT_DIM, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return unit_phase(self.head(h[-1]))


class FastCausalConv1d(nn.Module):
    """Conv1d-equivalent causal kernel evaluated as lagged channel projections.

    This preserves the exact kernel, dilation, parameter layout, and default
    Conv1d initialization while avoiding a very slow deterministic cuDNN path
    observed for short dilated sequences on the available Windows GPU.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        fan_in = in_channels * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]. Each kernel tap maps the matching past
        # position to the output position; this is cross-correlation exactly
        # as in Conv1d with left padding of dilation * (kernel_size - 1).
        time_major = x.transpose(1, 2).contiguous()
        batch, steps, _channels = time_major.shape
        output = self.bias.view(1, 1, -1)
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag == 0:
                source = time_major
            else:
                source = time_major[:, :steps - lag]
            projected = F.linear(source, self.weight[:, :, tap])
            if lag:
                projected = F.pad(projected, (0, 0, lag, 0))
            output = output + projected
        return output.transpose(1, 2)


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = FastCausalConv1d(in_channels, HIDDEN, kernel_size=3, dilation=dilation)
        self.skip = nn.Identity() if in_channels == HIDDEN else nn.Linear(in_channels, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv(x)
        skip = x if isinstance(self.skip, nn.Identity) else self.skip(x.transpose(1, 2)).transpose(1, 2)
        return F.relu(main + skip)


class TCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(CausalResidualBlock(JOINT_DIM, 1), CausalResidualBlock(HIDDEN, 2))
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2))[:, :, -1]
        return unit_phase(self.head(h))


class TemporalCNNBackbone(nn.Module):
    """Same CNN feature stack, preserving all ten time positions for head ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(JOINT_DIM, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(HIDDEN, HIDDEN, kernel_size=5, padding=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class DirectSequenceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TemporalCNNBackbone()
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unit_phase(self.head(self.backbone(x)))


class IntegratedVelocityHead(nn.Module):
    """Predict a first-frame phase and positive adjacent increments, then integrate."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TemporalCNNBackbone()
        self.anchor = nn.Linear(HIDDEN, 2)
        self.rate = nn.Linear(HIDDEN * 2, 1)
        nn.init.constant_(self.rate.bias, -2.3)  # about 0.1 rad per 50 Hz frame at start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        anchor = unit_phase(self.anchor(h[:, 0]))
        pairs = torch.cat((h[:, :-1], h[:, 1:]), dim=-1)
        increments = F.softplus(self.rate(pairs).squeeze(-1))
        theta0 = torch.atan2(anchor[:, 0], anchor[:, 1])
        theta = theta0[:, None] + torch.cat(
            (torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=1)), dim=1
        )
        return torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def phase_windows(data: dict[str, np.ndarray], full_labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build all ten-frame phase targets by sequence/frame key; preserve sample order."""
    lookup: dict[tuple[str, int], float] = {}
    for seq, frame, y in zip(full_labels["sequence"], full_labels["frame"], full_labels["y"]):
        key = (str(seq), int(frame))
        if key in lookup:
            raise ValueError(f"duplicate frame label: {key}")
        lookup[key] = float(y)
    values = np.zeros((len(data["frame"]), WINDOW), dtype=np.float32)
    valid = np.ones(len(data["frame"]), dtype=bool)
    for i, (seq, frame) in enumerate(zip(data["sequence"], data["frame"])):
        name, end = str(seq), int(frame)
        for j, current in enumerate(range(end - WINDOW + 1, end + 1)):
            value = lookup.get((name, current))
            if value is None:
                valid[i] = False
                break
            values[i, j] = value
    return values, valid


def predict(model: nn.Module, x: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(DEVICE)
            predictions.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions, axis=0)


def angle(pred: np.ndarray) -> np.ndarray:
    return np.arctan2(pred[..., 0], pred[..., 1])


def circular_abs_error(pred: np.ndarray, target_cycles: np.ndarray) -> np.ndarray:
    delta = angle(pred) - (target_cycles.astype(np.float64) * 2.0 * np.pi)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def summarize_errors(errors: np.ndarray, sequences: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    if not len(selected):
        return {"n": 0, "macro_sequence_mae_rad": None, "by_sequence": {}}
    rows: dict[str, Any] = {}
    for sequence in sorted({str(sequences[i]) for i in selected}):
        indices = np.asarray([i for i in selected if str(sequences[i]) == sequence], dtype=np.int64)
        value = errors[indices]
        rows[sequence] = {
            "n": int(len(indices)),
            "mae_rad": float(value.mean()),
            "median_rad": float(np.median(value)),
            "rmse_rad": float(np.sqrt(np.mean(value * value))),
            "p90_rad": float(np.quantile(value, 0.9)),
        }
    return {
        "n": int(len(selected)),
        "macro_sequence_mae_rad": float(np.mean([row["mae_rad"] for row in rows.values()])),
        "micro_mae_rad": float(errors[selected].mean()),
        "by_sequence": rows,
    }


def build_masks(validation: dict[str, np.ndarray], filtered_validation: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    keys = [(str(s), int(f)) for s, f in zip(validation["sequence"], validation["frame"])]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate validation sequence/frame keys")
    consistent_keys = {(str(s), int(f)) for s, f in zip(filtered_validation["sequence"], filtered_validation["frame"])}
    consistent = np.asarray([key in consistent_keys for key in keys], dtype=bool)
    if int(consistent.sum()) != len(consistent_keys) or int(consistent.sum()) != len(filtered_validation["frame"]):
        raise ValueError("consistency subset does not map one-to-one to original validation frames")

    masks: dict[str, np.ndarray] = {"full_validation": np.ones(len(keys), dtype=bool), "consistency_validation": consistent}
    manual_counts: dict[str, int] = {}
    with MANUAL_PATH.open(encoding="utf-8-sig", newline="") as stream:
        manual_rows = list(csv.DictReader(stream))
    if len(manual_rows) != 144:
        raise ValueError(f"expected 144 locked manual cycles, found {len(manual_rows)}")
    for name in ("manual_review_all", "event_count_mismatch", "long_retained_gt_2_5x", "long_retained_gt_1_8x", "typical_control"):
        masks[name] = np.zeros(len(keys), dtype=bool)
    sequence = np.asarray([str(s) for s in validation["sequence"]])
    frames = validation["frame"].astype(np.int64)
    for row in manual_rows:
        seq = row["sequence_id"]
        start, end = int(row["start_frame_inclusive"]), int(row["end_frame_exclusive"])
        selected = (sequence == seq) & (frames >= start) & (frames < end)
        if not selected.any():
            raise ValueError(f"manual cycle has no validation frames: {row['cycle_id']}")
        masks["manual_review_all"] |= selected
        category = row["category"]
        masks[category] |= selected
    for name, mask in masks.items():
        manual_counts[name] = int(mask.sum())
    if int(masks["manual_review_all"].sum()) != sum(manual_counts[k] for k in ("event_count_mismatch", "long_retained_gt_2_5x", "long_retained_gt_1_8x", "typical_control")):
        raise ValueError("manual review category frame sets overlap or are incomplete")
    return masks, {"validation_consistency_samples": int(consistent.sum()), "validation_full_samples": len(keys), "manual_review_frame_counts": manual_counts}


def train_run(
    *, run_id: str, factory: type[nn.Module], seed: int,
    train: dict[str, np.ndarray], full_train_labels: dict[str, np.ndarray],
    validation: dict[str, np.ndarray], validation_targets_all: np.ndarray,
    validation_target_windows: np.ndarray, validation_window_valid: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    seed_everything(seed)
    model = factory().to(DEVICE)
    window_head = factory in (DirectSequenceHead, IntegratedVelocityHead)
    if window_head:
        y_windows, valid = phase_windows(train, full_train_labels)
        n_before = len(train["x"])
        train_x = train["x"][valid]
        train_y = y_windows[valid]
        if not len(train_x):
            raise ValueError("no training rows have complete ten-frame target labels")
        target_tensor = phase_targets(torch.from_numpy(train_y))
    else:
        n_before = len(train["x"])
        train_x = train["x"]
        train_y = train["y"]
        target_tensor = phase_targets(torch.from_numpy(train_y))
        valid = np.ones(n_before, dtype=bool)

    ds = TensorDataset(torch.from_numpy(train_x), target_tensor)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_losses: list[float] = []
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            prediction = model(xb)
            loss = circular_loss(prediction, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(xb)
            seen += len(xb)
        epoch_losses.append(weighted_loss / seen)

    train_pred = predict(model, train_x)
    if train_pred.ndim == 3:
        train_last = train_pred[:, -1]
        train_target_last = train_y[:, -1]
    else:
        train_last = train_pred
        train_target_last = train_y
    train_errors = circular_abs_error(train_last, train_target_last)

    val_pred = predict(model, validation["x"])
    val_last = val_pred[:, -1] if val_pred.ndim == 3 else val_pred
    val_errors = circular_abs_error(val_last, validation_targets_all)
    val_reports = {name: summarize_errors(val_errors, validation["sequence"], mask) for name, mask in masks.items()}

    internal_window_report = None
    if val_pred.ndim == 3:
        valid_indices = np.flatnonzero(validation_window_valid)
        per_frame_errors = circular_abs_error(
            val_pred[valid_indices], validation_target_windows[valid_indices]
        )
        internal_window_report = {
            "evaluable_window_count": int(len(valid_indices)),
            "mean_circular_abs_error_rad_over_all_window_positions": float(per_frame_errors.mean()),
            "final_position_mae_rad_on_same_rows": float(per_frame_errors[:, -1].mean()),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / f"{run_id}.pt")
    np.savez_compressed(OUT_DIR / f"{run_id}_predictions.npz", val_prediction=val_last.astype(np.float32))
    if val_pred.ndim == 3:
        np.savez_compressed(OUT_DIR / f"{run_id}_window_predictions.npz", val_prediction=val_pred.astype(np.float32))

    return {
        "run_id": run_id,
        "seed": seed,
        "architecture": factory.__name__,
        "device": str(DEVICE),
        "train_samples_available": int(n_before),
        "train_samples_used": int(len(train_x)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_width": HIDDEN,
        "final_epoch_training_loss": epoch_losses[-1],
        "epoch_training_loss": epoch_losses,
        "training_final_frame_mae_rad": float(train_errors.mean()),
        "validation": val_reports,
        "window_head_internal_validation": internal_window_report,
    }


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"preserving existing result directory: {OUT_DIR}")
    if not PROTOCOL_PATH.is_file() or not MANUAL_PATH.is_file():
        raise FileNotFoundError("frozen protocol or mapped manual review set is missing")

    train_original = load_split(ORIGINAL_PATH, "train")
    validation = load_split(ORIGINAL_PATH, "validation")
    train_filtered = load_split(FILTERED_PATH, "train")
    validation_filtered = load_split(FILTERED_PATH, "validation")
    if not np.array_equal(train_original["mean"], train_filtered["mean"]) or not np.array_equal(train_original["std"], train_filtered["std"]):
        raise ValueError("normalization statistics differ between original and filtered archives")
    masks, mask_summary = build_masks(validation, validation_filtered)
    validation_windows, validation_window_valid = phase_windows(validation, validation)
    target_cycles = validation["y"].astype(np.float32)
    specs = [
        ("original_train_cnn", CNNEncoder, train_original, train_original),
        ("filtered_train_cnn", CNNEncoder, train_filtered, train_original),
        ("filtered_train_lstm", LSTMEncoder, train_filtered, train_original),
        ("filtered_train_tcn", TCNEncoder, train_filtered, train_original),
        ("filtered_train_cnn_direct_sequence_head", DirectSequenceHead, train_filtered, train_original),
        ("filtered_train_cnn_integrated_velocity_head", IntegratedVelocityHead, train_filtered, train_original),
    ]

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name, factory, training_data, full_labels in specs:
            run_id = f"{name}_seed_{seed}"
            record = train_run(
                run_id=run_id, factory=factory, seed=seed,
                train=training_data, full_train_labels=full_labels,
                validation=validation, validation_targets_all=target_cycles,
                validation_target_windows=validation_windows,
                validation_window_valid=validation_window_valid,
                masks=masks,
            )
            records.append(record)
            print(json.dumps({"completed": run_id, "train_mae": record["training_final_frame_mae_rad"],
                              "primary_validation": record["validation"]["consistency_validation"]["macro_sequence_mae_rad"]},
                             ensure_ascii=False), flush=True)

    manifest = {
        "schema": "g1.contact_followup_comparison.v1",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_archive_sha256": sha256(ORIGINAL_PATH),
        "filtered_archive_sha256": sha256(FILTERED_PATH),
        "manual_mapping_sha256": sha256(MANUAL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "test_split_loaded": False,
        "normalizer": "original full training split train_mean/train_std; reused unchanged in every condition",
        "sample_masks": mask_summary,
        "training": {"seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "learning_rate": LEARNING_RATE, "hidden_width": HIDDEN, "device": str(DEVICE)},
        "runs": records,
        "interpretation_limits": [
            "Touchdown labels and displayed kinematics share the same detector; consistency is not ground truth.",
            "Validation contains one held-out subject, so results do not establish cross-subject generalization.",
            "Overlapping windows and training seeds are not independent subjects or independent samples.",
            "Full validation is supplementary because it includes detector-inconsistent and edge samples.",
        ],
    }
    (OUT_DIR / "comparison_results.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "run_count": len(records), "test_split_loaded": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
