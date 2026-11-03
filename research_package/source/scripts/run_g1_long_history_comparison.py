"""Run the preregistered 10/20/40-frame by architecture comparison.

Only the detector-consistency train and validation arrays are loaded. The
previously viewed test split is not opened. See the dated extension protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import run_g1_contact_followup_comparison as base


DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "contact_consistency_filter_v1" / "phase_dataset_train_validation.npz"
PROTOCOL = ROOT / "results" / "g1_contact_followup_20261022" / "followup_extension_preregistration_20261022.md"
DEFAULT_OUT = ROOT / "results" / "g1_contact_followup_20261022" / "long_history_v1_replay_fixed"
WINDOWS = (10, 20, 40)
SEEDS = (20261021, 20261022, 20261023)
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_WINDOW = max(WINDOWS)


class LongContextTCNEncoder(nn.Module):
    """Fixed five-block causal TCN with receptive field 63 frames."""

    def __init__(self) -> None:
        super().__init__()
        dilations = (1, 2, 4, 8, 16)
        blocks: list[nn.Module] = []
        in_channels = base.JOINT_DIM
        for dilation in dilations:
            blocks.append(ContextCausalResidualBlock(in_channels, dilation))
            in_channels = base.HIDDEN
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(base.HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x.transpose(1, 2))[:, :, -1]
        return base.unit_phase(self.head(h))


class ContextFastCausalConv1d(nn.Module):
    """Conv1d-equivalent lag projections, including lags beyond input length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        bound = 1.0 / np.sqrt(in_channels * kernel_size)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_major = x.transpose(1, 2).contiguous()
        batch, steps, _channels = time_major.shape
        output = self.bias.view(1, 1, -1).expand(batch, steps, -1)
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag >= steps:
                # Left padding makes this tap zero at every available timestep.
                continue
            source = time_major if lag == 0 else time_major[:, : steps - lag]
            projected = torch.nn.functional.linear(source, self.weight[:, :, tap])
            if lag:
                projected = torch.nn.functional.pad(projected, (0, 0, lag, 0))
            output = output + projected
        return output.transpose(1, 2)


class ContextCausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = ContextFastCausalConv1d(in_channels, base.HIDDEN, kernel_size=3, dilation=dilation)
        self.skip = nn.Identity() if in_channels == base.HIDDEN else nn.Linear(in_channels, base.HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv(x)
        skip = x if isinstance(self.skip, nn.Identity) else self.skip(x.transpose(1, 2)).transpose(1, 2)
        return torch.relu(main + skip)


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "cnn": base.CNNEncoder,
    "lstm": base.LSTMEncoder,
    "tcn": LongContextTCNEncoder,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_split(archive: Any, split: str) -> dict[str, Any]:
    x = archive[f"x_{split}"].astype(np.float32)
    y = archive[f"y_{split}_cycles"].astype(np.float32)
    sequences = archive[f"sequence_{split}"].astype(str)
    end_frames = archive[f"frame_{split}"].astype(np.int64)
    cycle_ids = archive[f"cycle_id_{split}"].astype(str)
    if x.shape[1:] != (10, base.JOINT_DIM):
        raise ValueError(f"{split}: expected [N, 10, 29] windows, got {x.shape}")
    if not (len(x) == len(y) == len(sequences) == len(end_frames) == len(cycle_ids)):
        raise ValueError(f"{split}: filtered archive arrays are not row-aligned")

    frame_features: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    target_by_sequence: dict[str, dict[int, tuple[float, str]]] = defaultdict(dict)
    for i, (sequence, end, phase, cycle_id) in enumerate(zip(sequences, end_frames, y, cycle_ids)):
        name = str(sequence)
        end = int(end)
        if end in target_by_sequence[name]:
            raise ValueError(f"{split}/{name}: duplicate target frame {end}")
        target_by_sequence[name][end] = (float(phase), str(cycle_id))
        for offset, frame in enumerate(range(end - 9, end + 1)):
            value = x[i, offset]
            previous = frame_features[name].get(frame)
            if previous is not None and not np.allclose(previous, value, rtol=0.0, atol=1e-6):
                raise ValueError(f"{split}/{name}/{frame}: overlapping windows disagree")
            frame_features[name][frame] = value.copy()

    candidates: list[tuple[str, int, float, str]] = []
    for name in sorted(target_by_sequence):
        targets = target_by_sequence[name]
        for end in sorted(targets):
            phase, cycle_id = targets[end]
            if not cycle_id:
                continue
            history_frames = range(end - MAX_WINDOW + 1, end + 1)
            if all(
                frame in frame_features[name]
                and frame in targets
                and targets[frame][1] == cycle_id
                for frame in history_frames
            ):
                candidates.append((name, end, phase, cycle_id))
    if not candidates:
        raise ValueError(f"{split}: no target frames have a complete {MAX_WINDOW}-frame same-cycle history")

    windows: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        windows[window] = np.stack([
            np.stack([frame_features[name][frame] for frame in range(end - window + 1, end + 1)])
            for name, end, _phase, _cycle in candidates
        ]).astype(np.float32)
    return {
        "windows": windows,
        "y": np.asarray([row[2] for row in candidates], dtype=np.float32),
        "sequence": np.asarray([row[0] for row in candidates], dtype=str),
        "frame": np.asarray([row[1] for row in candidates], dtype=np.int64),
        "cycle_id": np.asarray([row[3] for row in candidates], dtype=str),
    }


def report_by_sequence(errors: np.ndarray, sequences: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sequence in sorted(set(sequences.tolist())):
        values = errors[sequences == sequence]
        result[str(sequence)] = {
            "n": int(len(values)),
            "mae_rad": float(values.mean()),
            "median_rad": float(np.median(values)),
            "p90_rad": float(np.quantile(values, 0.9)),
        }
    return result


def train_one(
    *,
    output_dir: Path,
    architecture: str,
    window: int,
    seed: int,
    train_data: dict[str, Any],
    validation_data: dict[str, Any],
) -> dict[str, Any]:
    base.seed_everything(seed)
    factory = ARCHITECTURES[architecture]
    model = factory().to(base.DEVICE)
    train_x = train_data["windows"][window]
    train_y_cycles = train_data["y"]
    validation_x = validation_data["windows"][window]
    validation_y_cycles = validation_data["y"]

    dataset = TensorDataset(torch.from_numpy(train_x), base.phase_targets(torch.from_numpy(train_y_cycles)))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=base.DEVICE.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_losses: list[float] = []
    for _epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(base.DEVICE, non_blocking=True)
            yb = yb.to(base.DEVICE, non_blocking=True)
            prediction = model(xb)
            loss = base.circular_loss(prediction, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            seen += len(xb)
        epoch_losses.append(loss_sum / seen)

    train_prediction = base.predict(model, train_x)
    validation_prediction = base.predict(model, validation_x)
    train_errors = base.circular_abs_error(train_prediction, train_y_cycles)
    validation_errors = base.circular_abs_error(validation_prediction, validation_y_cycles)
    train_by_sequence = report_by_sequence(train_errors, train_data["sequence"])
    validation_by_sequence = report_by_sequence(validation_errors, validation_data["sequence"])
    validation_macro = float(np.mean([row["mae_rad"] for row in validation_by_sequence.values()]))

    run_id = f"{architecture}_window{window}_seed{seed}"
    checkpoint_path = output_dir / f"{run_id}.pt"
    prediction_path = output_dir / f"{run_id}_predictions.npz"
    torch.save(model.state_dict(), checkpoint_path)
    np.savez_compressed(
        prediction_path,
        train_prediction=train_prediction.astype(np.float32),
        validation_prediction=validation_prediction.astype(np.float32),
        validation_error_rad=validation_errors.astype(np.float32),
        validation_sequence=validation_data["sequence"],
        validation_frame=validation_data["frame"],
    )
    return {
        "run_id": run_id,
        "architecture": architecture,
        "model_class": factory.__name__,
        "window_frames": window,
        "window_seconds_at_50_hz": window / 50.0,
        "seed": seed,
        "device": str(base.DEVICE),
        "train_n": int(len(train_x)),
        "validation_n": int(len(validation_x)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_width": base.HIDDEN,
        "epoch_training_loss": epoch_losses,
        "training_mae_rad": float(train_errors.mean()),
        "training_by_sequence": train_by_sequence,
        "validation_macro_sequence_mae_rad": validation_macro,
        "validation_by_sequence": validation_by_sequence,
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions_sha256": sha256(prediction_path),
    }


def write_summaries(output_dir: Path, records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["architecture"], record["window_frames"])].append(record)
    with (output_dir / "summary_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "architecture", "window_frames", "n_seeds",
            "training_mae_mean_rad", "training_mae_sd_rad",
            "consistency_val_macro_mean_rad", "consistency_val_macro_sd_rad",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (architecture, window), rows in sorted(groups.items()):
            writer.writerow({
                "architecture": architecture,
                "window_frames": window,
                "n_seeds": len(rows),
                "training_mae_mean_rad": float(np.mean([r["training_mae_rad"] for r in rows])),
                "training_mae_sd_rad": float(np.std([r["training_mae_rad"] for r in rows], ddof=1)),
                "consistency_val_macro_mean_rad": float(np.mean([r["validation_macro_sequence_mae_rad"] for r in rows])),
                "consistency_val_macro_sd_rad": float(np.std([r["validation_macro_sequence_mae_rad"] for r in rows], ddof=1)),
            })

    with (output_dir / "per_sequence_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["architecture", "window_frames", "seed", "split", "sequence_id", "n", "mae_rad", "median_rad", "p90_rad"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for split, key in (("train", "training_by_sequence"), ("validation", "validation_by_sequence")):
                for sequence, metrics in record[key].items():
                    writer.writerow({
                        "architecture": record["architecture"],
                        "window_frames": record["window_frames"],
                        "seed": record["seed"],
                        "split": split,
                        "sequence_id": sequence,
                        **metrics,
                    })

    with (output_dir / "paired_window_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["architecture", "seed", "comparison", "delta_rad"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        lookup = {(r["architecture"], r["seed"], r["window_frames"]): r["validation_macro_sequence_mae_rad"] for r in records}
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                baseline = lookup[(architecture, seed, 10)]
                for window in (20, 40):
                    writer.writerow({
                        "architecture": architecture,
                        "seed": seed,
                        "comparison": f"window{window}_minus_window10",
                        "delta_rad": lookup[(architecture, seed, window)] - baseline,
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="reconstruct common samples without training")
    args = parser.parse_args()
    if not DATA_PATH.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError("filtered data archive or frozen extension protocol is missing")
    if args.out_dir.exists() and not args.dry_run:
        raise FileExistsError(f"preserving existing output directory: {args.out_dir}")

    with np.load(DATA_PATH, allow_pickle=False) as archive:
        train_data = reconstruct_split(archive, "train")
        validation_data = reconstruct_split(archive, "validation")
        train_mean, train_std = archive["train_mean"].copy(), archive["train_std"].copy()
    for data in (train_data, validation_data):
        keys = list(zip(data["sequence"].tolist(), data["frame"].tolist()))
        if len(keys) != len(set(keys)):
            raise ValueError("common target-frame set contains duplicates")
        if any(data["windows"][length].shape[0] != len(keys) for length in WINDOWS):
            raise ValueError("window lengths do not share the same target-frame set")

    counts = {
        split: {
            "n": int(len(data["y"])),
            "by_sequence": {
                sequence: int(np.count_nonzero(data["sequence"] == sequence))
                for sequence in sorted(set(data["sequence"].tolist()))
            },
            "cycle_count": int(len(set(data["cycle_id"].tolist()))),
        }
        for split, data in (("train", train_data), ("validation", validation_data))
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "window_frames": WINDOWS, "common_samples": counts}, ensure_ascii=False, indent=2))
        return

    args.out_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    planned_runs = [
        f"{architecture}_window{window}_seed{seed}"
        for seed in SEEDS
        for window in WINDOWS
        for architecture in ARCHITECTURES
    ]
    status_path = args.out_dir / "run_status.json"
    status = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_run_ids": planned_runs,
        "completed_run_ids": [],
        "test_split_loaded": False,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for window in WINDOWS:
            for architecture in ARCHITECTURES:
                run_id = f"{architecture}_window{window}_seed{seed}"
                try:
                    record = train_one(
                        output_dir=args.out_dir,
                        architecture=architecture,
                        window=window,
                        seed=seed,
                        train_data=train_data,
                        validation_data=validation_data,
                    )
                except Exception as exc:
                    status["status"] = "FAILED"
                    status["failed_run_id"] = run_id
                    status["error"] = f"{type(exc).__name__}: {exc}"
                    status["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                    raise
                records.append(record)
                status["completed_run_ids"].append(run_id)
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "completed": record["run_id"],
                    "train_mae_rad": record["training_mae_rad"],
                    "validation_macro_mae_rad": record["validation_macro_sequence_mae_rad"],
                }, ensure_ascii=False), flush=True)

    status["status"] = "COMPLETED"
    status["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    write_summaries(args.out_dir, records)
    result = {
        "schema": "g1.long_history_comparison.v1",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "source_filtered_archive_sha256": sha256(DATA_PATH),
        "training_script_sha256": sha256(Path(__file__).resolve()),
        "base_architecture_script_sha256": sha256(Path(base.__file__).resolve()),
        "test_split_loaded": False,
        "normalization": {
            "fit_source": "original full training split only",
            "mean_sha256": hashlib.sha256(train_mean.tobytes()).hexdigest(),
            "std_sha256": hashlib.sha256(train_std.tobytes()).hexdigest(),
        },
        "common_sample_counts": counts,
        "factor_matrix": {
            "architectures": list(ARCHITECTURES),
            "window_frames": list(WINDOWS),
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "runs": records,
        "interpretation_limits": [
            "Only one held-out validation subject is represented.",
            "Overlapping windows and training seeds are not independent subjects.",
            "Phase targets remain labels from the shared kinematic touchdown detector.",
            "The five-block context TCN is the preregistered extension model and differs from the earlier two-block TCN.",
        ],
    }
    (args.out_dir / "comparison_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.out_dir), "run_count": len(records), "test_split_loaded": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
