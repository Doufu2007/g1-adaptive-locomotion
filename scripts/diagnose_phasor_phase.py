"""Audit the circular coverage of a PHASOR encoder on its motion sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.velocity.phasor.diagnostics import circular_coverage_rad, unwrap_angles
from src.tasks.velocity.phasor.encoder import PhaseEncoder
from src.tasks.velocity.phasor.train_encoder import resample_joint_positions


DEFAULT_MOTION = PROJECT_ROOT / "src/assets/motions/g1/dance1_subject2.csv"
DEFAULT_ENCODER = PROJECT_ROOT / "src/tasks/velocity/phasor/phasor_encoder_v2.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_windows(joint_pos: torch.Tensor, window: int) -> torch.Tensor:
    if len(joint_pos) < window:
        raise ValueError(f"sequence has fewer than {window} frames")
    return torch.stack(
        [joint_pos[index : index + window] for index in range(len(joint_pos) - window + 1)]
    )


def summarize(latent: np.ndarray, bins: int = 32) -> dict[str, float | int]:
    angle = np.arctan2(latent[:, 1], latent[:, 0])
    unwrapped = unwrap_angles(angle)
    circular_steps = np.angle(np.exp(1j * np.diff(angle)))
    occupied = np.unique(
        np.floor(np.mod(angle, 2 * np.pi) * bins / (2 * np.pi)).astype(int)
    )
    return {
        "n_windows": int(len(latent)),
        "radius_mean": float(np.linalg.norm(latent, axis=1).mean()),
        "radius_std": float(np.linalg.norm(latent, axis=1).std()),
        "circular_coverage_rad": circular_coverage_rad(angle),
        "occupied_bins": int(len(occupied)),
        "total_bins": bins,
        "unwrapped_span_rad": float(unwrapped.max() - unwrapped.min()),
        "net_winding_cycles": float((unwrapped[-1] - unwrapped[0]) / (2 * np.pi)),
        "median_abs_step_rad": float(np.median(np.abs(circular_steps))),
        "p95_abs_step_rad": float(np.quantile(np.abs(circular_steps), 0.95)),
        "max_abs_step_rad": float(np.max(np.abs(circular_steps))),
        "net_to_total_angular_travel": float(
            abs(circular_steps.sum()) / (np.abs(circular_steps).sum() + 1e-12)
        ),
    }


def encode_sequence(
    encoder: PhaseEncoder,
    joint_pos: torch.Tensor,
    window: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    windows = build_windows(joint_pos, window).to(device)
    normalized = (windows - mean) / (std + 1e-8)
    outputs = []
    with torch.no_grad():
        for batch in normalized.split(512):
            outputs.append(encoder(batch).cpu())
    return torch.cat(outputs).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/phase_diagnostics"
    )
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--policy-fps", type=float, default=50.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion = np.loadtxt(args.motion, delimiter=",", dtype=np.float32)
    joint_pos = torch.from_numpy(motion[:, 7:36].copy())
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    window = int(checkpoint["window"])
    joint_dim = int(checkpoint["joint_dim"])
    if joint_pos.shape[1] != joint_dim:
        raise ValueError("motion joint dimension does not match encoder checkpoint")

    encoder = PhaseEncoder(joint_dim=joint_dim, window=window).to(device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    mean = checkpoint["mean"].to(device).squeeze(0)
    std = checkpoint["std"].to(device).squeeze(0)

    resampled = resample_joint_positions(
        joint_pos, source_fps=args.source_fps, target_fps=args.policy_fps
    )
    sequences = {
        "source_rate": joint_pos,
        "policy_rate": resampled,
    }
    latent_by_rate = {
        name: encode_sequence(encoder, seq, window, mean, std, device)
        for name, seq in sequences.items()
    }
    report = {
        "checkpoint": str(args.checkpoint.name),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "motion": str(args.motion.name),
        "motion_sha256": sha256_file(args.motion),
        "checkpoint_training_fps": checkpoint.get("policy_fps", "unknown"),
        "evaluated_rates_fps": {
            "source_rate": args.source_fps,
            "policy_rate": args.policy_fps,
        },
        "device": str(device),
        "sequences": {
            name: summarize(latent) for name, latent in latent_by_rate.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.checkpoint.stem}_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    for name, latent in latent_by_rate.items():
        angles = np.arctan2(latent[:, 1], latent[:, 0])
        axes[0].scatter(
            latent[:, 0], latent[:, 1],
            c=np.linspace(0, 1, len(latent)), s=2, alpha=0.45,
            cmap="viridis", label=name.replace("_", " "),
        )
        axes[1].plot(
            unwrap_angles(angles), linewidth=0.7, label=name.replace("_", " ")
        )
        axes[2].hist(
            np.mod(angles, 2 * np.pi), bins=32, range=(0, 2 * np.pi),
            histtype="step", linewidth=1.2, label=name.replace("_", " "),
        )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="$z_1$", ylabel="$z_2$")
    axes[0].set_title("Latent trajectory")
    axes[1].set(xlabel="Window index", ylabel="Unwrapped angle (rad)")
    axes[1].set_title("Angle by sequence")
    axes[2].set(xlabel="Angle (rad)", ylabel="Window count")
    axes[2].set_title("Circular occupancy")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    figure_path = args.output_dir / f"{args.checkpoint.stem}_diagnostics.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"figure={figure_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
