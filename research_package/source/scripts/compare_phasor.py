"""Compare baseline (fixed CPG) vs PHASOR (learned phase) training results.

Usage:
    cd ~/unitree_rl_mjlab
    python scripts/compare_phasor.py \\
        --baseline_dir logs/rsl_rl/g1_velocity/<baseline_timestamp> \\
        --phasor_dir logs/rsl_rl/g1_velocity/<phasor_timestamp>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tb_scalars(log_dir: Path) -> dict[str, np.ndarray]:
    """Extract all scalar time series from a TensorBoard events file."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = {"step": steps, "value": values}
    return data


def plot_training_curves(baseline: dict, phasor: dict, save_dir: Path):
    """Side-by-side training curve comparison."""
    keys_of_interest = [
        ("Episode_Reward/track_linear_velocity", "Velocity Tracking Reward"),
        ("Episode_Reward/track_angular_velocity", "Angular Velocity Tracking Reward"),
        ("Episode_Reward/foot_gait", "Gait Reward"),
        ("Episode_Reward/stand_still", "Stand Still Penalty"),
        ("Episode_Termination/fell_over", "Fall Rate"),
        ("Metrics/twist/error_vel_xy", "Velocity Error XY"),
        ("Metrics/twist/error_vel_yaw", "Velocity Error Yaw"),
        ("Episode_Metrics/mean_action_acc", "Action Acceleration"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(keys_of_interest):
        ax = axes[idx]

        if key in baseline:
            b = baseline[key]
            # Smooth with rolling average for cleaner plots.
            window = max(1, len(b["value"]) // 200)
            if window > 1:
                smoothed = np.convolve(b["value"], np.ones(window)/window, mode="valid")
                ax.plot(b["step"][window-1:], smoothed, "b-", alpha=0.8, label="Baseline (fixed CPG)", linewidth=1)
            else:
                ax.plot(b["step"], b["value"], "b-", alpha=0.8, label="Baseline (fixed CPG)", linewidth=1)

        if key in phasor:
            p = phasor[key]
            window = max(1, len(p["value"]) // 200)
            if window > 1:
                smoothed = np.convolve(p["value"], np.ones(window)/window, mode="valid")
                ax.plot(p["step"][window-1:], smoothed, "r-", alpha=0.8, label="PHASOR (learned)", linewidth=1)
            else:
                ax.plot(p["step"], p["value"], "r-", alpha=0.8, label="PHASOR (learned)", linewidth=1)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "training_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved: {save_path}")


def compute_summary(baseline: dict, phasor: dict) -> str:
    """Generate a text summary comparing final performance."""
    lines = []
    lines.append("=" * 70)
    lines.append("PHASOR vs Baseline Comparison")
    lines.append("=" * 70)

    metrics = [
        ("Episode_Reward/track_linear_velocity", "Vel Track Reward", "higher"),
        ("Episode_Reward/track_angular_velocity", "Ang Track Reward", "higher"),
        ("Episode_Reward/foot_gait", "Gait Reward", "higher"),
        ("Episode_Termination/fell_over", "Fall Rate", "lower"),
        ("Metrics/twist/error_vel_xy", "Vel Error XY (m/s)", "lower"),
        ("Metrics/twist/error_vel_yaw", "Vel Error Yaw (rad/s)", "lower"),
    ]

    lines.append(f"{'Metric':<35s} {'Baseline':>12s} {'PHASOR':>12s} {'Δ':>10s}")
    lines.append("-" * 70)

    for key, name, direction in metrics:
        b_val = "N/A"
        p_val = "N/A"
        delta = "N/A"

        if key in baseline:
            # Average over last 10% of training.
            b_arr = baseline[key]["value"]
            n_last = max(1, len(b_arr) // 10)
            b_val = f"{b_arr[-n_last:].mean():.4f}"

        if key in phasor:
            p_arr = phasor[key]["value"]
            n_last = max(1, len(p_arr) // 10)
            p_val = f"{p_arr[-n_last:].mean():.4f}"

        if key in baseline and key in phasor:
            b_mean = b_arr[-n_last:].mean()
            p_mean = p_arr[-n_last:].mean()
            d = p_mean - b_mean
            delta = f"{d:+.4f}"

        lines.append(f"{name:<35s} {b_val:>12s} {p_val:>12s} {delta:>10s}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare PHASOR vs Baseline training")
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--phasor_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/phasor_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline from: {args.baseline_dir}")
    baseline_data = load_tb_scalars(args.baseline_dir)
    print(f"  → {len(baseline_data)} metrics found")

    print(f"Loading PHASOR from: {args.phasor_dir}")
    phasor_data = load_tb_scalars(args.phasor_dir)
    print(f"  → {len(phasor_data)} metrics found")

    # Summary table.
    summary = compute_summary(baseline_data, phasor_data)
    print(summary)

    with open(args.output_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Training curves.
    plot_training_curves(baseline_data, phasor_data, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
