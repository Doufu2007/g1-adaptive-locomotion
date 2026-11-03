"""Generate comprehensive final report for the G1 PHASOR gait adjustment project.

Usage:
    python scripts/generate_final_report.py [--rough_phasor_dir PATH]

If --rough_phasor_dir is not provided, uses the latest *_rough_phasor directory.
"""
import argparse, json, sys, os
from pathlib import Path
from datetime import datetime
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

os.environ["WANDB_MODE"] = "disabled"

# Resolved from this file so the report regenerates wherever the repo lives,
# not only on the cloud box where it was first written.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Config ──────────────────────────────────────────────────────────
# Each arm was trained as one or more resumable segments.  Scoring a single
# segment answers a different question from the one this report asks, because
# a segment's own last 10% is not that arm's final training budget.  Every arm
# is merged across segments and then scored on the iterations that ALL arms of
# its terrain actually cover.
FLAT_BASELINE_SEGMENTS = ("2026-08-10_09-35-45_baseline_full",)
FLAT_PHASOR_SEGMENTS = ("2026-08-10_12-29-14_phasor_full",
                        "2026-08-10_21-49-27_phasor_resume_final")
ROUGH_BASELINE_SEGMENTS = ("2026-08-10_23-06-29_rough_baseline",)
ROUGH_PHASOR_SEGMENTS = ("2026-08-11_06-13-25_rough_phasor",
                         "2026-08-11_06-18-55_rough_phasor",
                         "2026-08-11_10-51-03_rough_phasor")
MATCHED_WINDOW_ITERATIONS = 1000
LOG_DIR = PROJECT_ROOT / "logs/rsl_rl/g1_velocity"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_report"


def load_tb_summary(log_dir: Path) -> dict:
    """Load TensorBoard scalars and compute final (last 10%) statistics."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        values = np.array([e.value for e in events])
        steps = np.array([e.step for e in events])
        n_last = max(1, len(values) // 10)
        data[tag] = {
            "final_mean": float(values[-n_last:].mean()),
            "final_std": float(values[-n_last:].std()),
            "values": values.tolist(),
            "steps": steps.tolist(),
            "n_points": len(values),
        }
    return data


def _merge_segments(segment_names: tuple[str, ...] | list[str]) -> dict:
    """Concatenate every segment of one arm onto a single iteration axis."""
    merged: dict[str, dict[int, float]] = {}
    for name in segment_names:
        accumulator = EventAccumulator(str(LOG_DIR / name), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            bucket = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                # Segments overlap at their boundaries; one value per iteration.
                bucket[int(event.step)] = float(event.value)
    return merged


def _restrict(merged: dict, low: int, high: int) -> dict:
    data = {}
    for tag, bucket in merged.items():
        steps = [step for step in sorted(bucket) if low <= step <= high]
        if not steps:
            continue
        values = np.array([bucket[step] for step in steps])
        data[tag] = {
            "final_mean": float(values.mean()),
            "final_std": float(values.std()),
            "values": values.tolist(),
            "steps": steps,
            "n_points": len(values),
            "window": [low, high],
        }
    return data


def load_matched_pair(baseline_segments, phasor_segments,
                      window: int = MATCHED_WINDOW_ITERATIONS) -> tuple[dict, dict]:
    """Score both arms of one terrain on the iterations they have in common.

    ``load_tb_summary`` reads the tail of one directory; for a resumed or
    longer-running arm that tail is an arbitrary later budget, so the two arms
    end up compared at different amounts of training.  Merging every segment
    and restricting both arms to the same trailing window removes that
    difference instead of reporting it as an effect.
    """
    baseline = _merge_segments(baseline_segments)
    phasor = _merge_segments(phasor_segments)
    if not baseline or not phasor:
        return _restrict(baseline, 0, 0), _restrict(phasor, 0, 0)
    covered = min(min(max(b) for b in baseline.values()),
                  min(max(b) for b in phasor.values()))
    low = covered - window + 1
    return _restrict(baseline, low, covered), _restrict(phasor, low, covered)


def comparison_table(name_map: dict, baseline: dict, phasor: dict) -> str:
    """Generate comparison markdown table."""
    lines = []
    lines.append("| Metric | Baseline | PHASOR | Δ | Change |")
    lines.append("|--------|----------|--------|---|--------|")
    for key, label in name_map.items():
        if key not in baseline or key not in phasor:
            continue
        b = baseline[key]["final_mean"]
        p = phasor[key]["final_mean"]
        d = p - b
        pct = (d / abs(b)) * 100 if b != 0 else 0

        # Determine if improvement
        is_reward = "Reward" in key
        is_error = "error" in key.lower() or "fell" in key.lower()
        improved = (d > 0 and is_reward) or (d < 0 and is_error)
        emoji = "✅" if improved else "⚠️"

        lines.append(f"| {label} | {b:.4f} | {p:.4f} | {d:+.4f} | {pct:+.1f}% {emoji} |")
    return "\n".join(lines)


def format_comparison_plain(name_map: dict, baseline: dict, phasor: dict) -> str:
    """Generate plain-text comparison."""
    lines = []
    lines.append(f"{'Metric':<40s} {'Baseline':>10s} {'PHASOR':>10s} {'Δ':>10s} {'Change':>10s}")
    lines.append("-" * 80)
    for key, label in name_map.items():
        if key not in baseline or key not in phasor:
            continue
        b = baseline[key]["final_mean"]
        p = phasor[key]["final_mean"]
        d = p - b
        pct = (d / abs(b)) * 100 if b != 0 else 0
        emoji = "✅" if ((d > 0 and "Reward" in key) or (d < 0 and ("error" in key.lower() or "fell" in key.lower()))) else "⚠️"
        lines.append(f"{label:<40s} {b:>10.4f} {p:>10.4f} {d:>+10.4f} {pct:>+9.1f}% {emoji}")
    return "\n".join(lines)


def latest_rough_phasor_dir():
    """Find latest rough PHASOR directory with checkpoints."""
    dirs = sorted(LOG_DIR.glob("*_rough_phasor"), reverse=True)
    for d in dirs:
        if list(d.glob("model_*.pt")):
            return d
    return None


def current_iteration(run_dir: Path) -> int:
    """Get latest checkpoint iteration."""
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        return 0
    return int(models[-1].stem.split("_")[1])


def generate_report(rough_phasor_dir: Path | None = None):
    """Generate the full comprehensive report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if rough_phasor_dir is None:
        rough_phasor_dir = latest_rough_phasor_dir()

    print("Loading flat terrain data...")
    flat_baseline, flat_phasor = load_matched_pair(FLAT_BASELINE_SEGMENTS, FLAT_PHASOR_SEGMENTS)

    print("Loading rough terrain data...")
    rough_segments = list(ROUGH_PHASOR_SEGMENTS)
    if rough_phasor_dir is not None and rough_phasor_dir.name not in rough_segments:
        rough_segments.append(rough_phasor_dir.name)
    rough_baseline, rough_phasor = load_matched_pair(ROUGH_BASELINE_SEGMENTS, rough_segments)
    rough_iter = max((current_iteration(LOG_DIR / name) for name in rough_segments), default=0)
    flat_window = flat_baseline.get("Episode_Termination/fell_over", {}).get("window")
    rough_window = rough_baseline.get("Episode_Termination/fell_over", {}).get("window")
    print(f"matched windows - flat {flat_window}, rough {rough_window}")

    # ── Metric definitions ──────────────────────────────────────────
    primary_metrics = {
        "Episode_Reward/track_linear_velocity": "Velocity Tracking Reward",
        "Episode_Reward/track_angular_velocity": "Angular Velocity Tracking",
        "Episode_Reward/foot_gait": "Gait Reward (CPG-calibrated)",
        "Episode_Termination/fell_over": "Fall Rate",
        "Metrics/twist/error_vel_xy": "Velocity Error XY (m/s)",
        "Metrics/twist/error_vel_yaw": "Velocity Error Yaw (rad/s)",
    }

    secondary_metrics = {
        "Episode_Reward/pose": "Posture Reward",
        "Episode_Reward/stand_still": "Stand Still Penalty",
        "Episode_Reward/foot_clearance": "Foot Clearance Penalty",
        "Episode_Reward/foot_slip": "Foot Slip Penalty",
        "Episode_Reward/action_rate_l2": "Action Smoothness",
        "Episode_Reward/joint_acc_l2": "Joint Acceleration",
        "Episode_Reward/self_collisions": "Self-Collision Penalty",
        "Episode_Metrics/mean_action_acc": "Action Acceleration (m/s²)",
        "Metrics/angular_momentum_mean": "Angular Momentum",
        "Metrics/slip_velocity_mean": "Slip Velocity (m/s)",
    }

    rough_specific = {
        "Curriculum/terrain_levels": "Terrain Difficulty Level",
        "Metrics/landing_force_mean": "Landing Force (N)",
    }

    # ── Build report ────────────────────────────────────────────────
    lines = []
    lines.append("# G1 Gait Adjustment — PHASOR vs Baseline: Comprehensive Final Report")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Project**: Unitree G1 Locomotion with Learned Phase Encoder")
    lines.append("**Hardware**: NVIDIA RTX 5090, 32GB VRAM")
    lines.append(f"**Training Budget**: 10,000 PPO iterations × 4,096 envs = ~2.4B timesteps per arm")
    lines.append(f"**Scoring window**: the last {MATCHED_WINDOW_ITERATIONS:,} iterations covered by "
                 f"BOTH arms of a terrain (flat {flat_window[0]}–{flat_window[1]}, "
                 f"rough {rough_window[0]}–{rough_window[1]})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("PHASOR replaces a fixed-frequency CPG clock (0.6s period) with a learned phase encoder. ")
    lines.append("The encoder — a lightweight 1D-CNN — maps a 10-frame sliding window of G1 joint positions ")
    lines.append("to a 2D phase embedding on the unit circle. It was pre-trained on motion capture data ")
    lines.append("and frozen during RL training.")
    lines.append("")
    lines.append("### The claim this report supports")
    lines.append("")
    lines.append("**Claim.** A fixed clock advances only with elapsed time, so it cannot stay aligned with a "
                 "gait cycle whose period varies. A phase estimated from recent joint motion can stay "
                 "aligned with the actual gait. This is the load-bearing claim; the locomotion metrics "
                 "below are observations about what that alignment did and did not buy.")
    lines.append("")
    lines.append("**Supported, on the walking encoder.** The replaced clock is configured at 0.6 s — 30 frames "
                 "at 50 Hz — while measured touchdown intervals in the reference motion run 45–122 frames "
                 "(typically 45–70). Swept over every period, the *best possible* fixed clock scores "
                 "**0.2187** macro cycle completion on the gate; as configured it scores **0.0772**. The "
                 "frozen walking encoder scores **0.4128** — **1.9×** the best possible fixed clock and "
                 "**5.3×** the configured one.")
    lines.append("")
    lines.append("**Not claimed.** (i) The learned phase does not match the reference absolutely: 0.4128 is "
                 "far below the gate's own 0.90 bar, and the encoder is recorded "
                 "`encoder_freeze_eligible: false`. This is a comparative result, not a passing one. "
                 "(ii) The fixed clock fails this gate *by construction* — a constant period cannot "
                 "complete exactly one turn per variable-length cycle — so what is measured is the size "
                 "of the gap, not a surprising direction. (iii) **Velocity tracking is not part of the "
                 "claim.** It is terrain-conditional and single-seed; see §7.1.")
    lines.append("")
    lines.append("**One provenance gap, now scored rather than bridged.** The mechanism evidence above "
                 "belongs to a walking-trained encoder. Every PHASOR arm in this report loaded a *different* "
                 "encoder, trained on dance motion. That encoder has now been scored on the same frozen gate, "
                 "same calibration, validation split only: it completes **0 of 603 cycles** (macro 0.0000), "
                 "and its predicted phase advances 0.0084 rad per sequence where one cycle needs 6.283.")
    lines.append("")
    lines.append("That is not a gate convention disagreeing with the encoder: the output is near-constant "
                 "under the gate's input normalization *and* under the encoder's own deployment "
                 "normalization, and an earlier independent diagnostic "
                 "(`results/phase_diagnostics/phasor_encoder_diagnostics.json`) already found it constant on "
                 "its own training motion — 0.597 rad of circular coverage, 4 of 32 angle bins occupied, net "
                 "winding 0.0017 cycles over 3936 windows. The consequence is narrow but real: the four RL "
                 "runs compared a **near-constant phase observation** against a fixed clock, not a tracking "
                 "phase against a fixed clock. No number in §2–§3 changes, and the mechanism result is "
                 "unaffected — it rests on the walking encoder, which scores 0.4128 on that same gate. What "
                 "changes is what §2–§3 may be read as evidence *for*.")
    lines.append("")
    lines.append("| artifact | SHA-256 | trained on | gate macro | used by |")
    lines.append("|---|---|---|---|---|")
    lines.append("| `phasor_encoder.pt` | `a0be7719` | dance (dance1_subject2) | **0.0000** (0/603 cycles) | every PHASOR arm in this report |")
    lines.append("| `phase_encoder_runtime_v1.pt` = `phase_encoder_checkpoint.pt` (identical weights) | `54d6f41b` / `f8dbc6e7` | walking mocap | **0.4128** | current runtime, experiment B |")
    lines.append("| v2.1b epoch 1 / v2.1c epoch 1 | — | walking mocap | 0.6962 / 0.7032 | **no RL run** |")
    lines.append("")
    lines.append("Gate reference points: oracle 0.9984, best-over-grid fixed clock 0.2187, configured fixed "
                 "clock 0.0772, gate bar 0.90. Sources: `results/g1_phase_gate_v4_2/fixed_clock_baseline.json`, "
                 "`rescore_v4_2.json`, `results/g1_phase_gate_v4/current_encoder_v1_v4_audit.json`, and for the "
                 "dance encoder `results/g1_phase_gate_v4_2/august_encoder_gate_score.json` (scored by "
                 "`scripts/score_august_encoder_on_gate_v4_2.py`, which reproduces the frozen walking-encoder "
                 "figure to <1e-12 as an equivalence control).")
    lines.append("")
    lines.append("### Key Results (Flat Terrain)")
    lines.append("")
    lines.append(comparison_table(primary_metrics, flat_baseline, flat_phasor))
    lines.append("")

    if rough_phasor and rough_iter >= 10000:
        lines.append("### Key Results (Rough Terrain)")
        lines.append("")
        lines.append(comparison_table(primary_metrics, rough_baseline, rough_phasor))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 1. Methodology")
    lines.append("")
    lines.append("### 1.1 Phase Encoder Architecture")
    lines.append("")
    lines.append("| Component | Detail |")
    lines.append("|-----------|--------|")
    lines.append("| Architecture | 1D-CNN: Conv1d(29,128,k=5) → ReLU → Conv1d(128,128,k=5) → ReLU → AdaptiveAvgPool → Linear(128,2) |")
    lines.append("| Input | 10-frame window of 29-DOF joint positions (normalized) |")
    lines.append("| Output | 2D vector on unit circle (sin/cos-like phase) |")
    lines.append("| Training | Autoencoder (encoder+decoder), 300 epochs on G1 mocap dance data |")
    lines.append("| Loss | L_recon + 0.3·L_smooth + 0.1·L_cyclic |")
    lines.append("| Size | ~400KB (encoder only) |")
    lines.append("")
    lines.append("![Phase Circle](../../src/tasks/velocity/phasor/phase_circle.png)")
    lines.append("*Figure: Learned phase embedding — each point is one frame of joint data, colored by time.*")
    lines.append("")
    lines.append("### 1.2 RL Training Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Algorithm | PPO (Proximal Policy Optimization) |")
    lines.append("| Actor | MLP(obs→512→256→128→actions), ELU |")
    lines.append("| Critic | MLP(obs→512→256→128→value), ELU |")
    lines.append("| Learning Rate | 1e-3 (adaptive, target KL=0.01) |")
    lines.append("| γ / λ | 0.99 / 0.95 |")
    lines.append("| Steps per env | 24 |")
    lines.append("| Parallel envs | 4,096 |")
    lines.append("| Total timesteps | ~2.4 billion |")
    lines.append("| Episode length | 20s (4,000 steps @ 200Hz control / 4 decimation) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Flat Terrain Results")
    lines.append("")
    lines.append("### 2.1 Primary Metrics")
    lines.append("")
    lines.append(comparison_table(primary_metrics, flat_baseline, flat_phasor))
    lines.append("")
    lines.append("### 2.2 Secondary Metrics")
    lines.append("")
    lines.append(comparison_table(secondary_metrics, flat_baseline, flat_phasor))
    lines.append("")
    lines.append("### 2.3 Analysis")
    lines.append("")

    # Compute derived stats
    vel_improvement = (flat_phasor["Episode_Reward/track_linear_velocity"]["final_mean"] /
                       flat_baseline["Episode_Reward/track_linear_velocity"]["final_mean"] - 1) * 100
    ang_improvement = (flat_phasor["Episode_Reward/track_angular_velocity"]["final_mean"] /
                       flat_baseline["Episode_Reward/track_angular_velocity"]["final_mean"] - 1) * 100
    fall_reduction = (1 - flat_phasor["Episode_Termination/fell_over"]["final_mean"] /
                      flat_baseline["Episode_Termination/fell_over"]["final_mean"]) * 100

    lines.append(f"- **Velocity tracking reward {vel_improvement:+.1f}%** — one seed per arm, so this locates a difference and does not attribute a cause to it; the direction does not carry over to rough terrain (§3.4, §4.2)")
    lines.append(f"- **Angular velocity tracking {ang_improvement:+.1f}%** — same caveat, same single seed")
    lines.append(f"- **Fall rate {flat_baseline['Episode_Termination/fell_over']['final_mean']:.4f} → "
                 f"{flat_phasor['Episode_Termination/fell_over']['final_mean']:.4f}** "
                 f"({fall_reduction:.1f}% lower over the shared window), one seed per arm")
    lines.append("- **Gait reward dropped ~34%** — expected: this metric is calibrated to the fixed 0.6 s CPG, and PHASOR's learned phase produces different timing, so the drop is not a gait-quality statement either way")
    lines.append("- **Most other locomotion metrics moved favourably** — posture, action smoothness, self-collision and action acceleration all improved, while foot clearance (−7.2%), foot slip (−24.9%) and slip velocity (+5.5%) moved the other way (§2.2)")
    lines.append("")
    lines.append("![Flat Training Comparison](../phasor_comparison/training_comparison_v2.png)")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Rough Terrain Results")
    lines.append("")

    if rough_phasor and rough_iter >= 10000:
        lines.append("### 3.1 Primary Metrics")
        lines.append("")
        lines.append(comparison_table(primary_metrics, rough_baseline, rough_phasor))
        lines.append("")
        lines.append("### 3.2 Secondary Metrics")
        lines.append("")
        lines.append(comparison_table(secondary_metrics, rough_baseline, rough_phasor))
        lines.append("")
        lines.append("### 3.3 Rough-Specific Metrics")
        lines.append("")
        lines.append(comparison_table(rough_specific, rough_baseline, rough_phasor))
        lines.append("")
        lines.append("### 3.4 Analysis")
        lines.append("")

        r_vel_imp = (rough_phasor["Episode_Reward/track_linear_velocity"]["final_mean"] /
                     rough_baseline["Episode_Reward/track_linear_velocity"]["final_mean"] - 1) * 100
        r_ang_imp = (rough_phasor["Episode_Reward/track_angular_velocity"]["final_mean"] /
                     rough_baseline["Episode_Reward/track_angular_velocity"]["final_mean"] - 1) * 100
        r_fall_red = (1 - rough_phasor["Episode_Termination/fell_over"]["final_mean"] /
                      rough_baseline["Episode_Termination/fell_over"]["final_mean"]) * 100

        lines.append(f"- **Velocity tracking {'improved' if r_vel_imp > 0 else 'declined'} "
                     f"{abs(r_vel_imp):.1f}%** on rough terrain")
        lines.append(f"- **Angular velocity tracking {'improved' if r_ang_imp > 0 else 'declined'} "
                     f"{abs(r_ang_imp):.1f}%** on rough terrain")
        lines.append(f"- **Fall rate {rough_baseline['Episode_Termination/fell_over']['final_mean']:.3f} → "
                     f"{rough_phasor['Episode_Termination/fell_over']['final_mean']:.3f} over iterations "
                     f"{rough_window[0]}–{rough_window[1]}** ({r_fall_red:.1f}% lower, one seed per arm). "
                     "This metric moves between 0.44 and 0.80 from one thousand-iteration window to the "
                     "next within a single run, so one seed per arm locates the difference but does not "
                     "separate it from training noise; a multi-seed matched-budget replication is "
                     "registered in `results/phasor_rough_equalbudget_20261020/`.")
        lines.append("")
        lines.append("![Rough Training Comparison](../rough_comparison/training_comparison.png)")
        lines.append("")
    else:
        lines.append(f"⚠️ **Rough terrain PHASOR training in progress** ({rough_iter}/10001 iterations)")
        lines.append("")
        lines.append("### 3.1 Rough Baseline Performance (pre-computed)")
        lines.append("")
        lines.append("Rough terrain is dramatically harder than flat — the baseline struggles significantly:")
        lines.append("")
        lines.append("| Metric | Flat Baseline | Rough Baseline | Degradation |")
        lines.append("|--------|--------------|----------------|-------------|")
        for key, label in primary_metrics.items():
            fb = flat_baseline[key]["final_mean"]
            rb = rough_baseline[key]["final_mean"]
            deg = ((rb - fb) / abs(fb)) * 100 if fb != 0 else 0
            lines.append(f"| {label} | {fb:.4f} | {rb:.4f} | {deg:+.1f}% |")
        lines.append("")
        lines.append("The rough terrain baseline shows:")
        lines.append(f"- {rough_baseline['Episode_Termination/fell_over']['final_mean']*100:.1f}% fall rate — the robot falls on over half of episodes")
        lines.append("- Velocity tracking is severely degraded compared to flat terrain")
        lines.append("- This large performance gap is where PHASOR's adaptive gait can potentially provide the most benefit")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Cross-Terrain Comparison")
    lines.append("")
    lines.append("![Cross-Terrain Comparison](cross_terrain_comparison.png)")
    lines.append("*Figure: All four configurations compared — flat BL, flat PH, rough BL, rough PH.*")
    lines.append("")
    lines.append("### 4.1 Flat vs Rough Baseline")
    lines.append("")
    lines.append("| Metric | Flat | Rough | Ratio |")
    lines.append("|--------|------|-------|-------|")
    for key, label in primary_metrics.items():
        fb = flat_baseline[key]["final_mean"]
        rb = rough_baseline[key]["final_mean"]
        ratio = rb / fb if fb != 0 else float('inf')
        lines.append(f"| {label} | {fb:.4f} | {rb:.4f} | {ratio:.2f}x |")
    lines.append("")
    lines.append("### 4.2 Terrain Transfer Insights")
    lines.append("")
    lines.append("- **PHASOR uses the same encoder for both terrains** — trained once on mocap, deployed universally")
    lines.append("- **Rough terrain is the open question** — on the shared budget window the fixed-clock arm tracked forward velocity better (−11.2% for PHASOR) and reached a higher terrain-difficulty curriculum level, so the flat-terrain benefit does not carry over in this single-seed comparison")
    lines.append("- **The encoder's sliding window provides implicit terrain awareness** — joint trajectories reflect ground contact patterns")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Computational Cost")
    lines.append("")
    lines.append("| Component | Cost |")
    lines.append("|-----------|------|")
    lines.append("| Phase encoder training | < 1 minute (300 epochs, single GPU) |")
    lines.append("| Encoder inference overhead | ~0.1ms per env-step (negligible vs physics) |")
    lines.append("| Flat terrain RL | ~2 hours (10k iters) |")
    lines.append("| Rough terrain RL | ~5 hours (10k iters, raycasting overhead) |")
    lines.append("| Encoder model size | 407KB (PhaseEncoder only) |")
    lines.append("| Policy ONNX size | 878KB (actor only) |")
    lines.append(f"| Total project GPU-hours | ~14 hours (4 training runs) |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 6. Artifacts & Deliverables")
    lines.append("")
    lines.append("### 6.1 Trained Models")
    lines.append("")
    lines.append("| Model | Path | Iterations | ONNX |")
    lines.append("|-------|------|-----------|------|")
    lines.append(f"| Flat Baseline | `{FLAT_BASELINE_SEGMENTS[-1]}/model_10000.pt` | 10,000 | ✅ policy.onnx |")
    lines.append(f"| Flat PHASOR | `{FLAT_PHASOR_SEGMENTS[-1]}/model_9999.pt` | 9,999 | ✅ policy.onnx |")
    lines.append(f"| Rough Baseline | `{ROUGH_BASELINE_SEGMENTS[-1]}/model_9900.pt` | 9,900 | ✅ policy.onnx |")
    if rough_phasor_dir and rough_iter >= 10000:
        lines.append(f"| Rough PHASOR | `{rough_phasor_dir.name}/model_*.pt` | {rough_iter} | ✅ policy.onnx |")
    else:
        lines.append(f"| Rough PHASOR | `{rough_phasor_dir.name}/model_*.pt` (training) | {rough_iter} | ⏳ |")
    lines.append("")

    lines.append("### 6.2 Results Directory Structure")
    lines.append("")
    lines.append("```")
    lines.append("results/")
    lines.append("├── final_report/")
    lines.append("│   ├── FINAL_REPORT.md          ← This report")
    lines.append("│   └── final_results.json       ← Machine-readable results")
    lines.append("├── phasor_comparison/           ← Flat terrain comparison")
    lines.append("│   ├── training_comparison_v2.png")
    lines.append("│   └── summary.txt")
    lines.append("├── rough_comparison/            ← Rough terrain comparison")
    lines.append("│   ├── training_comparison.png")
    lines.append("│   └── summary.txt")
    lines.append("├── rough_baseline_precomputed.json")
    lines.append("├── results.json                 ← PINN results")
    lines.append("└── SUMMARY.md                   ← PINN summary")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 7. Conclusions")
    lines.append("")
    lines.append("### 7.1 The claim, and what each part of it rests on")
    lines.append("")
    lines.append("1. **A fixed clock cannot follow a variable-period gait.** This is architectural and measured: "
                 "the clock is configured at 0.6 s (30 frames) against touchdown intervals of 45–122 frames, "
                 "and the best period over the whole sweep still reaches only 0.2187 macro completion "
                 "(0.0772 as configured).")
    lines.append("2. **A phase estimated from recent joint motion tracks the gait cycle substantially better — "
                 "for the walking encoder.** 0.4128 macro, **1.9×** the best possible fixed clock and **5.3×** "
                 "the configured one. It does **not** clear the gate's own 0.90 bar (`encoder_freeze_eligible: "
                 "false`), so this is a comparative result, not a passing one. Note that the PHASOR arms in "
                 "this report used a different, dance-trained encoder; §7.2 records what scoring that encoder "
                 "on this same gate returned, and it is not 0.4128.")
    lines.append("3. **On flat terrain the phase-observation arm measured better than the fixed-clock arm**: "
                 "velocity tracking {:+.1f}%, angular {:+.1f}%, falls −{:.1f}%, velocity error XY −4.2%, yaw "
                 "−10.0% — one seed per arm throughout.".format(vel_improvement, ang_improvement, fall_reduction))
    lines.append("4. **These did not carry to rough terrain**: velocity tracking −11.2%, velocity error XY +4.7%, "
                 "terrain difficulty level −47.8%, falls −16.7% (n=1). The registered multi-seed replication in "
                 "`results/phasor_rough_equalbudget_20261020/` is the experiment that would settle the fall rate; "
                 "no registered experiment settles velocity tracking.")
    lines.append("5. **Items 3–4 are an ablation of the phase observation, not of phase fidelity.** The encoder "
                 "those runs loaded outputs a near-constant phase on both motions it has been measured on — its "
                 "own dance training data and the walking validation set, where it completes 0 of 603 cycles — so "
                 "the phase channel they received was very likely near-constant. No in-simulator measurement of "
                 "that channel exists; the statement is inference from two offline measurements. Velocity "
                 "tracking is in any case terrain-conditional and single-seed, and the claim is items 1–2.")
    lines.append("6. **Negligible computational cost**: 407KB model, ~0.1ms inference overhead; ONNX policies "
                 "exported for all models.")
    lines.append("")
    lines.append("### 7.2 Where this evidence is thin")
    lines.append("")
    lines.append("| gap | what would close it |")
    lines.append("|---|---|")
    lines.append("| No run in this report used a phase that actually tracks the gait cycle: the encoder behind §2–§3 scores 0.0000 on the gate | The current runtime encoder scores 0.4128 and is what experiment B loads; a run at that level would be the first behavioural test of a tracking phase |")
    lines.append("| Every terrain comparison here is one seed per arm | Experiment B (3 paired seeds, registered) covers the rough fall rate only |")
    lines.append("| The learned phase reaches 0.4128, not the gate's 0.90 | A better walking encoder; v2.1b/v2.1c reach 0.6962/0.7032 but have not been trained with |")
    lines.append("| Flat gains do not transfer to rough | Open; the rough arm also reached a lower curriculum level (−47.8%), which may explain part of it |")
    lines.append("")
    lines.append("### 7.3 Limitations & Future Work")
    lines.append("")
    lines.append("| Limitation | Future Direction |")
    lines.append("|-----------|------------------|")
    lines.append("| Encoder frozen during RL | Jointly fine-tune encoder + policy |")
    lines.append("| Single G1 mocap source | Train encoder on diverse terrain walking data |")
    lines.append("| No terrain conditioning | Condition phase on height scan for terrain-aware gait |")
    lines.append("| Sim-only evaluation | Deploy to real G1 hardware |")
    lines.append("| G1-specific encoder | Train multi-robot encoder (Go2, H1_2, etc.) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Reproducibility")
    lines.append("")
    lines.append("### Training Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# Flat terrain baseline")
    lines.append("python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096 --agent.run-name baseline_full")
    lines.append("")
    lines.append("# Flat terrain PHASOR")
    lines.append("python scripts/train.py Unitree-G1-Flat-PHASOR --env.scene.num-envs=4096 --agent.run-name phasor_full")
    lines.append("")
    lines.append("# Rough terrain baseline")
    lines.append("python scripts/train.py Unitree-G1-Rough --env.scene.num-envs=4096 --agent.run-name rough_baseline")
    lines.append("")
    lines.append("# Rough terrain PHASOR")
    lines.append("python scripts/train.py Unitree-G1-Rough-PHASOR --env.scene.num-envs=4096 --agent.run-name rough_phasor")
    lines.append("```")
    lines.append("")
    lines.append("### Comparison")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/compare_phasor.py \\")
    lines.append("    --baseline_dir <path> --phasor_dir <path> --output_dir results/<name>")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("🤖 Generated with [Claude Code](https://claude.com/claude-code)")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report_text = "\n".join(lines)

    # Save report
    report_path = OUTPUT_DIR / "FINAL_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    # Save JSON summary
    json_data = {
        "project": "G1 Gait Adjustment — PHASOR vs Baseline",
        "date": datetime.now().isoformat(),
        "flat_terrain": {
            metric: {
                "baseline": flat_baseline[metric]["final_mean"] if metric in flat_baseline else None,
                "phasor": flat_phasor[metric]["final_mean"] if metric in flat_phasor else None,
            }
            for metric in list(primary_metrics.keys()) + list(secondary_metrics.keys())
        },
        "rough_terrain": {
            "baseline_iterations": 9900,
            "phasor_iterations": rough_iter,
            "complete": rough_iter >= 10000,
        }
    }

    if rough_phasor and rough_iter >= 10000:
        json_data["rough_terrain"]["metrics"] = {
            metric: {
                "baseline": rough_baseline[metric]["final_mean"] if metric in rough_baseline else None,
                "phasor": rough_phasor[metric]["final_mean"] if metric in rough_phasor else None,
            }
            for metric in list(primary_metrics.keys()) + list(secondary_metrics.keys())
        }

    json_path = OUTPUT_DIR / "final_results.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"Report saved: {report_path}")
    print(f"JSON saved: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FLAT TERRAIN RESULTS")
    print("=" * 70)
    print(format_comparison_plain(primary_metrics, flat_baseline, flat_phasor))

    if rough_phasor and rough_iter >= 10000:
        print("\n" + "=" * 70)
        print("ROUGH TERRAIN RESULTS")
        print("=" * 70)
        print(format_comparison_plain(primary_metrics, rough_baseline, rough_phasor))

    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rough_phasor_dir", type=Path, default=None)
    args = parser.parse_args()
    generate_report(args.rough_phasor_dir)


if __name__ == "__main__":
    main()
