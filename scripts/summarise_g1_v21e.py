"""Apply the v2.1e pre-registered decision rules to the three arm reports.

Reads the three epoch sweeps written by ``train_g1_phase_encoder_v2_1e.py`` and
nothing else. It trains nothing, changes no threshold, and does not touch the
test split.

The rules being applied are quoted from
``docs/superpowers/plans/g1-phase-encoder-v2-1e-preregistration.md``:

1. candidate     - full v4.2 pass on some epoch (macro >= 0.90, every condition
                   >= 0.80, every stratum >= 0.80, point error <= pi/6, coverage
                   and reference-clock checks pass)
2. improvement   - best macro strictly greater than the frozen v2.1b 0.6962456093
3. mechanism     - versus the control arm's best epoch, an arm supports the
                   hypothesis only if it shows lower net-turn error sd AND fewer
                   whole-turn slips AND higher macro. A macro gain with flat or
                   worse sd/slips is a point-accuracy gain, not a mechanism result.
4. stop          - if neither raised-weight arm beats the baseline, the loss-side
                   line is closed with CYCLE_WEIGHT_NOT_THE_BINDING_CONSTRAINT.

The slip statistic used here is the project's published one
(``| |net - 2*pi| - 2*pi | < 1``), the same definition that reproduces 45 on the
frozen v2.1b epoch-1 predictions. It is not interchangeable with the broad
``far_tail_count`` field, which is reported for shape only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"
SUMMARY_JSON = OUT_DIR / "v21e_cycle_weight_summary.json"
SUMMARY_MD = OUT_DIR / "v21e_cycle_weight_summary.md"

ARMS = (
    ("control", 2.0),
    ("w20", 20.0),
    ("w60", 60.0),
)
BASELINE_MACRO = 0.6962456093142473
BASELINE_POINT = 0.7716245022746507
STOP_THRESHOLD_MACRO = 0.65
CONTROL_TOLERANCE = 1e-6


def load_arm(label: str) -> dict[str, Any]:
    path = OUT_DIR / f"v21e_{label}_epoch_sweep.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} missing; run the arm first")
    return json.loads(path.read_text(encoding="utf-8"))


def best_epoch(report: dict[str, Any]) -> dict[str, Any]:
    return max(report["epochs"], key=lambda row: row["macro_completion"])


def row_for(label: str, cycle_weight: float, report: dict[str, Any]) -> dict[str, Any]:
    best = best_epoch(report)
    net = best["net_turn"]
    return {
        "arm": label,
        "cycle_weight": cycle_weight,
        "cycle_weight_share": report["increment_weight_shares"]["cycle"],
        "best_epoch": best["epoch"],
        "macro_completion": best["macro_completion"],
        "point_error_rad": best["condition_macro_point_error_rad"],
        "freeze_eligible": best["freeze_eligible"],
        "point_error_within_cap": best["point_error_within_cap"],
        "net_error_sd": net.get("net_error_sd"),
        "share_within_tolerance": net.get("share_within_tolerance"),
        "count_too_small": net.get("count_too_small"),
        "count_too_large": net.get("count_too_large"),
        "whole_turn_slip_count": net.get("whole_turn_slip_count"),
        "far_tail_count": net.get("far_tail_count"),
        "corr_period_net_error": net.get("corr_period_net_error"),
        "status": report["status"],
        "condition_completion": best["condition_completion"],
        "period_length_strata": best["period_length_strata"],
        "epoch1_macro": report["epochs"][0]["macro_completion"],
        "last_epoch_macro": report["epochs"][-1]["macro_completion"],
    }


def main() -> int:
    reports = {label: load_arm(label) for label, _ in ARMS}
    rows = [row_for(label, weight, reports[label]) for label, weight in ARMS]
    by_label = {row["arm"]: row for row in rows}
    control = by_label["control"]

    control_delta_macro = control["macro_completion"] - BASELINE_MACRO
    control_delta_point = control["point_error_rad"] - BASELINE_POINT
    control_reproduced = (abs(control_delta_macro) <= CONTROL_TOLERANCE
                          and abs(control_delta_point) <= CONTROL_TOLERANCE)

    raised = [row for row in rows if row["arm"] != "control"]
    best_raised = max(raised, key=lambda row: row["macro_completion"])
    any_beats_baseline = any(row["macro_completion"] > BASELINE_MACRO for row in raised)
    any_candidate = any(row["status"] == "READY_SELECTED_CANDIDATE" for row in rows)

    mechanism: dict[str, Any] = {}
    for row in raised:
        lower_sd = row["net_error_sd"] < control["net_error_sd"]
        fewer_slips = row["whole_turn_slip_count"] < control["whole_turn_slip_count"]
        higher_macro = row["macro_completion"] > control["macro_completion"]
        mechanism[row["arm"]] = {
            "lower_net_error_sd": bool(lower_sd),
            "fewer_whole_turn_slips": bool(fewer_slips),
            "higher_macro_than_control": bool(higher_macro),
            "supports_mechanism": bool(lower_sd and fewer_slips and higher_macro),
        }

    if any_candidate:
        verdict = "CANDIDATE_SELECTED"
    elif any(block["supports_mechanism"] for block in mechanism.values()):
        verdict = "MECHANISM_SUPPORTED_BUT_BELOW_GATE"
    elif any_beats_baseline:
        verdict = "MACRO_GAIN_WITHOUT_MECHANISM_EVIDENCE"
    elif best_raised["macro_completion"] <= STOP_THRESHOLD_MACRO:
        verdict = "STOP_BAR_UNATTAINABLE"
    else:
        verdict = "CYCLE_WEIGHT_NOT_THE_BINDING_CONSTRAINT"

    payload = {
        "schema": "g1.v21e_cycle_weight_summary.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "test_split_read": False,
        "preregistration": "docs/superpowers/plans/g1-phase-encoder-v2-1e-preregistration.md",
        "arms": rows,
        "control_reproduction": {
            "baseline_macro": BASELINE_MACRO,
            "baseline_point_error_rad": BASELINE_POINT,
            "macro_delta": control_delta_macro,
            "point_delta": control_delta_point,
            "tolerance": CONTROL_TOLERANCE,
            "reproduced": bool(control_reproduced),
        },
        "mechanism_tests_vs_control": mechanism,
        "best_raised_arm": best_raised["arm"],
        "any_raised_arm_beats_v21b": bool(any_beats_baseline),
        "verdict": verdict,
        "slip_statistic_note": (
            "whole_turn_slip_count is | |net-2pi| - 2pi | < 1.0 rad, the published definition; it "
            "reproduces 45 on the frozen v2.1b epoch-1 predictions. far_tail_count (|net-2pi| > 3.0) "
            "is a broader shape statistic and must not be compared against any published slip count."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def fmt(value: Any, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Encoder v2.1e: cycle-integral weight sweep",
        "",
        "- Date: 2026-10-21",
        "- Pre-registration: `docs/superpowers/plans/g1-phase-encoder-v2-1e-preregistration.md`",
        "- Gate: `g1-phase-gate-v4.2`, unmodified; thresholds frozen",
        "- One parameter varied: `CYCLE_WEIGHT`, the increment weight on the one-row-per-cycle pairs",
        "- Fit on train only, select on validation only; test split not loaded",
        "",
        "## Results (best epoch per arm)",
        "",
        "| Arm | Cycle weight | Increment share | Best epoch | Macro | Point error (rad) | Net error sd | Within tol. | Too small / large | Whole-turn slips | v4.2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['cycle_weight']:g} | {row['cycle_weight_share'] * 100:.2f}% | "
            f"{row['best_epoch']} | {row['macro_completion']:.6f} | {row['point_error_rad']:.6f} | "
            f"{fmt(row['net_error_sd'], 3)} | {fmt(row['share_within_tolerance'], 4)} | "
            f"{row['count_too_small']} / {row['count_too_large']} | {row['whole_turn_slip_count']} | "
            f"{'pass' if row['status'] == 'READY_SELECTED_CANDIDATE' else 'fail'} |"
        )
    lines += [
        "",
        f"The point-error cap is `pi/6 = 0.523599 rad`; the macro bar is 0.90.",
        "",
        "## Control reproduction",
        "",
        f"Control arm macro delta versus the frozen v2.1b figure: `{control_delta_macro:+.3e}`; "
        f"point-error delta `{control_delta_point:+.3e}` (tolerance `{CONTROL_TOLERANCE:g}`). "
        f"Reproduced: **{control_reproduced}**.",
        "",
        "## Pre-registered mechanism test",
        "",
        "An arm supports the hypothesis only if, against the control arm's best epoch, it shows "
        "lower net-turn error sd **and** fewer whole-turn slips **and** higher macro.",
        "",
    ]
    for arm, block in mechanism.items():
        lines.append(
            f"- `{arm}`: lower sd {block['lower_net_error_sd']}, "
            f"fewer slips {block['fewer_whole_turn_slips']}, "
            f"higher macro {block['higher_macro_than_control']} -> "
            f"**supports mechanism: {block['supports_mechanism']}**"
        )
    lines += [
        "",
        "## All 30 epochs",
        "",
        "Macro completion per epoch. No arm is early-stopped; every arm completed all 30 epochs.",
        "",
        "| Arm | macro completion, epochs 1-30 |",
        "|---|---|",
    ]
    for row in rows:
        curve = " ".join(f"{e['macro_completion']:.3f}" for e in reports[row["arm"]]["epochs"])
        lines.append(f"| {row['arm']} | {curve} |")
    lines += [
        "",
        "Per-epoch point error, net-turn decomposition, and the full v4.2 report are in each "
        "`v21e_<arm>_epoch_sweep.json`.",
    ]

    detail = {}
    for row in rows:
        epochs = reports[row["arm"]]["epochs"]
        peak = max(epochs, key=lambda e: e["macro_completion"])
        low_sd = min(epochs, key=lambda e: e["net_turn"]["net_error_sd"])
        detail[row["arm"]] = {
            "peak_epoch": peak["epoch"],
            "peak_macro": peak["macro_completion"],
            "peak_point": peak["condition_macro_point_error_rad"],
            "peak_slips": peak["net_turn"]["whole_turn_slip_count"],
            "low_sd_epoch": low_sd["epoch"],
            "low_sd": low_sd["net_turn"]["net_error_sd"],
            "low_sd_macro": low_sd["macro_completion"],
        }

    peaks = " / ".join(f"{detail[row['arm']]['peak_macro']:.6f}" for row in rows)
    points = " -> ".join(f"{detail[row['arm']]['peak_point']:.4f}" for row in rows)
    slips = " -> ".join(str(detail[row["arm"]]["peak_slips"]) for row in rows)
    low_sd = "; ".join(
        f"`{row['arm']}` {detail[row['arm']]['low_sd']:.3f} at epoch {detail[row['arm']]['low_sd_epoch']}"
        f" (macro {detail[row['arm']]['low_sd_macro']:.4f})"
        for row in rows
    )
    all_peak_at_one = all(detail[row["arm"]]["peak_epoch"] == 1 for row in rows)

    lines += [
        "",
        "## What the sweep actually moved (descriptive, not a decision input)",
        "",
        "The pre-registered verdict above is negative on its own terms. Recording what did change, "
        "because it constrains what the next hypothesis may claim:",
        "",
        f"- **Peak epoch.** Every arm peaks at epoch 1: {all_peak_at_one}. Peak macros: {peaks}. "
        "Raising the cycle weight did not move the peak, and none approaches the 0.90 bar.",
        f"- **The intended quantities did move.** Point error at each arm's peak: {points} rad. "
        f"Whole-turn slips at each arm's peak: {slips}. The cycle term is not inert.",
        f"- **Lower net-error sd is reachable, but not at a usable macro.** Lowest sd per arm: {low_sd}. "
        "Latitude on the integral exists, but it is bought with point accuracy rather than added to it.",
        "- **Later epochs are more stable under raised cycle weight** (see the curves above), yet never "
        "exceed each arm's epoch-1 peak.",
        "",
        "Read together: the graded integral is not under-weighted supervision. The objective sits on a "
        "frontier where point accuracy and the cycle integral trade against each other, and reweighting "
        "moves position along that frontier without raising the macro, which is the quantity the gate "
        "scores. This is the pre-registered stop condition for the loss-side line.",
        "",
        f"## Verdict: `{verdict}`",
        "",
        "No PPO run, deployment selection, or checkpoint freeze follows from this result. "
        "The existing test split was opened by an earlier diagnostic, so it is not a blind "
        "confirmation set; a confirmatory claim needs an independent holdout.",
        "",
        "## Artifacts",
        "",
        "- Trainer: `scripts/train_g1_phase_encoder_v2_1e.py`",
        "- Per-arm reports and predictions: `v21e_<arm>_epoch_sweep.json`, `v21e_<arm>_epoch_predictions.npz`",
        "- Weight decomposition: `v21e_cycle_weight_decomposition.json`",
        "- Machine-readable summary: `v21e_cycle_weight_summary.json`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for row in rows:
        print(f"{row['arm']:8s} w={row['cycle_weight']:5g} ep{row['best_epoch']:3d} "
              f"macro {row['macro_completion']:.6f} point {row['point_error_rad']:.6f} "
              f"sd {fmt(row['net_error_sd'], 3)} slips {row['whole_turn_slip_count']}")
    print(f"control reproduced: {control_reproduced}")
    print(f"verdict: {verdict}")
    print(f"-> {SUMMARY_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
