"""Apply the pre-registered decision rules to the four v2.1f runs.

Reads only on-disk artifacts written by ``train_g1_phase_encoder_v2_1f_within.py``
and the pre-registration it cites. Every number in the output is read from those
files; nothing is typed by hand.

Rules (fixed before the runs; see the pre-registration):

1. TRANSFER_IS_THE_LIMIT  any run passes the full v4.2 gate
2. SUBJECT_CHANGE_CONTRIBUTES  W_mean - C_mean >= 0.05 macro
3. NOT_TRANSFER_LIMIT  no pass and both W_mean and C_mean <= 0.65
4. UNRESOLVED  anything else
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"
MACRO_BAR = 0.90
CONDITION_BAR = 0.80
STRATUM_BAR = 0.80
EFFECT_SIZE = 0.05
STOP_BAR = 0.65

ARMS = ("within", "cross")
FOLDS = ("A", "B")


def load(arm: str, fold: str) -> dict:
    path = OUT_DIR / f"v21f_{arm}_fold{fold}_epoch_sweep.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def best_epoch(payload: dict) -> dict:
    return next(r for r in payload["epochs"] if r["epoch"] == payload["best_epoch_by_macro"])


def main() -> int:
    payloads = {(arm, fold): load(arm, fold) for arm in ARMS for fold in FOLDS}
    rows = []
    for arm in ARMS:
        for fold in FOLDS:
            payload = payloads[(arm, fold)]
            record = best_epoch(payload)
            rows.append({
                "arm": arm,
                "fold": fold,
                "train_sequences": payload["sequences"]["train"],
                "validation_sequences": payload["sequences"]["validation"],
                "train_rows": payload["rows"]["train"],
                "train_touchdowns": payload["touchdowns"]["train"],
                "validation_rows": payload["rows"]["validation"],
                "validation_touchdowns": payload["touchdowns"]["validation"],
                "cross_selection": payload.get("cross_arm_selection"),
                "pair_count": payload["pair_count"],
                "best_epoch": record["epoch"],
                "macro": record["macro_completion"],
                "point_error_rad": record["condition_macro_point_error_rad"],
                "min_condition": min(record["condition_completion"].values()),
                "min_stratum": min(record["period_length_strata"].values()),
                "net_error_sd": record["net_turn"].get("net_error_sd"),
                "share_within_tolerance": record["net_turn"].get("share_within_tolerance"),
                "whole_turn_slip_count": record["net_turn"].get("whole_turn_slip_count"),
                "cycle_count": record["net_turn"].get("cycle_count"),
                "train_loss": record["train"]["loss"],
                "train_point_loss": record["train"]["point_loss"],
                "full_v42_pass": (
                    record["macro_completion"] >= MACRO_BAR
                    and record["condition_macro_point_error_rad"] <= payload["preregistered"]["point_error_cap_rad"]
                    and all(v >= CONDITION_BAR for v in record["condition_completion"].values())
                    and all(v >= STRATUM_BAR for v in record["period_length_strata"].values())
                ),
                "status": payload["status"],
            })

    within = [r["macro"] for r in rows if r["arm"] == "within"]
    cross = [r["macro"] for r in rows if r["arm"] == "cross"]
    w_mean = sum(within) / len(within)
    c_mean = sum(cross) / len(cross)
    passes = [r for r in rows if r["full_v42_pass"]]

    if passes:
        verdict = "TRANSFER_IS_THE_LIMIT"
        reason = ("a run passed the full v4.2 gate: " +
                  ", ".join(f"{r['arm']}-{r['fold']} macro {r['macro']:.6f}" for r in passes))
    elif w_mean - c_mean >= EFFECT_SIZE:
        verdict = "SUBJECT_CHANGE_CONTRIBUTES"
        reason = (f"W_mean {w_mean:.6f} - C_mean {c_mean:.6f} = {w_mean - c_mean:.6f} "
                  f">= pre-declared {EFFECT_SIZE}")
    elif w_mean <= STOP_BAR and c_mean <= STOP_BAR:
        verdict = "NOT_TRANSFER_LIMIT"
        reason = (f"no pass; W_mean {w_mean:.6f} and C_mean {c_mean:.6f} both <= {STOP_BAR}")
    else:
        verdict = "UNRESOLVED"
        reason = (f"no pass; W_mean {w_mean:.6f}, C_mean {c_mean:.6f}, "
                  f"gap {w_mean - c_mean:.6f} < {EFFECT_SIZE}, and not both <= {STOP_BAR}")

    summary = {
        "schema": "g1.phase_encoder_v21f_transfer_summary.v1",
        "preregistration": "docs/superpowers/plans/g1-phase-within-subject-transfer-preregistration.md",
        "test_split_read": False,
        "ruling": {"verdict": verdict, "reason": reason},
        "aggregates": {
            "W_mean_macro": w_mean,
            "C_mean_macro": c_mean,
            "W_minus_C": w_mean - c_mean,
            "effect_size_threshold": EFFECT_SIZE,
            "stop_bar": STOP_BAR,
            "within_macros": within,
            "cross_macros": cross,
        },
        "runs": rows,
        "caveats": [
            "One seed per cell. The 0.05 gap threshold is a declared effect size, not a measured noise band.",
            "One subject (subject 1) only; nothing about subjects 2-5.",
            "The stored feature normalisation was fit on the training subjects and was deliberately NOT refit, so a negative result cannot separate 'subject is not the limit' from 'the limit acts through the normalisation statistics'.",
            "Varies the data partition only. Says nothing about a structurally different learner such as a rate-and-integrate (PINN / neural-ODE) model.",
        ],
    }

    json_path = OUT_DIR / "v21f_transfer_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def fmt(value, digits=6):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# v2.1f: does the change of subject cause the 0.70 ceiling?",
        "",
        "Pre-registered in `docs/superpowers/plans/g1-phase-within-subject-transfer-preregistration.md`",
        "(plus Amendment 1, appended before any run). Generated from on-disk JSON only.",
        "",
        f"## Ruling: **{verdict}**",
        "",
        reason,
        "",
        "## The four runs",
        "",
        "| arm | fold | train sequences | train rows | train td | val rows | val td | best ep | macro | point err | min cond | min stratum | net err sd | slips | full v4.2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['fold']} | {', '.join(r['train_sequences'])} | "
            f"{r['train_rows']:,} | {r['train_touchdowns']} | {r['validation_rows']:,} | "
            f"{r['validation_touchdowns']} | {r['best_epoch']} | {r['macro']:.6f} | "
            f"{r['point_error_rad']:.4f} | {r['min_condition']:.4f} | {r['min_stratum']:.4f} | "
            f"{fmt(r['net_error_sd'], 3)} | {r['whole_turn_slip_count']} | "
            f"{'PASS' if r['full_v42_pass'] else 'no'} |"
        )
    lines += [
        "",
        "`td` is the total left-foot touchdown count, the quantity the cycle term counts;",
        "it is the matching variable for the cross arm. `slips` is the project's published",
        "whole-turn slip definition, not the broad tail count.",
        "",
        "## Aggregates",
        "",
        f"- `W_mean` = {w_mean:.6f}",
        f"- `C_mean` = {c_mean:.6f}",
        f"- `W_mean - C_mean` = {w_mean - c_mean:.6f} (threshold {EFFECT_SIZE})",
        "",
        "## Reference points",
        "",
        "| point | macro |",
        "|---|---|",
        "| oracle | 0.998418 |",
        "| v2.1b cross-subject baseline | 0.696246 |",
        "| v2.1c best prior arm | 0.703175 |",
        "| best-over-grid fixed clock | 0.218664 |",
        "",
        "## Caveats carried from the registration",
        "",
    ]
    lines += [f"- {c}" for c in summary["caveats"]]
    lines.append("")

    md_path = OUT_DIR / "v21f_transfer_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "reason": reason,
                      "outputs": [str(json_path.relative_to(ROOT)),
                                  str(md_path.relative_to(ROOT))]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
