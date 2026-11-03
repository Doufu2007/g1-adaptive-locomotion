"""Apply the locked decision rule of experiment B to the finished runs.

Implements `results/phasor_rough_equalbudget_20261020/preregistration.md`
sections 3, 4, 4.1 and 4.2 and nothing else.  It cannot choose metrics, seeds
or thresholds: every constant below is transcribed from that file.

Primary outcome
    ``Episode_Termination/fell_over``, mean over iterations 9001-10000 of each
    run's own TensorBoard trajectory (``deterministic_final_window``).

Paired differences
    ``d_s = phasor_s - baseline_s`` per seed, negative meaning fewer falls.

This script refuses to run unless every run's recorded source hashes match the
values locked in ``analysis_plan.json``; a silently different code path would
make the comparison non-comparable.

Usage:
    python scripts/analyze_g1_rough_equalbudget.py --runs runs.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results" / "phasor_rough_equalbudget_20261020"
PLAN_PATH = EXPERIMENT / "analysis_plan.json"
OUT_PATH = EXPERIMENT / "analysis.json"

PRIMARY_METRIC = "Episode_Termination/fell_over"
FINAL_WINDOW = (9001.0, 10000.0)
EFFECT_FLOOR = 0.05
DEGENERATE_FALL_FRACTION = 0.95
SD_BANDS = {"resolves": 0.028, "extend": 0.045}
T_MULTIPLIER_N3 = 1.77  # as used in results/phasor_pilot_20261016/analysis.json

# Two-sided 95% t critical values, df = n - 1.
T95 = {2: 4.302652729749464, 3: 3.182446305284263, 4: 2.7764451051977987,
       5: 2.570581835636314, 6: 2.446911848791681}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_final_window(log_dir: Path, metric: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if metric not in set(accumulator.Tags()["scalars"]):
        raise KeyError(f"{metric} missing from {log_dir}")
    events = accumulator.Scalars(metric)
    low, high = FINAL_WINDOW
    values = [event.value for event in events if low <= event.step <= high]
    if len(values) < 10:
        raise ValueError(f"{log_dir} has only {len(values)} points inside the final window")
    return {
        "n_points": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "iteration_min": min(event.step for event in events if low <= event.step <= high),
        "iteration_max": max(event.step for event in events if low <= event.step <= high),
    }


def t_interval(differences: list[float]) -> tuple[float, float, float, float]:
    n = len(differences)
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    if n < 2:
        return mean, sd, float("nan"), float("nan")
    half = T95[n - 1] * sd / math.sqrt(n)
    return mean, sd, mean - half, mean + half


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="JSON mapping condition -> seed -> log dir")
    args = parser.parse_args()

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    observed = {name: sha256_file(ROOT / name) for name in plan["pinned_sources"]}
    mismatched = {
        name: {"expected": plan["pinned_sources"][name], "observed": digest}
        for name, digest in observed.items()
        if digest != plan["pinned_sources"][name]
    }
    if mismatched:
        raise SystemExit(f"source hashes changed since the plan was locked: {mismatched}")

    runs = json.loads(Path(args.runs).read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in plan["seeds"]]

    rows, degenerate, paired = [], [], []
    for seed in seeds:
        row = {"ppo_seed": seed}
        for condition in ("baseline", "phasor"):
            log_dir = Path(runs[condition][str(seed)])
            stats = read_final_window(log_dir, PRIMARY_METRIC)
            row[condition] = {"log_dir": str(log_dir), **stats}
            if stats["mean"] >= DEGENERATE_FALL_FRACTION:
                degenerate.append({"ppo_seed": seed, "condition": condition, "value": stats["mean"]})
        if {"baseline", "phasor"} <= row.keys():
            difference = row["phasor"]["mean"] - row["baseline"]["mean"]
            row["difference_phasor_minus_baseline"] = difference
            is_degenerate = any(
                entry["ppo_seed"] == seed for entry in degenerate
            )
            row["used_in_paired_test"] = not is_degenerate
            if not is_degenerate:
                paired.append(difference)
        rows.append(row)

    if len(degenerate) >= 2:
        verdict = "INCONCLUSIVE"
        reason = "two or more runs did not reach a walking regime"
    elif len(paired) < 2:
        verdict = "INCONCLUSIVE"
        reason = "fewer than two non-degenerate paired seeds remain"
    else:
        mean, sd, low, high = t_interval(paired)
        if not (mean < 0):
            verdict, reason = "NOT_ESTABLISHED", "paired mean is not negative"
        elif not (high < 0):
            verdict, reason = "NOT_ESTABLISHED", "95% interval includes zero"
        elif not (abs(mean) >= EFFECT_FLOOR):
            verdict, reason = "NOT_ESTABLISHED", f"|mean| below the {EFFECT_FLOOR} floor"
        else:
            verdict, reason = "LEARNED_PHASE_REDUCES_FALLS", "all three section-4 conditions hold"

    mean, sd, low, high = t_interval(paired) if len(paired) >= 2 else (float("nan"),) * 4
    if len(paired) >= 2:
        if sd <= SD_BANDS["resolves"]:
            power = "DESIGN_RESOLVES_THE_EFFECT_FLOOR"
        elif sd <= SD_BANDS["extend"]:
            power = "UNRESOLVED_AT_N3_EXTEND_TO_N6"
        else:
            power = "UNDERPOWERED_FOR_A_0.05_EFFECT_ON_THIS_METRIC"
    else:
        power = "NOT_COMPUTABLE"

    payload = {
        "schema": "g1.rough_equalbudget_analysis.v1",
        "preregistration": str(Path("results/phasor_rough_equalbudget_20261020/preregistration.md")),
        "primary_metric": PRIMARY_METRIC,
        "final_window": list(FINAL_WINDOW),
        "seeds": seeds,
        "rows": rows,
        "degenerate_runs": degenerate,
        "paired_differences": paired,
        "paired_mean": mean,
        "paired_sd": sd,
        "paired_ci95": [low, high],
        "mde_80pct_at_current_n": T_MULTIPLIER_N3 * sd if math.isfinite(sd) else None,
        "effect_floor": EFFECT_FLOOR,
        "power_classification": power,
        "verdict": verdict,
        "verdict_reason": reason,
        "source_hashes_verified": observed,
        "test_split_read": False,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("verdict", "verdict_reason", "paired_mean",
                                              "paired_ci95", "power_classification")}, indent=2))
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
