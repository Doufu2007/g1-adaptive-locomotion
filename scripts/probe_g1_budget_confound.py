"""Re-read the August rough-terrain comparison at matched training budget.

``results/final_report/final_results.json`` compares the *final* tensorboard
window of two runs that did not train for the same number of iterations:

    rough baseline   0 .. 10000
    rough PHASOR     ~1000 .. 15026   (three resumable segments)

Any quantity read off the tail therefore compares iteration 10000 against
iteration 15026.  This script reads both event files, interpolates every
scalar onto a common iteration grid, and reports the same quantities at
**matched** iterations so the confound is a number rather than a suspicion.

It also reports the flat pair, which *is* budget-matched (10000 vs 9999), as a
control: whatever this method does to a fair comparison it must also do there.

Read-only.  Nothing under ``results/final_report`` is modified.

Usage:
    python scripts/probe_g1_budget_confound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "rsl_rl" / "g1_velocity"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "budget_confound.json"

# Both PHASOR runs were continued across several directories; the trajectory
# only exists once every segment is merged in iteration order.
RUNS: dict[str, tuple[str, ...]] = {
    "rough_baseline": ("2026-08-10_23-06-29_rough_baseline",),
    "rough_phasor": (
        "2026-08-11_06-13-25_rough_phasor",
        "2026-08-11_06-18-55_rough_phasor",
        "2026-08-11_10-51-03_rough_phasor",
    ),
    "flat_baseline": ("2026-08-10_09-35-45_baseline_full",),
    "flat_phasor": (
        "2026-08-10_12-29-14_phasor_full",
        "2026-08-10_21-49-27_phasor_resume_final",
    ),
}

METRICS = (
    "Episode_Termination/fell_over",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/track_angular_velocity",
    "Episode_Reward/foot_gait",
)

MATCHED_ITERATIONS = (3000, 5000, 7000, 9000, 10000)


def load_scalars(log_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in METRICS:
        if metric not in available:
            continue
        events = accumulator.Scalars(metric)
        series[metric] = (
            np.asarray([event.step for event in events], dtype=np.float64),
            np.asarray([event.value for event in events], dtype=np.float64),
        )
    return series


def summarize(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Iteration coverage plus the trailing statistic the report actually quoted.

    ``generate_final_report.load_tb_summary`` calls its headline number
    ``final_mean`` and defines it as the mean of the last ``n // 10`` logged
    points of that run.  Reproducing it here is the self-check: the merged
    segments must reproduce ``final_results.json`` before any matched-budget
    number means anything.
    """
    out: dict = {"metrics": {}, "iteration_min": None, "iteration_max": None}
    for metric, (steps, values) in series.items():
        n_last = max(1, values.size // 10)
        out["metrics"][metric] = {
            "n_points": int(steps.size),
            "first_step": float(steps[0]),
            "last_step": float(steps[-1]),
            "reported_final_mean_last_decile": float(values[-n_last:].mean()),
            "tail_mean_last_10_points": float(values[-10:].mean()) if values.size >= 10 else None,
        }
        out["iteration_min"] = float(steps[0]) if out["iteration_min"] is None else min(out["iteration_min"], float(steps[0]))
        out["iteration_max"] = float(steps[-1]) if out["iteration_max"] is None else max(out["iteration_max"], float(steps[-1]))
    return out


def interpolate(series: dict[str, tuple[np.ndarray, np.ndarray]], metric: str, iteration: int) -> float | None:
    if metric not in series:
        return None
    steps, values = series[metric]
    if iteration < steps.min() or iteration > steps.max():
        return None
    # Trailing window of 200 iterations smooths the per-iteration curriculum noise.
    window = (steps > iteration - 200) & (steps <= iteration)
    if window.sum() < 3:
        window = np.abs(steps - iteration) <= 200
    if window.sum() == 0:
        return None
    return float(values[window].mean())


def merge_segments(dirnames: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Concatenate a run's event files and sort by iteration."""
    merged: dict[str, list[np.ndarray]] = {}
    for dirname in dirnames:
        path = LOG_DIR / dirname
        if not path.is_dir():
            raise FileNotFoundError(path)
        for metric, (steps, values) in load_scalars(path).items():
            merged.setdefault(metric, [[], []])
            merged[metric][0].append(steps)
            merged[metric][1].append(values)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric, (step_parts, value_parts) in merged.items():
        steps = np.concatenate(step_parts)
        values = np.concatenate(value_parts)
        order = np.argsort(steps, kind="stable")
        series[metric] = (steps[order], values[order])
    return series


def main() -> int:
    loaded = {}
    for name, dirnames in RUNS.items():
        loaded[name] = merge_segments(dirnames)
        coverage = summarize(loaded[name])
        print(
            f"{name:16s} iterations {coverage['iteration_min']:.0f}..{coverage['iteration_max']:.0f}"
            f"  ({len(dirnames)} event file(s))",
            flush=True,
        )

    pairs = (("rough", "rough_baseline", "rough_phasor"), ("flat", "flat_baseline", "flat_phasor"))

    payload: dict = {
        "schema": "g1.budget_confound.v1",
        "purpose": "quantify how much of the reported terrain improvement is iteration budget",
        "metrics": list(METRICS),
        "matched_iteration_grid": list(MATCHED_ITERATIONS),
        "runs": {name: summarize(series) for name, series in loaded.items()},
        "pairs": {},
    }

    for label, base_name, phasor_name in pairs:
        base = loaded[base_name]
        phasor = loaded[phasor_name]
        rows = []
        for iteration in MATCHED_ITERATIONS:
            row = {"iteration": iteration}
            for metric in METRICS:
                row[metric] = {
                    "baseline": interpolate(base, metric, iteration),
                    "phasor": interpolate(phasor, metric, iteration),
                }
            rows.append(row)

        # Last window both runs cover, which is the widest fair comparison:
        # 1000 iterations ending where the shorter run stopped.
        stop = min(
            payload["runs"][base_name]["iteration_max"],
            payload["runs"][phasor_name]["iteration_max"],
        )
        start = stop - 1000.0
        matched_window: dict = {"iteration_start": start, "iteration_end": stop}
        for metric in METRICS:
            entry: dict = {}
            for arm, series in (("baseline", base), ("phasor", phasor)):
                if metric not in series:
                    entry[arm] = None
                    continue
                steps, values = series[metric]
                mask = (steps >= start) & (steps <= stop)
                entry[arm] = float(values[mask].mean()) if mask.any() else None
            if entry["baseline"] not in (None, 0.0) and entry["phasor"] is not None:
                entry["relative_change"] = entry["phasor"] / entry["baseline"] - 1.0
            matched_window[metric] = entry
        tail = {}
        for metric in METRICS:
            base_tail = payload["runs"][base_name]["metrics"].get(metric, {}).get("reported_final_mean_last_decile")
            phasor_tail = payload["runs"][phasor_name]["metrics"].get(metric, {}).get("reported_final_mean_last_decile")
            tail[metric] = {
                "baseline_final_mean": base_tail,
                "phasor_final_mean": phasor_tail,
                "relative_change": (
                    None
                    if base_tail in (None, 0.0) or phasor_tail is None
                    else phasor_tail / base_tail - 1.0
                ),
            }
        payload["pairs"][label] = {
            "baseline_run": RUNS[base_name],
            "phasor_run": RUNS[phasor_name],
            "baseline_iterations_covered": [
                payload["runs"][base_name]["iteration_min"],
                payload["runs"][base_name]["iteration_max"],
            ],
            "phasor_iterations_covered": [
                payload["runs"][phasor_name]["iteration_min"],
                payload["runs"][phasor_name]["iteration_max"],
            ],
            "as_reported_final_mean": tail,
            "matched_end_window": matched_window,
            "matched_iteration_grid": rows,
        }

        fell = "Episode_Termination/fell_over"
        print(f"\n--- {label} (as reported = unmatched last-decile means)")
        print(f"  fell_over : baseline {tail[fell]['baseline_final_mean']}  phasor {tail[fell]['phasor_final_mean']}")
        mw = matched_window[fell]
        print(f"--- {label} (matched window {matched_window['iteration_start']:.0f}..{matched_window['iteration_end']:.0f})")
        print(f"  fell_over : baseline {mw['baseline']}  phasor {mw['phasor']}  rel {mw.get('relative_change')}")
        print(f"--- {label} (matched iteration grid)")
        for row in rows:
            b = row[fell]["baseline"]
            p = row[fell]["phasor"]
            print(f"  iter {row['iteration']:6d}  baseline {b if b is None else f'{b:.4f}'}  phasor {p if p is None else f'{p:.4f}'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\n-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
