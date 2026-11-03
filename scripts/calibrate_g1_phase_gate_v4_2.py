"""Re-score the frozen validation estimators under the v4.2 endpoint correction.

Reads only the frozen validation split (subject 1). The test split is never
opened. Nothing here trains, selects a new checkpoint, or changes a threshold:
it takes the v4.1 predictions that are already on disk and applies v4.2.

Outputs live in `results/g1_phase_gate_v4_2/`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.phase_gate_v4 import score_phase_cycles_v4
from src.tasks.velocity.phasor.phase_gate_v4_2 import (
    NET_TOLERANCE_RAD,
    PHASE_GATE_VERSION,
    TERMINAL_SAMPLE_POLICY,
    calibrate_phase_gate_v4_2,
    score_phase_cycles_v4_2,
)


ROOT_DIR = ROOT
V4_DIR = ROOT / "results" / "g1_phase_gate_v4"
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
DATA_PATH = DATA_DIR / "phase_dataset.npz"
PREDICTIONS_PATH = DATA_DIR / "phase_predictions.npz"
CANDIDATES_PATH = V4_DIR / "candidate_validation_v2_epoch_selection_predictions.npz"
V4_CALIBRATION_PATH = V4_DIR / "calibration_v4_amendment1.json"
V4_MODULE_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4.py"
V4_2_MODULE_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4_2.py"

DATA_SHA256 = "e204c9dd0a48527ac4c82cfd0c539fba7d584f1faea6242ad0eba66970008b38"
#: v4.1 must stay byte-identical; the correction is a new module, not an edit.
V4_MODULE_SHA256 = None  # filled from the frozen v4 calibration record

#: Declared now, before any v2.1 training run, and not applied retroactively.
#: A clock whose mean per-frame error exceeds the contract's own per-cycle net
#: tolerance (pi/6 rad, one sixth of a turn) cannot time gait events, and the
#: cycle gate cannot see a constant offset because net and max-step are both
#: differences (`phase = reference + 1.0` scores macro 1.0000 in v4.1 and v4.2).
#: This adds no new constant; it reuses the frozen one.
POINT_ERROR_CAP_RAD = NET_TOLERANCE_RAD
POINT_ERROR_CAP_SOURCE = "reuses pre-registered NET_TOLERANCE_RAD (pi/6); declared before v2.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_cycle_completion_fraction": report["macro_cycle_completion_fraction"],
        "micro_cycle_completion_fraction": report["encoder_micro_cycle_completion_fraction"],
        "reference_clock_completion_fraction": report["reference_clock_completion_fraction"],
        "condition_completion": {
            name: row["completion_fraction"] for name, row in report["condition_completion"].items()
        },
        "period_length_strata": {
            name: row["completion_fraction"] for name, row in report["period_length_strata"].items()
        },
        "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
        "observed_cycle_count": report["observed_cycle_count"],
        "evaluable_cycle_count": report["evaluable_cycle_count"],
        "coverage_pass": report["coverage_pass"],
        "reference_clock_pass": report["reference_clock_pass"],
        "condition_completion_pass": report["condition_completion_pass"],
        "period_length_completion_pass": report["period_length_completion_pass"],
        "encoder_freeze_eligible": report["encoder_freeze_eligible"],
        "point_error_within_cap": bool(
            report["condition_macro_point_error_rad"] <= POINT_ERROR_CAP_RAD
        ),
    }


def _score(prediction: np.ndarray, base: dict[str, Any], scorer) -> dict[str, Any]:
    return scorer(
        prediction,
        base["reference_phase"],
        base["sequence_ids"],
        base["frame_ids"],
        base["touchdowns"],
        condition_ids=base["sequence_ids"],
        episode_ids=base["sequence_ids"],
        history_valid=np.ones(len(base["frame_ids"]), dtype=bool),
        period_cuts_frames=base["period_cuts"],
        required_conditions=base["conditions"],
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )


def _candidate_key(name: str) -> tuple[float, int, str]:
    match = re.fullmatch(r"context_(\d+)_epoch_(\d+)", name)
    if match is None:
        raise ValueError(f"unexpected candidate name {name!r}")
    return -float(match.group(1)), int(match.group(2)), name


def main() -> int:
    required = [DATA_PATH, PREDICTIONS_PATH, CANDIDATES_PATH, V4_CALIBRATION_PATH, V4_MODULE_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required v4.2 re-score inputs are missing: " + ", ".join(missing))

    data_hash = sha256_file(DATA_PATH)
    if data_hash != DATA_SHA256:
        raise ValueError("dataset hash differs from the frozen experiment input")

    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    v4_module_hash = sha256_file(V4_MODULE_PATH)
    recorded_hash = frozen.get("source_sha256", {}).get(str(V4_MODULE_PATH.relative_to(ROOT)))
    if recorded_hash is not None and recorded_hash != v4_module_hash:
        raise ValueError(
            "phase_gate_v4.py changed since the v4.1 calibration; the v4.1 record is no longer valid"
        )
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    dataset = np.load(DATA_PATH, allow_pickle=False)
    predictions = np.load(PREDICTIONS_PATH, allow_pickle=False)
    candidates = np.load(CANDIDATES_PATH, allow_pickle=False)

    sequence_ids = dataset["sequence_validation"]
    frame_ids = dataset["frame_validation"]
    reference_phase = dataset["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi)
    touchdowns = dataset["left_touchdown_validation"].astype(bool)
    conditions = sorted(np.unique(sequence_ids).tolist())
    base = {
        "reference_phase": reference_phase,
        "sequence_ids": sequence_ids,
        "frame_ids": frame_ids,
        "touchdowns": touchdowns,
        "period_cuts": period_cuts,
        "conditions": conditions,
    }

    # Calibrate 4.2 on the reference clock, then confirm the frozen v4.1 cuts
    # reproduce (the correction must not move the strata).
    calibration = calibrate_phase_gate_v4_2(
        np.stack((np.sin(reference_phase), np.cos(reference_phase)), axis=-1),
        reference_phase,
        sequence_ids,
        frame_ids,
        touchdowns,
        condition_ids=sequence_ids,
        episode_ids=sequence_ids,
        history_valid=np.ones(len(frame_ids), dtype=bool),
        required_conditions=conditions,
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    v4_2_cuts = tuple(calibration["period_length_cut_points_frames"])
    if max(abs(a - b) for a, b in zip(v4_2_cuts, period_cuts)) > 1e-9:
        raise ValueError("v4.2 recalibration moved the period strata; the scorer is not equivalent")

    reference_prediction = np.stack((np.sin(reference_phase), np.cos(reference_phase)), axis=-1)
    v1_prediction = predictions["val_pred"]

    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for name in sorted((key for key in candidates.files if key.startswith("context_")), key=_candidate_key):
        report = _score(candidates[name], base, score_phase_cycles_v4_2)
        reports[name] = report
        v41 = _score(candidates[name], base, score_phase_cycles_v4)
        rows.append({
            "candidate": name,
            "context_frames": int(name.split("_")[1]),
            "epoch": int(name.split("_epoch_")[1]),
            "macro_completion_v4_2": report["macro_cycle_completion_fraction"],
            "macro_completion_v4_1": v41["macro_cycle_completion_fraction"],
            "delta_v4_1_to_v4_2": (
                report["macro_cycle_completion_fraction"] - v41["macro_cycle_completion_fraction"]
            ),
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            **_summary(report),
        })

    reference_report = _score(reference_prediction, base, score_phase_cycles_v4_2)
    v1_report = _score(v1_prediction, base, score_phase_cycles_v4_2)
    v1_report_v41 = _score(v1_prediction, base, score_phase_cycles_v4)

    ranked = sorted(rows, key=lambda row: -row["macro_completion_v4_2"])
    best = ranked[0] if ranked else None
    passing = [row for row in rows if row["encoder_freeze_eligible"]]
    passing_with_cap = [row for row in rows if row["encoder_freeze_eligible"] and row["point_error_within_cap"]]

    def scalar(report: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in _summary(report).items()}

    payload = {
        "schema": "g1.phase_gate_v4_2_rescore.v1",
        "gate_version": PHASE_GATE_VERSION,
        "terminal_sample_policy": TERMINAL_SAMPLE_POLICY,
        "source_split": "validation_only",
        "test_split_read": False,
        "purpose": (
            "Re-score the frozen v4.1 predictions under the v4.2 endpoint correction. "
            "No training, no new candidate, no threshold change, no promotion."
        ),
        "thresholds_changed": False,
        "dataset_sha256": data_hash,
        "v4_module_sha256": v4_module_hash,
        "v4_module_sha256_matches_v4_record": recorded_hash == v4_module_hash,
        "v4_2_module_sha256": sha256_file(V4_2_MODULE_PATH),
        "candidate_predictions_sha256": sha256_file(CANDIDATES_PATH),
        "period_length_cut_points_frames": list(v4_2_cuts),
        "period_length_cuts_unchanged": True,
        "point_error_cap_rad": POINT_ERROR_CAP_RAD,
        "point_error_cap_source": POINT_ERROR_CAP_SOURCE,
        "point_error_cap_applied_to_v4_2_freeze_rule": False,
        "estimators": {
            "reference_clock": scalar(reference_report),
            "encoder_v1": {
                **scalar(v1_report),
                "macro_completion_v4_1": v1_report_v41["macro_cycle_completion_fraction"],
            },
            "encoder_v2_best_of_60": {
                **(scalar(reports[best["candidate"]]) if best else {}),
                "candidate": best["candidate"] if best else None,
            },
        },
        "candidate_count": len(rows),
        "candidates_passing_v4_2": len(passing),
        "candidates_passing_v4_2_and_point_cap": len(passing_with_cap),
        "best_candidate_v4_2": best,
        "candidates": rows,
        "status": (
            "READY_CANDIDATE_PASSED"
            if passing_with_cap
            else "BLOCKED_NO_CANDIDATE_PASSED_V4_2_UNDER_POINT_ERROR_CAP"
            if not passing
            else "BLOCKED_CANDIDATES_PASS_CYCLE_GATE_ONLY_WITH_POINT_ERROR_ABOVE_CAP"
        ),
        "ppo_blocked": True,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rescore_v4_2.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    calibration_payload = {
        "schema": "g1.phase_gate_calibration_record.v4_2",
        "gate_version": PHASE_GATE_VERSION,
        "terminal_sample_policy": TERMINAL_SAMPLE_POLICY,
        "source_split": "validation_only",
        "period_length_cut_points_frames": list(v4_2_cuts),
        "period_length_cuts_reuse_v4_1": list(period_cuts),
        "reference_clock_completion_fraction": calibration["reference_clock_completion_fraction"],
        "reference_clock_pass": calibration["reference_clock_pass"],
        "coverage_pass": calibration["coverage_pass"],
        "encoder_v1_macro_completion_fraction": v1_report["encoder_completion_fraction"],
        "encoder_v1_freeze_eligible": v1_report["encoder_freeze_eligible"],
        "thresholds_valid": calibration["thresholds_valid"],
        "calibration_status": calibration["calibration_status"],
        "source_sha256": {
            str(DATA_PATH.relative_to(ROOT)): data_hash,
            str(V4_MODULE_PATH.relative_to(ROOT)): v4_module_hash,
            str(V4_2_MODULE_PATH.relative_to(ROOT)): sha256_file(V4_2_MODULE_PATH),
        },
    }
    (OUT_DIR / "calibration_v4_2.json").write_text(
        json.dumps(calibration_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "status": payload["status"],
        "reference_clock": reference_report["reference_clock_completion_fraction"],
        "v1_v4_2": v1_report["macro_cycle_completion_fraction"],
        "v1_v4_1": v1_report_v41["macro_cycle_completion_fraction"],
        "best_candidate": best["candidate"] if best else None,
        "best_v4_2": best["macro_completion_v4_2"] if best else None,
        "best_v4_1": best["macro_completion_v4_1"] if best else None,
        "candidates_passing_v4_2": len(passing),
        "point_error_cap_rad": POINT_ERROR_CAP_RAD,
        "outputs": [str((OUT_DIR / "rescore_v4_2.json").relative_to(ROOT)),
                    str((OUT_DIR / "calibration_v4_2.json").relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
