"""Freeze the G1 rollout point-error gate with validation sequence macros."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_phase_study import evaluate_phase_predictions, select_macro_phase_threshold
from src.tasks.velocity.phasor.confirmation import CONDITION_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
OUTPUT_DIR = ROOT / "results" / "g1_phase_gate_v3"
PREVIOUS_CYCLE_CALIBRATION = ROOT / "results" / "g1_phase_gate_v2" / "rollout_gate_calibration.json"
LEGACY_PHASE_REPORT = DATA_DIR / "phase_encoder_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_report(
    predicted: np.ndarray,
    target: np.ndarray,
    sequence_ids: np.ndarray,
    fixed_clock: np.ndarray | None = None,
) -> dict[str, Any]:
    sequence_reports: dict[str, dict[str, Any]] = {}
    for sequence in dict.fromkeys(np.asarray(sequence_ids).tolist()):
        mask = np.asarray(sequence_ids) == sequence
        sequence_reports[str(sequence)] = evaluate_phase_predictions(
            predicted[mask],
            target[mask],
            fixed_clock_phase=None if fixed_clock is None else fixed_clock[mask],
        )
    return {
        "aggregation": "equal-weight mean of per-sequence mean circular errors",
        "sequence_count": len(sequence_reports),
        "sample_count": int(len(predicted)),
        "macro_circular_phase_error_rad": float(np.mean([
            item["circular_phase_error_rad"] for item in sequence_reports.values()
        ])),
        "macro_fixed_clock_circular_phase_error_rad": (
            float(np.mean([
                item["fixed_clock_circular_phase_error_rad"]
                for item in sequence_reports.values()
            ]))
            if fixed_clock is not None else None
        ),
        "per_sequence": sequence_reports,
    }


def calibrate() -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = DATA_DIR / "phase_dataset.npz"
    prediction_path = DATA_DIR / "phase_predictions.npz"
    checkpoint_path = DATA_DIR / "phase_encoder_checkpoint.pt"
    legacy_report_path = LEGACY_PHASE_REPORT
    previous = json.loads(PREVIOUS_CYCLE_CALIBRATION.read_text(encoding="utf-8"))
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=True) as dataset, np.load(
        prediction_path, allow_pickle=False
    ) as predictions:
        threshold = select_macro_phase_threshold(
            predictions["val_pred"],
            predictions["y_val"],
            dataset["sequence_validation"],
            quantile=0.95,
            task_resolution_rad=np.pi / 6.0,
        )
        validation = macro_report(
            predictions["val_pred"],
            predictions["y_val"],
            dataset["sequence_validation"],
        )
        test = macro_report(
            predictions["test_pred"],
            predictions["y_test"],
            dataset["sequence_test"],
            predictions["test_clock"],
        )

    if int(threshold["validation_sequence_count"]) < 4:
        raise ValueError("validation must contain at least four independent sequences")
    if set(CONDITION_NAMES) != {"slow_constant", "medium_constant", "fast_constant", "speed_stand_transition"}:
        raise ValueError("the frozen confirmation conditions changed")
    if float(previous["tolerance_rad"]) != 0.20943951023931953:
        raise ValueError("the existing cycle tolerance is not the expected frozen value")

    calibration = {
        **previous,
        "schema": "g1.phase_rollout_calibration.v3",
        "dataset_sha256": sha256_file(dataset_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration_script_sha256": sha256_file(Path(__file__)),
        "phase_gate_threshold_rad": threshold["phase_gate_threshold_rad"],
        "phase_threshold_quantile": threshold["threshold_quantile"],
        "phase_threshold_source": threshold["threshold_source"],
        "validation_sequence_count": threshold["validation_sequence_count"],
        "validation_sequence_macro_errors_rad": threshold["validation_sequence_macro_errors_rad"],
        "validation_macro_error_rad": threshold["validation_macro_error_rad"],
        "phase_error_aggregation": "per-sequence mean circular error, then equal-weight macro mean across scheduled conditions",
        "minimum_cycles_per_condition": 10,
        "minimum_episodes_per_condition": 2,
        "required_condition_ids": list(range(len(CONDITION_NAMES))),
        "required_condition_names": list(CONDITION_NAMES),
    }
    phase_report = {
        "schema": "g1.phase_encoder_macro_report.v1",
        "source_split": "validation_only_for_threshold; test_evaluated_after_freeze",
        "phase_gate_threshold_rad": threshold["phase_gate_threshold_rad"],
        "phase_threshold_source": threshold["threshold_source"],
        "legacy_pointwise_threshold_rad": legacy_report["phase_gate_threshold_rad"],
        "legacy_threshold_use": "retained for audit; superseded because per-frame q95 and rollout macro mean used different aggregation units",
        "validation_report": validation,
        "test_report": {
            **test,
            "phase_pass": bool(
                test["macro_circular_phase_error_rad"]
                <= threshold["phase_gate_threshold_rad"]
            ),
        },
        "dataset_sha256": sha256_file(dataset_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "legacy_report_sha256": sha256_file(legacy_report_path),
    }
    return calibration, phase_report


def main() -> None:
    calibration, report = calibrate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rollout_gate_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "phase_encoder_report_macro.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"calibration": calibration, "phase_report": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
