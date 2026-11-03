"""Freeze the v4 period-level calibration and audit the existing v1 encoder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.phase_gate_v4 import calibrate_phase_gate_v4, score_phase_cycles_v4


V4_DIR = ROOT / "results" / "g1_phase_gate_v4"
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
DATA_PATH = DATA_DIR / "phase_dataset.npz"
PREDICTIONS_PATH = DATA_DIR / "phase_predictions.npz"
CHECKPOINT_PATH = DATA_DIR / "phase_encoder_checkpoint.pt"
PROVENANCE_PATH = ROOT / "docs" / "data" / "g1_phase_data_provenance.json"
V3_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v3" / "rollout_gate_calibration.json"
ROLLOUT_PATH = (
    ROOT / "results" / "g1_phase_confirmation_20261019"
    / "historical_full_baseline_seed42_confirm_seed_20261051"
    / "phase_predictions.npz"
)
CONTRACT_PATH = V4_DIR / "gate_contract_v4.json"
GATE_MODULE_PATH = ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4.py"
CALIBRATION_SCRIPT_PATH = ROOT / "scripts" / "calibrate_g1_phase_gate_v4.py"
V2_TRAINER_PATH = ROOT / "scripts" / "train_g1_phase_encoder_v2.py"
OUTPUT_PATH = V4_DIR / "calibration_v4_amendment1.json"
AUDIT_PATH = V4_DIR / "current_encoder_v1_v4_audit_amendment1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _without_cycles(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "cycle_records"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    required = [
        DATA_PATH, PREDICTIONS_PATH, CHECKPOINT_PATH, PROVENANCE_PATH,
        V3_CALIBRATION_PATH, CONTRACT_PATH, GATE_MODULE_PATH,
        CALIBRATION_SCRIPT_PATH, V2_TRAINER_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required v4 calibration inputs are missing: " + ", ".join(missing))

    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in required}
    if hashes[str(DATA_PATH.relative_to(ROOT))] != "e204c9dd0a48527ac4c82cfd0c539fba7d584f1faea6242ad0eba66970008b38":
        raise ValueError("source dataset hash differs from the frozen experiment input")
    if hashes[str(CHECKPOINT_PATH.relative_to(ROOT))] != "f8dbc6e78c16468308cd72169262fb82daab4816e708fe1b0d83bdea2266ce6b":
        raise ValueError("v1 checkpoint hash differs from the recorded source checkpoint")

    dataset = np.load(DATA_PATH, allow_pickle=False)
    predictions = np.load(PREDICTIONS_PATH, allow_pickle=False)
    sequence_ids = dataset["sequence_validation"]
    frame_ids = dataset["frame_validation"]
    reference_phase = dataset["y_validation_cycles"].astype(np.float64) * (2.0 * np.pi)
    touchdowns = dataset["left_touchdown_validation"].astype(bool)
    validation_prediction = predictions["val_pred"]
    if len(validation_prediction) != len(reference_phase):
        raise ValueError("validation prediction and frozen validation rows do not align")

    calibration = calibrate_phase_gate_v4(
        validation_prediction,
        reference_phase,
        sequence_ids,
        frame_ids,
        touchdowns,
        condition_ids=sequence_ids,
        episode_ids=sequence_ids,
        history_valid=np.ones(len(frame_ids), dtype=bool),
        required_conditions=sorted(np.unique(sequence_ids).tolist()),
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    period_cuts = tuple(calibration["period_length_cut_points_frames"])
    reference_validation = score_phase_cycles_v4(
        np.stack((np.sin(reference_phase), np.cos(reference_phase)), axis=-1),
        reference_phase,
        sequence_ids,
        frame_ids,
        touchdowns,
        condition_ids=sequence_ids,
        episode_ids=sequence_ids,
        history_valid=np.ones(len(frame_ids), dtype=bool),
        period_cuts_frames=period_cuts,
        required_conditions=sorted(np.unique(sequence_ids).tolist()),
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    calibration["source_sha256"] = hashes
    calibration["calibration_revision"] = "g1-phase-gate-v4.1-amendment-1"
    calibration["gate_contract_sha256"] = sha256_file(CONTRACT_PATH)
    calibration["source_git"] = {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "worktree_status": "pre-existing dirty state; v4 is additive",
    }
    calibration["reference_clock_calibration"] = _without_cycles(reference_validation)
    calibration["baseline_encoder_checkpoint_sha256"] = hashes[str(CHECKPOINT_PATH.relative_to(ROOT))]
    calibration["baseline_encoder_freeze_decision"] = "REJECTED" if not calibration["encoder_freeze_eligible"] else "ELIGIBLE"
    calibration["calibration_status"] = (
        "READY_REFERENCE_CLOCK_PASS_V1_ENCODER_REJECTED"
        if calibration["reference_clock_pass"] and not calibration["encoder_freeze_eligible"]
        else calibration["calibration_status"]
    )
    calibration["reference_report"] = _without_cycles(calibration["reference_report"])
    calibration["encoder_report"] = _without_cycles(calibration["encoder_report"])
    _write_json(OUTPUT_PATH, calibration)

    validation_audit = score_phase_cycles_v4(
        validation_prediction,
        reference_phase,
        sequence_ids,
        frame_ids,
        touchdowns,
        condition_ids=sequence_ids,
        episode_ids=sequence_ids,
        history_valid=np.ones(len(frame_ids), dtype=bool),
        period_cuts_frames=period_cuts,
        required_conditions=sorted(np.unique(sequence_ids).tolist()),
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    rollout_audit = None
    rollout_hash = None
    if ROLLOUT_PATH.is_file():
        rollout_hash = sha256_file(ROLLOUT_PATH)
        rollout = np.load(ROLLOUT_PATH, allow_pickle=False)
        condition_ids = rollout["condition_ids"]
        env_ids = rollout["env_ids"]
        rollout_audit = score_phase_cycles_v4(
            rollout["predicted_phase"],
            rollout["target_phase"],
            env_ids,
            rollout["frame_indices"],
            rollout["left_touchdown"],
            condition_ids=condition_ids,
            episode_ids=rollout["episode_ids"],
            history_valid=np.ones(len(rollout["frame_indices"]), dtype=bool),
            reference_valid=rollout["target_valid_mask"],
            period_cuts_frames=period_cuts,
            required_conditions=[0, 1, 2, 3],
            minimum_cycles_per_condition=10,
            minimum_cycles_per_length_stratum=10,
            minimum_episode_segments_per_condition=2,
        )

    audit = {
        "schema": "g1.phase_gate_v4_baseline_audit.v1",
        "gate_version": calibration["gate_version"],
        "purpose": "read-only v4 re-score of frozen encoder v1; no v2 tuning input",
        "calibration_sha256": sha256_file(OUTPUT_PATH),
        "validation_prediction_sha256": hashes[str(PREDICTIONS_PATH.relative_to(ROOT))],
        "validation_encoder_v1": validation_audit,
        "historical_confirmation_seed_20261051": {
            "path": str(ROLLOUT_PATH.relative_to(ROOT)),
            "sha256": rollout_hash,
            "report": rollout_audit,
            "source_split": "historical_rollout_audit_only",
        } if rollout_audit is not None else None,
        "v1_freeze_decision": "REJECTED_BELOW_V4_VALIDATION_COMPLETION_REQUIREMENTS",
        "v1_validation_macro_completion": validation_audit["macro_cycle_completion_fraction"],
        "required_macro_completion": 0.90,
    }
    _write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "calibration_status": calibration["calibration_status"],
        "period_length_cut_points_frames": calibration["period_length_cut_points_frames"],
        "reference_clock_completion_fraction": calibration["reference_clock_completion_fraction"],
        "v1_encoder_macro_completion_fraction": validation_audit["macro_cycle_completion_fraction"],
        "v1_freeze_eligible": validation_audit["encoder_freeze_eligible"],
        "rollout_v1_macro_completion_fraction": (
            rollout_audit["macro_cycle_completion_fraction"] if rollout_audit is not None else None
        ),
        "outputs": [str(OUTPUT_PATH.relative_to(ROOT)), str(AUDIT_PATH.relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
