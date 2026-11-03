"""Freeze rollout progression rules from held-out validation trajectories only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_phase_study import calibrate_cycle_progression


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
OUTPUT_DIR = ROOT / "results" / "g1_phase_gate_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dataset_path = DATA_DIR / "phase_dataset.npz"
    prediction_path = DATA_DIR / "phase_predictions.npz"
    checkpoint_path = DATA_DIR / "phase_encoder_checkpoint.pt"
    dataset = np.load(dataset_path, allow_pickle=True)
    predictions = np.load(prediction_path, allow_pickle=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("phase_unit")) != "radians":
        raise ValueError("checkpoint phase unit must be radians")
    if checkpoint.get("output_convention") != "(sin(phi), cos(phi))":
        raise ValueError("checkpoint phase vector convention is incompatible")

    calibration = calibrate_cycle_progression(
        predictions["val_pred"],
        dataset["sequence_validation"],
        dataset["frame_validation"],
        dataset["left_touchdown_validation"].astype(bool),
        task_resolution_rad=float(np.pi / 6.0),
        quantile=0.95,
        validation_minimum_completion_fraction=0.90,
    )
    result = {
        "schema": "g1.phase_rollout_calibration.v1",
        "source_split": "validation_only",
        "dataset_sha256": sha256_file(dataset_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration_script_sha256": sha256_file(Path(__file__)),
        "evaluation_script_sha256": sha256_file(ROOT / "scripts" / "evaluate_phase_study.py"),
        "contact_rule": {
            "min_contact_frames": 2,
            "min_release_frames": 2,
            "source": "src/tasks/velocity/phasor/data.py:ContactDetectionConfig",
        },
        "completion_rule": "positive_net_progression; absolute one-turn error within frozen tolerance; per-step forward/reverse limits within validation quantiles",
        "macro_completion_fraction_required": 0.90,
        **calibration,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "rollout_gate_calibration.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
