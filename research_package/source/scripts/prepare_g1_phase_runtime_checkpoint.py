"""Bind a trained G1 phase checkpoint to its frozen runtime contract.

This copies the selected weights without retraining, adds the frozen joint
order, and records hashes for the exact training data and reward template.
The source checkpoint is never modified and an existing output is not replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.velocity.phasor.data import G1_JOINT_ORDER


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "results" / "g1_phase_dataset_20261019"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_stats(checkpoint: dict[str, Any]) -> None:
    mean = torch.as_tensor(checkpoint.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(checkpoint.get("std"), dtype=torch.float32)
    if mean.ndim == 2 and mean.shape[0] == 1:
        mean = mean.squeeze(0)
    if std.ndim == 2 and std.shape[0] == 1:
        std = std.squeeze(0)
    if tuple(mean.shape) != (len(G1_JOINT_ORDER),) or tuple(std.shape) != (len(G1_JOINT_ORDER),):
        raise ValueError("checkpoint normalization shape does not match the 29-joint contract")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("checkpoint normalization values must be finite")
    if not torch.all(std > 0):
        raise ValueError("checkpoint standard deviations must be positive")


def build_runtime_checkpoint(
    source_checkpoint_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    reward_template_path: Path,
    output_path: Path,
) -> dict[str, str]:
    """Validate provenance, then write a new runtime checkpoint copy."""
    source_checkpoint_path = source_checkpoint_path.resolve(strict=True)
    dataset_path = dataset_path.resolve(strict=True)
    metadata_path = metadata_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    reward_template_path = reward_template_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace frozen runtime checkpoint: {output_path}")

    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("source checkpoint must contain a mapping")
    if int(checkpoint.get("joint_dim", -1)) != len(G1_JOINT_ORDER) or int(checkpoint.get("window", -1)) != 10:
        raise ValueError("source checkpoint does not match the 10-frame, 29-joint input contract")
    if checkpoint.get("output_convention") != "(sin(phi), cos(phi))":
        raise ValueError("source checkpoint uses an unexpected phase output convention")
    if "state_dict" not in checkpoint:
        raise ValueError("source checkpoint is missing state_dict")
    _validate_stats(checkpoint)

    dataset_sha256 = sha256_file(dataset_path)
    metadata_sha256 = sha256_file(metadata_path)
    manifest_sha256 = sha256_file(manifest_path)
    template_sha256 = sha256_file(reward_template_path)
    template = _json(reward_template_path)
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset hash does not match source checkpoint")
    if checkpoint.get("data_manifest_sha256") != manifest_sha256:
        raise ValueError("manifest hash does not match source checkpoint")
    if template.get("status") != "FROZEN" or template.get("fit_split") != "train":
        raise ValueError("reward template must be frozen from the training split")
    if template.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset hash does not match reward template")
    if template.get("dataset_metadata_sha256") != metadata_sha256:
        raise ValueError("dataset metadata hash does not match reward template")
    if template.get("provenance_manifest_sha256") != manifest_sha256:
        raise ValueError("manifest hash does not match reward template")
    if tuple(template.get("joint_order", ())) != G1_JOINT_ORDER:
        raise ValueError("reward template joint order does not match the G1 runtime contract")
    if sorted(checkpoint.get("train_sequences", [])) != sorted(template.get("train_sequences", [])):
        raise ValueError("training sequence identities differ between checkpoint and reward template")
    if template.get("test_split_used_for_fit_or_validation") not in (None, False):
        raise ValueError("reward template indicates that test data influenced fitting")

    runtime_checkpoint = dict(checkpoint)
    runtime_checkpoint["joint_names"] = list(G1_JOINT_ORDER)
    runtime_checkpoint["runtime_contract"] = {
        "schema": "g1.phase_runtime_checkpoint.v1",
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "dataset_sha256": dataset_sha256,
        "metadata_sha256": metadata_sha256,
        "manifest_sha256": manifest_sha256,
        "reward_template_sha256": template_sha256,
        "joint_order": list(G1_JOINT_ORDER),
        "joint_dim": len(G1_JOINT_ORDER),
        "window": 10,
        "phase_output_convention": "(sin(phi), cos(phi))",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(runtime_checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "source_checkpoint_sha256": runtime_checkpoint["runtime_contract"]["source_checkpoint_sha256"],
        "dataset_sha256": dataset_sha256,
        "metadata_sha256": metadata_sha256,
        "manifest_sha256": manifest_sha256,
        "reward_template_sha256": template_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=ARTIFACT_DIR / "phase_encoder_checkpoint.pt")
    parser.add_argument("--dataset", type=Path, default=ARTIFACT_DIR / "phase_dataset.npz")
    parser.add_argument("--dataset-metadata", type=Path, default=ARTIFACT_DIR / "dataset_metadata.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "data" / "g1_phase_data_provenance.json")
    parser.add_argument("--reward-template", type=Path, default=ARTIFACT_DIR / "g1_reward_template_v1.json")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "phase_encoder_runtime_v1.pt")
    args = parser.parse_args()
    print(json.dumps(build_runtime_checkpoint(
        args.source_checkpoint,
        args.dataset,
        args.dataset_metadata,
        args.manifest,
        args.reward_template,
        args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
