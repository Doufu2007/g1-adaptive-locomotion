"""Wrap the architecture-sweep TCN weights into a G1 runtime checkpoint.

The sweep at ``results/g1_contact_followup_20261021/run_v1_replay_optimized/``
saved a bare ``state_dict``.  The runtime loader in
``src/tasks/velocity/mdp/observations.py`` requires the frozen contract fields
(``joint_dim``, ``window``, ``joint_names``, ``mean``/``std``,
``output_convention`` and ``runtime_contract``).  This script adds them without
retraining and without touching the source weights.

Every provenance hash that describes the *data* or the *reward template* is
copied from the existing v1 runtime checkpoint, because the TCN arm was trained
on the same dataset with the same statistics.  ``source_checkpoint_sha256`` is
recomputed from the TCN file itself.

Refuses to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.data import G1_JOINT_ORDER  # noqa: E402
from src.tasks.velocity.phasor.encoder_tcn import TCNEncoder  # noqa: E402
V1_RUNTIME = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_encoder_runtime_v1.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "results"
        / "g1_contact_followup_20261021"
        / "run_v1_replay_optimized"
        / "filtered_train_tcn_seed_20261021.pt",
    )
    parser.add_argument("--reference-runtime", type=Path, default=V1_RUNTIME)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    reference = args.reference_runtime.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing checkpoint: {output}")

    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("source must be a state_dict mapping")

    model = TCNEncoder(joint_dim=len(G1_JOINT_ORDER))
    model.load_state_dict(state, strict=True)
    model.eval()

    ref = torch.load(reference, map_location="cpu", weights_only=False)
    if ref.get("output_convention") != "(sin(phi), cos(phi))":
        raise ValueError("reference runtime checkpoint has an unexpected output convention")
    if tuple(ref.get("joint_names", ())) != G1_JOINT_ORDER:
        raise ValueError("reference runtime checkpoint joint order does not match the G1 order")

    mean = torch.as_tensor(ref["train_mean"], dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(ref["train_std"], dtype=torch.float32).reshape(-1)
    if tuple(mean.shape) != (len(G1_JOINT_ORDER),) or tuple(std.shape) != (len(G1_JOINT_ORDER),):
        raise ValueError("reference normalization does not match the 29-joint contract")
    if not torch.all(std > 0):
        raise ValueError("reference standard deviations must be positive")

    source_sha = sha256_file(source)
    contract: dict[str, Any] = {
        "schema": "g1.phase_runtime_checkpoint.v1",
        "source_checkpoint_sha256": source_sha,
        "dataset_sha256": ref["dataset_sha256"],
        "metadata_sha256": ref["runtime_contract"]["metadata_sha256"],
        "manifest_sha256": ref["data_manifest_sha256"],
        "reward_template_sha256": ref["runtime_contract"]["reward_template_sha256"],
        "joint_order": list(G1_JOINT_ORDER),
        "joint_dim": len(G1_JOINT_ORDER),
        "window": 10,
        "phase_output_convention": "(sin(phi), cos(phi))",
    }

    checkpoint = {
        "state_dict": state,
        "architecture": "tcn",
        "tcn_hidden": 64,
        "joint_dim": len(G1_JOINT_ORDER),
        "window": 10,
        "preserve_time": False,
        "selected_architecture": "tcn_causal_residual",
        "train_mean": mean,
        "train_std": std,
        "mean": mean,
        "std": std,
        "seed": 20261021,
        "phase_unit": "radians",
        "output_convention": "(sin(phi), cos(phi))",
        "data_manifest_sha256": ref["data_manifest_sha256"],
        "dataset_sha256": ref["dataset_sha256"],
        "train_sequences": ref["train_sequences"],
        "validation_sequences": ref["validation_sequences"],
        "test_sequences": ref["test_sequences"],
        "joint_names": list(G1_JOINT_ORDER),
        "runtime_contract": contract,
        "source_checkpoint_path": str(source.relative_to(ROOT)),
        "source_checkpoint_sha256": source_sha,
        "note": (
            "Weights from the 2026-10-21 architecture sweep, window 10, filtered "
            "training subset, 8 epochs. Contract fields copied from "
            "phase_encoder_runtime_v1.pt because the data and reward template are "
            "unchanged. Gate score 0.4804 (v1: 0.4163); still below the 0.90 bar, "
            "so this does not authorise a PPO run."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    written = torch.load(output, map_location="cpu", weights_only=False)
    TCNEncoder(joint_dim=len(G1_JOINT_ORDER)).load_state_dict(written["state_dict"], strict=True)
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": source_sha,
        "contract_schema": contract["schema"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
