"""Fit phase alignment on one gait-diagnostic seed and score another seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tasks.velocity.phasor.diagnostics import evaluate_phase_alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path, Path]:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trajectory_name = summary.get("trajectory_file")
    if not trajectory_name or Path(trajectory_name).name != trajectory_name:
        raise ValueError(f"invalid trajectory_file in {summary_path}")
    trajectory_path = directory / trajectory_name
    actual_hash = sha256_file(trajectory_path)
    if summary.get("trajectory_sha256") != actual_hash:
        raise ValueError(f"trajectory hash mismatch: {trajectory_path}")

    with np.load(trajectory_path, allow_pickle=False) as archive:
        if not {"phase", "contacts"}.issubset(archive.files):
            raise ValueError(f"trajectory bundle lacks phase/contact arrays: {trajectory_path}")
        arrays = {
            "phase": np.asarray(archive["phase"], dtype=np.float64),
            "contacts": np.asarray(archive["contacts"], dtype=bool),
        }
    return summary, arrays, summary_path, trajectory_path


def _locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def build_report(
    calibration_dir: Path,
    heldout_dir: Path,
    grid_size: int = 1000,
) -> dict[str, Any]:
    calibration, calibration_arrays, calibration_summary_path, calibration_trajectory_path = (
        _load_bundle(calibration_dir)
    )
    heldout, heldout_arrays, heldout_summary_path, heldout_trajectory_path = _load_bundle(
        heldout_dir
    )

    if calibration.get("seed") == heldout.get("seed"):
        raise ValueError("calibration and held-out bundles must use different simulator seeds")
    for key in ("checkpoint_sha256", "encoder_sha256"):
        if calibration.get(key) != heldout.get(key):
            raise ValueError(f"calibration and held-out bundles have different {key}")

    offsets = (0.0, 0.5)
    threshold = 0.56
    alignment = evaluate_phase_alignment(
        calibration_arrays["phase"],
        calibration_arrays["contacts"],
        heldout_arrays["phase"],
        heldout_arrays["contacts"],
        offsets=offsets,
        threshold=threshold,
        grid_size=grid_size,
    )
    input_paths = (
        ("calibration_summary", calibration_summary_path),
        ("calibration_trajectories", calibration_trajectory_path),
        ("heldout_summary", heldout_summary_path),
        ("heldout_trajectories", heldout_trajectory_path),
    )

    return {
        "schema": "phasor.phase_alignment_report.v1",
        "analysis_method": (
            "Fit one phase direction and global rotation on the calibration seed; "
            "apply the same transform unchanged to the held-out simulator seed."
        ),
        "calibration_seed": calibration["seed"],
        "heldout_seed": heldout["seed"],
        "checkpoint_sha256": calibration["checkpoint_sha256"],
        "encoder_sha256": calibration["encoder_sha256"],
        "phase_direction": alignment["phase_direction"],
        "phase_shift_cycles": alignment["phase_shift_cycles"],
        "threshold": threshold,
        "offsets": list(offsets),
        "grid_size": grid_size,
        "calibration_contact_agreement": alignment[
            "calibration_contact_agreement"
        ],
        "heldout_contact_agreement": alignment["heldout_contact_agreement"],
        "heldout_contact_agreement_by_env": alignment[
            "heldout_contact_agreement_by_env"
        ],
        "input_artifacts": [
            {
                "role": role,
                "locator": _locator(path),
                "sha256": f"sha256:{sha256_file(path)}",
            }
            for role, path in input_paths
        ],
        "interpretation_limit": (
            "Calibration agreement is measured on the same simulator seed used to fit "
            "the transform. Held-out traces share one frozen policy checkpoint and "
            "are not independent policy-training replicates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=1000)
    args = parser.parse_args()

    report = build_report(args.calibration_dir, args.heldout_dir, args.grid_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
