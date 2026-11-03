"""Run the preregistered fixed-clock controller feasibility pilot."""

from __future__ import annotations

import json
import hashlib
import traceback
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np

from scripts.collect_g1_phase_confirmation import collect
from scripts.evaluate_g1_phase_confirmation import evaluate_confirmation
from src.tasks.velocity.phasor.confirmation import choose_feasible_policy_candidate


ROOT = Path(__file__).resolve().parents[1]
POLICY_RUNS = ROOT / "results" / "phasor_pilot_20261016" / "runs"
PILOT_ROOT = ROOT / "results" / "g1_controller_feasibility_20261019"
CONFIRMATION_ROOT = ROOT / "results" / "g1_phase_confirmation_20261019"
ENVIRONMENT_SEED = 20261050
CONFIRMATION_SEED = 20261051
POLICY_SEEDS = (20261031, 20261032, 20261033)
CHECKPOINT = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_encoder_runtime_v1.pt"
CALIBRATION = ROOT / "results" / "g1_phase_gate_v3" / "rollout_gate_calibration.json"
HELDOUT_REPORT = ROOT / "results" / "g1_phase_gate_v3" / "phase_encoder_report_macro.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_manifest(policy_seed: int) -> Path:
    path = POLICY_RUNS / f"pilot_baseline_seed_{policy_seed}_attempt_03" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing preregistered policy candidate manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("condition") != "baseline"
        or manifest.get("task_id") != "Unitree-G1-Flat"
        or int(manifest.get("seed", -1)) != policy_seed
    ):
        raise ValueError(f"candidate manifest does not match fixed-clock seed {policy_seed}: {path}")
    return path


def _evaluate_and_save(run_dir: Path) -> dict[str, Any]:
    rollout_path = run_dir / "trajectories.npz"
    manifest_path = run_dir / "manifest.json"
    report, predictions = evaluate_confirmation(
        rollout_path,
        CHECKPOINT,
        CALIBRATION,
        HELDOUT_REPORT,
    )
    prediction_path = run_dir / "phase_predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    calibration_path = CALIBRATION
    confirmation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = CHECKPOINT
    report.update({
        "schema": "g1.phase_confirmation_gate.v1",
        "run_kind": json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["run_kind"],
        "status": report["phase_gate_status"],
        "decision": (
            "PASS_PHASE_CONFIRMATION" if report["phase_gate_status"] == "pass" else
            "INCONCLUSIVE_CYCLE_COVERAGE" if report["phase_gate_status"] == "insufficient_coverage" else
            "FAIL_PHASE_CONFIRMATION"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rollout_path": str(rollout_path.resolve()),
        "rollout_sha256": _sha256_file(rollout_path),
        "rollout_manifest_sha256": _sha256_file(manifest_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "policy_checkpoint_path": confirmation_manifest["policy_checkpoint_path"],
        "policy_onnx_path": confirmation_manifest["policy_onnx_path"],
        "pilot_manifest_path": confirmation_manifest["pilot_manifest_path"],
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": _sha256_file(calibration_path),
        "heldout_report_path": str(HELDOUT_REPORT.resolve()),
        "heldout_report_sha256": _sha256_file(HELDOUT_REPORT),
        "formal_ppo_started": False,
    })
    (run_dir / "phase_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def run_pilot() -> dict[str, Any]:
    if not CALIBRATION.is_file() or not HELDOUT_REPORT.is_file():
        raise FileNotFoundError("the v3 validation-only phase gate must be frozen before controller trials")
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_reports: list[dict[str, Any]] = []
    candidate_paths: dict[int, Path] = {}

    for policy_seed in POLICY_SEEDS:
        manifest_path = _find_manifest(policy_seed)
        candidate_paths[policy_seed] = manifest_path
        run_dir = PILOT_ROOT / f"policy_seed_{policy_seed}_pilot_seed_{ENVIRONMENT_SEED}"
        result: dict[str, Any] = {
            "policy_seed": policy_seed,
            "manifest_path": str(manifest_path),
            "run_dir": str(run_dir),
        }
        try:
            manifest = collect(
                run_dir,
                seed=ENVIRONMENT_SEED,
                pilot_manifest_path=manifest_path,
                run_kind="controller_feasibility_pilot",
            )
            phase_report = _evaluate_and_save(run_dir)
            condition_coverage = phase_report["condition_coverage"]
            result.update({
                "status": "completed",
                "termination_count": int(manifest["termination_count"]),
                "truncation_count": int(manifest["truncation_count"]),
                "coverage_status": phase_report["cycle_coverage_status"],
                "complete_cycle_count": int(sum(
                    row["complete_cycle_count"] for row in condition_coverage.values()
                )),
                "condition_coverage": condition_coverage,
                "phase_gate_status": phase_report["phase_gate_status"],
                "rollout_manifest_sha256": manifest["trajectory_sha256"],
            })
        except Exception as error:
            result.update({
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "coverage_status": "insufficient_coverage",
                "termination_count": 10**9,
                "complete_cycle_count": 0,
            })
        candidate_reports.append(result)
        (PILOT_ROOT / "candidate_reports.json").write_text(
            json.dumps(candidate_reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    selected_seed = choose_feasible_policy_candidate(candidate_reports)
    result: dict[str, Any] = {
        "schema": "g1.controller_feasibility_pilot.v1",
        "status": "candidate_feasible" if selected_seed is not None else "no_feasible_candidate",
        "environment_seed": ENVIRONMENT_SEED,
        "policy_candidates": candidate_reports,
        "selection_rule": "adequate 10-cycle/2-episode coverage in all four conditions, then fewer terminations, then more complete cycles, then lower policy seed",
        "selected_policy_seed": selected_seed,
        "formal_ppo_started": False,
        "encoder_retrained": False,
    }

    if selected_seed is not None:
        manifest_path = candidate_paths[selected_seed]
        confirmation_dir = CONFIRMATION_ROOT / f"fixed_clock_seed_{CONFIRMATION_SEED}_policy_{selected_seed}"
        confirmation_manifest = collect(
            confirmation_dir,
            seed=CONFIRMATION_SEED,
            pilot_manifest_path=manifest_path,
            run_kind="confirmation",
        )
        confirmation_report = _evaluate_and_save(confirmation_dir)
        result["independent_confirmation"] = {
            "seed": CONFIRMATION_SEED,
            "policy_seed": selected_seed,
            "manifest_path": str(confirmation_dir / "manifest.json"),
            "termination_count": int(confirmation_manifest["termination_count"]),
            "truncation_count": int(confirmation_manifest["truncation_count"]),
            "cycle_coverage_status": confirmation_report["cycle_coverage_status"],
            "phase_gate_status": confirmation_report["phase_gate_status"],
            "condition_coverage": confirmation_report["condition_coverage"],
            "formal_ppo_started": False,
        }

    output_path = PILOT_ROOT / "feasibility_report.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_pilot(), indent=2, ensure_ascii=False))
