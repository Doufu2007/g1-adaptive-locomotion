"""Evaluate one historical full baseline checkpoint as a controller scout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import onnxruntime as ort
import yaml

from scripts.collect_g1_phase_confirmation import collect
from scripts.run_g1_controller_feasibility_pilot import (
    CALIBRATION,
    CONFIRMATION_ROOT,
    HELDOUT_REPORT,
    _evaluate_and_save,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUN = ROOT / "logs" / "rsl_rl" / "g1_velocity" / "2026-08-10_09-35-45_baseline_full"
POLICY_PATH = TRAINING_RUN / "policy.onnx"
CHECKPOINT_PATH = TRAINING_RUN / "model_9900.pt"
AGENT_CONFIG_PATH = TRAINING_RUN / "params" / "agent.yaml"
ENV_CONFIG_PATH = TRAINING_RUN / "params" / "env.yaml"
SOURCE_MANIFEST_PATH = ROOT / "results" / "g1_controller_feasibility_20261019" / "historical_full_baseline_seed42_source.json"
PILOT_ROOT = ROOT / "results" / "g1_controller_feasibility_20261019"
ENVIRONMENT_SEED = 20261050
CONFIRMATION_SEED = 20261051


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest() -> dict[str, Any]:
    for path in (POLICY_PATH, CHECKPOINT_PATH, AGENT_CONFIG_PATH, ENV_CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    agent_config = yaml.load(AGENT_CONFIG_PATH.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if int(agent_config.get("seed", -1)) != 42:
        raise ValueError("the historical baseline agent config no longer identifies seed 42")

    session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    metadata = session.get_modelmeta().custom_metadata_map
    if (
        len(inputs) != 1
        or inputs[0].name != "obs"
        or inputs[0].shape != [1, 98]
        or len(outputs) != 1
        or outputs[0].name != "actions"
        or outputs[0].shape != [1, 29]
    ):
        raise ValueError("historical baseline ONNX input/output contract does not match the collector")
    expected_observations = "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
    if metadata.get("observation_names") != expected_observations:
        raise ValueError("historical baseline ONNX observation order does not match the current fixed-clock actor")

    source = {
        "schema": "g1.controller_policy_source.v1",
        "status": "completed",
        "condition": "baseline",
        "task_id": "Unitree-G1-Flat",
        "policy_name": "historical_full_fixed_clock_baseline",
        "seed": 42,
        "training_iterations": 9900,
        "source_type": "historical_full_baseline_checkpoint_scout",
        "training_run_dir": str(TRAINING_RUN.resolve()),
        "policy_onnx_path": str(POLICY_PATH.resolve()),
        "policy_onnx_sha256": _sha256_file(POLICY_PATH),
        "policy_checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "policy_checkpoint_sha256": _sha256_file(CHECKPOINT_PATH),
        "agent_config_path": str(AGENT_CONFIG_PATH.resolve()),
        "agent_config_sha256": _sha256_file(AGENT_CONFIG_PATH),
        "environment_config_path": str(ENV_CONFIG_PATH.resolve()),
        "environment_config_sha256": _sha256_file(ENV_CONFIG_PATH),
        "onnx_input": {"name": inputs[0].name, "shape": inputs[0].shape},
        "onnx_output": {"name": outputs[0].name, "shape": outputs[0].shape},
        "observation_names": metadata.get("observation_names"),
        "joint_names": metadata.get("joint_names"),
        "action_scale": metadata.get("action_scale"),
        "evaluation_role": "exploratory controller feasibility only; not independent confirmation",
    }
    SOURCE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_MANIFEST_PATH.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return source


def _run_and_evaluate(run_dir: Path, *, seed: int, run_kind: str) -> dict[str, Any]:
    manifest = collect(
        run_dir,
        seed=seed,
        policy_source_path=SOURCE_MANIFEST_PATH,
        run_kind=run_kind,
    )
    phase_report = _evaluate_and_save(run_dir)
    return {
        "run_kind": run_kind,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "phase_report_path": str((run_dir / "phase_gate.json").resolve()),
        "termination_count": int(manifest["termination_count"]),
        "truncation_count": int(manifest["truncation_count"]),
        "coverage_status": phase_report["cycle_coverage_status"],
        "condition_coverage": phase_report["condition_coverage"],
        "complete_cycle_count": int(sum(
            row["complete_cycle_count"] for row in phase_report["condition_coverage"].values()
        )),
        "phase_gate_status": phase_report["phase_gate_status"],
        "phase_rollout_gate_pass": phase_report["phase_rollout_gate_pass"],
        "rollout_sha256": phase_report["rollout_sha256"],
    }


def run_scout() -> dict[str, Any]:
    if not CALIBRATION.is_file() or not HELDOUT_REPORT.is_file():
        raise FileNotFoundError("the frozen v3 phase gates must exist before the controller scout")
    source = _source_manifest()
    pilot_dir = PILOT_ROOT / f"historical_full_baseline_seed42_pilot_seed_{ENVIRONMENT_SEED}_attempt_02"
    pilot = _run_and_evaluate(pilot_dir, seed=ENVIRONMENT_SEED, run_kind="controller_feasibility_pilot")

    result: dict[str, Any] = {
        "schema": "g1.historical_controller_scout.v1",
        "status": "controller_infeasible",
        "hypothesis": "the three recent 1000-iteration pilot policies fall because they have not learned a stable gait; the historical 9900-iteration fixed-clock baseline is an existing checkpoint with a matching observation/action contract and is evaluated as a single-variable controller scout",
        "source_manifest_path": str(SOURCE_MANIFEST_PATH.resolve()),
        "source_manifest_sha256": _sha256_file(SOURCE_MANIFEST_PATH),
        "policy_name": source["policy_name"],
        "policy_seed": source["seed"],
        "training_iterations": source["training_iterations"],
        "environment_seed": ENVIRONMENT_SEED,
        "controller_feasibility_pilot": pilot,
        "independent_confirmation": None,
        "formal_ppo_started": False,
        "encoder_retrained": False,
    }

    if pilot["coverage_status"] == "adequate":
        confirmation_dir = CONFIRMATION_ROOT / f"historical_full_baseline_seed42_confirm_seed_{CONFIRMATION_SEED}"
        confirmation = _run_and_evaluate(
            confirmation_dir,
            seed=CONFIRMATION_SEED,
            run_kind="confirmation",
        )
        result["independent_confirmation"] = confirmation
        result["status"] = (
            "independent_confirmation_passed"
            if confirmation["coverage_status"] == "adequate" and confirmation["phase_rollout_gate_pass"] is True
            else "independent_confirmation_failed"
        )

    output_path = PILOT_ROOT / "historical_controller_scout.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_scout(), indent=2, ensure_ascii=False))
