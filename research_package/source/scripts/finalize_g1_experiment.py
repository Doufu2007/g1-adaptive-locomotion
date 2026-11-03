"""Assemble the final machine-readable status for the G1 phase study."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/g1_phase_dataset_20261019"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    data = load("dataset_metadata.json")
    phase = load("phase_encoder_report.json")
    rollout = load("onpolicy_phase_gate.json")
    config = load("config_audit.json")
    status = {
        "schema": "g1.phase_experiment_status.v1",
        "data_gate": {"status": data.get("status"), "provenance": "docs/data/g1_phase_data_provenance.json"},
        "heldout_phase_gate": {
            "status": "PASS" if phase["test_report"]["phase_pass"] else "FAIL",
            "circular_error_rad": phase["test_report"]["circular_phase_error_rad"],
            "fixed_clock_error_rad": phase["test_report"]["fixed_clock_circular_phase_error_rad"],
            "threshold_rad": phase["phase_gate_threshold_rad"],
        },
        "onpolicy_phase_gate": {
            "status": "PASS" if rollout["gate"]["phase_rollout_gate_pass"] else "FAIL",
            "circular_error_rad": rollout["macro_circular_phase_error_rad"],
            "complete_cycle_fraction": rollout["macro_cycle_completed_one_turn_fraction"],
            "rule": rollout["progression_gate_rule"],
        },
        "configuration_audit": {"status": "PASS" if config["configuration_differences_audited"] else "FAIL", "artifact": "config_audit.json"},
        "gpu_smoke": {"status": "FAIL", "reason": "mjlab installation missing scene/scene.xml", "artifact": "gpu_smoke.log"},
        "formal_ppo": {"status": "NOT_RUN", "reason": "on-policy phase progression gate failed"},
        "control_claim": "BLOCKED",
        "control_metrics_available": False,
        "interpretation": "Offline held-out phase is better than fixed clock, but the predicted rollout phase does not continuously wind through gait cycles.",
    }
    (OUT / "experiment_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
