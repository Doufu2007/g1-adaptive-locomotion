from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_g1_experiment_startup as startup
from scripts.check_g1_experiment_startup import verify_hash_entries


class G1ExperimentStartupTests(unittest.TestCase):
    def test_preflight_blocks_failed_phase_gate_when_coverage_is_adequate(self) -> None:
        blockers = startup.startup_gate_blockers(
            {"status": "completed", "run_kind": "confirmation"},
            {
                "status": "fail",
                "cycle_coverage_status": "adequate",
                "phase_rollout_gate_pass": False,
            },
        )

        self.assertEqual(blockers, ["independent confirmation phase gate failed"])

    def test_preflight_reports_current_hashes_without_writing_a_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def artifact(name: str) -> Path:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
                return path

            configs = tuple(artifact(f"config-{index}.py") for index in range(3))
            encoder = artifact("encoder.pt")
            reward = artifact("reward.json")
            calibration = artifact("calibration.json")
            matrix = artifact("matrix.md")
            contract = artifact("contract.json")
            failure_tree = artifact("failure-tree.json")
            encoder_candidates = artifact("encoder-candidates.json")
            policy_checkpoint = artifact("policy.pt")
            policy_onnx = artifact("policy.onnx")
            agent_config = artifact("agent.yaml")
            environment_config = artifact("environment.yaml")
            policy_manifest = artifact("policy-manifest.json")
            policy_manifest.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "agent_config_path": str(agent_config),
                        "environment_config_path": str(environment_config),
                    }
                ),
                encoding="utf-8",
            )
            phase_report = artifact("phase-gate.json")
            phase_report.write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "cycle_coverage_status": "adequate",
                        "phase_rollout_gate_pass": False,
                    }
                ),
                encoding="utf-8",
            )
            confirmation = artifact("confirmation-manifest.json")
            confirmation.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_kind": "confirmation",
                        "seed": 20261051,
                        "pilot_manifest_path": str(policy_manifest),
                        "policy_checkpoint_path": str(policy_checkpoint),
                        "policy_onnx_path": str(policy_onnx),
                    }
                ),
                encoding="utf-8",
            )
            lock_path = root / "startup_hashes.json"

            with patch.multiple(
                startup,
                ROOT=root,
                DEFAULT_LOCK=lock_path,
                CONFIG_PATHS=configs,
                ENCODER_CHECKPOINT=encoder,
                REWARD_TEMPLATE=reward,
                CALIBRATION=calibration,
                MATRIX=matrix,
                ANALYSIS_CONTRACT=contract,
                FAILURE_TREE_REPORT=failure_tree,
                ENCODER_CANDIDATE_REPORT=encoder_candidates,
            ):
                result = startup.preflight_startup_hashes(
                    confirmation,
                    phase_report,
                )

            self.assertEqual(result["status"], "blocked")
            self.assertIn("independent confirmation phase gate failed", result["blockers"])
            self.assertIn(
                "encoder_runtime_checkpoint",
                {entry["name"] for entry in result["artifacts"]},
            )
            self.assertFalse(result["lock_written"])
            self.assertFalse(lock_path.exists())

    def test_startup_hash_check_detects_changed_and_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "config.json"
            artifact.write_text("locked", encoding="utf-8")
            digest = hashlib.sha256(b"locked").hexdigest()
            entries = [
                {"name": "config", "path": "config.json", "sha256": digest},
                {"name": "checkpoint", "path": "missing.pt", "sha256": "0" * 64},
            ]

            self.assertEqual(verify_hash_entries(entries[:1], root), [])
            self.assertEqual(
                verify_hash_entries(entries, root),
                ["checkpoint: missing artifact missing.pt"],
            )
            artifact.write_text("changed", encoding="utf-8")
            self.assertEqual(
                verify_hash_entries(entries[:1], root),
                ["config: SHA-256 mismatch for config.json"],
            )


if __name__ == "__main__":
    unittest.main()
