from __future__ import annotations

import unittest

from scripts.audit_g1_phase_configs import audit_configurations


class G1PhaseConfigAuditTests(unittest.TestCase):
    def test_registered_factor_matrix_uses_frozen_runtime_artifacts(self) -> None:
        report = audit_configurations(write_report=False)

        self.assertEqual(report["status"], "PASS", report["checks"])
        self.assertEqual(
            set(report["groups"]),
            {"fixed_clock", "actor_ablation", "reward_ablation", "joint_phase", "random_vector"},
        )
        expected = {
            "fixed_clock": ("fixed_clock", "fixed_clock"),
            "actor_ablation": ("learned", "fixed_clock"),
            "reward_ablation": ("fixed_clock", "learned"),
            "joint_phase": ("learned", "learned"),
            "random_vector": ("random_vector", "fixed_clock"),
        }
        budget_hashes = set()
        checkpoint_hashes = set()
        template_hashes = set()
        for name, (actor_mode, reward_mode) in expected.items():
            row = report["groups"][name]
            self.assertTrue(row["registered"], name)
            self.assertEqual(row["actor_mode"], actor_mode)
            self.assertEqual(row["reward_mode"], reward_mode)
            self.assertTrue(row["critic_fixed_clock"], name)
            budget_hashes.add(row["training_budget_sha256"])
            if actor_mode == "learned" or reward_mode == "learned":
                checkpoint_hashes.add(row["checkpoint_sha256"])
                template_hashes.add(row["reward_template_sha256"])

        self.assertEqual(len(budget_hashes), 1)
        self.assertEqual(len(checkpoint_hashes), 1)
        self.assertEqual(len(template_hashes), 1)


if __name__ == "__main__":
    unittest.main()
