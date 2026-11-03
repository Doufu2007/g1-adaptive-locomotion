import sys
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = WORKSPACE_ROOT / "paper"
sys.path.insert(0, str(PAPER_DIR))

from freeze_phasor_protocol import evaluation_scenarios, training_run_order


class PhasorProtocolAssetTests(unittest.TestCase):
    def test_evaluation_scenarios_are_fixed_balanced_and_in_range(self):
        rows = evaluation_scenarios()
        self.assertEqual(rows, evaluation_scenarios())
        self.assertEqual([row["scenario_id"] for row in rows], list(range(1001, 1101)))
        self.assertEqual(len({row["reset_seed"] for row in rows}), 100)

        standing = [row for row in rows if row["scenario_type"] == "standing"]
        moving = [row for row in rows if row["scenario_type"] == "moving"]
        self.assertEqual(len(standing), 5)
        self.assertEqual(len(moving), 95)
        self.assertTrue(
            all(
                row["vx_mps"] == row["vy_mps"] == row["yaw_rate_radps"] == 0.0
                for row in standing
            )
        )

        vx_bins = {int((row["vx_mps"] + 1.0) / 3.0 * 95) for row in moving}
        vy_bins = {int((row["vy_mps"] + 1.0) / 2.0 * 95) for row in moving}
        yaw_bins = {
            int((row["yaw_rate_radps"] + 1.0) / 2.0 * 95) for row in moving
        }
        self.assertEqual(vx_bins, set(range(95)))
        self.assertEqual(vy_bins, set(range(95)))
        self.assertEqual(yaw_bins, set(range(95)))

        for row in rows:
            self.assertGreaterEqual(row["vx_mps"], -1.0)
            self.assertLessEqual(row["vx_mps"], 2.0)
            self.assertGreaterEqual(row["vy_mps"], -1.0)
            self.assertLessEqual(row["vy_mps"], 1.0)
            self.assertGreaterEqual(row["yaw_rate_radps"], -1.0)
            self.assertLessEqual(row["yaw_rate_radps"], 1.0)
            self.assertEqual(
                (row["ood_vx_mps"], row["ood_vy_mps"], row["ood_yaw_rate_radps"]),
                (2.2, 0.0, 1.1),
            )

    def test_training_schedule_is_a_reproducible_complete_block_design(self):
        rows = training_run_order()
        self.assertEqual(rows, training_run_order())
        self.assertEqual([row["run_order"] for row in rows], list(range(1, 16)))

        expected_conditions = {"E0", "E1", "E2", "E3", "E4"}
        for seed in (11, 42, 73):
            block = [row for row in rows if row["training_seed"] == seed]
            self.assertEqual(len(block), 5)
            self.assertEqual({row["condition"] for row in block}, expected_conditions)


if __name__ == "__main__":
    unittest.main()
