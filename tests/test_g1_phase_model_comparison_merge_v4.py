import json
import unittest

from scripts import merge_g1_phase_model_comparison_v4 as merge
from scripts.run_g1_tctst_residual_v4_comparison import PRETRAIN_DIR


class TestG1PhaseModelComparisonMergeV4(unittest.TestCase):
    def test_seed_20261023_points_to_the_completed_recovery_run(self):
        expected = (
            merge.OUT
            / "recovery_seed_20261023_attempt01"
            / "tctst_repaired_seed_20261023"
        )

        self.assertEqual(merge.TCTST_RUNS[20261023], expected)
        self.assertTrue((expected / "completed.json").is_file())

    def test_seed_20261022_points_to_a_usable_recovery_run(self):
        run_dir = merge.TCTST_RUNS[20261022]
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["seed"], 20261022)
        self.assertEqual(manifest["model"], "tctst_repaired")
        self.assertIn(manifest["status"], {"RUNNING", "COMPLETED"})

    def test_merge_reads_the_frozen_pretraining_verdict(self):
        loader = getattr(merge, "load_pretraining_verdict", None)
        self.assertTrue(callable(loader), "merge must expose the pretraining artifact it consumes")
        producer_path, verdict = loader()

        self.assertEqual(producer_path, PRETRAIN_DIR / "verdict.json")
        self.assertEqual(verdict["verdict"], "PRETRAIN_COLLAPSE")


if __name__ == "__main__":
    unittest.main()
