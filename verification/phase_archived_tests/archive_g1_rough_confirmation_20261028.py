import importlib
import unittest

import numpy as np


class ConfirmationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = importlib.import_module('scripts.g1_rough_confirmation_contract_20261028')
        cls.ex = importlib.import_module('scripts.g1_rough_confirmation_experiment_20261028')

    def test_method_mapping_and_disjoint_design(self):
        self.assertEqual(self.c.base_method('RE_live'), 'RE')
        self.assertEqual(self.c.base_method('RE_frozen'), 'RE')
        self.assertEqual(self.c.base_method('RC_clock'), 'RC')
        self.assertEqual(self.c.base_method('FC_clock'), 'FC')
        self.assertEqual(set(self.c.TERRAINS).intersection(self.c.EVAL_SEEDS), set())
        self.assertEqual(len(self.c.build_jobs()), 12)
        self.assertEqual(len(self.c.world_slots()), 15)
        self.assertEqual(len({(j['method'], j['terrain_seed']) for j in self.c.build_jobs()}), 12)

    def test_frozen_phase_reuses_episode_initial_value(self):
        import torch
        first = torch.tensor([[1., 0.], [0., 1.]])
        cached = self.ex.freeze_phase(first, None, torch.tensor([True, True]))
        changed = torch.tensor([[0., 1.], [-1., 0.]])
        used = self.ex.freeze_phase(changed, cached, torch.tensor([False, False]))
        torch.testing.assert_close(used, first)
        reset = self.ex.freeze_phase(changed, used, torch.tensor([True, False]))
        torch.testing.assert_close(reset, torch.tensor([[0., 1.], [0., 1.]]))

    def test_frozen_phase_rejects_bad_shape_and_nonfinite(self):
        import torch
        with self.assertRaises(ValueError):
            self.ex.freeze_phase(torch.zeros(2, 3), None, torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.ex.freeze_phase(torch.tensor([[float('nan'), 0.]]), None, torch.ones(1, dtype=torch.bool))

    def test_paired_analysis_rejects_incomplete_rows(self):
        with self.assertRaises(ValueError):
            self.c.paired_analysis([])

    def test_import_has_no_sampling_side_effect(self):
        self.assertTrue(hasattr(self.ex, 'rollout'))
        self.assertFalse((self.ex.OUT / 'queue_started.json').exists())


if __name__ == '__main__':
    unittest.main()
