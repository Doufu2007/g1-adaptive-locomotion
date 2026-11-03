"""Configuration boundary tests; no performance assertions on random training."""
import copy
from types import SimpleNamespace
import unittest

from scripts.diagnose_g1_speed_priority_20261026 import configure_speed


class TreatmentBoundary(unittest.TestCase):
    def test_exact_single_weight_change(self):
        cfg=SimpleNamespace(rewards={name:SimpleNamespace(weight=w,params={'std':.5})
            for name,w in [('track_linear_velocity',1.),('pose',1.),('action_rate_l2',-.05)]},other=[3,7])
        expected=copy.deepcopy(cfg)
        expected.rewards['track_linear_velocity'].weight=2.
        self.assertEqual(vars(configure_speed(cfg)),vars(expected))

    def test_reject_wrong_baseline(self):
        for value in (0.,.5,2.,float('nan')):
            cfg=SimpleNamespace(rewards={'track_linear_velocity':SimpleNamespace(weight=value)})
            with self.assertRaises(AssertionError):
                configure_speed(cfg)

    def test_extra_reward_terms_preserved(self):
        for n in range(1,10):
            rewards={str(i):SimpleNamespace(weight=i*.17) for i in range(n)}
            cfg=SimpleNamespace(rewards=dict(rewards,track_linear_velocity=SimpleNamespace(weight=1.)))
            configure_speed(cfg)
            self.assertEqual({k:v for k,v in cfg.rewards.items() if k!='track_linear_velocity'},rewards)


if __name__=='__main__':
    unittest.main()
