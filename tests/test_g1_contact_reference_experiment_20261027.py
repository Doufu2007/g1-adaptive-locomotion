import unittest
try:
    from scripts.g1_contact_reference_experiment_20261027 import check_initialization,paired_differences
except ImportError:
    check_initialization=paired_differences=None

class ContractTest(unittest.TestCase):
    def test_initialization_requires_all_five_exact_fields(self):
        self.assertIsNotNone(check_initialization,'reference experiment interface is not implemented')
        row={k:k for k in ('initial_actor_sha256','initial_critic_sha256','common_cfg_sha256','common_agent_sha256','startup_model_hashes')}
        self.assertTrue(check_initialization(row,dict(row)))
        for key in row:
            other=dict(row);other[key]='different'
            with self.assertRaises(ValueError):check_initialization(row,other)
            other=dict(row);other.pop(key)
            with self.assertRaises(ValueError):check_initialization(row,other)
    def test_effect_pairs_by_training_seed_and_rejects_missing_or_duplicate_episodes(self):
        self.assertIsNotNone(paired_differences,'paired comparisons are not implemented')
        rows=[dict(arm=a,seed=s,eval_seed=e,mae_forward_velocity_mps=v) for s,v in [(1,.1),(2,.3),(3,.5)] for e in (8,9) for a in ('reference','clock')]
        for r in rows:
            if r['arm']=='reference':r['mae_forward_velocity_mps']-=.02
        self.assertTrue(all(abs(x+.02)<1e-9 for x in paired_differences(rows,'reference','clock')))
        with self.assertRaises(ValueError):paired_differences(rows[:-1],'reference','clock')
        with self.assertRaises(ValueError):paired_differences(rows+[rows[0]],'reference','clock')
if __name__=='__main__':unittest.main()
