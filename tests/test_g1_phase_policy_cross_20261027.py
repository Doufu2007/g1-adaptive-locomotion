import unittest
try:
    from scripts.g1_phase_policy_cross_20261027 import build_jobs,world_slots,mechanism_contrasts,route_actions
except ImportError:
    build_jobs=world_slots=mechanism_contrasts=route_actions=None

class MatrixTest(unittest.TestCase):
    def test_fixed_matrix_and_world_slot_routing(self):
        self.assertIsNotNone(build_jobs,'cross matrix is missing')
        jobs=build_jobs();self.assertEqual(len(jobs),12);self.assertEqual(jobs,build_jobs())
        self.assertEqual({(r['controller'],r['phase']) for r in jobs},{(p,q) for p in ('source','clock','learned','reference') for q in ('clock','learned','reference')})
        slots=world_slots();self.assertEqual(len(slots),9)
        self.assertEqual(slots[0],dict(train_seed=20261901,eval_seed=20263611))
        self.assertEqual(slots[8],dict(train_seed=20261903,eval_seed=20263613))
    def test_each_world_uses_its_own_seed_policy_in_order(self):
        self.assertIsNotNone(route_actions,'policy router is missing')
        import numpy as np
        obs=np.arange(9*98,dtype=np.float32).reshape(9,98)
        class Session:
            def __init__(self,k):self.k=k
            def run(self,outputs,feed):return [np.full((1,29),float(feed['obs'][0,0])+self.k,dtype=np.float32)]
        got=route_actions(obs,[Session(i//3) for i in range(9)],['obs']*9)
        self.assertEqual(got.shape,(9,29))
        np.testing.assert_array_equal(got[:,0],obs[:,0]+np.arange(9)//3)
        with self.assertRaises(ValueError):route_actions(obs[:8],[Session(0)]*9,['obs']*9)
    def test_contrasts_pair_by_seed_and_reject_missing_or_duplicate_slot(self):
        self.assertIsNotNone(mechanism_contrasts,'mechanism comparisons are missing')
        rows=[dict(controller=j['controller'],phase=j['phase'],**slot,mae_forward_velocity_mps=.1+(.01 if j['controller']=='learned' else 0)+(.02 if j['phase']=='learned' else 0)) for j in build_jobs() for slot in world_slots()]
        c=mechanism_contrasts(rows);self.assertEqual(len(c),6)
        self.assertTrue(all(abs(v+.02)<1e-9 for v in c['E_input_clock_minus_E']['per_seed']))
        self.assertTrue(all(abs(v-.01)<1e-9 for v in c['E_weights_under_clock']['per_seed']))
        with self.assertRaises(ValueError):mechanism_contrasts(rows[:-1])
        with self.assertRaises(ValueError):mechanism_contrasts(rows+[rows[0]])
if __name__=='__main__':unittest.main()
