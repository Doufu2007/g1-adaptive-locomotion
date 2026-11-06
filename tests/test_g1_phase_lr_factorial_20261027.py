import unittest
from scripts import g1_phase_lr_factorial_20261027 as study

class FactorialTests(unittest.TestCase):
    def rows(self):
        values={'clock_default':10.,'clock_low':8.,'learned_default':14.,'learned_low':9.}
        return [dict(seed=s,arm=k.rsplit('_',1)[0],lr_mode=k.rsplit('_',1)[1],eval_seed=e,mae=v+s+e)
                for s in (1,2,3) for k,v in values.items() for e in (11,12)]
    def test_known_interaction_and_nested_seed_unit(self):
        result=study.factorial_contrasts(self.rows(),'mae')
        self.assertEqual(result['interaction']['per_seed'],[-3.,-3.,-3.])
        self.assertEqual(result['E_low_minus_clock_low']['per_seed'],[1.,1.,1.])
        self.assertEqual(result['E_low_minus_default']['per_seed'],[-5.,-5.,-5.])
        self.assertEqual(result['interaction']['n'],3)
    def test_episode_replication_does_not_create_training_seeds(self):
        rows=self.rows(); copies=[dict(r,eval_seed=r['eval_seed']+100) for r in rows]
        self.assertEqual(study.factorial_contrasts(rows,'mae'),study.factorial_contrasts(rows+copies,'mae'))
    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):study.factorial_contrasts(self.rows()[:-1],'mae')
    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):study.factorial_contrasts(self.rows()+self.rows()[:1],'mae')
    def test_nonfinite_metric_rejected(self):
        rows=self.rows();rows[0]['mae']=float('nan')
        with self.assertRaises(ValueError):study.factorial_contrasts(rows,'mae')
    def test_guardrail_blocks_speed_only_candidate(self):
        self.assertFalse(study.candidate_gate([-.01,-.02,-.03],{'falls':1,'tilt':0.,'heading':0.,'displacement':0.})['proceed'])
        self.assertTrue(study.candidate_gate([-.01,-.02,-.03],{'falls':0,'tilt':0.,'heading':0.,'displacement':0.})['proceed'])
    def test_checkpoint_name_matches_native_velocity_export_contract(self):
        from pathlib import Path
        for point in (0,1,501):
            path=study.checkpoint_path(Path('trial'),point)
            self.assertEqual(Path(str(path).split('model')[0]),Path('trial'))
            self.assertEqual(path.name,'model_'+str(point)+'.pt')
    def test_matrix_complete_equal_budget(self):
        jobs=study.build_jobs();self.assertEqual(len(jobs),12)
        self.assertEqual(len({(j['arm'],j['lr_mode'],j['seed']) for j in jobs}),12)
        self.assertTrue(all(j['updates']==501 and j['transitions']==769536 for j in jobs))

if __name__=='__main__':unittest.main()
