import itertools
import unittest
import tempfile
from pathlib import Path
import numpy as np

from scripts import g1_rough_transfer_contract_20261028 as c


class RoughContractTest(unittest.TestCase):
    def test_matrix_and_routing(self):
        self.assertEqual(len(c.build_jobs()), 16)
        self.assertEqual(len({(j['method'], j['terrain_seed']) for j in c.build_jobs()}), 16)
        slots = c.world_slots()
        self.assertEqual(len(slots), 15)
        self.assertEqual([x['patch_col'] for x in slots], list(range(5))*3)
        class Session:
            def __init__(self, i): self.i = i
            def run(self, _, feed):
                assert list(feed) == ['actor']
                return [np.full((1,29), self.i, dtype=np.float32)]
        act=c.route_actions(np.zeros((15,98),np.float32),[Session(i) for i in range(15)],['actor']*15)
        np.testing.assert_equal(act[:,0], np.arange(15))

    def test_contacts_world_patch_and_constraint(self):
        result = c.contact_exposure(
            geom=np.array([[1,10],[11,1],[1,10],[1,10],[1,10],[2,3]]),
            world=np.array([0,0,1,0,0,1]),
            positions=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[.9,0,0],[0,0,0]]),
            dist=np.array([-.01]*6),margin=np.zeros(6),
            efc=np.array([[0],[1],[2],[-1],[3],[4]]),foot_ids={1,2},terrain_ids={10,11},
            assigned=[{10},{11}], bounds=np.array([[-1,1,-1,1]]*2),num_envs=2)
        np.testing.assert_equal(result['total'], [3,1])
        np.testing.assert_equal(result['interior'], [1,0])
        self.assertEqual(result['indices'].tolist(),[0,1,2,4])
        with self.assertRaises(ValueError):
            c.contact_exposure(np.array([[1,10]]),np.array([2]),np.zeros((1,3)),np.zeros(1),np.ones(1),np.ones((1,1)),{1},{10},[{10}],np.array([[-1,1,-1,1]]),1)

    def raw(self,n=1000,speed=.5):
        t=np.arange(n)*.02
        return dict(time=t,position=np.stack((speed*t,np.zeros(n),np.zeros(n)),1),
                    quaternion=np.tile([1.,0,0,0],(n,1)),velocity=np.tile([speed,0.,0.],(n,1)),
                    gravity=np.tile([0.,0.,-1.],(n,1)),angle=t*2*np.pi,
                    contact=np.zeros((n,2),bool),foot_contact_total=np.ones(n,int),
                    foot_contact_interior=np.ones(n,int))

    def test_early_fall_and_stationary_are_not_success(self):
        early=c.episode_metrics(self.raw(100),20.,'steady',True,True)
        self.assertFalse(early['completed'])
        self.assertAlmostEqual(early['mae_forward_velocity_mps'],.45)
        self.assertFalse(c.episode_metrics(self.raw(speed=0),20.,'steady',False,True)['completed'])
        self.assertTrue(c.episode_metrics(self.raw(),20.,'steady',False,True)['completed'])
        no_contact=self.raw();no_contact['foot_contact_total'][:]=0;no_contact['foot_contact_interior'][:]=0
        r=c.episode_metrics(no_contact,20.,'steady',False,True)
        self.assertIsNone(r['rough_contact_fraction']);self.assertFalse(r['completed'])
        edge=self.raw();edge['foot_contact_interior'][49:]=0
        self.assertFalse(c.episode_metrics(edge,20.,'steady',False,True)['completed'])

    def test_direction_and_push_censoring(self):
        r=self.raw();r['quaternion'][:]=[2**-.5,0,0,2**-.5]
        m=c.episode_metrics(r,20.,'push',False,False)
        self.assertAlmostEqual(m['forward_displacement_m'],0.,places=8)
        self.assertFalse(m['completed'])
        m=c.episode_metrics(self.raw(100),20.,'push',True,True)
        self.assertTrue(all(not x['exposed'] for x in m['push_recovery']))

    def test_exact_seed_p_never_invents_significance(self):
        self.assertEqual(c.seed_comparison([.2,.2,.2])['p_exact'],.25)
        self.assertEqual(c.seed_comparison([0,0,0])['p_exact'],1.)
        p=c.holm([.01,.04,.02]);np.testing.assert_allclose(p,[.03,.04,.04])

    def test_native_terminal_raw_at_horizon_and_no_concatenation(self):
        raw=dict(time=np.arange(1000)*.02,terminated=np.zeros(1000,bool),timed_out=np.zeros(1000,bool))
        self.assertFalse(c.termination_from_raw(raw))
        raw['terminated'][-1]=True
        self.assertTrue(c.termination_from_raw(raw))
        raw['terminated'][1]=True
        with self.assertRaises(ValueError):c.termination_from_raw(raw)

    def test_missing_duplicate_and_full_factorial(self):
        rows=[]
        for j in c.build_jobs():
            for s in c.world_slots():
                for scenario in ('steady','push'):
                    rows.append(dict(j,**s,scenario=scenario,completed=True,fell=False,
                        mae_forward_velocity_mps=.05,forward_displacement_m=9.,tilt_rms_rad=.02,abs_heading_change_rad=.1))
        result=c.paired_analysis(rows)
        self.assertEqual(result['independent_training_seeds'],3)
        self.assertEqual(len(result['family']),8)
        self.assertFalse(result['candidate']['passed'])
        with self.assertRaises(ValueError):c.paired_analysis(rows[:-1])
        with self.assertRaises(ValueError):c.paired_analysis(rows+[rows[0]])
        favorable=[dict(r,completed=r['method']=='DE',fell=r['method']!='DE') for r in rows]
        self.assertTrue(c.paired_analysis(favorable)['candidate']['passed'])
        for r in favorable:
            if r['method']=='DE':r['mae_forward_velocity_mps']=.10
        self.assertFalse(c.paired_analysis(favorable)['candidate']['passed'])

    def test_hash_arrays_ignore_names_but_not_physics(self):
        a={'hfield_data':np.array([0.,1.]),'geom_pos':np.zeros((1,3))}
        b={k:v.copy() for k,v in a.items()}
        self.assertEqual(c.array_digest(a),c.array_digest(b))
        b['hfield_data'][1]=.9
        self.assertNotEqual(c.array_digest(a),c.array_digest(b))


class RoughExecutionTest(unittest.TestCase):
    def test_refuse_unfinished_output(self):
        from scripts import g1_rough_transfer_experiment_20261028 as r
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'job';r.prepare_output(p)
            with self.assertRaises(FileExistsError):r.prepare_output(p)

    def test_pairing_rejects_changed_state(self):
        from scripts import g1_rough_transfer_experiment_20261028 as r
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pair.json'
            r.pairing_guard(p,{'position':[0,0,1]})
            r.pairing_guard(p,{'position':[0,0,1]})
            with self.assertRaises(ValueError):r.pairing_guard(p,{'position':[0,0,2]})

    def test_assigned_geometry_requires_actual_patch(self):
        from scripts import g1_rough_transfer_experiment_20261028 as r
        p=np.array([[0,0,0],[0,16,0],[0,32,0],[0,48,0],[0,64,0]])
        origins=np.array([[0,16,.06],[0,0,.06]])
        assigned,bounds=r.assigned_patches(p,np.arange(5),origins,True)
        self.assertEqual(assigned,[{1},{0}])
        np.testing.assert_equal(bounds,[[ -16,16,8,24],[-16,16,-8,8]])
        with self.assertRaises(ValueError):r.assigned_patches(p,np.arange(5),np.array([[0,8,0]]),True)


if __name__=='__main__':unittest.main()
