"""Behavioral contracts for a causal, contact-locked reference; no simulator needed."""
import math
import unittest
import torch
try:
    from scripts.g1_contact_phase_reference import ContactPhaseReference
except ImportError:
    ContactPhaseReference=None
try:
    from scripts.g1_contact_phase_reference import calibrate_batches
except ImportError:
    calibrate_batches=None

class ContactReferenceTest(unittest.TestCase):
    def make(self,n=2):
        self.assertIsNotNone(ContactPhaseReference,'causal reference is not implemented')
        return ContactPhaseReference(n,'cpu',(0.,math.pi))
    def run_step(self,s,k,contact,ep=None):
        return s.update(k,torch.tensor([k]*s.num_envs if ep is None else ep),torch.tensor(contact))
    def test_no_contact_is_nominal_and_repeated_reads_are_idempotent(self):
        s=self.make()
        for k in range(40):
            v=self.run_step(s,k,[[False,False]]*2)
            expected=torch.tensor([[math.sin(k*.02*2*math.pi/.6),math.cos(k*.02*2*math.pi/.6)]]*2)
            torch.testing.assert_close(v,expected,atol=2e-6,rtol=0)
            before=s.theta.clone()
            torch.testing.assert_close(self.run_step(s,k,[[True,True]]*2),v,atol=0,rtol=0)
            torch.testing.assert_close(s.theta,before,atol=0,rtol=0)
    def test_runner_randomizes_episode_length_after_cached_initial_observation(self):
        s=self.make();self.run_step(s,0,[[False,False]]*2,ep=[0,0])
        got=self.run_step(s,1,[[True,False]]*2,ep=[182,716])
        theta=torch.tensor([182,716],dtype=torch.float64)*.02*2*math.pi/.6
        expected=torch.stack((theta.sin(),theta.cos()),dim=-1).float()
        torch.testing.assert_close(got,expected,atol=1e-7,rtol=0)
        self.assertEqual(s.touchdowns.sum().item(),0)
        with self.assertRaises(ValueError):self.run_step(s,2,[[True,False]]*2,ep=[400,717])
    def test_initial_contact_does_not_fabricate_touchdown(self):
        s=self.make(1)
        for k in range(9):self.run_step(s,k,[[True,False]])
        self.assertEqual(s.touchdowns.tolist(),[[0,0]])
    def test_debounce_is_causal_and_confirmation_is_not_backdated(self):
        s=self.make(1); self.run_step(s,0,[[False,False]])
        for k in (1,2):
            self.run_step(s,k,[[True,False]])
            self.assertEqual(s.touchdowns[0,0].item(),0)
        self.run_step(s,3,[[True,False]])
        self.assertEqual(s.touchdowns[0,0].item(),1)
        self.assertEqual(s.last_touch[0,0].item(),3)
    def test_partial_reset_preserves_other_world_and_removes_old_events(self):
        s=self.make()
        for k in range(10):self.run_step(s,k,[[k>=2,False],[False,False]])
        second=s.theta[1].clone();s.reset(torch.tensor([0]))
        self.run_step(s,9,[[True,False],[False,False]],ep=[0,9])
        self.assertEqual(s.touchdowns[0,0].item(),0)
        self.assertEqual(s.theta[0].item(),0)
        torch.testing.assert_close(s.theta[1],second)
    def test_period_uses_same_foot_confirmations_and_bounded_monotone_correction(self):
        s=self.make(1); prev=None
        for k in range(110):
            self.run_step(s,k,[[k%40>=5 and k%40<20,False]])
            if prev is not None:
                delta=s.theta[0].item()-prev
                self.assertGreater(delta,0)
                self.assertGreaterEqual(delta,2*math.pi*.02/1.*.75-1e-8)
                self.assertLessEqual(delta,2*math.pi*.02/.4*1.25+1e-8)
            prev=s.theta[0].item()
        self.assertGreater(s.period[0].item(),.6)
        self.assertLessEqual(s.period[0].item(),1.)
    def test_simultaneous_touchdown_is_not_used_to_correct_phase(self):
        s=self.make(1)
        for k in range(30):self.run_step(s,k,[[k%10>=5]*2])
        self.assertEqual(s.pending[0].item(),0)
        self.assertEqual(s.period[0].item(),.6)
        self.assertEqual(s.touchdowns.sum().item(),0)
    def test_prefix_does_not_depend_on_future_contacts(self):
        a,b=self.make(1),self.make(1)
        for k in range(25):
            x=[[7<=k<15,False]]
            torch.testing.assert_close(self.run_step(a,k,x),self.run_step(b,k,x),atol=0,rtol=0)
        frozen=a.theta.clone()
        for k in range(25,60):self.run_step(b,k,[[k%8<4,False]])
        torch.testing.assert_close(a.theta,frozen,atol=0,rtol=0)
    def test_rejects_nonfinite_contacts_and_nonreset_step_gap(self):
        s=self.make(1);self.run_step(s,0,[[False,False]])
        with self.assertRaises(ValueError):self.run_step(s,1,[[float('nan'),0.]])
        with self.assertRaises(ValueError):s.update(3,torch.tensor([1]),torch.zeros(1,2,dtype=torch.bool))
        with self.assertRaises(ValueError):s.update(1,torch.tensor([1]),torch.zeros(1,3))

class CalibrationTest(unittest.TestCase):
    def packs(self):
        packs=[]
        for _ in range(3):
            ticks=torch.arange(24)[:,None]+torch.arange(64)[None,:]
            theta=ticks.double()*2*math.pi/30
            critic=torch.zeros(24,64,113)
            critic[:,:,9]=theta.sin();critic[:,:,10]=theta.cos()
            cycle=ticks%30
            critic[:,:,105]=(cycle>=4)&(cycle<16)
            critic[:,:,106]=(cycle>=19)&(cycle<29)
            packs.append(dict(observations=dict(critic=critic),buffers=dict(dones=torch.zeros(24,64,1))))
        return packs
    def test_training_only_calibration_uses_confirmation_phase_and_a_single_global_target(self):
        self.assertIsNotNone(calibrate_batches,'training calibration is not implemented')
        r=calibrate_batches(self.packs())
        self.assertTrue(r['passed']);self.assertEqual(len(r['targets']),2)
        self.assertGreaterEqual(min(r['event_counts']),50)
        self.assertGreater(min(r['concentrations']),.99)
        self.assertAlmostEqual(r['targets'][0],6*2*math.pi/30,places=5)
        self.assertAlmostEqual(r['targets'][1],-9*2*math.pi/30,places=5)
    def test_calibration_refuses_low_coverage_and_invalid_contact(self):
        self.assertIsNotNone(calibrate_batches,'training calibration is not implemented')
        p=self.packs();p[0]['observations']['critic'][:,:,105:107]=0
        self.assertFalse(calibrate_batches(p[:1])['passed'])
        p[0]['observations']['critic'][0,0,105]=.5
        with self.assertRaises(ValueError):calibrate_batches(p)

class ContinuousCalibrationTest(unittest.TestCase):
    def test_burnin_excludes_initial_unlocked_events_and_resets_are_causal(self):
        import scripts.g1_contact_phase_reference as m
        self.assertTrue(hasattr(m,'calibrate_trajectory'),'continuous training calibration is not implemented')
        ticks=torch.arange(500)[:,None].repeat(1,64)
        theta=ticks.double()*2*math.pi/30
        phase=torch.stack((theta.sin(),theta.cos()),dim=-1)
        cycle=ticks%30;contact=torch.stack(((cycle>=4)&(cycle<16),(cycle>=19)&(cycle<29)),dim=-1)
        phase[:150]=torch.tensor([1.,0.],dtype=torch.float64)
        r=m.calibrate_trajectory(phase,contact,torch.zeros(500,64,dtype=torch.bool),150)
        self.assertTrue(r['passed']);self.assertGreater(min(r['concentrations']),.99)
        self.assertAlmostEqual(r['targets'][0],6*2*math.pi/30,places=5)
        self.assertEqual(r['start_step'],150)

if __name__=='__main__':unittest.main()
