import copy,unittest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
try:
 from scripts.g1_source_anchor_ppo_20261027 import SourceAnchoredPPO
except ImportError:SourceAnchoredPPO=None

def make_algorithm(cls,coefficient=None):
 torch.manual_seed(784)
 obs=TensorDict({'actor':torch.randn(8,5),'critic':torch.randn(8,7)},batch_size=[8])
 groups={'actor':['actor'],'critic':['critic']}
 actor=MLPModel(obs,groups,'actor',3,hidden_dims=(8,),obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=(8,),obs_normalization=True)
 storage=RolloutStorage('rl',8,4,obs,[3],'cpu')
 kw=dict(num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-5,schedule='adaptive')
 if coefficient is not None:kw['anchor_coefficient']=coefficient
 alg=cls(actor,critic,storage,**kw)
 for _ in range(4):
  alg.act(obs)
  obs=TensorDict({'actor':torch.randn(8,5),'critic':torch.randn(8,7)},batch_size=[8])
  alg.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.bool),{})
 alg.compute_returns(obs)
 return alg,obs

def assert_tree(test,left,right):
 if isinstance(left,torch.Tensor):test.assertTrue(torch.equal(left,right))
 elif isinstance(left,dict):
  test.assertEqual(set(left),set(right))
  for k in left:assert_tree(test,left[k],right[k])
 elif isinstance(left,(tuple,list)):
  test.assertEqual(len(left),len(right))
  for a,b in zip(left,right):assert_tree(test,a,b)
 else:test.assertEqual(left,right)

class SourceAnchorTest(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(SourceAnchoredPPO,'source-anchor implementation missing')
 def test_zero_coefficient_matches_native_multi_minibatch_weights_adam_and_rng(self):
  native,_=make_algorithm(PPO);state=torch.get_rng_state();native.update();native_rng=torch.get_rng_state()
  anchored,_=make_algorithm(SourceAnchoredPPO,0.);self.assertTrue(torch.equal(state,torch.get_rng_state()));anchored.update()
  assert_tree(self,native.actor.state_dict(),anchored.actor.state_dict());assert_tree(self,native.critic.state_dict(),anchored.critic.state_dict())
  assert_tree(self,native.optimizer.state_dict(),anchored.optimizer.state_dict());self.assertTrue(torch.equal(native_rng,torch.get_rng_state()))
 def test_source_kl_has_known_value_current_gradient_no_teacher_gradient_or_rng_draw(self):
  alg,obs=make_algorithm(SourceAnchoredPPO,.1);rng=torch.get_rng_state().clone()
  with torch.no_grad():mu=alg.source_teacher(obs)
  current=((mu+.5).detach().requires_grad_(True),torch.full_like(mu,.5).requires_grad_(True))
  kl=alg.source_kl(obs,current);torch.testing.assert_close(kl,torch.full((8,),1.5));kl.mean().backward()
  self.assertGreater(float(current[0].grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in alg.source_teacher.parameters()))
  self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_teacher_stays_frozen_and_is_preserved_across_checkpoint_restore(self):
  alg,obs=make_algorithm(SourceAnchoredPPO,.1);teacher=copy.deepcopy(alg.source_teacher.state_dict());alg.update()
  assert_tree(self,teacher,alg.source_teacher.state_dict());payload=copy.deepcopy(alg.save())
  restored,_=make_algorithm(SourceAnchoredPPO,.1);restored.load(payload,None,True)
  assert_tree(self,teacher,restored.source_teacher.state_dict());assert_tree(self,alg.actor.state_dict(),restored.actor.state_dict())
  self.assertTrue(any(not torch.equal(v,teacher[k]) for k,v in alg.actor.state_dict().items() if v.is_floating_point()))
 def test_anchor_does_not_pollute_twenty_native_kl_trace_entries(self):
  from scripts.g1_phase_lr_factorial_20261027 import record_update
  alg,_=make_algorithm(SourceAnchoredPPO,.1);losses,trace=record_update(alg,alg.update)
  self.assertEqual(len(trace),20);self.assertGreaterEqual(losses['source_kl'],0);self.assertTrue(all('effective_lr' in r for r in trace))
 def test_missing_teacher_is_only_allowed_for_explicit_initial_actor_only_load(self):
  alg,_=make_algorithm(SourceAnchoredPPO,.1);native_payload=PPO.save(alg)
  before=copy.deepcopy(alg.actor.state_dict())
  self.assertRaises(ValueError,alg.load,native_payload,None,True)
  assert_tree(self,before,alg.actor.state_dict())
  alg.load(native_payload,{'actor':True},True)
  assert_tree(self,alg.actor.state_dict(),alg.source_teacher.state_dict())
  incomplete=alg.save();incomplete.pop('source_anchor_coefficient')
  self.assertRaises(ValueError,alg.load,incomplete,None,True)

try:
 from scripts.g1_source_anchor_experiment_20261027 import common_agent_digest,paired_contrasts,dual_candidate,build_jobs
except ImportError:common_agent_digest=paired_contrasts=dual_candidate=build_jobs=None

class ExperimentContractTest(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(common_agent_digest,'anchor experiment contract missing')
 def test_agent_normalizes_only_declared_class_and_learning_rate(self):
  native={'algorithm':{'class_name':'PPO','learning_rate':1e-5,'gamma':.99},'seed':7}
  anchored=copy.deepcopy(native);anchored['algorithm']['class_name']='scripts.g1_source_anchor_ppo_20261027:SourceAnchoredPPO'
  self.assertEqual(common_agent_digest(native),common_agent_digest(anchored))
  anchored['algorithm']['gamma']=.95;self.assertNotEqual(common_agent_digest(native),common_agent_digest(anchored))
  anchored['algorithm']['class_name']='other';self.assertRaises(ValueError,common_agent_digest,anchored)
 def test_launch_hash_pairing_uses_startup_record_and_actual_interpreter_probe(self):
  from scripts.g1_source_anchor_experiment_20261027 import check_launch_seed
  launch=dict(startup_pythonhashseed='20263300',launcher_hash_probe=123)
  self.assertTrue(check_launch_seed(launch,dict(launch)))
  self.assertRaises(ValueError,check_launch_seed,dict(launch,launcher_hash_probe=124),launch)
  self.assertRaises(ValueError,check_launch_seed,dict(launch,startup_pythonhashseed='20261901'),launch)
 def test_speed_gate_against_original_clock_prevents_degrading_the_control(self):
  guards=dict(falls=0,tilt=0,heading=0,displacement=0)
  gate=dual_candidate([-.02]*3,guards,[.01]*3,guards);self.assertFalse(gate['proceed'])
  self.assertTrue(dual_candidate([-.01]*3,guards,[-.006]*3,guards)['proceed'])
  self.assertFalse(dual_candidate([-.01]*3,dict(guards,heading=.3),[-.006]*3,guards)['proceed'])
 def test_paired_family_and_complete_six_job_budget(self):
  self.assertEqual(len(build_jobs()),6);self.assertEqual(sum(x['transitions'] for x in build_jobs()),4617216)
  rows=[dict(seed=s,arm=arm,variant=v,eval_seed=e,mae_forward_velocity_mps=value) for s in (1,2,3) for e in (11,12) for arm,v,value in [('clock','base',.05),('learned','base',.07),('clock','anchor',.04),('learned','anchor',.03)]]
  c=paired_contrasts(rows)
  for key,value in {'E_anchor_minus_clock_anchor':-.01,'E_anchor_minus_E_base':-.04,'clock_anchor_minus_clock_base':-.01,'interaction':-.03}.items():
   for observed in c[key]:self.assertAlmostEqual(observed,value)
  self.assertRaises(ValueError,paired_contrasts,rows[:-1]);self.assertRaises(ValueError,paired_contrasts,rows+[rows[0]])

if __name__=='__main__':unittest.main()
