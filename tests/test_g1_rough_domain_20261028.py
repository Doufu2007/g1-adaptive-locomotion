"""Finite S design and analysis checks; no environment stepping."""
import copy
import unittest
from scripts import g1_rough_domain_contract_20261028 as s


def rows_for(completions=None, falls=None):
    completions = completions or dict(FC=10, FE=10, RC=10, RE=13)
    falls = falls or dict(FC=20, FE=20, RC=20, RE=17)
    rows = []
    for m in s.METHODS:
        for t in s.TERRAINS:
            for seed in s.TRAIN_SEEDS:
                for ei, e in enumerate(s.EVAL_SEEDS):
                    for qi, q in enumerate(s.SCENARIOS):
                        i = (s.TERRAINS.index(t)-1)*10 + ei*2 + qi
                        rows.append(dict(method=m, terrain_seed=t, train_seed=seed,
                                         eval_seed=e, scenario=q,
                                         completed=True if t == 0 else i < completions[m],
                                         fell=False if t == 0 else i >= 30-falls[m],
                                         mae_forward_velocity_mps=.1,
                                         forward_displacement_m=6.,
                                         tilt_rms_rad=.1, abs_heading_change_rad=.1))
    return rows


class RoughDomainContractTests(unittest.TestCase):
    def test_complete_fixed_matrix(self):
        jobs = s.training_jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({(j['method'], j['source_seed']) for j in jobs}), 12)
        self.assertEqual(sum(j['transitions'] for j in jobs), 4626432)
        self.assertTrue(all(j['updates'] == 251 for j in jobs))
        self.assertEqual(jobs, s.training_jobs())
        self.assertEqual(len(s.evaluation_jobs()), 16)
        self.assertEqual(len(s.slots()), 15)
        self.assertEqual(len(rows_for()), 480)
        self.assertTrue(set(s.TRAIN_MAPS).isdisjoint(s.TERRAINS))
        self.assertTrue(set(s.TRAIN_MAPS).isdisjoint((20264201, 20264202, 20264203)))
        self.assertEqual(s.CHECKPOINTS, (0, 1, 25, 100, 251))

    def test_explicit_domain_phase_and_source_routing(self):
        self.assertEqual(s.source_phase('RE'), 'learned')
        self.assertEqual(s.source_phase('FC'), 'clock')
        self.assertEqual(s.training_map('RC', 20261902), 20264602)
        self.assertEqual(s.training_map('FC', 20261902), 0)
        for m, seed in [('DE', 20261901), ('RE', 999)]:
            with self.assertRaises(ValueError):
                s.training_map(m, seed)

    def test_missing_duplicate_seed_rejected(self):
        rows = rows_for()
        with self.assertRaises(ValueError):
            s.paired_analysis(rows[:-1])
        with self.assertRaises(ValueError):
            s.paired_analysis(rows + [rows[0]])
        rows[0]['mae_forward_velocity_mps'] = float('nan')
        with self.assertRaises(ValueError):
            s.paired_analysis(rows)

    def test_holm_eight_and_exact_seed_p(self):
        result = s.paired_analysis(rows_for())
        self.assertEqual(len(result['family']), 8)
        for r in result['family'].values():
            self.assertGreaterEqual(r['p_exact'], .25)
            self.assertGreaterEqual(r['p_holm'], r['p_exact'])
            self.assertEqual(r['n'], 3)
        self.assertEqual(result['family']['RE_minus_RC:completed']['p_exact'], .25)
        self.assertEqual(result['family']['RE_minus_RC:completed']['p_holm'], 1.)

    def test_candidate_requires_strong_clock_and_both_endpoints(self):
        self.assertTrue(s.paired_analysis(rows_for())['candidate']['passed'])
        r = s.paired_analysis(rows_for(dict(FC=10, FE=10, RC=13, RE=13)))
        self.assertFalse(r['candidate']['passed'])
        self.assertTrue(r['candidate']['comparators']['FC']['passed'])
        self.assertFalse(r['candidate']['comparators']['RC']['passed'])
        r = s.paired_analysis(rows_for(falls=dict(FC=20, FE=20, RC=20, RE=19)))
        self.assertFalse(r['candidate']['passed'])

    def test_candidate_requires_flat_guards(self):
        rows = rows_for()
        for r in rows:
            if r['method'] == 'RE' and r['terrain_seed'] == 0:
                r['mae_forward_velocity_mps'] += .0201
        result = s.paired_analysis(rows)
        self.assertFalse(result['candidate']['passed'])
        self.assertFalse(result['candidate']['flat_guards']['mae_forward_velocity_mps'])
        self.assertTrue(result['candidate']['comparators']['RC']['passed'])

    def test_real_training_cfg_inserts_terrain_before_reset(self):
        from scripts import g1_phase_lr_factorial_20261027 as a
        study, _ = a.setup()
        for seed in s.TRAIN_SEEDS:
            cfgs = {m:s.make_training_cfg(study, m, seed) for m in s.METHODS}
            for cfg in cfgs.values():
                order = list(cfg.events)
                self.assertLess(order.index('randomize_terrain'), order.index('reset_base'))
                self.assertEqual(cfg.scene.num_envs, 64)
                self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x, (.3, .8))
                self.assertTrue(cfg.observations['actor'].enable_corruption)
                self.assertEqual(cfg.sim.njmax, 1500)
                self.assertNotIn('height_scan', cfg.observations['actor'].terms)
            self.assertEqual(study.common_cfg_digest(cfgs['FC']), study.common_cfg_digest(cfgs['FE']))
            self.assertEqual(study.common_cfg_digest(cfgs['RC']), study.common_cfg_digest(cfgs['RE']))
            from dataclasses import replace
            for cfg in cfgs.values():
                cfg.scene.terrain = cfgs['FC'].scene.terrain
            self.assertEqual(len({study.common_cfg_digest(cfg) for cfg in cfgs.values()}), 1)

    def test_domain_benefit_does_not_imply_phase_candidate(self):
        result = s.paired_analysis(rows_for(dict(FC=3, FE=3, RC=13, RE=13),
                                           dict(FC=27, FE=27, RC=17, RE=17)))
        self.assertTrue(result['training_domain_effect']['learned']['point_gate_passed'])
        self.assertTrue(result['training_domain_effect']['clock']['point_gate_passed'])
        self.assertFalse(result['candidate']['passed'])


class RoughDomainAdapterTests(unittest.TestCase):
    def test_adapter_routes_every_source_and_never_writes_R(self):
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        from scripts import g1_rough_transfer_experiment_20261028 as r
        old_out, old_c, old_lock, old_policy = r.OUT, r.c, r.lock, r.policy_entry
        import tempfile
        from pathlib import Path
        import numpy as np
        class Policy:
            def __init__(self, value): self.value=value
            def run(self, _, inputs): return [np.full((1,29), self.value, np.float32)]
        sessions=[Policy(100*s.METHODS.index(m)+i//5) for m in s.METHODS for i in range(15)]
        obs=np.zeros((60,98),np.float32)
        actions=ev.ROUTE_ACTIONS(obs,sessions,['actor']*60)
        for i in range(60): self.assertEqual(actions[i,0], 100*(i//15)+(i%15)//5)
        with tempfile.TemporaryDirectory() as tmp:
            proxy=ev.make_adapter(Path(tmp), lambda:None, lambda m,seed:dict(method=m,seed=seed))
            self.assertEqual(proxy.OUT, Path(tmp))
            self.assertEqual(proxy.c.world_slots()[0]['eval_seed'], 20264811)
            self.assertEqual(proxy.c.world_slots()[-1]['train_seed'], 20261903)
            self.assertEqual(r.OUT,old_out);self.assertIs(r.c,old_c)
            self.assertIs(r.lock,old_lock);self.assertIs(r.policy_entry,old_policy)
        for method,seed in [('DE',20261901),('RE',111)]:
            with self.assertRaises(ValueError): ev.policy_entry(method,seed)

    def test_adapter_pairing_writes_only_S_and_rejects_changed_binding(self):
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        from scripts import g1_rough_transfer_experiment_20261028 as r
        import tempfile
        from pathlib import Path
        original=r.a.sha(r.OUT/'prospective_protocol.json')
        with tempfile.TemporaryDirectory() as tmp:
            proxy=ev.make_adapter(Path(tmp), lambda:None, lambda m,seed:dict(method=m,seed=seed))
            path=proxy.OUT/'pairing'/'terrain0.json'
            proxy.pairing_guard(path,dict(physics='shared'))
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):proxy.pairing_guard(path,dict(physics='changed'))
        self.assertEqual(r.a.sha(r.OUT/'prospective_protocol.json'),original)

    def test_no_analysis_or_independent_PASS_before_complete_data(self):
        import importlib.util
        from scripts import g1_rough_domain_experiment_20261028 as ex
        spec=importlib.util.spec_from_file_location('s_independent_cpu',ex.OUT/'independent_audit.py')
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            audit.O=Path(tmp)
            with self.assertRaises(FileNotFoundError):audit.run()
            self.assertFalse((audit.O/'independent_audit.json').exists())

    def test_incomplete_new_policy_is_not_usable(self):
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'training'/'RE_20261901';folder.mkdir(parents=True)
            (folder/'manifest.json').write_text(json.dumps(dict(status='RUNNING',updates=1)),encoding='utf8')
            with self.assertRaises(ValueError): ev.policy_entry('RE',20261901,base=Path(tmp))


class RoughDomainEngineeringGuards(unittest.TestCase):
    def test_alignment_after_terrain_before_managers(self):
        from scripts import g1_rough_domain_experiment_20261028 as ex
        import torch,random,numpy as np
        class Base:
            def __init__(self,terrain_draws):
                torch.rand(terrain_draws);np.random.rand(terrain_draws);random.random()
                self.load_managers()
            def load_managers(self):
                self.timers=torch.rand(5);self.physics=torch.rand(9);self.command=np.random.rand(3)
        Aligned=ex.aligned_environment_type(Base,torch,20261901)
        flat,rough=Aligned(0),Aligned(64)
        torch.testing.assert_close(flat.timers,rough.timers,rtol=0,atol=0)
        torch.testing.assert_close(flat.physics,rough.physics,rtol=0,atol=0)
        np.testing.assert_array_equal(flat.command,rough.command)

    def test_invalid_training_never_writes_completed_analysis(self):
        from scripts import g1_rough_domain_evaluation_20261028 as ev
        from unittest.mock import patch
        from types import SimpleNamespace
        import tempfile
        from pathlib import Path
        writes=[]
        with tempfile.TemporaryDirectory() as tmp:
            def write():
                writes.append('COMPLETED');ev.a.write(Path(tmp)/'analysis.json',dict(status='COMPLETED'))
            proxy=SimpleNamespace(analyze=write)
            with patch.object(ev.ex,'OUT',Path(tmp)),patch.object(ev,'make_adapter',return_value=proxy), patch.object(ev.ex,'verify_training',side_effect=ValueError('bad budget')):
                with self.assertRaises(ValueError):ev.analyze()
        self.assertEqual(writes,[])

    def test_fresh_numpy_proxy_keeps_lazy_testing_and_write_guard(self):
        import subprocess,sys,numpy
        from pathlib import Path
        code="import site;site.addsitedir("+repr(str(Path(numpy.__file__).parent.parent))+");from scripts import g1_rough_domain_evaluation_20261028 as e;import numpy as np,io;calls=[];p=e.guarded_numpy(np,lambda:calls.append(1));p.testing.assert_array_equal([1],[1]);p.savez_compressed(io.BytesIO(),x=np.ones(2));assert calls==[1]"
        result=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True,cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(result.returncode,0,result.stderr)

    def test_disk_floor_stops_before_asset_write(self):
        from scripts import g1_rough_domain_experiment_20261028 as ex
        from unittest.mock import patch
        from types import SimpleNamespace
        with patch.object(ex.shutil,'disk_usage',return_value=SimpleNamespace(free=9*1024**3)):
            with self.assertRaises(ValueError):ex.check_disk()
        with patch.object(ex.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)):
            ex.check_disk()


class RoughDomainRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from scripts import g1_rough_domain_experiment_20261028 as ex
        cls.torch, cls.ex = torch, ex
        cls.entry = ex.source_entry('FC', 20261902)
        cls.saved = torch.load(cls.entry['checkpoint'], map_location='cpu', weights_only=False)

    def runner(self):
        from tensordict import TensorDict
        from rsl_rl.models import MLPModel
        from mjlab.rl import MjlabOnPolicyRunner
        from scripts.g1_source_anchor_ppo_20261027 import SourceAnchoredPPO
        torch = self.torch
        obs = TensorDict({name:torch.zeros(1, next(v.shape[1] for v in self.saved[key].values() if v.ndim == 2))
            for name,key in [('actor','actor_state_dict'), ('critic','critic_state_dict')]}, batch_size=[1])
        groups = {'actor':['actor'], 'critic':['critic']}
        kw = dict(hidden_dims=(512,256,128), activation='elu', obs_normalization=True)
        actor = MLPModel(obs, groups, 'actor', 29, **kw,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar'))
        critic = MLPModel(obs, groups, 'critic', 1, **kw)
        from types import SimpleNamespace
        runner = object.__new__(MjlabOnPolicyRunner)
        runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=-99))
        runner.alg = SourceAnchoredPPO(actor, critic, None, device='cpu', learning_rate=1e-5)
        runner.current_learning_iteration = -999
        return runner

    def test_full_native_source_restore_including_lr_and_adam(self):
        runner = self.runner()
        result = self.ex.restore_source(runner, self.entry['checkpoint'], self.torch, restore_cuda=False)
        for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict','source_teacher_state_dict'):
            self.assertEqual(self.ex.state_digest(self.saved[key]), self.ex.state_digest(runner.alg.save()[key]))
        self.assertAlmostEqual(runner.alg.learning_rate, 2.25e-5)
        self.assertAlmostEqual(runner.alg.optimizer.param_groups[0]['lr'], 2.25e-5)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(runner.env.unwrapped.common_step_counter, 0)
        self.assertEqual(result['source_native_iteration'], 500)
        self.assertFalse(runner.alg.source_teacher.training)
        self.assertTrue(all(not p.requires_grad for p in runner.alg.source_teacher.parameters()))

    def test_restored_rng_is_source_rng_after_construction(self):
        import random
        import numpy as np
        state = self.saved['infos']['factorial_state']
        local = random.Random(); local.setstate(state['python_rng']); want_py = local.random()
        local_np = np.random.RandomState(); local_np.set_state(state['numpy_rng']); want_np = local_np.rand()
        gen = self.torch.Generator(); gen.set_state(state['torch_rng']); want_t = self.torch.rand(3, generator=gen)
        runner = self.runner()
        self.ex.restore_source(runner, self.entry['checkpoint'], self.torch, restore_cuda=False)
        self.assertEqual(random.random(), want_py)
        self.assertEqual(np.random.rand(), want_np)
        self.torch.testing.assert_close(self.torch.rand(3), want_t, atol=0, rtol=0)

    def test_incomplete_or_inconsistent_source_rejected_before_load(self):
        for missing in ('critic_state_dict','optimizer_state_dict','source_teacher_state_dict'):
            saved = dict(self.saved); saved.pop(missing)
            with self.assertRaises(ValueError): self.ex.validate_source_state(saved)
        saved = copy.deepcopy(self.saved)
        saved['optimizer_state_dict']['param_groups'][0]['lr'] = .001
        with self.assertRaises(ValueError): self.ex.validate_source_state(saved)
        saved = copy.deepcopy(self.saved)
        saved['infos']['factorial_state'].pop('cuda_rng')
        with self.assertRaises(ValueError): self.ex.validate_source_state(saved)

    def test_named_robot_physics_is_independent_of_terrain_compiled_ids(self):
        import mujoco
        import numpy as np
        from types import SimpleNamespace
        def raw(extra):
            model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+extra+
                '<body name="robot/base"><freejoint name="robot/root"/><geom type="sphere" size=".1" mass="1"/>'+
                '<body name="robot/leg" pos="0 0 -.2"><joint name="robot/hinge" type="hinge" damping=".1" armature=".2"/>'+
                '<geom name="robot/foot" type="sphere" size=".05" mass=".2"/></body></body></worldbody></mujoco>')
            def tensor(field):return self.torch.from_numpy(np.asarray(getattr(model,field)).copy())
            idx=SimpleNamespace(body_ids=self.torch.tensor([model.body('robot/base').id,model.body('robot/leg').id]),
                geom_ids=self.torch.tensor([0 if not extra else 1,model.geom('robot/foot').id]),
                joint_v_adr=self.torch.tensor([model.jnt_dofadr[model.joint('robot/hinge').id]]),
                free_joint_v_adr=self.torch.arange(6))
            runtime=SimpleNamespace(**{k:tensor(k) for k in ('body_mass','body_ipos','body_inertia','geom_friction','dof_damping','dof_armature')})
            return SimpleNamespace(sim=SimpleNamespace(mj_model=model,model=runtime),scene={'robot':SimpleNamespace(indexing=idx)})
        flat=self.ex.robot_physics(raw(''))
        rough=self.ex.robot_physics(raw('<geom name="terrain" type="plane" size="3 3 .1"/>'))
        self.assertEqual(self.ex.named_array_digest(flat), self.ex.named_array_digest(rough))
        self.assertEqual(len(flat['dof_names']),7)
        rough['dof_damping'][0]+=1.
        self.assertNotEqual(self.ex.named_array_digest(flat), self.ex.named_array_digest(rough))

    def test_frozen_source_tamper_and_duplicate_queue_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'source';p.write_bytes(b'original')
            entries = [dict(path=str(p),sha256=self.ex.a.sha(p))]
            self.ex.verify_files(entries)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): self.ex.verify_files(entries)
            folder = Path(tmp)/'queue';folder.mkdir()
            self.ex.claim_queue(folder)
            with self.assertRaises(FileExistsError): self.ex.claim_queue(folder)


if __name__ == '__main__':
    unittest.main()
