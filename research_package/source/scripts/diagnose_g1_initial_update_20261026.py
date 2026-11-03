"""Frozen first-rollout counterfactuals: initial LR and critic initialization."""
from __future__ import annotations
import argparse
import contextlib
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import traceback
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/g1_ppo_initialization_probe_20261026/local_reconstruction_v3'
PLAN=ROOT/'docs/superpowers/plans/2026-10-26-g1-initial-update-probe.md'
SOURCE=ROOT/'logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt'
ORIGINAL=ROOT/'results/g1_stateful_flat_ppo_best_e_20261026/remote_download/snapshot/results/g1_stateful_flat_ppo_best_e_20261026/training'
SEEDS=(20261901,20261902,20261903)
EVAL=(20263111,20263112,20263113)
VARIANTS=('default_fresh','low_lr_fresh','default_source_critic','low_lr_source_critic')
EVALUATED=('before_update','default_fresh','low_lr_fresh','default_source_critic')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    from scripts.audit_g1_training_mechanism_20261026 import sha as digest
    return digest(path)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def bootstrap_rewards(rewards,values,timeouts,gamma):
    if values.shape!=rewards.shape+(1,) or timeouts.shape!=rewards.shape:
        raise ValueError('Expected rewards/timeouts[T,N] and values[T,N,1]')
    return (rewards+gamma*values[...,0]*timeouts)[...,None]


def state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def digest_tensor(value):
    return hashlib.sha256(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()).hexdigest()


def lock():
    import rsl_rl.algorithms.ppo as ppo
    import rsl_rl.modules.distribution as distribution
    import rsl_rl.storage.rollout_storage as storage
    from scripts import diagnose_g1_best_e_closed_loop_20261026 as previous
    paths=[Path(__file__),PLAN,SOURCE,SOURCE.parent/'params/env.yaml',
        Path(ppo.__file__),Path(distribution.__file__),Path(storage.__file__),
        ROOT/'scripts/g1_stateful_flat_ppo.py',ROOT/'scripts/g1_stateful_online.py',
        ROOT/'scripts/g1_phase_feedback_probe_20261026.py',
        ROOT/'scripts/run_g1_cause_audit_windows_20261026.py',
        ROOT/'scripts/g1_best_phase_batched.py',ROOT/'scripts/diagnose_g1_best_e_closed_loop_20261026.py',
        ROOT/'scripts/analyze_g1_initial_update_20261026.py',
        ROOT/'tests/test_g1_initial_update_probe_20261026.py',previous.ENCODER]
    paths += [ORIGINAL/f'flat_{arm}_seed{seed}'/'manifest.json' for seed in SEEDS for arm in ('clock','learned')]
    sources=[dict(path=str(p),sha256=sha(p)) for p in paths]
    path=OUT/'prospective_protocol.json'
    if path.exists():
        old=read(path);assert old['sources']==sources,'Frozen source changed';return old
    jobs=[(arm,seed) for seed in SEEDS for arm in ('clock','learned')]
    random.Random(20263201).shuffle(jobs)
    evaluate=[(seed,condition) for seed in SEEDS for condition in EVALUATED]
    random.Random(20263202).shuffle(evaluate)
    result=dict(status='LOCKED_BEFORE_NEW_ROLLOUTS',at=now(),classification='POST_RESULTS_EXPLORATORY',
        sources=sources,capture_order=jobs,evaluation_order=evaluate,train_seeds=SEEDS,eval_seeds=EVAL,
        encoder_sha256=previous.ENCODER_SHA,actor_source_sha256=sha(SOURCE),
        variants=VARIANTS,evaluated_variants=EVALUATED,
        fixed_budget=dict(environment_training_transitions=6*64*24,actual_first_updates=6,
            frozen_batch_replay_updates=24,optimizer_minibatches_total=30*20,
            formal_evaluation_episodes=36,technical_smoke_episodes=2,
            failed_collection_overhead_transitions=1536,total_including_failed_collection=10752),
        primary_family=['low_lr_fresh_minus_default_fresh','default_source_critic_minus_default_fresh'],
        metric='Full20s body-forward velocity MAE; remainder after fall velocity0',
        unit='Three original PPO seeds, evaluation slots nested within policy',
        branch_rule='All three lower MAE, CI below0 and BHq<.05, no new falls, tilt delta<=.02rad, '
                    'heading delta<=.2rad, direction displacement delta>=-.5m: conditional support. '
                    'Otherwise mixed/imprecise or guardrail tradeoff. No added budget.',
        scope='Only first-update mechanisms; no conclusion about full501-update control superiority. '
              'Inherited critic need not be calibrated to the new reward/phase state distribution.',
        technical_amendment=dict(previous_protocol_sha256=sha(OUT.parent/'prospective_protocol.json'),
            failed_manifest_sha256=sha(OUT.parent/'captures/clock_20261901/manifest.json'),
            previous_training_transitions=1536,previous_optimizer_updates=0,
            second_failure_protocol_sha256=sha(OUT.parent/'local_reconstruction_v2/prospective_protocol.json'),
            second_failure_manifest_sha256=sha(OUT.parent/'local_reconstruction_v2/captures/clock_20261901/manifest.json'),
            original_critic_identity='Not reproduced across Windows torch2.6 and Linux torch2.8; exact source not isolated',
            scope='Matched local initialization experiment; not exact remote first-update reconstruction'),
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','mjlab','mujoco','warp-lang','rsl-rl-lib','onnxruntime')})
    write(path,result);return result


def trace_update(alg,function):
    """Observe native update without any extra stochastic forward or gradient mutation."""
    import torch
    rows=[];current={}
    native_kl=alg.actor.get_kl_divergence
    native_step=alg.optimizer.step
    native_generator=alg.storage.mini_batch_generator
    native_clip=torch.nn.utils.clip_grad_norm_
    actor_ids={id(p) for p in alg.actor.parameters()}
    def generator(*args,**kwargs):
        for index,batch in enumerate(native_generator(*args,**kwargs)):
            current['index']=index;current['order_sha256']=digest_tensor(batch.actions)
            yield batch
    def kl(*args,**kwargs):
        result=native_kl(*args,**kwargs)
        rows.append(dict(index=current['index'],order_sha256=current['order_sha256'],
            measured_training_minibatch_KL_mean=float(result.mean()),
            measured_training_minibatch_KL_max=float(result.max())))
        return result
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        result=native_clip(parameters,*args,**kwargs)
        name='actor' if parameters and id(parameters[0]) in actor_ids else 'critic'
        rows[-1][name+'_gradient_norm_before_clipping']=float(result)
        return result
    def step(*args,**kwargs):
        before=[p.detach().clone() for p in alg.actor.parameters()]
        rows[-1]['effective_learning_rate']=alg.optimizer.param_groups[0]['lr']
        result=native_step(*args,**kwargs)
        square=sum(float((p.detach()-b).square().sum()) for p,b in zip(alg.actor.parameters(),before))
        count=sum(p.numel() for p in alg.actor.parameters())
        rows[-1]['actor_parameter_step_rms']=float(np.sqrt(square/count))
        return result
    alg.storage.mini_batch_generator=generator;alg.actor.get_kl_divergence=kl
    alg.optimizer.step=step;torch.nn.utils.clip_grad_norm_=clip
    try:
        losses=function()
        assert len(rows)==20
        return losses,rows
    finally:
        alg.storage.mini_batch_generator=native_generator;alg.actor.get_kl_divergence=native_kl
        alg.optimizer.step=native_step;torch.nn.utils.clip_grad_norm_=native_clip


def observation_compatibility(path,arm):
    import yaml
    def tree(node):
        if isinstance(node,yaml.MappingNode):return {k.value:tree(v) for k,v in node.value}
        if isinstance(node,yaml.SequenceNode):return [tree(v) for v in node.value]
        return [node.tag,node.value]
    a,b=[tree(yaml.compose(p.read_text(encoding='utf-8'))) for p in (SOURCE.parent/'params/env.yaml',path)]
    a,b=a['observations']['critic'],b['observations']['critic']
    assert list(a['terms'])==list(b['terms']),'Critic observation order differs'
    for name in a['terms']:
        if name=='phase' and arm=='learned':continue
        for field in ('func','params','scale','clip','history_length'):
            assert a['terms'][name][field]==b['terms'][name][field],(name,field)
    return dict(status='PASS',observation_terms=list(a['terms']),source_and_current_width=113,
                allowed_change='phase uses frozen E' if arm=='learned' else None,
                limitation='Names/config compatible; transferred value need not predict new task returns correctly')


def capture(arm,seed):
    from scripts.g1_phase_feedback_probe_20261026 import setup
    study,torch=setup()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    folder=OUT/'captures'/f'{arm}_{seed}'
    if folder.exists():
        assert read(folder/'manifest.json')['status']=='COMPLETED','Preserve incomplete capture'
        return
    folder.mkdir(parents=True)
    record=dict(status='RUNNING',started_at=now(),arm=arm,train_seed=seed,
        source_sha256=sha(SOURCE),script_sha256=sha(__file__),transitions=0,
        pythonhashseed=__import__('os').environ.get('PYTHONHASHSEED'),
        backend=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_deterministic=torch.backends.cudnn.deterministic,
                     cudnn_benchmark=torch.backends.cudnn.benchmark))
    write(folder/'manifest.json',record)
    env=None
    try:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cfg=study.make_cfg('flat',arm,seed,play=False,num_envs=64,horizon=20.)
        other=study.make_cfg('flat','clock' if arm=='learned' else 'learned',seed,play=False,num_envs=64,horizon=20.)
        assert study.common_cfg_digest(cfg)==study.common_cfg_digest(other)
        dump_yaml(folder/'env.yaml',asdict(cfg))
        record['observation_compatibility']=observation_compatibility(folder/'env.yaml',arm)
        agent=load_rl_cfg('Unitree-G1-Flat');agent.seed=seed;agent.logger='tensorboard'
        agent.save_interval=50;agent.upload_model=False;agent.algorithm.learning_rate=1e-3
        assert agent.num_steps_per_env==24
        raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        bundle=observations._load_phasor_encoder(raw,str(study.RUNTIME));bundle.encoder.eval().requires_grad_(False)
        encoder_before=study.base.state_sha(bundle.encoder)
        record['startup_model_hashes']={k:digest_tensor(getattr(raw.sim.model,k)) for k in ('body_mass','body_ipos','geom_friction')}
        torch.manual_seed(seed+1000);torch.cuda.manual_seed_all(seed+1000)
        cls=load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
        runner=cls(env,asdict(agent),str(folder),device='cuda:0')
        runner.load(str(SOURCE),load_cfg={'actor':True},strict=True,map_location='cuda:0')
        runner.current_learning_iteration=0;raw.common_step_counter=0
        assert not runner.alg.optimizer.state
        record['initial_actor_sha256']=study.base.state_sha(runner.alg.actor)
        record['initial_critic_sha256']=study.base.state_sha(runner.alg.critic)
        old=read(ORIGINAL/f'flat_{arm}_seed{seed}'/'manifest.json')
        assert old['initial_actor_state_sha256']==record['initial_actor_sha256']
        record['original_initial_actor_matches']=True
        record['original_initial_critic_matches']=old['initial_critic_state_sha256']==record['initial_critic_sha256']
        record['original_initial_critic_sha256']=old['initial_critic_state_sha256']
        paired=OUT/'captures'/f'{"clock" if arm=="learned" else "learned"}_{seed}'/'manifest.json'
        if paired.exists():
            assert read(paired)['initial_critic_sha256']==record['initial_critic_sha256'],'Local phase-pair critic mismatch'
        record['initialization_scope']='Matched local fresh critic; remote initial critic SHA differs'
        critic_initial=state(runner.alg.critic)
        env_rewards=[];timeouts=[];pre_observations=[];native_step=env.step
        def checked_step(actions):
            assert torch.isfinite(actions).all()
            pre_observations.append({k:v.detach().cpu().clone() for k,v in runner.alg.transition.observations.items()})
            result=native_step(actions)
            env_rewards.append(result[1].detach().cpu().clone())
            timeouts.append(result[3].get('time_outs',torch.zeros_like(result[2])).detach().cpu().clone())
            record['transitions']+=64
            return result
        env.step=checked_step
        native_compute=runner.alg.compute_returns;last={}
        def compute(obs):
            last['obs']={k:v.detach().cpu().clone() for k,v in obs.items()}
            return native_compute(obs)
        runner.alg.compute_returns=compute
        native_update=runner.alg.update
        def update():
            alg=runner.alg;st=alg.storage
            pack=dict(arm=arm,seed=seed,agent=asdict(agent),
                observations={k:v.detach().cpu().clone() for k,v in st.observations.items()},last_obs=last['obs'],
                buffers={k:getattr(st,k).detach().cpu().clone() for k in
                    ('actions','rewards','dones','values','actions_log_prob','returns','advantages')},
                distribution_params=tuple(p.detach().cpu().clone() for p in st.distribution_params),
                env_rewards=torch.stack(env_rewards),timeouts=torch.stack(timeouts),
                actor_before=state(alg.actor),critic_before_collection=critic_initial,
                critic_before_update=state(alg.critic),rng_cpu=torch.get_rng_state().clone(),
                rng_cuda=torch.cuda.get_rng_state().cpu().clone(),pre_step_snapshots=pre_observations)
            torch.save(pack,folder/'batch.pt')
            for key in st.observations.keys():
                torch.testing.assert_close(st.observations[key].cpu(),torch.stack([r[key] for r in pre_observations]),rtol=0,atol=0)
            losses,trace=trace_update(alg,native_update)
            torch.save(dict(actor=state(alg.actor),critic=state(alg.critic),optimizer=alg.optimizer.state_dict(),
                rng_cpu=torch.get_rng_state().clone(),rng_cuda=torch.cuda.get_rng_state().cpu().clone()),folder/'actual_post.pt')
            write(folder/'actual_trace.json',dict(losses=losses,minibatches=trace,final_lr=alg.learning_rate))
            return losses
        runner.alg.update=update
        dump_yaml(folder/'agent.yaml',asdict(agent));write(folder/'manifest.json',record)
        runner.learn(num_learning_iterations=1,init_at_random_ep_len=True)
        assert record['transitions']==1536 and study.base.state_sha(bundle.encoder)==encoder_before
        record.update(status='COMPLETED',ended_at=now(),encoder_frozen=True,
                      batch_sha256=sha(folder/'batch.pt'),actual_post_sha256=sha(folder/'actual_post.pt'))
        write(folder/'manifest.json',record)
    except BaseException as error:
        record.update(status='FAILED',ended_at=now(),error=repr(error),traceback=traceback.format_exc())
        write(folder/'manifest.json',record);raise
    finally:
        if env is not None:env.close()


def make_algorithm(pack,variant,torch):
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    obs=TensorDict({k:v[0].to('cuda:0') for k,v in pack['observations'].items()},batch_size=[64])
    cfg=copy.deepcopy(pack['agent']);groups={'actor':['actor'],'critic':['critic']}
    ac,cc=cfg['actor'],cfg['critic'];ac.pop('class_name');cc.pop('class_name')
    actor=MLPModel(obs,groups,'actor',29,**ac).to('cuda:0')
    critic=MLPModel(obs,groups,'critic',1,**cc).to('cuda:0')
    storage=RolloutStorage('rl',64,24,obs,[29],device='cuda:0')
    kw=cfg['algorithm'];kw.pop('class_name');kw.pop('share_cnn_encoders',None)
    kw['learning_rate']=1e-5 if variant.startswith('low_lr') else 1e-3
    alg=PPO(actor,critic,storage,device='cuda:0',**kw)
    actor.load_state_dict(pack['actor_before'],strict=True)
    critic.load_state_dict(pack['critic_before_update'],strict=True)
    for k,v in pack['observations'].items():storage.observations[k].copy_(v)
    for k,v in pack['buffers'].items():getattr(storage,k).copy_(v)
    storage.distribution_params=tuple(v.to('cuda:0') for v in pack['distribution_params']);storage.step=24
    last=TensorDict({k:v.to('cuda:0') for k,v in pack['last_obs'].items()},batch_size=[64])
    alg.train_mode()
    if variant.endswith('source_critic'):
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        critic.load_state_dict(saved['critic_state_dict'],strict=True)
    else:
        critic.load_state_dict(pack['critic_before_collection'],strict=True)
    values=[]
    with torch.inference_mode():
        for step in range(24):
            current=storage.observations[step]
            values.append(critic(current).clone())
            following=storage.observations[step+1] if step<23 else last
            critic.update_normalization(following)
        storage.values.copy_(torch.stack(values))
        storage.rewards.copy_(bootstrap_rewards(pack['env_rewards'].to('cuda:0'),storage.values,
                                               pack['timeouts'].to('cuda:0'),alg.gamma))
    with torch.inference_mode():alg.compute_returns(last)
    if not variant.endswith('source_critic'):
        torch.testing.assert_close(storage.values,pack['buffers']['values'].to('cuda:0'),rtol=0,atol=1e-6)
        for key,value in critic.state_dict().items():
            torch.testing.assert_close(value.cpu(),pack['critic_before_update'][key],rtol=0,atol=1e-7)
        torch.testing.assert_close(storage.returns,pack['buffers']['returns'].to('cuda:0'),rtol=0,atol=1e-6)
        torch.testing.assert_close(storage.advantages,pack['buffers']['advantages'].to('cuda:0'),rtol=0,atol=1e-6)
        expected=bootstrap_rewards(pack['env_rewards'].to('cuda:0'),storage.values,pack['timeouts'].to('cuda:0'),alg.gamma)
        torch.testing.assert_close(expected,storage.rewards,rtol=0,atol=1e-7)
    return alg


def export_actor(actor_state,path,torch):
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    import onnxruntime as ort
    model=MLPModel(TensorDict({'actor':torch.zeros(1,98)},batch_size=[1]),{'actor':['actor']},'actor',29,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar'))
    model.load_state_dict(actor_state,strict=True);model.eval()
    exporter=model.as_onnx(verbose=False).eval();path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(exporter,exporter.get_dummy_inputs(),str(path),opset_version=18,
        input_names=exporter.input_names,output_names=exporter.output_names,dynamo=False)
    from scripts.audit_g1_training_mechanism_20261026 import pool
    obs,_=pool();obs=obs[:64]
    with torch.inference_mode():expected=model(TensorDict({'actor':obs},batch_size=[len(obs)])).numpy()
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None].numpy()})[0] for r in obs])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    return dict(policy_sha256=sha(path),onnx_max_abs_error=float(abs(actual-expected).max()))


def replay(arm,seed,variant):
    from scripts.g1_phase_feedback_probe_20261026 import setup
    _,torch=setup()
    capture_folder=OUT/'captures'/f'{arm}_{seed}'
    record=read(capture_folder/'manifest.json');assert record['status']=='COMPLETED'
    assert sha(capture_folder/'batch.pt')==record['batch_sha256']
    folder=OUT/'replays'/f'{arm}_{seed}_{variant}'
    if folder.exists():
        assert read(folder/'manifest.json')['status']=='COMPLETED','Preserve incomplete replay';return
    folder.mkdir(parents=True)
    manifest=dict(status='RUNNING',started_at=now(),arm=arm,train_seed=seed,variant=variant,
        batch_sha256=record['batch_sha256'],script_sha256=sha(__file__))
    write(folder/'manifest.json',manifest)
    try:
        pack=torch.load(capture_folder/'batch.pt',map_location='cpu',weights_only=False)
        alg=make_algorithm(pack,variant,torch)
        before_actor=state(alg.actor);before_critic=state(alg.critic)
        torch.save(dict(values=alg.storage.values.detach().cpu(),returns=alg.storage.returns.detach().cpu(),
            advantages=alg.storage.advantages.detach().cpu(),critic_before=before_critic),folder/'targets.pt')
        torch.set_rng_state(pack['rng_cpu']);torch.cuda.set_rng_state(pack['rng_cuda'])
        if variant=='default_fresh':
            losses=alg.update();trace=read(capture_folder/'actual_trace.json')['minibatches']
            actual=torch.load(capture_folder/'actual_post.pt',map_location='cpu',weights_only=False)
            max_errors={}
            for name,module in (('actor',alg.actor),('critic',alg.critic)):
                differences=[float((v.cpu()-actual[name][k]).abs().max()) for k,v in module.state_dict().items()]
                max_errors[name]=max(differences)
                assert max(differences)<=1e-7,('Recorded/unrecorded update mismatch',name,max(differences))
            assert torch.equal(torch.get_rng_state(),actual['rng_cpu'])
            assert torch.equal(torch.cuda.get_rng_state().cpu(),actual['rng_cuda'])
            manifest['instrumented_vs_native_max_error']=max_errors
            manifest['rng_exactly_matched']=True
        else:
            losses,trace=trace_update(alg,alg.update)
        assert len(trace)==20
        original_order=[r['order_sha256'] for r in read(capture_folder/'actual_trace.json')['minibatches']]
        assert [r['order_sha256'] for r in trace]==original_order
        post=state(alg.actor)
        assert all(torch.isfinite(v).all() for v in post.values())
        torch.save(dict(actor_state_dict=post,critic_state_dict=state(alg.critic),optimizer_state_dict=alg.optimizer.state_dict(),
            iter=0,infos=dict(variant=variant,batch_sha256=record['batch_sha256'])),folder/'model_0.pt')
        from scripts.audit_g1_training_mechanism_20261026 import forward,pool
        observations,_=pool()
        with torch.no_grad():
            before=forward(before_actor,observations);after=forward(post,observations)
            old_std=before_actor['distribution.std_param'];new_std=post['distribution.std_param']
            kl=(torch.log(new_std/old_std)+(old_std.square()+(before-after).square())/(2*new_std.square())-.5).sum(-1)
        write(folder/'trace.json',dict(minibatches=trace,losses=losses,final_lr=alg.learning_rate))
        manifest.update(status='COMPLETED',ended_at=now(),minibatches=20,
            action_rms_change=float((after-before).square().mean().sqrt()),
            audit_state_pool_conditional_KL_mean=float(kl.mean()),
            initial_critic_normalizer_count=int(before_critic['obs_normalizer.count']),
            original_values_std=float(pack['buffers']['values'].std()),
            checkpoint_sha256=sha(folder/'model_0.pt'),**export_actor(post,folder/'policy.onnx',torch))
        write(folder/'manifest.json',manifest)
        if arm=='learned' and variant=='default_fresh':
            pre=OUT/'replays'/f'learned_{seed}_before_update'
            pre.mkdir(parents=True,exist_ok=True)
            torch.save(dict(actor_state_dict=pack['actor_before']),pre/'model_before.pt')
            write(pre/'manifest.json',dict(status='COMPLETED',variant='before_update',train_seed=seed,
                reference='Actor after shared rollout normalization; no optimizer update',
                batch_sha256=record['batch_sha256'],**export_actor(pack['actor_before'],pre/'policy.onnx',torch)))
    except BaseException as error:
        manifest.update(status='FAILED',ended_at=now(),error=repr(error),traceback=traceback.format_exc())
        write(folder/'manifest.json',manifest);raise


def evaluate(seed,condition,*,smoke=False):
    from scripts.g1_phase_feedback_probe_20261026 import setup
    study,torch=setup()
    from scripts import diagnose_g1_best_e_closed_loop_20261026 as evaluator
    evaluator.OUT=OUT
    policy=OUT/'replays'/f'learned_{seed}_{condition}'/'policy.onnx'
    assert sha(policy)==read(policy.parent/'manifest.json')['policy_sha256']
    evaluator.checkpoint_policy=lambda arm,s,checkpoint:policy
    name=('smoke_' if smoke else '')+f'{condition}_{seed}'
    target=OUT/'rollouts'/name
    if target.exists():
        assert read(target/'manifest.json')['status']=='COMPLETED','Preserve incomplete evaluation';return
    evaluator.local_rollout(SimpleNamespace(name=name,trained='learned',train_seed=seed,phase='raw',offset=0.,
        checkpoint=0,eval_seeds=list(EVAL[:1] if smoke else EVAL),scenarios=['steady'],horizon=.2 if smoke else 20.))
    value=read(target/'manifest.json');value.update(initial_update_condition=condition,initial_update_probe_sha256=sha(__file__))
    write(target/'manifest.json',value)


def queue():
    protocol=lock();completed=[]
    def status(state,**fields):write(OUT/'queue_status.json',dict(status=state,at=now(),completed=completed,**fields))
    def job(name,function,*args,**kwargs):
        status('RUNNING',current=name);(OUT/'logs').mkdir(parents=True,exist_ok=True)
        with (OUT/'logs'/(name+'.log')).open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            function(*args,**kwargs)
        completed.append(name);gc.collect()
        import torch
        torch.cuda.empty_cache();print(json.dumps(dict(completed=name)),flush=True)
    try:
        for arm,seed in protocol['capture_order']:
            job(f'capture_{arm}_{seed}',capture,arm,seed)
            for variant in VARIANTS:job(f'replay_{arm}_{seed}_{variant}',replay,arm,seed,variant)
        job('smoke_default',evaluate,SEEDS[0],'default_fresh',smoke=True)
        job('smoke_low_lr',evaluate,SEEDS[0],'low_lr_fresh',smoke=True)
        for seed,condition in protocol['evaluation_order']:job(f'evaluate_{condition}_{seed}',evaluate,seed,condition)
        status('COMPLETED',training_transitions=9216,failed_collection_overhead_transitions=1536,
               total_including_failed_collection=10752,actual_updates=6,replay_updates=24,formal_episodes=36)
    except BaseException as error:
        status('FAILED',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['lock','queue'],required=True)
    args=parser.parse_args();lock() if args.stage=='lock' else queue()
