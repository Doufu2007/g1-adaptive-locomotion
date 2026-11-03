"""Matched phase x initial-LR experiment, with immutable inputs and raw evidence."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/g1_phase_lr_factorial_20261027'
SPEC=ROOT/'docs/superpowers/specs/2026-10-27-g1-phase-lr-factorial-design.md'
PLAN=ROOT/'docs/superpowers/plans/2026-10-27-g1-phase-lr-factorial.md'
SEEDS=(20261901,20261902,20261903)
CHECKPOINTS=(0,1,25,100,251,501)
LR={'default':.001,'low':.00001}
SOURCE=ROOT/'logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt'
SOURCE_SHA='949fae33632b9a3592e6346a6155eea059ee2fd39487207a0a0cea5d35edd8b4'
ENCODER=ROOT/'deliveries/g1_stateful_phase_20261025/snapshot/results/g1_stateful_phase_20261025/training/observer_sensors_mixed_seed20261026/final.pt'
ENCODER_SHA='8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346'

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def checkpoint_path(folder,completed):
    return Path(folder)/f'model_{completed}.pt'

def build_jobs():
    jobs=[dict(arm=a,lr_mode=m,seed=s,updates=501,transitions=501*64*24) for s in SEEDS for a in ('clock','learned') for m in LR]
    random.Random(20263301).shuffle(jobs)
    return jobs

def factorial_contrasts(rows,metric):
    groups={};keys=set()
    for r in rows:
        k=(r['seed'],r['arm'],r['lr_mode'],r['eval_seed'])
        if k in keys:raise ValueError('Duplicate paired episode')
        keys.add(k);v=float(r[metric])
        if not math.isfinite(v):raise ValueError('Nonfinite metric')
        groups.setdefault(k[:3],{})[k[3]]=v
    seeds=sorted({r['seed'] for r in rows});differences={k:[] for k in ('E_low_minus_clock_low','interaction','E_low_minus_default','clock_low_minus_default')}
    if not seeds:raise ValueError('Empty rows')
    for s in seeds:
        try:g=[groups[(s,a,m)] for a,m in [('clock','default'),('clock','low'),('learned','default'),('learned','low')]]
        except KeyError as error:raise ValueError('Missing group') from error
        if not all(set(x)==set(g[0]) for x in g):raise ValueError('Missing paired evaluation')
        c0,c1,e0,e1=[statistics.mean(x.values()) for x in g]
        for k,v in zip(differences,(e1-c1,(e1-e0)-(c1-c0),e1-e0,c1-c0)):differences[k].append(v)
    return {k:dict(n=len(seeds),seeds=seeds,per_seed=v,mean=statistics.mean(v)) for k,v in differences.items()}

def candidate_gate(per_seed,guards):
    if not per_seed or not all(math.isfinite(v) for v in per_seed):raise ValueError('Invalid seed differences')
    speed=statistics.mean(per_seed)<=-.005 and sum(v<0 for v in per_seed)>=math.ceil(2*len(per_seed)/3)
    protection=guards['falls']<=0 and guards['tilt']<=.02 and guards['heading']<=.2 and guards['displacement']>=-.5
    return dict(proceed=bool(speed and protection),speed=bool(speed),guardrails=bool(protection),
        interpretation='Exploratory candidate only; independent confirmation required')

def setup():
    from scripts.g1_phase_feedback_probe_20261026 import setup as original
    study,torch=original()
    assert sha(SOURCE)==SOURCE_SHA and sha(ENCODER)==ENCODER_SHA
    study.verify_runtime()
    return study,torch

def lock():
    paths=[Path(__file__),SPEC,PLAN,SOURCE,ENCODER,
        ROOT/'tests/test_g1_phase_lr_factorial_20261027.py']
    paths += [ROOT/'scripts'/n for n in ('g1_stateful_flat_ppo.py','g1_stateful_online.py',
        'g1_phase_feedback_probe_20261026.py','run_g1_cause_audit_windows_20261026.py',
        'run_g1_best_e_ppo_20261026.py','run_g1_stateful_flat_ppo_remote_20261025.py',
        'g1_best_phase_batched.py','g1_closed_loop_metrics.py','diagnose_g1_initial_update_20261026.py')]
    paths += sorted((ROOT/'src/tasks/velocity').rglob('*.py'))
    paths += [ROOT/'results/phasor_pilot_20261016/run_pilot.py']
    for module in ('rsl_rl.algorithms.ppo','rsl_rl.runners.on_policy_runner','mjlab.rl.runner'):
        paths.append(Path(__import__(module,fromlist=['x']).__file__))
    sources=[dict(path=str(p),sha256=sha(p)) for p in paths]
    identity=dict(sources=sources,jobs=build_jobs(),checkpoints=list(CHECKPOINTS),
        source_sha256=SOURCE_SHA,encoder_sha256=ENCODER_SHA,
        packages={p:importlib.metadata.version(p) for p in ('torch','numpy','mjlab','mujoco','warp-lang','rsl-rl-lib','onnxruntime')},
        primary='Full20s body-forward velocity MAE; zero velocity after fall',
        final_family=['E_low_minus_clock_low','interaction','E_low_minus_default','clock_low_minus_default'],
        train_transitions=9234432,smoke_transitions=12288,technical_frozen_replay_updates=2,formal_episodes=540,
        initial_lr=LR,adaptive_schedule=True,eval_initial_seeds=list(range(20263411,20263421)),
        confirmatory=False,statistical_unit='3 paired training seeds, episodes nested; fixed simulation cohort',
        user_authorization='2026-10-27: 按照上述说法进行核查，完成任务目标',
        conditional_next_steps='Reference-phase training or independently frozen confirmation as specified in design; not auto-selected for good scores')
    path=OUT/'prospective_protocol.json'
    if path.exists():
        old=read(path);assert old['identity']==identity,'Frozen code, configuration or inputs changed'
        return old
    value=dict(status='LOCKED_BEFORE_TRAINING',at=now(),identity=identity)
    write(path,value)
    for item in sources:
        p=Path(item['path'])
        if p.is_relative_to(ROOT) and p.suffix in ('.py','.md'):
            target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    return value

def record_update(alg,native):
    """Read native KL and effective LR; no additional forward or random draws."""
    rows=[];old_kl=alg.actor.get_kl_divergence;old_step=alg.optimizer.step
    def kl(*a,**kw):
        result=old_kl(*a,**kw);rows.append(dict(kl_mean=float(result.mean()),kl_max=float(result.max())))
        return result
    def step(*a,**kw):
        rows[-1]['effective_lr']=float(alg.optimizer.param_groups[0]['lr'])
        return old_step(*a,**kw)
    alg.actor.get_kl_divergence=kl;alg.optimizer.step=step
    try:
        losses=native();assert len(rows)==alg.num_learning_epochs*alg.num_mini_batches
        return losses,rows
    finally:alg.actor.get_kl_divergence=old_kl;alg.optimizer.step=old_step

def train(arm,lr_mode,seed,*,smoke=False):
    protocol=lock();study,torch=setup()
    import numpy as np
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    from scripts import diagnose_g1_initial_update_20261026 as probe
    name=f'{arm}_{lr_mode}_{seed}';folder=OUT/('smoke' if smoke else 'training')/name
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json');assert m['status']=='COMPLETED','Incomplete attempt preserved'
        assert m['protocol_sha256']==sha(OUT/'prospective_protocol.json')
        return
    folder.mkdir(parents=True,exist_ok=False)
    target=2 if smoke else 501;points=(0,1,2) if smoke else CHECKPOINTS
    record=dict(status='RUNNING',started_at=now(),arm=arm,lr_mode=lr_mode,seed=seed,updates=0,transitions=0,
        source_sha256=SOURCE_SHA,encoder_sha256=ENCODER_SHA,protocol_sha256=sha(OUT/'prospective_protocol.json'),
        checkpoint_save_method='native MjlabOnPolicyRunner; deferred ONNX export',
        initial_learning_rate=LR[lr_mode],adaptive_schedule=True,planned_updates=target,planned_transitions=target*64*24,
        launcher_hash_probe=hash('g1_phase_lr_hash_probe'),startup_pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        argv=sys.argv,checkpoints={},backend=dict(tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
    write(folder/'manifest.json',record);env=None
    try:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cfg=study.make_cfg('flat',arm,seed,play=False,num_envs=64,horizon=20.)
        other=study.make_cfg('flat','learned' if arm=='clock' else 'clock',seed,play=False,num_envs=64,horizon=20.)
        assert study.common_cfg_digest(cfg)==study.common_cfg_digest(other)
        record['common_cfg_sha256']=study.common_cfg_digest(cfg)
        agent=load_rl_cfg('Unitree-G1-Flat');agent.seed=seed;agent.logger='tensorboard';agent.upload_model=False
        agent.save_interval=10**9;agent.algorithm.learning_rate=LR[lr_mode]
        assert agent.num_steps_per_env==24
        agent_dict=asdict(agent)
        common_agent=json.loads(json.dumps(study.canonical(agent_dict)))
        common_agent['algorithm'].pop('learning_rate')
        record['common_agent_sha256']=hashlib.sha256(json.dumps(common_agent,sort_keys=True).encode()).hexdigest()
        dump_yaml(folder/'env.yaml',asdict(cfg));dump_yaml(folder/'agent.yaml',agent_dict)
        raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        bundle=observations._load_phasor_encoder(raw,str(ENCODER));bundle.encoder.eval().requires_grad_(False)
        encoder_before=study.base.state_sha(bundle.encoder)
        record['startup_model_hashes']={k:probe.digest_tensor(getattr(raw.sim.model,k)) for k in ('body_mass','body_ipos','geom_friction')}
        torch.manual_seed(seed+1000);torch.cuda.manual_seed_all(seed+1000)
        cls=load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
        runner=cls(env,agent_dict,str(folder),device='cuda:0')
        runner.load(str(SOURCE),load_cfg={'actor':True},strict=True,map_location='cuda:0')
        runner.current_learning_iteration=0;raw.common_step_counter=0
        assert not runner.alg.optimizer.state
        assert not ({id(p) for p in bundle.encoder.parameters()} & {id(p) for g in runner.alg.optimizer.param_groups for p in g['params']})
        record['initial_actor_sha256']=study.base.state_sha(runner.alg.actor)
        record['initial_critic_sha256']=study.base.state_sha(runner.alg.critic)
        for a in ('clock','learned'):
            for mode in LR:
                paired=folder.parent/f'{a}_{mode}_{seed}'/'manifest.json'
                if paired.exists() and paired!=folder/'manifest.json':
                    q=read(paired)
                    for key in ('initial_actor_sha256','initial_critic_sha256','common_cfg_sha256','common_agent_sha256','startup_model_hashes'):
                        if key in q:assert q[key]==record[key],('Initialization/configuration mismatch',key)
        original_step=env.step
        def checked_step(action):
            assert torch.isfinite(action).all()
            command=raw.command_manager.get_command('twist')
            assert bool(((command[:,0]>=.3-1e-6)&(command[:,0]<=.8+1e-6)&(command[:,1:].abs().amax(dim=1)<1e-6)).all())
            result=original_step(action);record['transitions']+=64;return result
        env.step=checked_step
        # The task override also exports and assumes an initialized logging writer.
        # Persist with the native mjlab parent; export all frozen actors after training.
        native_save=lambda path,infos=None:MjlabOnPolicyRunner.save(runner,path,infos)
        def save_checkpoint(_path=None,infos=None):
            completed=record['updates']
            if completed not in points or str(completed) in record['checkpoints']:return
            path=checkpoint_path(folder,completed)
            assert study.base.state_sha(bundle.encoder)==encoder_before
            details=dict(completed_updates=completed,learning_rate=runner.alg.learning_rate,arm=arm,lr_mode=lr_mode,seed=seed,
                encoder_sha256=ENCODER_SHA,protocol_sha256=record['protocol_sha256'],python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            native_save(str(path),dict(infos or {},factorial_state=details))
            record['checkpoints'][str(completed)]=dict(path=str(path),sha256=sha(path))
            record['updated_at']=now();write(folder/'manifest.json',record)
        runner.save=save_checkpoint
        native_update=runner.alg.update
        trace_path=folder/'updates.jsonl'
        def update():
            st=runner.alg.storage
            data=dict(completed_update=record['updates']+1,reward_mean=float(st.rewards.mean()),
                return_mean=float(st.returns.mean()),return_std=float(st.returns.std()),
                value_mean=float(st.values.mean()),advantage_std=float(st.advantages.std()),
                lr_before=runner.alg.learning_rate,at=now())
            losses,trace=record_update(runner.alg,native_update)
            record['updates']+=1;data.update(losses=losses,minibatches=trace,lr_after=runner.alg.learning_rate)
            with trace_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(data,allow_nan=False)+'\n')
            write(folder/'progress.json',dict(updates=record['updates'],transitions=record['transitions'],at=now(),lr=runner.alg.learning_rate))
            return losses
        runner.alg.update=update
        old_log=runner.logger.log
        def log(*a,**kw):
            result=old_log(*a,**kw);save_checkpoint();return result
        runner.logger.log=log
        save_checkpoint()
        write(folder/'manifest.json',record)
        runner.learn(num_learning_iterations=target,init_at_random_ep_len=True)
        assert record['updates']==target and record['transitions']==target*1536
        assert set(record['checkpoints'])==set(map(str,points))
        assert study.base.state_sha(bundle.encoder)==encoder_before
        record['encoder_frozen']=True
        for k,entry in record['checkpoints'].items():
            saved=torch.load(entry['path'],map_location='cpu',weights_only=False)
            policy=folder/f'policy_{k}.onnx'
            entry.update(probe.export_actor(saved['actor_state_dict'],policy,torch),policy=str(policy))
            assert saved['infos']['factorial_state']['completed_updates']==int(k)
        record.update(status='COMPLETED',ended_at=now(),updates=target,transitions=target*1536)
    except BaseException as error:
        record.update(status='FAILED',ended_at=now(),error=repr(error),traceback=traceback.format_exc());raise
    finally:
        write(folder/'manifest.json',record)
        if env is not None:env.close()

def evaluate(arm,lr_mode,seed,updates,*,smoke=False):
    lock();study,torch=setup()
    from scripts import g1_best_phase_batched as batch
    training=OUT/('smoke' if smoke else 'training')/f'{arm}_{lr_mode}_{seed}'
    m=read(training/'manifest.json');assert m['status']=='COMPLETED'
    entry=m['checkpoints'][str(updates)]
    assert sha(entry['path'])==entry['sha256'] and sha(entry['policy'])==entry['policy_sha256']
    name=f'{arm}_{lr_mode}_{seed}_u{updates}';folder=OUT/('smoke_evaluation' if smoke else 'evaluation')/name
    if (folder/'manifest.json').exists():
        done=read(folder/'manifest.json');assert done['status']=='COMPLETED';return
    final=updates==501
    eval_seeds=list(range(20263411,20263421)) if final else [20263411,20263412,20263413]
    scenarios=['steady','speed_change','push'] if final else ['steady']
    batch.bootstrap=study.bootstrap;batch.make_cfg=study.make_cfg;batch.ENCODER=ENCODER
    batch.rollout_batched(SimpleNamespace(output=str(folder),policy=entry['policy'],terrain='flat',arm=arm,
        label=name,seeds=eval_seeds[:1] if smoke else eval_seeds,horizon=.2 if smoke else 20.,scenarios=scenarios))
    value=read(folder/'manifest.json');value.update(train_seed=seed,lr_mode=lr_mode,completed_updates=updates,
        protocol_sha256=sha(OUT/'prospective_protocol.json'),training_manifest_sha256=sha(training/'manifest.json'))
    write(folder/'manifest.json',value)

def replay_check():
    lock();_,torch=setup()
    from scripts import diagnose_g1_initial_update_20261026 as probe
    from scripts.run_g1_initial_update_recovery_20261026 import filtered_pack
    path=probe.OUT/'captures/clock_20261901/batch.pt'
    pack=filtered_pack(torch.load(path,map_location='cpu',weights_only=False));outputs=[]
    for recorded in (False,True):
        alg=probe.make_algorithm(pack,'default_fresh',torch)
        torch.set_rng_state(pack['rng_cpu']);torch.cuda.set_rng_state(pack['rng_cuda'])
        if recorded:losses,rows=record_update(alg,alg.update)
        else:losses=alg.update()
        outputs.append(dict(actor=probe.state(alg.actor),critic=probe.state(alg.critic),rng=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),losses=losses))
    for name in ('actor','critic'):
        for key,value in outputs[0][name].items():torch.testing.assert_close(value,outputs[1][name][key],rtol=0,atol=0)
    for name in ('rng','cuda'):assert torch.equal(outputs[0][name],outputs[1][name])
    assert outputs[0]['losses']==outputs[1]['losses']
    write(OUT/'instrumentation_parity.json',dict(status='PASS',at=now(),batch_sha256=sha(path),
        actor_critic_exact=True,cpu_cuda_rng_exact=True,technical_updates=2,minibatches_recorded=len(rows)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['lock','train','evaluate','replay-check'],required=True)
    parser.add_argument('--arm',choices=['clock','learned']);parser.add_argument('--lr-mode',choices=list(LR));parser.add_argument('--seed',type=int)
    parser.add_argument('--updates',type=int);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.stage=='lock':lock()
    elif args.stage=='replay-check':replay_check()
    elif args.stage=='train':train(args.arm,args.lr_mode,args.seed,smoke=args.smoke)
    else:evaluate(args.arm,args.lr_mode,args.seed,args.updates,smoke=args.smoke)
