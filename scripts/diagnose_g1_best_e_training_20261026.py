"""Paired 100-update continuation: original reward versus pose reward removed."""
import argparse
import contextlib
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit

OUT=audit.OUT/'training_interventions'
SEEDS=(20261901,20261902,20261903)
EXTRA_UPDATES=100


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def source_checkpoint(arm,seed):
    p=audit.SNAPSHOT/'results/g1_stateful_flat_ppo_best_e_20261026/training'/f'flat_{arm}_seed{seed}'
    manifest=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
    source=p/'model_500.pt'
    assert audit.sha(source)==manifest['final_checkpoint_sha256']
    return source,manifest


def train(arm,seed,mode,*,smoke=False):
    from scripts import run_g1_best_phase_control as base
    native=base.bootstrap
    from scripts.run_g1_best_e_ppo_20261026 import install
    install()
    from scripts import g1_stateful_flat_ppo as study
    study.base.bootstrap=native;study.previous.bootstrap=native
    np,torch=study.bootstrap()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    updates=2 if smoke else EXTRA_UPDATES
    name=f'{arm}_{mode}_{seed}'
    directory=OUT/('smoke' if smoke else 'training')/name
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        record=json.loads(manifest_path.read_text(encoding='utf-8'))
        if record['status']=='COMPLETED' and audit.sha(directory/'policy.onnx')==record['policy_sha256']:
            return directory/'policy.onnx'
        raise FileExistsError('preserve incomplete training: '+str(directory))
    directory.mkdir(parents=True,exist_ok=False)
    source,original=source_checkpoint(arm,seed)
    cfg=study.make_cfg('flat',arm,seed,play=False,num_envs=64,horizon=20.)
    source_weight=float(cfg.rewards['pose'].weight)
    assert source_weight==1.
    if mode=='pose_removed':
        cfg.rewards['pose'].weight=0.
    elif mode!='original_reward':
        raise ValueError(mode)
    agent=load_rl_cfg('Unitree-G1-Flat')
    agent.seed=seed;agent.logger='tensorboard';agent.save_interval=25;agent.upload_model=False
    agent.algorithm.learning_rate=1e-3
    assert agent.num_steps_per_env==24
    control_seed=seed+5000
    random.seed(control_seed);np.random.seed(control_seed)
    torch.manual_seed(control_seed);torch.cuda.manual_seed_all(control_seed)
    record=dict(status='RUNNING',at=audit.now(),arm=arm,ppo_seed=seed,mode=mode,
        source_checkpoint=str(source),source_checkpoint_sha256=audit.sha(source),
        encoder_sha256=audit.ENCODER_SHA,initial_completed_updates=501,
        additional_updates=updates,additional_transitions=updates*64*24,num_envs=64,
        pose_weight=float(cfg.rewards['pose'].weight),only_reward_change='pose weight 1 to 0' if mode=='pose_removed' else None,
        actor_critic_optimizer_restored=True,physics_restarted_in_all_conditions=True,
        continuation_rng_seed=control_seed,script_sha256=audit.sha(__file__),
        device=torch.cuda.get_device_name(),collected_transitions=0)
    write(manifest_path,record)
    env=None;collected=0
    try:
        raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        bundle=observations._load_phasor_encoder(raw,str(study.RUNTIME))
        encoder_before=study.base.state_sha(bundle.encoder)
        record['startup_model_hashes']={k:__import__('hashlib').sha256(
            np.ascontiguousarray(getattr(raw.sim.model,k).detach().cpu().numpy()).tobytes()).hexdigest()
            for k in ('body_mass','body_ipos','geom_friction')}
        cls=load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
        runner=cls(env,asdict(agent),str(directory),device='cuda:0')
        infos=runner.load(str(source),strict=True,map_location='cuda:0')
        assert infos['adaptation_state']['completed_updates']==501
        runner.current_learning_iteration=501
        runner.alg.learning_rate=float(infos['adaptation_state']['learning_rate'])
        assert runner.alg.optimizer.state
        saved=torch.load(source,map_location='cpu',weights_only=False)
        for target,key in [(runner.alg.actor,'actor_state_dict'),(runner.alg.critic,'critic_state_dict')]:
            for name,value in target.state_dict().items():
                assert torch.equal(value.cpu(),saved[key][name]),name
        del saved
        record['initial_actor_sha256']=study.base.state_sha(runner.alg.actor)
        record['initial_critic_sha256']=study.base.state_sha(runner.alg.critic)
        record['initial_learning_rate']=runner.alg.learning_rate
        record['initial_optimizer_state_entries']=len(runner.alg.optimizer.state)
        optimizer_ids={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
        assert not ({id(p) for p in bundle.encoder.parameters()} & optimizer_ids)
        assert not any(p.requires_grad for p in bundle.encoder.parameters())
        random.seed(control_seed);np.random.seed(control_seed)
        torch.manual_seed(control_seed);torch.cuda.manual_seed_all(control_seed)
        raw.reset(seed=control_seed)
        raw.common_step_counter=0
        original_step=env.step
        def checked_step(actions):
            nonlocal collected
            if not torch.isfinite(actions).all():
                raise ValueError('nonfinite training action')
            result=original_step(actions)
            collected+=64
            return result
        env.step=checked_step
        original_save=runner.save
        def checked_save(path,infos=None):
            if study.base.state_sha(bundle.encoder)!=encoder_before:
                raise ValueError('frozen E changed')
            extra=runner.current_learning_iteration-501+1
            original_save(path,dict(infos or {},cause_audit_state=dict(
                additional_completed_updates=extra,learning_rate=runner.alg.learning_rate,
                source_checkpoint_sha256=record['source_checkpoint_sha256'],mode=mode)))
            record.update(additional_completed_updates=extra,collected_transitions=collected,
                latest_checkpoint=str(path),updated_at=audit.now())
            write(manifest_path,record)
        runner.save=checked_save
        dump_yaml(directory/'env.yaml',asdict(cfg));dump_yaml(directory/'agent.yaml',asdict(agent))
        write(manifest_path,record)
        runner.learn(num_learning_iterations=updates,init_at_random_ep_len=True)
        assert collected==updates*64*24
        assert runner.current_learning_iteration==501+updates-1
        assert study.base.state_sha(bundle.encoder)==encoder_before
        import onnxruntime as ort
        policy=runner.get_inference_policy(device='cpu')
        obs=env.get_observations().to('cpu')
        with torch.inference_mode():
            expected=policy(obs,stochastic_output=False).numpy()
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        exported=ort.InferenceSession(str(directory/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        inputs=obs['actor'].numpy().astype(np.float32)
        actual=np.concatenate([exported.run(None,{exported.get_inputs()[0].name:x[None]})[0] for x in inputs])
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
        record.update(status='COMPLETED',ended_at=audit.now(),collected_transitions=collected,
            additional_completed_updates=updates,total_completed_updates=501+updates,
            policy_sha256=audit.sha(directory/'policy.onnx'),encoder_frozen_verified=True,
            torch_onnx_max_abs_error=float(np.abs(actual-expected).max()))
        write(manifest_path,record)
        return directory/'policy.onnx'
    except BaseException as error:
        record.update(status='FAILED',ended_at=audit.now(),error=repr(error),traceback=traceback.format_exc(),collected_transitions=collected)
        write(manifest_path,record)
        raise
    finally:
        if env is not None:
            env.close()


def queue():
    from types import SimpleNamespace
    completed=[]
    def status(state,**fields):
        write(OUT/'queue_status.json',dict(status=state,at=audit.now(),completed=completed,**fields))
    try:
        while True:
            p=audit.OUT/'queue_status.json'
            state=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
            if state.get('status')=='COMPLETED':
                break
            if state.get('status')=='FAILED':
                raise RuntimeError('phase diagnostics failed; repair before further GPU work')
            status('WAITING_FOR_PHASE_DIAGNOSTICS',phase_status=state.get('status'))
            time.sleep(30)
        from scripts import run_g1_best_phase_control as base
        native=base.bootstrap
        def run_training(arm,seed,mode,smoke=False):
            name=('smoke_' if smoke else '')+f'{arm}_{mode}_{seed}'
            status('TRAINING',current=name)
            (OUT/'logs').mkdir(parents=True,exist_ok=True)
            base.bootstrap=native
            with (OUT/'logs'/(name+'.log')).open('a',encoding='utf-8') as stream:
                with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                    policy=train(arm,seed,mode,smoke=smoke)
            completed.append(name)
            gc.collect()
            print(json.dumps(dict(completed=name)),flush=True)
            return policy
        run_training('learned',20261903,'pose_removed',smoke=True)
        policies={}
        for index,seed in enumerate(SEEDS):
            for arm in (('clock','learned') if index%2==0 else ('learned','clock')):
                for mode in ('original_reward','pose_removed'):
                    policies[(arm,seed,mode)]=run_training(arm,seed,mode)
        for seed in SEEDS:
            for arm in ('clock','learned'):
                cases=[('baseline501',None),('original_reward',policies[(arm,seed,'original_reward')]),
                       ('pose_removed',policies[(arm,seed,'pose_removed')])]
                for mode,policy in cases:
                    name=f'training_holdout_{arm}_{mode}_{seed}'
                    status('EVALUATING',current=name)
                    target=audit.OUT/'rollouts'/name
                    if (target/'manifest.json').exists():
                        old=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
                        if old['status']=='COMPLETED':
                            completed.append(name);continue
                        raise FileExistsError('preserve incomplete evaluation: '+name)
                    original_policy=audit.checkpoint_policy
                    if policy is not None:
                        audit.checkpoint_policy=lambda a,s,k,p=policy:p
                    args=SimpleNamespace(name=name,trained=arm,train_seed=seed,
                        phase='clock' if arm=='clock' else 'raw',offset=0.,
                        checkpoint=500 if policy is None else 600,
                        eval_seeds=[20262711,20262712,20262713],scenarios=['steady'],horizon=20.)
                    base.bootstrap=native
                    try:
                        with (OUT/'logs'/(name+'.log')).open('w',encoding='utf-8') as stream:
                            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                                audit.local_rollout(args)
                    finally:
                        audit.checkpoint_policy=original_policy
                    completed.append(name);gc.collect()
                    print(json.dumps(dict(completed=name)),flush=True)
        status('COMPLETED',training_runs=12,extra_updates_per_run=100,steady_evaluation_episodes=54)
    except BaseException as error:
        status('FAILED',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['queue','train'],required=True)
    parser.add_argument('--arm',choices=['clock','learned'],default='learned')
    parser.add_argument('--seed',type=int,default=20261903)
    parser.add_argument('--mode',choices=['original_reward','pose_removed'],default='pose_removed')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage=='queue':
        queue()
    else:
        train(args.arm,args.seed,args.mode,smoke=args.smoke)
