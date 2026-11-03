"""One locked, exploratory reward-weight intervention, with cached equal-budget controls."""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/g1_training_mechanism_audit_20261026/speed_priority'
CACHE=ROOT/'results/g1_best_e_cause_audit_20261026'
PLAN=ROOT/'docs/superpowers/plans/2026-10-26-g1-speed-priority-probe.md'
SEEDS=(20261901,20261902,20261903)
EVAL=(20262711,20262712,20262713)


def configure_speed(cfg):
    assert cfg.rewards['track_linear_velocity'].weight==1.,'Expected original speed weight 1'
    cfg.rewards['track_linear_velocity'].weight=2.
    return cfg


def sha(path):
    from scripts.audit_g1_training_mechanism_20261026 import longpath
    with longpath(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def lock():
    from scripts import diagnose_g1_best_e_training_20261026 as helper
    from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit
    paths=[Path(__file__),PLAN,Path(helper.__file__),Path(audit.__file__),
        ROOT/'scripts/run_g1_cause_audit_windows_20261026.py',
        ROOT/'scripts/run_g1_best_e_ppo_20261026.py',ROOT/'scripts/g1_stateful_flat_ppo.py',
        ROOT/'scripts/g1_stateful_online.py',ROOT/'scripts/g1_best_phase_batched.py',
        ROOT/'src/tasks/velocity/mdp/observations.py',ROOT/'src/tasks/velocity/mdp/rewards.py',
        ROOT/'src/tasks/velocity/config/g1/env_cfgs.py',audit.ENCODER,
        ROOT/'tests/test_g1_speed_priority_probe_20261026.py']
    controls=[]
    for seed in SEEDS:
        source,_=helper.source_checkpoint('learned',seed)
        folder=CACHE/'training_interventions/training'/f'learned_original_reward_{seed}'
        evaluation=CACHE/'rollouts'/f'training_holdout_learned_original_reward_{seed}'
        m=read(folder/'manifest.json');ev=read(evaluation/'manifest.json')
        assert m['status']==ev['status']=='COMPLETED'
        assert m['additional_completed_updates']==100 and m['collected_transitions']==153600
        assert sha(source)==m['source_checkpoint_sha256']
        assert sha(folder/'policy.onnx')==m['policy_sha256']==ev['policy_sha256']
        controls.append(dict(train_seed=seed,source_checkpoint=str(source),source_sha256=sha(source),
            cached_training=str(folder),cached_evaluation=str(evaluation)))
        paths.extend([source,*[folder/n for n in ('manifest.json','model_600.pt','policy.onnx','env.yaml','agent.yaml')],
                      *[evaluation/n for n in ('manifest.json','episodes.json','env.yaml')]])
        episodes=read(evaluation/'episodes.json')
        assert {r['seed'] for r in episodes}==set(EVAL) and len(episodes)==3
        for row in episodes:
            p=evaluation/row['raw_path'];assert sha(p)==row['raw_sha256'];paths.append(p)
    sources=[dict(path=str(p),sha256=sha(p)) for p in paths]
    target=OUT/'prospective_protocol.json'
    if target.exists():
        old=read(target);assert old['sources']==sources,'Frozen experiment inputs changed'
        return old
    from datetime import datetime,timezone
    order=list(SEEDS);random.Random(20263001).shuffle(order)
    value=dict(status='LOCKED_BEFORE_NEW_TRAINING',at=datetime.now(timezone.utc).isoformat(),
        hypothesis='Relative speed tracking reward is insufficient for the adapted E gait',
        classification='POST_RESULTS_EXPLORATORY; cached control results already known',
        sources=sources,controls=controls,train_seeds=SEEDS,eval_seeds=EVAL,run_order=order,
        intervention='Only track_linear_velocity.weight 1 to 2; pose and all other terms retained',
        restore='Same source501 actor, critic, optimizer and adaptive LR as cached control',
        budget=dict(formal_training_runs=3,updates_each=100,transitions_each=153600,
                    smoke_runs=2,smoke_updates_each=2,formal_evaluation_episodes=9),
        primary_family=['speed_priority601_minus_original_reward601_full20_velocity_MAE'],
        inference_unit='Three original PPO training seeds; rollout slots nested within policy',
        guardrails=dict(new_falls=0,mean_tilt_increase_rad=.02,mean_heading_increase_rad=.2,
                        mean_initial_direction_displacement_loss_m=.5),
        branch_rules='All three MAE decreases and seed-level 95% CI below0 with all guardrails: '
                     'support for sensitivity. MAE benefit with guardrail failure: tradeoff. Mixed/CI crossing0: '
                     'inconclusive. No benefit: this weight does not repair. No extra weights/budget.',
        no_exact_physics_reproducibility_claim=True,
        software={n:importlib.metadata.version(n) for n in ('torch','numpy','mjlab','mujoco','warp-lang','rsl-rl-lib')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        user_authorization='User asked to continue diagnosing poor E results; local bounded intervention; no remote cost')
    write(target,value)
    return value


def verify_training(folder,seed):
    import yaml
    cached=CACHE/'training_interventions/training'/f'learned_original_reward_{seed}'
    a,b=read(cached/'manifest.json'),read(folder/'manifest.json')
    for field in ('source_checkpoint_sha256','initial_actor_sha256','initial_critic_sha256',
                  'initial_learning_rate','initial_optimizer_state_entries','continuation_rng_seed','startup_model_hashes'):
        assert a[field]==b[field],field
    def tree(node,path=()):
        if isinstance(node,yaml.MappingNode):
            return {k.value:tree(v,path+(k.value,)) for k,v in node.value}
        if isinstance(node,yaml.SequenceNode):
            return [tree(v,path+(str(i),)) for i,v in enumerate(node.value)]
        if path==('rewards','track_linear_velocity','weight'):
            return 'CONTROLLED_SPEED_WEIGHT'
        return [node.tag,node.value]
    an,bn=[yaml.compose((p/'env.yaml').read_text(encoding='utf-8')) for p in (cached,folder)]
    assert tree(an)==tree(bn),'Unplanned environment change'
    assert sha(cached/'agent.yaml')==sha(folder/'agent.yaml'),'Unplanned agent change'
    assert b['encoder_frozen_verified'] and b['status']=='COMPLETED'
    return dict(train_seed=seed,configuration_only_speed_weight_changed=True,
        source_actor_critic_learning_rate_physics_matched=True,
        source_checkpoint_sha256=b['source_checkpoint_sha256'])


def setup():
    from scripts import g1_phase_feedback_probe_20261026 as prior
    study,torch=prior.setup()
    from scripts import diagnose_g1_best_e_training_20261026 as helper
    from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit
    original=study.make_cfg
    def changed(*args,**kwargs):
        return configure_speed(original(*args,**kwargs))
    study.make_cfg=changed
    audit.OUT=OUT
    original_write=helper.write
    def tagged_write(path,record):
        if 'mode' in record:
            record=dict(record,execution_helper_mode=record['mode'],mode='speed_priority',
                scientific_treatment='track_linear_velocity.weight 1 to 2; pose retained',
                speed_weight=2.,intervention_script_sha256=sha(__file__),
                protocol_sha256=sha(OUT/'prospective_protocol.json'))
        return original_write(path,record)
    helper.write=tagged_write
    return study,torch,helper,audit,original


def queue():
    protocol=lock()
    study,torch,helper,audit,original_cfg=setup()
    completed=[]
    def status(state,**fields):
        write(OUT/'queue_status.json',dict(status=state,at=audit.now(),completed=completed,**fields))
    try:
        for index,seed in enumerate([SEEDS[-1],SEEDS[-1],*protocol['run_order']]):
            smoke=index<2
            label=f'smoke_{index+1}' if smoke else f'train_{seed}'
            helper.OUT=OUT/label if smoke else OUT
            folder=helper.OUT/('smoke' if smoke else 'training')/f'learned_original_reward_{seed}'
            status('TRAINING',current=label)
            (OUT/'logs').mkdir(parents=True,exist_ok=True)
            logpath=OUT/'logs'/(label+'.log')
            with logpath.open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                helper.train('learned',seed,'original_reward',smoke=smoke)
            check=verify_training(folder,seed)
            record=read(folder/'manifest.json')
            assert record['collected_transitions']==(2 if smoke else 100)*64*24
            write(folder/'comparison_integrity.json',check)
            completed.append(label);gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(completed=label)),flush=True)
        # Evaluation uses the original reward configuration, matching cached controls;
        # weights do not enter frozen-policy action inputs or dynamics.
        study.make_cfg=original_cfg
        original_policy=audit.checkpoint_policy
        for seed in SEEDS:
            name=f'speed_priority_{seed}'
            status('EVALUATING',current=name)
            folder=OUT/'training'/f'learned_original_reward_{seed}'
            audit.checkpoint_policy=lambda a,s,k,p=folder/'policy.onnx':p
            target=OUT/'rollouts'/name
            if target.exists():
                assert read(target/'manifest.json')['status']=='COMPLETED','Preserve incomplete evaluation'
            else:
                args=SimpleNamespace(name=name,trained='learned',train_seed=seed,phase='raw',offset=0.,
                    checkpoint=600,eval_seeds=list(EVAL),scenarios=['steady'],horizon=20.)
                with (OUT/'logs'/(name+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    audit.local_rollout(args)
                record=read(target/'manifest.json')
                record['scientific_treatment']='speed_priority601; frozen deterministic policy'
                record['intervention_script_sha256']=sha(__file__)
                record['evaluation_reward_weight']=1.
                record['evaluation_reward_note']='Original evaluation configuration; raw scores recomputed independently'
                write(target/'manifest.json',record)
            completed.append(name);gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(completed=name)),flush=True)
        audit.checkpoint_policy=original_policy
        status('COMPLETED',formal_training_runs=3,formal_episodes=9)
    except BaseException as error:
        status('FAILED',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['lock','queue'],required=True)
    args=parser.parse_args()
    lock() if args.stage=='lock' else queue()
