"""Locked paired comparison of deterministic and original Gaussian policies."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import traceback

import numpy as np

from scripts import g1_phase_feedback_probe_20261026 as previous
from scripts.audit_g1_training_mechanism_20261026 import longpath, read, sha

ROOT = previous.ROOT
OUT = ROOT / 'results/g1_training_mechanism_audit_20261026/policy_sampling'
SEEDS = (20261901, 20261902, 20261903)
EVAL = (20262811, 20262812, 20262813)
TRAIN = ROOT / 'results/g1_stateful_flat_ppo_best_e_20261026/remote_download/snapshot/results/g1_stateful_flat_ppo_best_e_20261026/training'


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    longpath(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def lock():
    path = OUT / 'prospective_protocol.json'
    jobs = [(arm, mode, seed) for seed in SEEDS for arm in ('clock', 'learned')
            for mode in ('deterministic', 'sampled')]
    random.Random(20262900).shuffle(jobs)
    value = dict(status='LOCKED_BEFORE_SAMPLING_ROLLOUTS',
        at=previous.prior.now(), script_sha256=sha(__file__),
        previous_probe_sha256=sha(previous.__file__),
        E_sha256=previous.prior.ENCODER_SHA, PPO_training=False,
        completed_policy_updates=501, train_seeds=SEEDS, eval_seeds=EVAL,
        scenario='steady', target_mps=.5, horizon_seconds=20., formal_episodes=36,
        conditions=['deterministic mean', 'mean + learned checkpoint sigma * independent standard normal'],
        noise_seed_rule='numpy SeedSequence([20262910, train_seed, eval_seed]); same epsilon across clock/E',
        run_order=jobs, inference_unit='three original PPO training seeds; three rollout slots nested',
        primary_comparison_family=['E_sampled_minus_deterministic', 'clock_sampled_minus_deterministic',
                                   'phase_by_sampling_interaction'],
        primary_metric='full 20-second body-forward velocity MAE; zeros for remainder after fall',
        guardrails=['falls', 'root tilt', 'absolute heading change', 'initial direction displacement'],
        interpretation='Exploratory fixed-policy deployment test. Same conditional Gaussian as training, '
                       'but observation noise and training physics/command distributions are not reproduced. '
                       'Not a causal test of training-time reliance on noise or other training settings.',
        source_policies=[dict(arm=arm, seed=seed,
            checkpoint_sha256=sha(TRAIN/f'flat_{arm}_seed{seed}'/'model_500.pt'),
            policy_sha256=sha(TRAIN/f'flat_{arm}_seed{seed}'/'policy.onnx'))
            for seed in SEEDS for arm in ('clock','learned')])
    if path.exists():
        old=read(path)
        assert old['script_sha256']==value['script_sha256']
        assert old['previous_probe_sha256']==value['previous_probe_sha256']
        assert old['source_policies']==value['source_policies']
        return old
    write('prospective_protocol.json',value)
    return value


def rollout(arm, mode, seed, *, smoke=False):
    study, torch = previous.setup()
    from scripts.g1_stateful_online import learned_phase
    from scripts.g1_closed_loop_metrics import summarize_episode
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    import onnxruntime as ort

    name=('smoke_' if smoke else '')+f'{arm}_{mode}_{seed}'
    directory=OUT/'rollouts'/name
    if directory.exists():
        raise FileExistsError('Preserve existing attempt: '+str(directory))
    directory.mkdir(parents=True)
    seeds=EVAL[:1] if smoke else EVAL
    steps=10 if smoke else 1000
    folder=TRAIN/f'flat_{arm}_seed{seed}'
    saved=torch.load(longpath(folder/'model_500.pt'),map_location='cpu',weights_only=False)
    sigma=saved['actor_state_dict']['distribution.std_param'].numpy().copy()
    assert sigma.shape==(29,) and np.isfinite(sigma).all() and (sigma>0).all()
    table=np.stack([np.random.default_rng(np.random.SeedSequence([20262910,seed,eval_seed])).standard_normal((1000,29)).astype(np.float32) for eval_seed in seeds])
    scale=0. if mode=='deterministic' else 1.

    def phase_input(env, encoder_path):
        live=learned_phase(env,encoder_path)
        return live if arm=='learned' else observations.phase(env,.6,'twist')

    cfg=study.make_cfg('flat','learned',20261200,play=True,num_envs=len(seeds),horizon=20.)
    for group in ('actor','critic'):
        cfg.observations[group].terms['phase']=replace(cfg.observations[group].terms['phase'],func=phase_input)
    dump_yaml(directory/'env.yaml',asdict(cfg))
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    policy=ort.InferenceSession(str(longpath(folder/'policy.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
    manifest=dict(status='RUNNING',started_at=previous.prior.now(),terrain='flat',arm=arm,mode=mode,
        train_seed=seed,seeds=list(seeds),scenarios=['steady'],horizon=steps*.02,
        parallel_worlds=len(seeds),policy=str(folder/'policy.onnx'),policy_sha256=sha(folder/'policy.onnx'),
        checkpoint_sha256=sha(folder/'model_500.pt'),encoder_sha256=previous.prior.ENCODER_SHA,
        action_sigma=sigma.tolist(),noise_table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        noise_scale=scale,experiment_script_sha256=sha(__file__),
        common_configuration_sha256=study.common_cfg_digest(cfg),
        packages={n:importlib.metadata.version(n) for n in ('torch','mjlab','mujoco','warp-lang','rsl-rl-lib')})
    write(Path('rollouts')/name/'manifest.json',manifest)
    env=None
    try:
        env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        manifest['startup_model_hashes']={key:hashlib.sha256(np.ascontiguousarray(getattr(env.sim.model,key).detach().cpu().numpy()).tobytes()).hexdigest() for key in ('body_mass','body_ipos','geom_friction')}
        robot,contacts=env.scene['robot'],env.scene['feet_ground_contact']
        for i,eval_seed in enumerate(seeds):
            random.seed(eval_seed);np.random.seed(eval_seed);torch.manual_seed(eval_seed);torch.cuda.manual_seed_all(eval_seed)
            env.reset(seed=eval_seed,env_ids=torch.tensor([i],device=env.device))
        origins=env.scene.env_origins.detach().cpu().numpy().copy()
        keys=('time','command','position','quaternion','velocity','angle','contact','joint_pos',
              'actions','mean_actions','standard_normal','gravity','observation','phase_vector','shadow_phase','episode_step')
        raw=[{k:[] for k in keys} for _ in seeds]
        done,fell=np.zeros(len(seeds),bool),np.zeros(len(seeds),bool)
        for step in range(steps):
            env.command_manager.get_command('twist')[:]=torch.tensor([.5,0.,0.],device=env.device)
            env.observation_manager._obs_buffer=None
            obs=env.observation_manager.compute()['actor'].detach().cpu().numpy().astype(np.float32)
            phase=phase_input(env,str(previous.prior.ENCODER)).detach().cpu().numpy().copy()
            np.testing.assert_allclose(obs[:,9:11],phase,atol=1e-7)
            mean=np.concatenate([policy.run(None,{policy.get_inputs()[0].name:x[None]})[0] for x in obs])
            action=mean+scale*sigma*table[:,step]
            assert np.isfinite(action).all() and np.isfinite(obs).all()
            if scale==0:
                np.testing.assert_array_equal(action,mean)
            values={key:value.detach().cpu().numpy().copy() for key,value in (
                ('position',robot.data.root_link_pos_w),('quaternion',robot.data.root_link_quat_w),
                ('velocity',robot.data.root_link_lin_vel_b),('gravity',robot.data.projected_gravity_b),
                ('joint_pos',observations._joint_positions_in_order(robot,robot.data.joint_pos,G1_JOINT_ORDER)),
                ('contact',contacts.data.current_contact_time>0),('shadow_phase',env._new_e_online_phase.phase),
                ('episode_step',env.episode_length_buf))}
            values.update(angle=np.arctan2(phase[:,0],phase[:,1]),phase_vector=phase,actions=action,
                          mean_actions=mean,standard_normal=table[:,step],observation=obs)
            for i in np.flatnonzero(~done):
                raw[i]['time'].append(step*.02);raw[i]['command'].append(.5)
                for key,value in values.items():
                    raw[i][key].append(value[i].copy())
            action[done]=0
            result=env.step(torch.as_tensor(action,device=env.device))
            terminated=result[2].detach().cpu().numpy().astype(bool)
            assert not np.any(result[3].detach().cpu().numpy().astype(bool)&~done)
            fell|=terminated&~done;done|=terminated
            if step%250==249:
                print(json.dumps(dict(name=name,seconds=(step+1)*.02,active=int((~done).sum()))),flush=True)
            if done.all():
                break
        episodes=[]
        for i,eval_seed in enumerate(seeds):
            record={k:np.asarray(v) for k,v in raw[i].items()}
            path=directory/f'steady_{eval_seed}.npz'
            np.savez_compressed(path,**record)
            row=summarize_episode(dict(record,scenario='steady'),np.full(steps,.5),.02,bool(fell[i]))
            row.update(scenario='steady',seed=eval_seed,raw_path=path.name,raw_sha256=sha(path),initial_env_origin=origins[i].tolist())
            episodes.append(row)
        write(Path('rollouts')/name/'episodes.json',episodes)
        manifest.update(status='COMPLETED',ended_at=previous.prior.now(),episode_count=len(episodes))
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write(Path('rollouts')/name/'manifest.json',manifest)
        if env is not None:
            env.close()


def queue():
    protocol=lock()
    completed=[]
    jobs=[('learned','deterministic',SEEDS[0],True),('learned','sampled',SEEDS[0],True)]
    jobs += [(arm,mode,seed,False) for arm,mode,seed in protocol['run_order']]
    try:
        for arm,mode,seed,smoke in jobs:
            name=('smoke_' if smoke else '')+f'{arm}_{mode}_{seed}'
            path=OUT/'rollouts'/name/'manifest.json'
            if path.exists():
                assert read(path)['status']=='COMPLETED','Preserve failed attempt: '+name
                completed.append(name)
                continue
            write('queue_status.json',dict(status='RUNNING',current=name,at=previous.prior.now(),completed=completed))
            (OUT/'logs').mkdir(parents=True,exist_ok=True)
            with (OUT/'logs'/f'{name}.log').open('w',encoding='utf-8') as log:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    rollout(arm,mode,seed,smoke=smoke)
            completed.append(name)
            print(json.dumps(dict(completed=name,groups=len(completed))),flush=True)
            gc.collect()
        write('queue_status.json',dict(status='COMPLETED',at=previous.prior.now(),completed=completed,formal_episodes=36))
    except BaseException as error:
        write('queue_status.json',dict(status='FAILED',at=previous.prior.now(),current=name,completed=completed,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['lock','queue'],required=True)
    args=parser.parse_args()
    lock() if args.stage=='lock' else queue()
