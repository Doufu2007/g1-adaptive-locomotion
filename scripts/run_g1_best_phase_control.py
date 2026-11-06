"""Exploratory TCN20/clock control comparison, with immutable per-run telemetry.

Stages: prepare; rollout (frozen-policy substitution); train (matched short
adaptation from the same clock actor, fresh critic/optimizer in both arms).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'results/g1_best_phase_control_20261022'
ENCODER = OUT/'tcn20_runtime.pt'
SWEEP = ROOT/'results/g1_contact_followup_20261022/long_history_v1_replay_fixed'
POLICIES = {
    'flat': ROOT/'logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full',
    'rough': ROOT/'logs/rsl_rl/g1_velocity/2026-08-10_23-06-29_rough_baseline',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def bootstrap():
    harness = ROOT/'results/phasor_pilot_20261016/run_pilot.py'
    spec = importlib.util.spec_from_file_location('_best_control_bootstrap', harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np, torch = module.import_project()
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    torch.set_num_threads(2)
    from src.tasks.velocity.phasor.mujoco_io import install_mjlab_scene_xml_unicode_compat
    install_mjlab_scene_xml_unicode_compat()
    return np, torch


def prepare():
    np, torch = bootstrap()
    from src.tasks.velocity.phasor.encoder_tcn_context import LongContextTCNEncoder
    from scripts.run_g1_long_history_comparison import LongContextTCNEncoder as TrainingModel
    if ENCODER.exists():
        raise FileExistsError(ENCODER)
    sweep = json.loads((SWEEP/'comparison_results.json').read_text(encoding='utf-8'))
    candidates = [r for r in sweep['runs'] if r['architecture']=='tcn' and r['window_frames']==20]
    selected = min(candidates, key=lambda r: (r['validation_macro_sequence_mae_rad'], r['seed']))
    source = SWEEP/(selected['run_id']+'.pt')
    if sha(source) != selected['checkpoint_sha256']:
        raise ValueError('selected source hash mismatch')
    ref_path = ROOT/'results/g1_phase_dataset_20261019/phase_encoder_runtime_v1.pt'
    ref = torch.load(ref_path, map_location='cpu', weights_only=False)
    for key, ref_key in [('mean', 'mean'), ('std', 'std')]:
        observed = hashlib.sha256(np.ascontiguousarray(ref[ref_key]).tobytes()).hexdigest()
        if observed != sweep['normalization'][key+'_sha256']:
            raise ValueError('normalization provenance mismatch: '+key)
    state = torch.load(source, map_location='cpu', weights_only=False)
    a, b = LongContextTCNEncoder().eval(), TrainingModel().eval()
    a.load_state_dict(state, strict=True)
    b.load_state_dict(state, strict=True)
    torch.manual_seed(20261021)
    x = torch.randn(16, 20, 29)
    with torch.inference_mode():
        torch.testing.assert_close(a(x), b(x), rtol=1e-6, atol=1e-7)
    contract = dict(ref['runtime_contract'])
    contract.update(schema='g1.phase_runtime_checkpoint.experimental.v2', window=20,
                    control_dt_seconds=.02, source_checkpoint_sha256=sha(source))
    ckpt = {k: ref[k] for k in ('joint_names','mean','std','dataset_sha256','data_manifest_sha256')}
    ckpt.update(state_dict=state, architecture='tcn_long_context', joint_dim=29, window=20,
                output_convention='(sin(phi), cos(phi))', runtime_contract=contract,
                source_checkpoint_path=str(source.relative_to(ROOT)), selection=selected,
                created_at=now(), encoder_freeze_eligible=False)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ENCODER)
    result = dict(created_at=now(), source=str(source), source_sha256=sha(source),
                  encoder=str(ENCODER), encoder_sha256=sha(ENCODER),
                  selected_run=selected['run_id'], validation_mae_rad=selected['validation_macro_sequence_mae_rad'],
                  runtime_parity_passed=True, original_weights_modified=False,
                  reference_runtime_sha256=sha(ref_path))
    write(OUT/'selection.json', result)
    print(json.dumps(result), flush=True)


def make_cfg(terrain, arm, seed, *, play, num_envs, horizon):
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.velocity.config.g1.env_cfgs import _configure_flat_phasor
    cfg = load_env_cfg('Unitree-G1-'+terrain.title(), play=play)
    if arm == 'learned':
        cfg = _configure_flat_phasor(cfg, actor_phase=True, reward_phase=True)
        cfg.observations['actor'].terms['phase'].params['encoder_path'] = str(ENCODER)
        cfg.rewards['foot_gait'].params['encoder_path'] = str(ENCODER)
        cfg.events['reset_phasor_phase_history'].params['encoder_path'] = str(ENCODER)
    elif not play:
        # Compare phase sources with the same frozen contact template.
        from src.tasks.velocity.config.g1.env_cfgs import _g1_phase_runtime_artifacts
        template = _g1_phase_runtime_artifacts()['reward_template']
        cfg.rewards['foot_gait'].params['threshold'] = template['stance_threshold']
        cfg.rewards['foot_gait'].params['offset'] = list(template['offset_cycles'])
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    if terrain == 'rough':
        generator = cfg.scene.terrain.terrain_generator
        generator.num_rows, generator.num_cols = 2, 5
        generator.seed = 20261200
        generator.sub_terrains = {'random_rough': generator.sub_terrains['random_rough']}
        generator.size = (32., 8.)
    if play:
        cfg.episode_length_s = horizon + .1
        # Commands are body-forward, and controlled pushes are world lateral.
        # Fix initial yaw so both use the same axes and signed distance is valid.
        cfg.events['reset_base'].params['pose_range']['yaw'] = (0., 0.)
        for group in cfg.observations.values():
            # A one-frame manager history has the same dimensions as no history,
            # but serves stale values when commands are intervened on manually.
            group.history_length = 0
        if 'randomize_terrain' in cfg.events:
            event = cfg.events.pop('randomize_terrain')
            cfg.events = {'randomize_terrain': event, **cfg.events}
        cmd = cfg.commands['twist']
        cmd.heading_command = False
        cmd.ranges.heading = None
        cmd.rel_standing_envs = 0.
        cmd.resampling_time_range = (horizon+1, horizon+1)
        cmd.ranges.lin_vel_x = (.5, .5)
        cmd.ranges.lin_vel_y = (0., 0.)
        cmd.ranges.ang_vel_z = (0., 0.)
        cfg.events.pop('push_robot', None)
    else:
        # This pilot studies forward walking, not turning or backward motion.
        cfg.curriculum = {}
        cmd = cfg.commands['twist']
        cmd.heading_command = False
        cmd.ranges.heading = None
        cmd.rel_standing_envs = 0.
        cmd.ranges.lin_vel_x = (.3, .8)
        cmd.ranges.lin_vel_y = (0., 0.)
        cmd.ranges.ang_vel_z = (0., 0.)
    return cfg


def rollout(args):
    if args.terrain == 'rough':
        from scripts.g1_best_phase_batched import rollout_batched
        return rollout_batched(args)
    np, torch = bootstrap()
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.velocity.mdp import observations
    from scripts.g1_closed_loop_metrics import command_schedule, summarize_episode
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    directory = Path(args.output) if args.output else OUT/args.label/(args.terrain+'_'+args.arm)
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    policy_path = Path(args.policy) if args.policy else POLICIES[args.terrain]/'policy.onnx'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ort.InferenceSession(str(policy_path), sess_options=options, providers=['CPUExecutionProvider'])
    cfg = make_cfg(args.terrain, args.arm, 20261200, play=True, num_envs=1, horizon=args.horizon)
    # Serialized cfg contains callables; keep the actual manager dump separately.
    from mjlab.utils.os import dump_yaml
    dump_yaml(directory/'env.yaml', asdict(cfg))
    manifest = dict(status='RUNNING', started_at=started, terrain=args.terrain, arm=args.arm,
                    label=args.label, policy=str(policy_path), policy_sha256=sha(policy_path),
                    encoder_sha256=sha(ENCODER), horizon=args.horizon, seeds=args.seeds,
                    scenarios=args.scenarios, argv=sys.argv,
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                      [Path(__file__), ROOT/'scripts/g1_closed_loop_metrics.py',
                       ROOT/'src/tasks/velocity/mdp/observations.py',
                       ROOT/'src/tasks/velocity/phasor/encoder_tcn_context.py']})
    write(directory/'manifest.json', manifest)
    env = None
    episodes = []
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
        manifest['startup_model_hashes'] = {
            key: hashlib.sha256(np.ascontiguousarray(getattr(env.sim.model, key).detach().cpu().numpy()).tobytes()).hexdigest()
            for key in ('body_mass', 'body_ipos', 'geom_friction')
        }
        dt = float(env.step_dt)
        if abs(dt-.02) > 1e-9:
            raise ValueError('50 Hz control required')
        for scenario in args.scenarios:
            for seed in args.seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                env.reset(seed=seed)
                robot = env.scene['robot']
                contacts = env.scene['feet_ground_contact']
                initial_origin = env.scene.env_origins[0].detach().cpu().tolist()
                commands = command_schedule(round(args.horizon/dt), dt, scenario)
                raw = {key: [] for key in ('time','position','quaternion','velocity','angle','contact','joint_pos','command','actions','gravity')}
                terminated = False
                for step, command in enumerate(commands):
                    env.command_manager.get_command('twist')[:] = torch.tensor([[command, 0., 0.]], device=env.device)
                    if scenario == 'push' and step in (round(5/dt), round(10/dt)):
                        velocity = robot.data.root_link_vel_w.clone()
                        velocity[:, 1] += .3 if step == round(5/dt) else -.3
                        robot.write_root_link_velocity_to_sim(velocity, env_ids=torch.tensor([0], device=env.device))
                        env.sim.forward()
                        env.sim.sense()
                    # The command is an intervention on this control state. Invalidate
                    # the manager's value cache without pushing observation histories.
                    env.observation_manager._obs_buffer = None
                    obs = env.observation_manager.compute()
                    actor = obs['actor'].detach().cpu().numpy().astype(np.float32)
                    np.testing.assert_allclose(actor[0, 6:9], [command, 0., 0.], atol=1e-7)
                    if list(actor.shape) != list(policy.get_inputs()[0].shape):
                        raise ValueError(f'policy/actor shape mismatch {actor.shape} vs {policy.get_inputs()[0].shape}')
                    # Both signals are read on the same current state; actor/reward reuse the cache.
                    vector = observations.learned_phase(env, str(ENCODER)) if args.arm == 'learned' else observations.phase(env,.6,'twist')
                    np.testing.assert_allclose(actor[0, 9:11], vector[0].detach().cpu().numpy(), atol=1e-7)
                    action = policy.run(None, {policy.get_inputs()[0].name: actor})[0]
                    if not np.isfinite(action).all() or not np.isfinite(actor).all():
                        raise ValueError('non-finite policy input/output')
                    raw['time'].append(step*dt)
                    for key, value in [('position',robot.data.root_link_pos_w),
                                       ('quaternion',robot.data.root_link_quat_w),
                                       ('velocity',robot.data.root_link_lin_vel_b),
                                       ('gravity',robot.data.projected_gravity_b)]:
                        raw[key].append(value[0].detach().cpu().numpy().copy())
                    raw['angle'].append(float(torch.atan2(vector[0,0],vector[0,1]).item()))
                    raw['contact'].append((contacts.data.current_contact_time[0]>0).detach().cpu().numpy().copy())
                    raw['joint_pos'].append(observations._joint_positions_in_order(robot,robot.data.joint_pos,G1_JOINT_ORDER)[0].detach().cpu().numpy().copy())
                    raw['command'].append(float(command)); raw['actions'].append(action[0].copy())
                    result = env.step(torch.as_tensor(action, device=env.device))
                    if bool(result[2][0].item()):
                        terminated = True
                        break
                raw = {k: np.asarray(v) for k,v in raw.items()}
                path = directory/f'{scenario}_{seed}.npz'
                np.savez_compressed(path, **raw)
                metric_raw = dict(raw, scenario=scenario)
                summary = summarize_episode(metric_raw, commands, dt, terminated)
                summary.update(scenario=scenario, seed=seed, raw_path=path.name, raw_sha256=sha(path),
                               initial_env_origin=initial_origin,
                               max_tilt_rad=float(np.arccos(np.clip(-raw['gravity'][:,2],-1,1)).max()))
                episodes.append(summary)
                write(directory/'episodes.json', episodes)
                print(json.dumps(dict(terrain=args.terrain,arm=args.arm,scenario=scenario,seed=seed,
                     survival=summary['survival_seconds'],distance=summary['forward_displacement_m'],
                     mae=summary['mae_forward_velocity_mps'],cycles=summary['phase']['complete_cycles'])),flush=True)
        manifest.update(status='COMPLETED', ended_at=now(), episode_count=len(episodes))
    except BaseException as error:
        manifest.update(status='FAILED', ended_at=now(), error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(directory/'manifest.json', manifest)
        if env is not None:
            env.close()


def train(args):
    np, torch = bootstrap()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.utils.os import dump_yaml
    directory = Path(args.output) if args.output else OUT/'matched_finetune'/f'{args.terrain}_{args.arm}_seed{args.train_seed}'
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(args.train_seed);np.random.seed(args.train_seed);torch.manual_seed(args.train_seed);torch.cuda.manual_seed_all(args.train_seed)
    task = 'Unitree-G1-'+args.terrain.title()
    cfg = make_cfg(args.terrain,args.arm,args.train_seed,play=False,num_envs=args.num_envs,horizon=20.)
    if args.terrain == 'rough':
        # Rough training startup needs >=55 contacts (the inherited cap is 48).
        # Allocate the same capacity in both arms; this does not change forces.
        cfg.sim.nconmax = max(cfg.sim.nconmax, 128)
    agent = load_rl_cfg(task)
    agent.seed = args.train_seed
    agent.logger = 'tensorboard'
    agent.save_interval = args.iterations
    source = POLICIES[args.terrain]/'model_9900.pt'
    record = dict(status='RUNNING',started_at=now(),arm=args.arm,terrain=args.terrain,
                  ppo_seed=args.train_seed,iterations=args.iterations,num_envs=args.num_envs,
                  source_actor_checkpoint=str(source),source_actor_sha256=sha(source),
                  encoder_sha256=sha(ENCODER),fresh_critic_and_optimizer=True,
                  scope='short matched adaptation; not training from scratch or convergence confirmation')
    record['source_hashes'] = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
        ROOT/'src/tasks/velocity/mdp/observations.py', ROOT/'src/tasks/velocity/mdp/rewards.py',
        ROOT/'src/tasks/velocity/phasor/runtime.py', ROOT/'src/tasks/velocity/config/g1/rl_cfg.py']}
    write(directory/'manifest.json',record)
    env = None
    completed_transitions = 0
    try:
        raw = ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        env = RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
        original_step = env.step
        def checked_step(action):
            nonlocal completed_transitions
            command = raw.command_manager.get_command('twist')
            if not bool(((command[:, 0] >= .3-1e-6) & (command[:, 0] <= .8+1e-6)
                         & (command[:, 1:].abs().amax(dim=1) < 1e-6)).all().item()):
                raise ValueError('training command left the frozen forward-walking domain')
            result = original_step(action)
            completed_transitions += args.num_envs
            return result
        env.step = checked_step
        cls = load_runner_cls(task) or MjlabOnPolicyRunner
        # Match model initialization independently of manager startup draws.
        torch.manual_seed(args.train_seed+1000)
        torch.cuda.manual_seed_all(args.train_seed+1000)
        runner = cls(env,asdict(agent),str(directory),device='cuda:0')
        runner.load(str(source),load_cfg={'actor':True},strict=True,map_location='cuda:0')
        runner.current_learning_iteration = 0
        # Actor-only adaptation starts a new curriculum counter in both arms.
        raw.common_step_counter = 0
        if runner.alg.optimizer.state:
            raise ValueError('adaptation optimizer must be fresh')
        record['load_cfg'] = {'actor': True, 'critic': False, 'optimizer': False, 'iteration': False}
        record['num_steps_per_env'] = agent.num_steps_per_env
        record['environment_transition_budget'] = args.iterations*args.num_envs*agent.num_steps_per_env
        record['initial_actor_state_sha256'] = state_sha(runner.alg.actor)
        record['initial_critic_state_sha256'] = state_sha(runner.alg.critic)
        write(directory/'manifest.json',record)
        dump_yaml(directory/'env.yaml',asdict(cfg));dump_yaml(directory/'agent.yaml',asdict(agent))
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        if completed_transitions != record['environment_transition_budget']:
            raise ValueError('training transition count does not match frozen budget')
        record.update(status='COMPLETED',ended_at=now(),policy_sha256=sha(directory/'policy.onnx'),
                      completed_learning_iterations=runner.current_learning_iteration+1)
    except BaseException as error:
        record.update(status='FAILED',ended_at=now(),error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record['completed_environment_transitions'] = completed_transitions
        write(directory/'manifest.json',record)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','rollout','train'],required=True)
    parser.add_argument('--terrain',choices=['flat','rough'],default='flat')
    parser.add_argument('--arm',choices=['clock','learned'],default='clock')
    parser.add_argument('--label',default='direct_swap')
    parser.add_argument('--policy')
    parser.add_argument('--output')
    parser.add_argument('--horizon',type=float,default=20.)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261201,20261202,20261203])
    parser.add_argument('--scenarios',nargs='+',choices=['steady','speed_change','push'],default=['steady','speed_change','push'])
    parser.add_argument('--iterations',type=int,default=101)
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--train-seed',type=int,default=20261210)
    args=parser.parse_args()
    {'prepare':prepare,'rollout':lambda:rollout(args),'train':lambda:train(args)}[args.stage]()


if __name__=='__main__':
    main()
