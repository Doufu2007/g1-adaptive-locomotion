"""Independent seeded episodes in parallel MuJoCo worlds, scored separately."""
from dataclasses import asdict
from pathlib import Path
import json
import random
import sys
import traceback

from scripts.run_g1_best_phase_control import (
    ROOT, OUT, ENCODER, POLICIES, bootstrap, make_cfg, now, sha, write,
)


def rollout_batched(args):
    np, torch = bootstrap()
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    from scripts.g1_closed_loop_metrics import command_schedule, summarize_episode

    directory = Path(args.output) if args.output else OUT/args.label/(args.terrain+'_'+args.arm)
    directory.mkdir(parents=True, exist_ok=False)
    policy_path = Path(args.policy) if args.policy else POLICIES[args.terrain]/'policy.onnx'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ort.InferenceSession(str(policy_path), sess_options=options, providers=['CPUExecutionProvider'])
    batch = len(args.seeds)
    cfg = make_cfg(args.terrain, args.arm, 20261200, play=True, num_envs=batch, horizon=args.horizon)
    dump_yaml(directory/'env.yaml', asdict(cfg))
    manifest = dict(status='RUNNING', started_at=now(), terrain=args.terrain, arm=args.arm,
        label=args.label, policy=str(policy_path), policy_sha256=sha(policy_path),
        encoder_sha256=sha(ENCODER), horizon=args.horizon, seeds=args.seeds,
        scenarios=args.scenarios, argv=sys.argv, parallel_worlds=batch,
        reset_protocol='public env.reset(seed=seed_i, env_ids=[i]); independent terminal truncation',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'scripts/run_g1_best_phase_control.py', ROOT/'scripts/g1_closed_loop_metrics.py',
            ROOT/'src/tasks/velocity/mdp/observations.py', ROOT/'src/tasks/velocity/phasor/runtime.py']})
    write(directory/'manifest.json', manifest)
    env, episodes = None, []
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
        manifest['startup_model_hashes'] = {
            key: __import__('hashlib').sha256(np.ascontiguousarray(getattr(env.sim.model, key).detach().cpu().numpy()).tobytes()).hexdigest()
            for key in ('body_mass', 'body_ipos', 'geom_friction')}
        dt = float(env.step_dt)
        if abs(dt-.02) > 1e-9:
            raise ValueError('50 Hz control required')
        robot, contacts = env.scene['robot'], env.scene['feet_ground_contact']
        for scenario in args.scenarios:
            for index, seed in enumerate(args.seeds):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                env.reset(seed=seed, env_ids=torch.tensor([index], device=env.device))
            origins = env.scene.env_origins.detach().cpu().numpy().copy()
            commands = command_schedule(round(args.horizon/dt), dt, scenario)
            keys = ('time','position','quaternion','velocity','angle','contact','joint_pos','command','actions','gravity')
            raw = [{key: [] for key in keys} for _ in args.seeds]
            done, fell = np.zeros(batch, bool), np.zeros(batch, bool)
            for step, command in enumerate(commands):
                env.command_manager.get_command('twist')[:] = torch.tensor([command, 0., 0.], device=env.device)
                if scenario == 'push' and step in (round(5/dt), round(10/dt)):
                    ids = torch.as_tensor(np.flatnonzero(~done), device=env.device)
                    velocity = robot.data.root_link_vel_w[ids].clone()
                    velocity[:, 1] += .3 if step == round(5/dt) else -.3
                    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                    env.sim.forward(); env.sim.sense()
                env.observation_manager._obs_buffer = None
                obs = env.observation_manager.compute()['actor'].detach().cpu().numpy().astype(np.float32)
                if list(obs[:1].shape) != list(policy.get_inputs()[0].shape):
                    raise ValueError('policy/actor shape mismatch')
                np.testing.assert_allclose(obs[:, 6:9], np.tile([command, 0., 0.], (batch, 1)), atol=1e-7)
                phase = observations.learned_phase(env, str(ENCODER)) if args.arm == 'learned' else observations.phase(env,.6,'twist')
                phase = phase.detach().cpu().numpy()
                np.testing.assert_allclose(obs[:, 9:11], phase, atol=1e-7)
                action = np.concatenate([policy.run(None, {policy.get_inputs()[0].name:obs[i:i+1]})[0] for i in range(batch)])
                if not np.isfinite(obs).all() or not np.isfinite(action).all():
                    raise ValueError('non-finite policy input/output')
                values = {key: value.detach().cpu().numpy().copy() for key, value in [
                    ('position', robot.data.root_link_pos_w), ('quaternion', robot.data.root_link_quat_w),
                    ('velocity', robot.data.root_link_lin_vel_b), ('gravity', robot.data.projected_gravity_b),
                    ('joint_pos', observations._joint_positions_in_order(robot, robot.data.joint_pos, G1_JOINT_ORDER)),
                    ('contact', contacts.data.current_contact_time>0)]}
                values.update(angle=np.arctan2(phase[:, 0], phase[:, 1]), actions=action)
                for index in np.flatnonzero(~done):
                    raw[index]['time'].append(step*dt); raw[index]['command'].append(command)
                    for key, value in values.items():
                        raw[index][key].append(value[index].copy())
                action[done] = 0.
                result = env.step(torch.as_tensor(action, device=env.device))
                terminated = result[2].detach().cpu().numpy().astype(bool)
                timed_out = result[3].detach().cpu().numpy().astype(bool)
                if np.any(timed_out & ~done):
                    raise ValueError('unexpected timeout before fixed evaluation horizon')
                fell |= terminated & ~done
                done |= terminated
                if step % 200 == 199:
                    print(json.dumps(dict(terrain=args.terrain, arm=args.arm, scenario=scenario,
                        elapsed_seconds=(step+1)*dt, active_worlds=int((~done).sum()))), flush=True)
                if done.all():
                    break
            for index, seed in enumerate(args.seeds):
                record = {key: np.asarray(value) for key, value in raw[index].items()}
                path = directory/f'{scenario}_{seed}.npz'
                np.savez_compressed(path, **record)
                summary = summarize_episode(dict(record, scenario=scenario), commands, dt, bool(fell[index]))
                summary.update(scenario=scenario, seed=seed, raw_path=path.name, raw_sha256=sha(path),
                    initial_env_origin=origins[index].tolist(),
                    max_tilt_rad=float(np.arccos(np.clip(-record['gravity'][:, 2], -1, 1)).max()))
                episodes.append(summary)
                write(directory/'episodes.json', episodes)
                print(json.dumps(dict(terrain=args.terrain, arm=args.arm, scenario=scenario, seed=seed,
                    survival=summary['survival_seconds'], distance=summary['forward_displacement_m'],
                    mae=summary['mae_forward_velocity_mps'], cycles=summary['phase']['complete_cycles'])), flush=True)
        manifest.update(status='COMPLETED', ended_at=now(), episode_count=len(episodes))
    except BaseException as error:
        manifest.update(status='FAILED', ended_at=now(), error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(directory/'manifest.json', manifest)
        if env is not None:
            env.close()
