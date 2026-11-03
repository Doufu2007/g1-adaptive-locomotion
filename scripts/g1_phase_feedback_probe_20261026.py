"""Prospective frozen-policy feedback, phase-shape and dependence experiments."""
import argparse
import contextlib
from dataclasses import asdict, replace
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import traceback

import numpy as np
from scripts import diagnose_g1_best_e_closed_loop_20261026 as prior

ROOT = prior.ROOT
OUT = ROOT/'results/g1_phase_feedback_probe_20261026'
SEEDS = (20261901, 20261902, 20261903)
EVAL = (20262811, 20262812, 20262813)
E_MODES = ('live', 'self_replay', 'clock_replay', 'linear_self', 'linear_nominal', 'clock', 'frozen')
C_MODES = ('clock', 'live', 'frozen')
PLAN = ROOT/'docs/superpowers/plans/2026-10-26-g1-phase-feedback-diagnosis.md'
CONTRASTS = (
    ('feedback_under_push', 'learned_self_replay', 'learned_live', 'push'),
    ('clock_donor_replay', 'learned_clock_replay', 'learned_live', 'steady'),
    ('phase_shape', 'learned_linear_self', 'learned_live', 'steady'),
    ('phase_rate', 'learned_linear_nominal', 'learned_linear_self', 'steady'),
    ('phase_origin', 'learned_clock', 'learned_linear_nominal', 'steady'),
    ('E_phase_freeze', 'learned_frozen', 'learned_live', 'steady'),
    ('clock_phase_freeze', 'clock_frozen', 'clock_clock', 'steady'),
    ('E_policy_gap', 'learned_live', 'clock_clock', 'steady'),
    ('clock_policy_live_E', 'clock_live', 'clock_clock', 'steady'),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def replay_vectors(table, indices):
    table, indices = np.asarray(table), np.asarray(indices, dtype=int)
    if table.ndim != 3 or table.shape[-1] != 2 or indices.shape != (len(table),):
        raise ValueError('replay requires [world,time,2] and one episode index per world')
    if np.any(indices < 0) or np.any(indices >= table.shape[1]):
        raise ValueError('replay trajectory has no requested sample; refusing to pad')
    return table[np.arange(len(table)), indices]


def linearize(time, angle, frequency=None):
    time, angle = np.asarray(time), np.unwrap(np.asarray(angle, dtype=float))
    if len(time) != len(angle) or len(time) < 2 or time[-1] < 19.98-1e-6:
        raise ValueError('linearization requires a complete 20-second source')
    if not np.isfinite(angle).all() or np.any(np.diff(angle) < -1e-5):
        raise ValueError('nonmonotone/nonfinite source phase')
    start = int(np.flatnonzero(time >= 3.)[0])
    measured = (angle[-1]-angle[start])/(2*np.pi*(time[-1]-time[start]))
    rate = float(measured if frequency is None else frequency)
    if not 0 < rate <= 4:
        raise ValueError('invalid average source frequency')
    theta = angle[0] + 2*np.pi*rate*(time-time[0])
    return np.stack((np.sin(theta), np.cos(theta)), -1).astype(np.float32), dict(
        initial_phase_rad=float(angle[0]), frequency_hz=rate, source_after3_frequency_hz=float(measured))


def setup():
    from scripts import run_g1_best_phase_control as base
    from scripts.run_g1_cause_audit_windows_20261026 import bootstrap
    from scripts.run_g1_best_e_ppo_20261026 import install
    install()
    from scripts import g1_stateful_flat_ppo as study
    base.bootstrap = study.base.bootstrap = study.previous.bootstrap = bootstrap
    np_, torch = study.bootstrap()
    assert prior.sha(prior.ENCODER) == prior.ENCODER_SHA
    return study, torch


def source_records(arm, mode, seed):
    name = f'{arm}_{mode}_{seed}'
    folder = OUT/'rollouts'/name
    manifest = read(folder/'manifest.json')
    assert manifest['status'] == 'COMPLETED'
    result = []
    for eval_seed in EVAL:
        row = next(r for r in read(folder/'episodes.json') if r['scenario'] == 'steady' and r['seed'] == eval_seed)
        path = folder/row['raw_path']
        assert prior.sha(path) == row['raw_sha256']
        with np.load(path, allow_pickle=False) as raw:
            record = {k:raw[k].copy() for k in ('time', 'phase_vector', 'shadow_phase', 'angle')}
        assert len(record['time']) == 1000 and not row['fell'], 'missing source tail'
        result.append((record, dict(source=name, eval_seed=eval_seed, raw_sha256=row['raw_sha256'])))
    return result


def treatment_table(mode, seed):
    if mode in ('live', 'clock'):
        return None, []
    if mode == 'frozen':
        return None, []
    donor = source_records('clock', 'clock', seed) if mode == 'clock_replay' else source_records('learned', 'live', seed)
    tables, info = [], []
    for record, provenance in donor:
        if mode == 'clock_replay':
            theta = record['shadow_phase']
            table = np.stack((np.sin(theta), np.cos(theta)), -1).astype(np.float32)
            provenance['phase_source'] = 'shadow E on clock-policy steady trajectory'
        elif mode == 'self_replay':
            table = record['phase_vector'].astype(np.float32)
        else:
            table, derived = linearize(record['time'], record['angle'],
                                      frequency=1/.6 if mode == 'linear_nominal' else None)
            provenance.update(derived)
        # The terminal observation at 20.00s is unused by the policy but is
        # computed by env.step. Give it the last value and record this explicitly.
        tables.append(np.concatenate((table, table[-1:]), axis=0))
        info.append(provenance)
    return np.stack(tables), info


def rollout(arm, mode, seed, *, smoke=False):
    study, torch = setup()
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from scripts.g1_stateful_online import learned_phase
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
    from scripts.g1_closed_loop_metrics import command_schedule, summarize_episode
    from scripts.g1_phase_adaptation_20261024 import common_cfg_digest

    name = ('smoke_' if smoke else '')+f'{arm}_{mode}_{seed}'
    directory = OUT/'rollouts'/name
    directory.mkdir(parents=True, exist_ok=False)
    eval_seeds = EVAL if not smoke else EVAL[:1]
    horizon = .2 if smoke else 20.
    table, source = treatment_table(mode, seed)
    table_t = torch.as_tensor(table, device='cuda:0') if table is not None else None
    if table_t is not None and smoke:
        table_t = table_t[:1]
    frozen = {}

    def phase_input(env, encoder_path):
        live = learned_phase(env, encoder_path)
        if mode == 'live':
            return live
        if mode == 'clock':
            return observations.phase(env, .6, 'twist')
        if mode == 'frozen':
            if 'value' not in frozen:
                frozen['value'] = live.clone()
            reset = env.episode_length_buf == 0
            initial = observations.phase(env, .6, 'twist') if arm == 'clock' else live
            frozen['value'][reset] = initial[reset]
            return frozen['value']
        idx = env.episode_length_buf.long()
        if bool(((idx < 0) | (idx >= table_t.shape[1])).any()):
            raise ValueError('episode index outside recorded replay; refusing to pad')
        return table_t[torch.arange(env.num_envs, device=env.device), idx]

    cfg = study.make_cfg('flat', 'learned', 20261200, play=True, num_envs=len(eval_seeds), horizon=horizon)
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['phase'] = replace(cfg.observations[group].terms['phase'], func=phase_input)
    dump_yaml(directory/'env.yaml', asdict(cfg))
    policy_path = prior.checkpoint_policy(arm, seed, 500)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ort.InferenceSession(str(policy_path), sess_options=options, providers=['CPUExecutionProvider'])
    manifest = dict(status='RUNNING', started_at=prior.now(), terrain='flat', arm=arm, mode=mode,
        train_seed=seed, seeds=list(eval_seeds), scenarios=['steady', 'push'] if not smoke else ['steady'],
        horizon=horizon, parallel_worlds=len(eval_seeds), policy=str(policy_path),
        policy_sha256=prior.sha(policy_path), encoder_sha256=prior.ENCODER_SHA, source=source,
        experiment_script_sha256=prior.sha(__file__), common_configuration_sha256=common_cfg_digest(cfg),
        terminal_replay_sample='unused terminal observation repeats last source value at index1000',
        packages={n:importlib.metadata.version(n) for n in ('torch', 'mjlab', 'mujoco', 'warp-lang', 'rsl-rl-lib')})
    write(Path('rollouts')/name/'manifest.json', manifest)
    env, episodes = None, []
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
        manifest['startup_model_hashes'] = {key:__import__('hashlib').sha256(
            np.ascontiguousarray(getattr(env.sim.model, key).detach().cpu().numpy()).tobytes()).hexdigest()
            for key in ('body_mass', 'body_ipos', 'geom_friction')}
        assert abs(float(env.step_dt)-.02) < 1e-9
        robot, contacts = env.scene['robot'], env.scene['feet_ground_contact']
        batch = len(eval_seeds)
        for scenario in manifest['scenarios']:
            for i, eval_seed in enumerate(eval_seeds):
                random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed); torch.cuda.manual_seed_all(eval_seed)
                env.reset(seed=eval_seed, env_ids=torch.tensor([i], device=env.device))
            origins = env.scene.env_origins.detach().cpu().numpy().copy()
            commands = command_schedule(round(horizon/.02), .02, scenario)
            keys = ('time', 'command', 'position', 'quaternion', 'velocity', 'angle', 'contact',
                    'joint_pos', 'actions', 'gravity', 'observation', 'phase_vector', 'shadow_phase', 'episode_step')
            raw = [{k:[] for k in keys} for _ in eval_seeds]
            done, fell = np.zeros(batch, bool), np.zeros(batch, bool)
            for step, command in enumerate(commands):
                env.command_manager.get_command('twist')[:] = torch.tensor([command, 0., 0.], device=env.device)
                if scenario == 'push' and step in (250, 500):
                    ids = torch.as_tensor(np.flatnonzero(~done), device=env.device)
                    velocity = robot.data.root_link_vel_w[ids].clone()
                    velocity[:, 1] += .3 if step == 250 else -.3
                    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                    env.sim.forward(); env.sim.sense()
                env.observation_manager._obs_buffer = None
                obs = env.observation_manager.compute()['actor'].detach().cpu().numpy().astype(np.float32)
                phase = phase_input(env, str(prior.ENCODER)).detach().cpu().numpy().copy()
                np.testing.assert_allclose(obs[:, 9:11], phase, atol=1e-7)
                np.testing.assert_allclose(obs[:, 6:9], np.tile([command, 0., 0.], (batch, 1)), atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(phase, axis=-1), 1., atol=2e-6)
                action = np.concatenate([policy.run(None, {policy.get_inputs()[0].name:obs[i:i+1]})[0] for i in range(batch)])
                if not np.isfinite(obs).all() or not np.isfinite(action).all():
                    raise ValueError('nonfinite policy input/output')
                values = {key:value.detach().cpu().numpy().copy() for key, value in (
                    ('position', robot.data.root_link_pos_w), ('quaternion', robot.data.root_link_quat_w),
                    ('velocity', robot.data.root_link_lin_vel_b), ('gravity', robot.data.projected_gravity_b),
                    ('joint_pos', observations._joint_positions_in_order(robot, robot.data.joint_pos, G1_JOINT_ORDER)),
                    ('contact', contacts.data.current_contact_time > 0),
                    ('shadow_phase', env._new_e_online_phase.phase), ('episode_step', env.episode_length_buf))}
                values.update(angle=np.arctan2(phase[:, 0], phase[:, 1]), phase_vector=phase,
                              actions=action, observation=obs)
                for i in np.flatnonzero(~done):
                    raw[i]['time'].append(step*.02); raw[i]['command'].append(command)
                    for key, value in values.items():
                        raw[i][key].append(value[i].copy())
                action[done] = 0
                result = env.step(torch.as_tensor(action, device=env.device))
                terminated = result[2].detach().cpu().numpy().astype(bool)
                timed_out = result[3].detach().cpu().numpy().astype(bool)
                if np.any(timed_out & ~done):
                    raise ValueError('unexpected timeout before fixed horizon')
                fell |= terminated & ~done; done |= terminated
                if step % 250 == 249:
                    print(json.dumps(dict(name=name, scenario=scenario, seconds=(step+1)*.02,
                                          active_worlds=int((~done).sum()))), flush=True)
                if done.all():
                    break
            for i, eval_seed in enumerate(eval_seeds):
                record = {k:np.asarray(v) for k,v in raw[i].items()}
                path = directory/f'{scenario}_{eval_seed}.npz'
                np.savez_compressed(path, **record)
                row = summarize_episode(dict(record, scenario=scenario), commands, .02, bool(fell[i]))
                row.update(scenario=scenario, seed=eval_seed, raw_path=path.name, raw_sha256=prior.sha(path),
                           initial_env_origin=origins[i].tolist())
                episodes.append(row)
                write(Path('rollouts')/name/'episodes.json', episodes)
            print(json.dumps(dict(name=name, scenario=scenario, episodes=len(episodes))), flush=True)
        manifest.update(status='COMPLETED', ended_at=prior.now(), episode_count=len(episodes))
    except BaseException as error:
        manifest.update(status='FAILED', ended_at=prior.now(), error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(Path('rollouts')/name/'manifest.json', manifest)
        if env is not None:
            env.close()


def lock():
    path = OUT/'prospective_protocol.json'
    hashes = {p:prior.sha(ROOT/p) for p in (
        'scripts/g1_phase_feedback_probe_20261026.py', 'scripts/g1_stateful_online.py',
        'scripts/g1_stateful_flat_ppo.py', 'scripts/run_g1_cause_audit_windows_20261026.py',
        'scripts/g1_closed_loop_metrics.py', 'src/tasks/velocity/mdp/observations.py')}
    if path.exists():
        value = read(path)
        assert value['source_hashes'] == hashes, 'locked experiment source changed'
        return value
    value = dict(status='LOCKED_BEFORE_NEW_ROLLOUTS', at=prior.now(), train_seeds=SEEDS,
        eval_seeds=EVAL, horizon=20., scenarios=['steady', 'push'], conditions={'learned':E_MODES, 'clock':C_MODES},
        expected_episodes=180, PPO_training=False, source_checkpoint_completed_updates=501,
        E_sha256=prior.ENCODER_SHA, source_hashes=hashes, plan_sha256=prior.sha(PLAN),
        primary_contrasts=CONTRASTS, family_id='phase_feedback_nine_contrasts',
        inference_unit='three original PPO training seeds, evaluations nested',
        controls='self replay must reproduce steady and pre-push trajectories; no outcome-selected donors',
        terminal_replay_sample='last source value at unused terminal index1000 only',
        scope='Frozen policies on local GPU; not a causal test of pretraining history or a new training comparison')
    write('prospective_protocol.json', value)
    return value


def queue():
    lock()
    completed = []
    def run(arm, mode, seed, smoke=False):
        name = ('smoke_' if smoke else '')+f'{arm}_{mode}_{seed}'
        folder = OUT/'rollouts'/name
        if (folder/'manifest.json').exists():
            assert read(folder/'manifest.json')['status'] == 'COMPLETED', 'preserve failed attempt: '+name
            completed.append(name)
            return
        write('queue_status.json', dict(status='RUNNING', at=prior.now(), current=name, completed=completed))
        (OUT/'logs').mkdir(parents=True, exist_ok=True)
        with (OUT/'logs'/(name+'.log')).open('w', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                rollout(arm, mode, seed, smoke=smoke)
        completed.append(name)
        print(json.dumps(dict(completed=name, groups_done=len(completed))), flush=True)
        gc.collect()
    try:
        run('learned', 'live', SEEDS[0], smoke=True)
        # Donors are generated for every seed before recipient results.
        for seed in SEEDS:
            for arm, mode in (('clock', 'clock'), ('learned', 'live')):
                run(arm, mode, seed)
        rng = random.Random(20262800)
        recipients = [(arm, mode, seed) for seed in SEEDS
                      for arm, modes in (('learned', E_MODES), ('clock', C_MODES))
                      for mode in modes if (arm, mode) not in (('clock', 'clock'), ('learned', 'live'))]
        rng.shuffle(recipients)
        for arm, mode, seed in recipients:
            run(arm, mode, seed)
        write('queue_status.json', dict(status='COMPLETED', at=prior.now(), completed=completed,
                                       formal_episodes=180, formal_groups=30))
    except BaseException as error:
        write('queue_status.json', dict(status='FAILED', at=prior.now(), completed=completed,
                                       error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['lock', 'queue'])
    args = parser.parse_args()
    lock() if args.stage == 'lock' else queue()
