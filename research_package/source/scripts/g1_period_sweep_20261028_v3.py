"""Version 3: the same sweep, run on a reproducible backend.

Version 1 failed its gate because it used the confirmation evaluation seeds on
the S maps (fixed in v2) and, more importantly, because the GPU backend is not
reproducible: two identical invocations of the *same* script, with bit-identical
initial state, diverged at the first ``env.step()``.  A fixed-action probe with
the policy removed reproduced that divergence, and the CPU backend did not
(bitwise identical over 200 steps x 2 worlds).

So v3 adds a ``--device`` switch and defaults the results to a separate
directory; the pre-registered comparisons are only meaningful on a reproducible
backend.  Everything else is v2: same slots, same periods, same maps, same
rollout, same metrics.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import random
import subprocess
import sys
import traceback

import numpy as np

from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_confirmation_contract_20261028 as c
from scripts import g1_rough_confirmation_experiment_20261028 as ce
from scripts import g1_rough_domain_contract_20261028 as d

ROOT = a.ROOT
OUT = ROOT / 'results/g1_period_sweep_cpu_20261028'
NOMINAL_PERIOD = .6
PERIODS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
S_MAPS = (20264701, 20264702, 20264703)
NEW_MAPS = tuple(range(20269101, 20269111))
PREFIX = ("import site;site.addsitedir('C:/Users/郑傅韬/AppData/Roaming/Python/"
          "Python313/site-packages');import runpy,sys;")


def probe_slots(tag):
    """P replays the audited S condition, so it must use the S initial states.

    The confirmation contract's ``world_slots()`` enumerates 20264911-20264915,
    which belong to the confirmation stage.  Reusing them against the S maps
    silently changes the condition the reproduction gate is supposed to check.
    """
    if tag == 'P' or tag.startswith('R'):
        if d.TRAIN_SEEDS != c.TRAIN_SEEDS:
            raise ValueError('training sources differ between S and confirmation')
        return [dict(train_seed=s, eval_seed=e, patch_col=i)
                for s in d.TRAIN_SEEDS for i, e in enumerate(d.EVAL_SEEDS)]
    return c.world_slots()


def plan():
    """No sampling: enumerate the exact job list and the reproduction gate."""
    p_jobs = [dict(probe='P', method='RC_clock', terrain=t, period=p)
              for p in PERIODS for t in S_MAPS]
    m_jobs = [dict(probe='M', method=m, terrain=t, period=NOMINAL_PERIOD)
              for t in NEW_MAPS for m in ('RE_live', 'RC_clock')]
    return dict(periods=list(PERIODS), s_maps=list(S_MAPS), new_maps=list(NEW_MAPS),
                nominal=NOMINAL_PERIOD, p_jobs=p_jobs, m_jobs=m_jobs,
                p_eval_seeds=list(d.EVAL_SEEDS), m_eval_seeds=list(c.EVAL_SEEDS),
                reproduction_gate=dict(period=.6, terrain=list(S_MAPS),
                                       expected_completed=22, expected_fell=46,
                                       expected_displacement_m=4.2437,
                                       tolerance_m=.01))


def folder(tag, method, terrain, period, suffix=''):
    if method.startswith('RE'):
        return OUT / tag / f'{method}_terrain{terrain}{suffix}'
    return OUT / tag / f'{method}_terrain{terrain}_p{period:.2f}{suffix}'


def scenario_suffix(scenarios):
    """A run restricted to some scenarios gets its own folder, so a full run
    and a split run can coexist without overwriting each other."""
    return '' if scenarios is None else '_' + '+'.join(scenarios)


def apply_period(cfg, period):
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['phase'].params['period'] = float(period)
    return cfg


def rollout(method, terrain_seed, period, tag, smoke=False, device='cuda:0',
            scenarios=None):
    study, torch = a.setup()
    import onnxruntime as ort
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from scripts.g1_stateful_online import learned_phase as live_learned_phase
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER

    learned = method.startswith('RE')
    name = f'{method}_terrain{terrain_seed}' + ('' if learned else f'_p{period:.2f}')
    out = folder(tag, method, terrain_seed, period, scenario_suffix(scenarios))
    if out.exists():
        prior = (json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
                 if (out / 'manifest.json').exists() else {})
        if prior.get('status') == 'COMPLETED':
            raise ValueError('completed output exists; refusing to overwrite: ' + str(out))
    else:
        out.mkdir(parents=True)
    all_slots = probe_slots(tag)
    slots = all_slots[:1] if smoke else all_slots
    batch = len(slots)
    limit = 10 if smoke else 1000
    scenarios = ('steady',) if smoke else tuple(scenarios or c.SCENARIOS)
    entries = [ce.policy_entry(method, s['train_seed']) for s in slots]

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions, names = [], []
    for entry in entries:
        session = ort.InferenceSession(entry['policy'], sess_options=options,
                                       providers=['CPUExecutionProvider'])
        if session.get_inputs()[0].shape != [1, 98]:
            raise ValueError('policy actor shape mismatch')
        sessions.append(session)
        names.append(session.get_inputs()[0].name)

    cfg = ce.make_cfg(study, method, terrain_seed, batch)
    if not learned:
        apply_period(cfg, period)
    dump_yaml(out / 'env.yaml', asdict(cfg))
    record = dict(status='RUNNING', started_at=a.now(), probe=tag, method=method,
                  device=device,
                  terrain_seed=terrain_seed, period=(None if learned else period),
                  smoke=smoke, policies=entries, slots=slots, scenarios=scenarios,
                  eval_seeds=sorted({s['eval_seed'] for s in slots}),
                  common_cfg_sha256=study.common_cfg_digest(cfg),
                  encoder_sha256=a.ENCODER_SHA, transitions=0, episode_count=0,
                  PPO_updates=0, Adam_steps=0)
    a.write(out / 'manifest.json', record)
    env = None
    episodes = []
    try:
        ce.seed_all(torch, 20264200)
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        if abs(float(env.step_dt) - .02) > 1e-9:
            raise ValueError('50Hz required')
        model = env.sim.mj_model
        foot_ids = {i for i in range(model.ngeom)
                    if any((model.geom(i).name or '').split('/')[-1]
                           == f'{side}_foot{k}_collision'
                           for side in ('left', 'right') for k in range(1, 8))}
        if len(foot_ids) != 14:
            raise ValueError('missing native 14 foot collision geometries')
        terrain_ids = set(int(x) for x in env.scene.terrain.indexing.geom_ids.tolist())
        hfield_ids = np.array([g for g in terrain_ids
                               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD],
                              dtype=int)
        if terrain_seed and len(hfield_ids) != 5:
            raise ValueError('exactly five generated hfields required')
        robot = env.scene['robot']
        sensor = env.scene['feet_ground_contact']
        site_ids, _ = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        for scenario in scenarios:
            for i, s in enumerate(slots):
                ce.seed_all(torch, s['eval_seed'])
                env.reset(seed=s['eval_seed'],
                          env_ids=torch.tensor([i], device=env.device))
            origins = env.scene.env_origins.cpu().numpy().copy()
            if terrain_seed:
                expected = env.scene.terrain.terrain_origins[
                    0, torch.arange(batch, device=env.device) % 5].cpu().numpy()
                np.testing.assert_array_equal(origins, expected)
            assigned, bounds = ce.assigned_patches(model.geom_pos, hfield_ids,
                                                  origins, bool(terrain_seed))
            physical = ce.physical_snapshot(env)
            physical_path = out / f'{scenario}_physics.npz'
            np.savez_compressed(physical_path, **physical)
            initial = {k: v.detach().cpu().numpy().copy() for k, v in (
                ('position', robot.data.root_link_pos_w),
                ('quaternion', robot.data.root_link_quat_w),
                ('joint_pos', robot.data.joint_pos),
                ('joint_vel', robot.data.joint_vel),
                ('root_velocity', robot.data.root_link_vel_w))}
            pairing = dict(common_cfg_sha256=record['common_cfg_sha256'],
                           physical_sha256=c.array_digest(physical),
                           initial={k: v.tolist() for k, v in initial.items()},
                           origins=origins.tolist(), slots=slots,
                           command=[.5, 0., 0.],
                           assigned=[sorted(x) for x in assigned],
                           bounds=bounds.tolist())
            ce.pairing_guard(OUT / tag / 'pairing' /
                             f'terrain{terrain_seed}_{scenario}.json', pairing)
            keys = ('time', 'velocity', 'gravity', 'joint_pos', 'joint_vel',
                    'angle', 'phase_used', 'contact', 'foot_contact_total',
                    'foot_contact_interior', 'terminated', 'timed_out')
            raw = [{k: [] for k in keys} for _ in slots]
            flattened = [{k: [] for k in ('geom', 'pos', 'dist', 'includemargin',
                                          'efc_address')} for _ in slots]
            offsets = [[0] for _ in slots]
            pos, quat = [], []
            done = np.zeros(batch, bool)
            fell = np.zeros(batch, bool)
            for step in range(limit):
                env.command_manager.get_command('twist')[:] = torch.tensor(
                    [.5, 0., 0.], device=env.device)
                if scenario == 'push' and step in (250, 500):
                    ids = torch.as_tensor(np.flatnonzero(~done), device=env.device)
                    velocity = robot.data.root_link_vel_w[ids].clone()
                    velocity[:, 1] += .3 if step == 250 else -.3
                    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                    env.sim.forward()
                    env.sim.sense()
                env.observation_manager._obs_buffer = None
                obs = env.observation_manager.compute()['actor'].cpu().numpy().astype(np.float32)
                live_phase = (live_learned_phase(env, str(a.ENCODER)) if learned
                              else observations.phase(env, period, 'twist'))
                phase_np = live_phase.cpu().numpy().copy()
                np.testing.assert_allclose(obs[:, 9:11], phase_np, atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(phase_np, axis=1), 1., atol=2e-6)
                np.testing.assert_allclose(obs[:, 6:9], np.tile([.5, 0., 0.], (batch, 1)),
                                           atol=1e-7)
                action = c.route_actions(obs, sessions, names)
                exposure, points, _ = ce.read_contacts(env, foot_ids, terrain_ids,
                                                       assigned, bounds)
                values = {k: v.detach().cpu().numpy().copy() for k, v in (
                    ('velocity', robot.data.root_link_lin_vel_b),
                    ('gravity', robot.data.projected_gravity_b),
                    ('joint_pos', observations._joint_positions_in_order(
                        robot, robot.data.joint_pos, G1_JOINT_ORDER)),
                    ('joint_vel', observations._joint_positions_in_order(
                        robot, robot.data.joint_vel, G1_JOINT_ORDER)),
                    ('contact', sensor.data.current_contact_time > 0))}
                values.update(angle=np.arctan2(phase_np[:, 0], phase_np[:, 1]),
                              phase_used=phase_np,
                              foot_contact_total=exposure['total'],
                              foot_contact_interior=exposure['interior'])
                if not all(np.isfinite(v).all() for v in values.values()):
                    raise ValueError('nonfinite raw state')
                position = robot.data.root_link_pos_w.detach().cpu().numpy().copy()
                quaternion = robot.data.root_link_quat_w.detach().cpu().numpy().copy()
                for i in np.flatnonzero(~done):
                    raw[i]['time'].append(step * .02)
                    for k, v in values.items():
                        raw[i][k].append(v[i].copy())
                    chosen = np.flatnonzero(exposure['world'] == i)
                    for k in ('geom', 'pos', 'dist', 'includemargin', 'efc_address'):
                        flattened[i][k].append(points[k][chosen].copy())
                    offsets[i].append(offsets[i][-1] + len(chosen))
                pos.append(position)
                quat.append(quaternion)
                active = int((~done).sum())
                action[done] = 0
                result = env.step(torch.as_tensor(action, device=env.device))
                record['transitions'] += active
                terminated = result[2].cpu().numpy().astype(bool)
                timed = result[3].cpu().numpy().astype(bool)
                for i in np.flatnonzero(~done):
                    raw[i]['terminated'].append(bool(terminated[i]))
                    raw[i]['timed_out'].append(bool(timed[i]))
                if np.any(timed & ~done):
                    raise ValueError('unexpected timeout before frozen horizon')
                fell |= terminated & ~done
                done |= terminated
                if done.all():
                    break
            # position/quaternion are recorded per step for every world, so index
            # them against the observation count of each (non-terminated) world.
            for i, slot in enumerate(slots):
                value = {k: np.asarray(v) for k, v in raw[i].items()}
                steps = len(value['time'])
                value['position'] = np.asarray(pos[:steps], dtype=np.float64)[:, i]
                value['quaternion'] = np.asarray(quat[:steps], dtype=np.float64)[:, i]
                value['contact_offsets'] = np.asarray(offsets[i], dtype=np.int64)
                for k, v in flattened[i].items():
                    value['points_' + k] = np.concatenate(v, axis=0)
                path = out / f'{scenario}_train{slot["train_seed"]}_eval{slot["eval_seed"]}.npz'
                np.savez_compressed(path, **value)
                if smoke:
                    score = dict(observed_steps=len(value['time']), fell=bool(fell[i]),
                                 foot_contact_total=int(np.sum(value['foot_contact_total'])),
                                 foot_contact_interior=int(np.sum(value['foot_contact_interior'])))
                else:
                    score = c.episode_metrics(value, 20., scenario, bool(fell[i]),
                                              bool(terrain_seed))
                score.update(probe=tag, method=method, terrain_seed=terrain_seed,
                             period=(None if learned else period), scenario=scenario,
                             raw_path=path.name, raw_sha256=a.sha(path), **slot)
                episodes.append(score)
            a.write(out / 'episodes.json', episodes)
        record.update(status='COMPLETED', ended_at=a.now(), episode_count=len(episodes))
    except BaseException as error:
        record.update(status='FAILED', ended_at=a.now(), error=repr(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        a.write(out / 'manifest.json', record)
        if env is not None:
            env.close()


def child(command, *args):
    return [sys.executable, '-X', 'utf8', '-c',
            PREFIX + "sys.argv=['g1_period_sweep_20261028_v2',"
            + ','.join(repr(x) for x in (command, *args))
            + "];runpy.run_module('scripts.g1_period_sweep_20261028_v3',run_name='__main__')"]


def run_child(item, log_path, device='cuda:0'):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = subprocess.os.environ.copy()
    # Match the S and confirmation parents so thread-count-dependent float
    # reduction cannot differ between the audited runs and this replay.
    env.update(PYTHONHASHSEED='20264200', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with open(log_path, 'w', encoding='utf-8') as handle:
        code = subprocess.call(child('run', '--tag', item['probe'], '--method',
                                     item['method'], '--terrain', str(item['terrain']),
                                     '--period', repr(item['period']),
                                     '--device', device),
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        raise SystemExit(f'{item} exited {code}; see {log_path}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'run', 'batch'))
    parser.add_argument('--tag', default='P')
    parser.add_argument('--methods', nargs='+', default=('RE_live', 'RC_clock'))
    parser.add_argument('--maps', nargs='+', type=int, default=None)
    parser.add_argument('--periods', nargs='+', type=float, default=None)
    parser.add_argument('--method')
    parser.add_argument('--terrain', type=int)
    parser.add_argument('--period', type=float, default=NOMINAL_PERIOD)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scenarios', nargs='+', default=None)
    args = parser.parse_args(argv)

    if args.command == 'plan':
        print(json.dumps(plan(), indent=1))
        return
    if args.command == 'run':
        rollout(args.method, args.terrain, args.period, args.tag, args.smoke, args.device,
            args.scenarios)
        return
    jobs = plan()['p_jobs'] if args.tag == 'P' else plan()['m_jobs']
    if args.tag == 'P':
        jobs = [j for j in jobs if j['period'] in set(args.periods or PERIODS)]
    jobs = [j for j in jobs
            if j['method'] in set(args.methods)
            and j['terrain'] in set(args.maps or (S_MAPS if args.tag == 'P' else NEW_MAPS))]
    done = 0
    for item in jobs:
        log = OUT / 'logs' / f"{item['probe']}_{item['method']}_t{item['terrain']}_p{item['period']:.2f}.log"
        target = folder(item['probe'], item['method'], item['terrain'], item['period'],
                        scenario_suffix(args.scenarios))
        if (target / 'manifest.json').exists():
            prior = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
            if prior.get('status') == 'COMPLETED':
                print(f'skip completed {item}', flush=True)
                continue
        print(f'run {item}', flush=True)
        run_child(item, log, args.device)
        done += 1
    print(json.dumps(dict(ran=done, total=len(jobs))), flush=True)


if __name__ == '__main__':
    main()
