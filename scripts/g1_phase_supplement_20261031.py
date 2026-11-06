"""Isolated rough supplement runner. Every simulation runs in a fresh child."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
import numpy as np
from scripts import g1_phase_lr_factorial_20261027 as a
from scripts import g1_rough_confirmation_contract_20261028 as oldc
from scripts import g1_rough_confirmation_experiment_20261028 as ce
from scripts import g1_rough_domain_experiment_20261028 as de
from scripts import g1_phase_supplement_contract_20261031 as c

ROOT = a.ROOT
OUT = Path('<local-data-root>/phasor_phase_supplement_20261031')
CONTROL = ROOT / 'results/g1_phase_supplement_20261031'
SPEC = ROOT / 'docs/superpowers/specs/2026-10-31-rough-phase-supplement-design.md'
PLAN = ROOT / 'docs/superpowers/plans/2026-10-31-rough-phase-supplement.md'
PREFIX = ("import site;site.addsitedir('<user-profile>/AppData/Roaming/Python/"
          "Python313/site-packages');import runpy,sys;")


def check_disk():
    for path, minimum in ((ROOT, 10), (OUT.anchor, 15)):
        if shutil.disk_usage(path).free < minimum * 1024 ** 3:
            raise ValueError(f'{path}: below locked {minimum}GiB free disk floor')


def reserve(folder):
    folder.mkdir(parents=True, exist_ok=False)


def setup():
    return a.setup()


def map_identity(seed, physical):
    arrays = {k: physical[k] for k in ('hfield_data', 'hfield_size', 'hfield_nrow', 'hfield_ncol', 'hfield_adr')}
    digest = oldc.array_digest(arrays)
    path = OUT / 'map_identities.json'
    known = a.read(path) if path.exists() else {}
    for key, prior in known.items():
        if (int(key) == seed) != (prior == digest):
            raise ValueError('map changed or distinct declared maps have identical physical arrays')
    known[str(seed)] = digest; a.write(path, known)


def policy_entry(split, mode, seed, smoke=False):
    method = 'learned' if mode in ('live', 'constant', 'replay') else mode
    folder = OUT / ('smoke_training' if smoke else 'training') / f'{method}_{seed}'
    m = a.read(folder / 'manifest.json')
    if m['status'] != 'COMPLETED':
        raise ValueError('policy training incomplete')
    point = '2' if smoke else str(c.UPDATES)
    entry = m['checkpoints'][point]
    for key in ('path', 'policy'):
        expected = entry['sha256' if key == 'path' else 'policy_sha256']
        if a.sha(entry[key]) != expected:
            raise ValueError('trained model source changed')
    return dict(method=method, train_seed=seed, checkpoint=entry['path'],
                checkpoint_sha256=entry['sha256'], policy=entry['policy'],
                policy_sha256=entry['policy_sha256'], training_manifest=str(folder / 'manifest.json'),
                training_manifest_sha256=a.sha(folder / 'manifest.json'))


def eval_cfg(study, mode, terrain, batch):
    learned = mode in ('live', 'constant', 'replay')
    cfg = ce.make_cfg(study, 'RE_live' if learned else 'RC_clock', terrain, batch)
    cfg.seed = c.CONSTRUCT_SEED
    if not learned:
        period = .55 if mode == 'clock55' else .60
        for group in ('actor', 'critic'):
            cfg.observations[group].terms['phase'].params['period'] = period
    return cfg


def train_cfg(study, method, seed):
    from mjlab.managers.event_manager import EventTermCfg
    if method not in c.ARMS or seed not in c.NEW_SEEDS:
        raise ValueError('outside finite training matrix')
    phase = 'learned' if method == 'learned' else 'clock'
    cfg = study.make_cfg('flat', phase, seed, play=False, num_envs=64, horizon=20.)
    native = study.previous.make_cfg('rough', 'clock', seed, play=False, num_envs=64, horizon=20.)
    cfg.scene.terrain = deepcopy(native.scene.terrain)
    g = cfg.scene.terrain.terrain_generator
    g.seed = c.MAPS['training'][c.NEW_SEEDS.index(seed)]
    g.curriculum = False; g.num_rows = 1; g.num_cols = 5; g.size = (32., 16.)
    rough = deepcopy(g.sub_terrains['random_rough']); rough.proportion = 1.
    rough.noise_range = (.02, .10); rough.noise_step = .02
    rough.horizontal_scale = .1; rough.vertical_scale = .005; rough.border_width = .25
    g.sub_terrains = {'random_rough': rough}
    cfg.sim.mujoco.ccd_iterations = 500; cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.njmax = 1500; cfg.sim.nconmax = 128
    cfg.events = {'randomize_terrain': EventTermCfg(func=ce.deterministic_terrain, mode='reset'),
                  **{k: v for k, v in cfg.events.items() if k != 'randomize_terrain'}}
    if method != 'learned':
        for group in ('actor', 'critic'):
            cfg.observations[group].terms['phase'].params['period'] = .55 if method == 'clock55' else .6
    return cfg


def bank(smoke=False):
    folder = OUT / ('smoke_bank' if smoke else 'phase_bank')
    manifest = a.read(folder / 'manifest.json')
    if manifest['status'] != 'LOCKED':
        raise ValueError('phase bank not locked')
    if a.sha(folder / 'increments.npz') != manifest['sha256']:
        raise ValueError('phase bank changed')
    with np.load(folder / 'increments.npz', allow_pickle=False) as v:
        return {int(k): v[k].copy() for k in v.files}


def build_bank(smoke=False):
    folder = OUT / ('smoke_bank' if smoke else 'phase_bank'); reserve(folder)
    seeds = c.NEW_SEEDS[:1] if smoke else c.NEW_SEEDS
    values = {s: [] for s in seeds}; sources = []
    if smoke:
        folders = [OUT / 'smoke_rollouts/live']
    else:
        folders = [rollout_folder('calibration', 'live', t, s)
                   for t in c.MAPS['calibration'] for s in seeds]
    for path in folders:
        manifest = a.read(path / 'manifest.json')
        if manifest['status'] != 'COMPLETED':
            raise ValueError('calibration incomplete')
        for row in a.read(path / 'episodes.json'):
            raw = path / row['raw_path']
            if a.sha(raw) != row['raw_sha256']:
                raise ValueError('calibration source changed')
            with np.load(raw, allow_pickle=False) as v:
                values[row['train_seed']].append(v['phase_used'].copy())
            sources.append(dict(path=str(raw), sha256=row['raw_sha256'], train_seed=row['train_seed']))
    if not smoke and len(sources) != 120:
        raise ValueError('calibration matrix incomplete')
    arrays = {str(s): c.phase_increments(values[s]) for s in seeds}
    np.savez_compressed(folder / 'increments.npz', **arrays)
    a.write(folder / 'manifest.json', dict(status='LOCKED', at=a.now(), sources=sources,
        sha256=a.sha(folder / 'increments.npz'),
        means={s: float(v.mean()) for s, v in arrays.items()},
        rule='all observed within-episode increments in map/start/time order; no outcome selection'))


def rollout_folder(split, mode, terrain, seed):
    if split == 'smoke':
        return OUT / 'smoke_rollouts' / mode
    return OUT / 'rollouts' / split / f'{mode}_map{terrain}_seed{seed}'


def raw_capacity(env):
    data = env.sim.wp_data
    if int(data.nacon.numpy()[0]) >= data.contact.geom.shape[0] or int(data.nefc.numpy().max()) >= 1500:
        raise ValueError('native contact/constraint capacity saturated')


def verify_seal():
    value = a.read(CONTROL / 'protocol.json')
    for entry in value['files']:
        if a.sha(entry['path']) != entry['sha256']:
            raise ValueError('sealed input changed: ' + entry['path'])
    return value


def child_args(command, **kwargs):
    args = [command]
    for key, value in kwargs.items():
        if value is True:
            args.append('--' + key.replace('_', '-'))
        elif value is not None and value is not False:
            args.extend(['--' + key.replace('_', '-'), str(value)])
    return [sys.executable, '-X', 'utf8', '-c', PREFIX +
            'sys.argv=' + repr(['g1_phase_supplement', *args]) +
            ";runpy.run_module('scripts.g1_phase_supplement_20261031',run_name='__main__')"]


def run_child(command, log, **kwargs):
    check_disk()
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(c.CONSTRUCT_SEED), PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf8') as stream:
        result = subprocess.call(child_args(command, **kwargs), cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, env=env,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result:
        raise RuntimeError(f'{command} failed code {result}; preserved {log}')


def train(method, seed, smoke=False, initialize_only=False):
    if not smoke and not initialize_only:
        verify_seal()
    check_disk(); study, torch = setup()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from src.tasks.velocity.mdp import observations
    from scripts import diagnose_g1_initial_update_20261026 as probe
    folder = OUT / ('initialization_preflight' if initialize_only else ('smoke_training' if smoke else 'training')) / f'{method}_{seed}'
    reserve(folder); target = 0 if initialize_only else (2 if smoke else c.UPDATES); points = (0,) if initialize_only else (0, 1, target)
    record = dict(status='RUNNING', at=a.now(), method=method, seed=seed, smoke=smoke,
                  source_sha256=a.SOURCE_SHA, encoder_sha256=a.ENCODER_SHA, updates=0,
                  transitions=0, PPO_updates=0, Adam_steps=0, checkpoints={}, learn_calls=0,
                  source_simulation_resume=False, planned_updates=target,
                  protocol_sha256=a.sha(CONTROL / 'protocol.json') if not smoke and not initialize_only else None)
    a.write(folder / 'manifest.json', record); env = None
    try:
        ce.seed_all(torch, seed)
        cfg = train_cfg(study, method, seed)
        if 'foot_gait' in cfg.rewards:
            raise ValueError('phase-specific gait reward must remain removed')
        agent = load_rl_cfg('Unitree-G1-Flat'); agent.seed = seed
        agent.logger = 'tensorboard'; agent.upload_model = False; agent.save_interval = 10**9
        agent.algorithm.class_name = 'scripts.g1_source_anchor_ppo_20261027:SourceAnchoredPPO'
        agent.algorithm.learning_rate = 1e-5
        agent_dict = asdict(agent)
        if agent.num_steps_per_env != 24:
            raise ValueError('native rollout length changed')
        dump_yaml(folder / 'env.yaml', asdict(cfg)); dump_yaml(folder / 'agent.yaml', agent_dict)
        Env = de.aligned_environment_type(ManagerBasedRlEnv, torch, seed)
        raw = Env(cfg=cfg, device='cuda:0'); env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        bundle = observations._load_phasor_encoder(raw, str(a.ENCODER))
        bundle.encoder.eval().requires_grad_(False)
        encoder_before = study.base.state_sha(bundle.encoder)
        ce.seed_all(torch, seed + 1000)
        cls = load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
        runner = cls(env, agent_dict, str(folder), device='cuda:0')
        runner.load(str(a.SOURCE), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        runner.current_learning_iteration = 0; raw.common_step_counter = 0
        if runner.alg.optimizer.state:
            raise ValueError('new continuation must start with empty Adam')
        teacher_before = study.base.state_sha(runner.alg.source_teacher)
        actor_before = study.base.state_sha(runner.alg.actor)
        if teacher_before != actor_before or runner.alg.anchor_coefficient != .1:
            raise ValueError('source teacher/beta mismatch')
        if {id(p) for p in bundle.encoder.parameters()} & {id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}:
            raise ValueError('encoder entered optimizer')
        obs = env.get_observations()
        if tuple(obs['actor'].shape) != (64, 98):
            raise ValueError('actor input mismatch')
        physical = ce.physical_snapshot(raw); np.savez_compressed(folder / 'startup_physics.npz', **physical)
        map_identity(c.MAPS['training'][c.NEW_SEEDS.index(seed)], physical)
        named = de.robot_physics(raw); np.savez_compressed(folder / 'robot_physics.npz', **named)
        robot = raw.scene['robot']
        initial_arrays = {k: v.detach().cpu().numpy().copy() for k, v in (
            ('position', robot.data.root_link_pos_w), ('quaternion', robot.data.root_link_quat_w),
            ('joint_pos', robot.data.joint_pos), ('joint_vel', robot.data.joint_vel),
            ('root_velocity', robot.data.root_link_vel_w))}
        np.savez_compressed(folder / 'initial_state.npz', **initial_arrays)
        ce.seed_all(torch, seed)
        paired = dict(initial_actor=actor_before, initial_critic=study.base.state_sha(runner.alg.critic),
            teacher=teacher_before, encoder=encoder_before, empty_Adam=True,
            RNG_sha256=de.state_digest(de.rng_state(torch)),
            common_cfg_sha256=study.common_cfg_digest(cfg),
            common_agent_sha256=hashlib.sha256(json.dumps(study.canonical(agent_dict), sort_keys=True).encode()).hexdigest(),
            physics_sha256=oldc.array_digest(physical), robot_physics_sha256=de.named_array_digest(named),
            initial_state_sha256=oldc.array_digest(initial_arrays))
        ce.pairing_guard(OUT / ('init_train_pairing' if initialize_only else ('smoke_train_pairing' if smoke else 'train_pairing')) / f'{seed}.json', paired)
        a.write(folder / 'pre_sampling_pairing.json', dict(status='PASS', transitions=0, **paired))
        record.update(initial=paired, phase_config=study.canonical({g: asdict(cfg.observations[g].terms['phase']) for g in ('actor', 'critic')}))
        feet, terrain, hfields, assigned, bounds = de.terrain_metadata(raw)
        if len(hfields) != 5:
            raise ValueError('five rough patches required')
        record.update(assigned_contact_total=0, exposure_boundary_samples=0)
        native_step = env.step
        def step(action):
            if not torch.isfinite(action).all():
                raise ValueError('nonfinite action')
            command = raw.command_manager.get_command('twist')
            if not bool(((command[:, 0] >= .3-1e-6) & (command[:, 0] <= .8+1e-6) & (command[:, 1:].abs().amax(dim=1) < 1e-6)).all()):
                raise ValueError('training command outside locked range')
            record['step_in_flight'] = True
            result = native_step(action); record['transitions'] += 64; record['step_in_flight'] = False
            if not all(torch.isfinite(v).all() for v in (result[0]['actor'], result[0]['critic'], result[1])):
                raise ValueError('nonfinite training observation/reward')
            raw_capacity(raw)
            return result
        env.step = step
        def save(_path=None, infos=None):
            completed = record['updates']
            if completed not in points or str(completed) in record['checkpoints']:
                return
            if study.base.state_sha(bundle.encoder) != encoder_before or study.base.state_sha(runner.alg.source_teacher) != teacher_before:
                raise ValueError('frozen teacher/encoder changed')
            if any(float(v['step']) != 20 * completed for v in runner.alg.optimizer.state.values()):
                raise ValueError('Adam budget mismatch')
            path = folder / f'model_{completed}.pt'
            MjlabOnPolicyRunner.save(runner, str(path), dict(infos or {}, supplement_state=dict(
                completed_updates=completed, method=method, seed=seed, learning_rate=runner.alg.learning_rate,
                **de.rng_state(torch))))
            record['checkpoints'][str(completed)] = dict(path=str(path), sha256=a.sha(path))
            a.write(folder / 'manifest.json', record)
        runner.save = save; native_update = runner.alg.update
        def update():
            completed = record['updates'] + 1; check_disk()
            if record['transitions'] != completed * 1536:
                raise ValueError('rollout budget drift')
            exposure, points_raw, _ = ce.read_contacts(raw, feet, terrain, assigned, bounds)
            np.savez_compressed(folder / f'exposure_{completed}.npz', **points_raw,
                                total=exposure['total'], interior=exposure['interior'])
            record['assigned_contact_total'] += int(exposure['interior'].sum())
            record['exposure_boundary_samples'] += 1
            before = runner.alg.learning_rate
            losses, trace = a.record_update(runner.alg, native_update)
            source_trace = runner.alg.last_source_kl
            if len(trace) != 20 or len(source_trace) != 20 or any(x['beta'] != .1 for x in source_trace):
                raise ValueError('native update/source KL budget drift')
            record.update(updates=completed, PPO_updates=completed, Adam_steps=20 * completed)
            row = dict(update=completed, transitions=record['transitions'], at=a.now(),
                       lr_before=before, lr_after=runner.alg.learning_rate, losses=losses,
                       minibatches=trace, source_kl=source_trace,
                       exposure_sha256=a.sha(folder / f'exposure_{completed}.npz'))
            with (folder / 'updates.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            a.write(folder / 'progress.json', dict(at=a.now(), updates=completed,
                transitions=record['transitions'], PPO_updates=completed, Adam_steps=20 * completed,
                lr=runner.alg.learning_rate))
            return losses
        runner.alg.update = update; native_log = runner.logger.log
        def log(*args, **kwargs):
            value = native_log(*args, **kwargs); save(); return value
        runner.logger.log = log
        save()
        if initialize_only:
            record.update(status='COMPLETED', ended_at=a.now(), encoder_frozen=True, teacher_frozen=True)
            return
        record['learn_calls'] += 1
        runner.learn(num_learning_iterations=target, init_at_random_ep_len=True)
        if record['updates'] != target or record['transitions'] != target * 1536 or record['learn_calls'] != 1:
            raise ValueError('native training budget incomplete')
        if record['assigned_contact_total'] == 0:
            raise ValueError('no assigned rough contact at update boundaries')
        for point, entry in record['checkpoints'].items():
            saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
            policy = folder / f'policy_{point}.onnx'
            entry.update(probe.export_actor(saved['actor_state_dict'], policy, torch), policy=str(policy))
        record.update(status='COMPLETED', ended_at=a.now(), encoder_frozen=True, teacher_frozen=True)
    except BaseException as error:
        record.update(status='FAILED', ended_at=a.now(), error=repr(error), traceback=traceback.format_exc())
        if 'runner' in locals():
            counts = [float(v['step']) for v in runner.alg.optimizer.state.values()]
            record['actual_Adam_step_range'] = [min(counts, default=0), max(counts, default=0)]
        a.write(folder / 'manifest.json', record)
        try:
            failure_evidence(error)
        except Exception as evidence_error:
            record['evidence_error'] = repr(evidence_error)
        raise
    finally:
        a.write(folder / 'manifest.json', record)
        if env is not None:
            env.close()


def preflight():
    import gc
    check_disk(); CONTROL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    study, torch = setup()
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.velocity.mdp import observations
    rows = []; common = []; rewards = []
    for method in c.ARMS:
        cfg = train_cfg(study, method, c.NEW_SEEDS[0])
        if 'foot_gait' in cfg.rewards:
            raise ValueError('phase-dependent reward present')
        common.append(study.common_cfg_digest(cfg))
        rewards.append(study.canonical({k: asdict(v) for k, v in cfg.rewards.items()}))
        phase = study.canonical({g: asdict(cfg.observations[g].terms['phase']) for g in ('actor', 'critic')})
        if method != 'learned':
            for group in ('actor', 'critic'):
                if cfg.observations[group].terms['phase'].params['period'] != (.55 if method == 'clock55' else .6):
                    raise ValueError('wrong fixed-period training config')
        else:
            for group in ('actor', 'critic'):
                encoder_path = Path(cfg.observations[group].terms['phase'].params['encoder_path'])
                if encoder_path.resolve() != a.ENCODER.resolve() or a.sha(encoder_path) != a.ENCODER_SHA:
                    raise ValueError('learned encoder path/hash mismatch')
        mode = 'live' if method == 'learned' else method
        ce.seed_all(torch, c.CONSTRUCT_SEED)
        raw = ManagerBasedRlEnv(cfg=eval_cfg(study, mode, 20271901, 1), device='cpu')
        try:
            obs = raw.observation_manager.compute()
            if tuple(obs['actor'].shape) != (1, 98) or not torch.isfinite(obs['actor']).all():
                raise ValueError('native actor preflight failed')
            physical = ce.physical_snapshot(raw)
            de.terrain_metadata(raw)
            named = de.robot_physics(raw)
            rows.append(dict(method=method, actor_shape=list(obs['actor'].shape),
                critic_shape=list(obs['critic'].shape), phase=phase,
                named_physics_sha256=de.named_array_digest(named), physics_sha256=oldc.array_digest(physical)))
        finally:
            raw.close(); del raw; gc.collect()
    if len(set(common)) != 1 or any(r != rewards[0] for r in rewards):
        raise ValueError('training common config/reward mismatch')
    a.write(CONTROL / 'cpu_preflight.json', dict(status='PASS', at=a.now(),
        source_sha256=a.SOURCE_SHA, encoder_sha256=a.ENCODER_SHA, rows=rows,
        train_transitions=0, eval_transitions=0, PPO_updates=0, Adam_steps=0))


def smoke():
    reserve(OUT / 'smoke_started')
    if a.read(CONTROL / 'cpu_preflight.json')['status'] != 'PASS':
        raise ValueError('CPU preflight missing')
    initialization_preflight()
    for method in c.ARMS:
        run_child('train', OUT / f'smoke_train_{method}.log', mode=method, seed=c.NEW_SEEDS[0], smoke=True)
    run_child('rollout', OUT / 'smoke_live.log', split='calibration', mode='live',
              terrain=20271901, seed=c.NEW_SEEDS[0], smoke=True)
    build_bank(True)
    for mode in ('constant', 'replay', 'clock55', 'clock60'):
        run_child('rollout', OUT / f'smoke_{mode}.log', split='test', mode=mode,
                  terrain=20271901, seed=c.NEW_SEEDS[0], smoke=True)
    run_child('rollout', OUT / 'smoke_live_repeat.log', split='calibration', mode='live',
              terrain=20271901, seed=c.NEW_SEEDS[0], smoke=True, repeat=True)
    audit_smoke()


def audit_smoke():
    verify_training(True)
    for method in c.ARMS:
        path = OUT / 'initialization_preflight' / f'{method}_{c.NEW_SEEDS[0]}' / 'manifest.json'
        m = a.read(path)
        if m['status'] != 'COMPLETED' or m['transitions'] or m['PPO_updates'] or m['Adam_steps'] or m['learn_calls']:
            raise ValueError('initial-state pairing preflight incomplete/nonzero budget')
    for folder in sorted((OUT / 'smoke_rollouts').iterdir()):
        audit_rollout(folder, True)
    first = a.read(OUT / 'smoke_rollouts/live/episodes.json')[0]
    second = a.read(OUT / 'smoke_rollouts/live_repeat/episodes.json')[0]
    with np.load(OUT / 'smoke_rollouts/live' / first['raw_path']) as x, np.load(OUT / 'smoke_rollouts/live_repeat' / second['raw_path']) as y:
        if set(x.files) != set(y.files):
            raise ValueError('CPU repeat raw keys differ')
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key], err_msg='CPU repeat differs: ' + key)
    transitions = sum(a.read(p / 'manifest.json')['transitions'] for p in (OUT / 'smoke_rollouts').iterdir())
    a.write(CONTROL / 'smoke_audit.json', dict(status='PASS', at=a.now(), train_transitions=9216,
        PPO_updates=6, Adam_steps=120, eval_transitions=transitions, CPU_bitwise_repeat=True,
        initial_state_pairing_preflight=True,
        semantics='engineering smoke; distinct from formal budget and scientific results'))


def initialization_preflight():
    for method in c.ARMS:
        run_child('train', OUT / f'initialization_preflight_{method}.log',
                  mode=method, seed=c.NEW_SEEDS[0], initialize_only=True)
    for method in c.ARMS:
        m = a.read(OUT / 'initialization_preflight' / f'{method}_{c.NEW_SEEDS[0]}' / 'manifest.json')
        if m['status'] != 'COMPLETED' or any(m[k] for k in ('transitions', 'PPO_updates', 'Adam_steps', 'learn_calls')):
            raise ValueError('zero-sampling paired initialization preflight failed')


def verify_training(smoke=False):
    import torch
    seeds = c.NEW_SEEDS[:1] if smoke else c.NEW_SEEDS
    target = 2 if smoke else c.UPDATES
    for seed in seeds:
        for method in c.ARMS:
            folder = OUT / ('smoke_training' if smoke else 'training') / f'{method}_{seed}'
            m = a.read(folder / 'manifest.json')
            if (m['status'], m['updates'], m['transitions'], m['PPO_updates'], m['Adam_steps'], m['learn_calls']) != ('COMPLETED', target, target * 1536, target, target * 20, 1):
                raise ValueError('training manifest budget mismatch')
            rows = [json.loads(s) for s in (folder / 'updates.jsonl').read_text(encoding='utf8').splitlines()]
            if len(rows) != target or [r['update'] for r in rows] != list(range(1, target + 1)):
                raise ValueError('training trace missing/duplicated update')
            for r in rows:
                if len(r['minibatches']) != 20 or len(r['source_kl']) != 20 or r['transitions'] != r['update'] * 1536:
                    raise ValueError('native minibatch/source KL budget mismatch')
                if a.sha(folder / f'exposure_{r["update"]}.npz') != r['exposure_sha256']:
                    raise ValueError('training contact trace changed')
            for point, entry in m['checkpoints'].items():
                if a.sha(entry['path']) != entry['sha256'] or a.sha(entry['policy']) != entry['policy_sha256']:
                    raise ValueError('training checkpoint/policy changed')
                saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
                if saved['infos']['supplement_state']['completed_updates'] != int(point):
                    raise ValueError('saved checkpoint update mismatch')
                if any(float(v['step']) != 20 * int(point) for v in saved['optimizer_state_dict']['state'].values()):
                    raise ValueError('saved cumulative Adam mismatch')
                if de.state_digest(saved['source_teacher_state_dict']) != de.state_digest(
                        torch.load(m['checkpoints']['0']['path'], map_location='cpu', weights_only=False)['actor_state_dict']):
                    raise ValueError('teacher not original source actor')


def audit_rollout(folder, smoke=False):
    m = a.read(folder / 'manifest.json')
    if m['status'] != 'COMPLETED' or m['PPO_updates'] or m['Adam_steps']:
        raise ValueError('incomplete/nonzero optimizer eval')
    bank_values = bank(smoke) if m['method'] in ('constant', 'replay') else None
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sessions = {}
    for item in m['policies']:
        if a.sha(item['policy']) != item['policy_sha256'] or a.sha(item['checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('raw audit model source changed')
        if item['train_seed'] not in sessions:
            session = ort.InferenceSession(item['policy'], sess_options=opts, providers=['CPUExecutionProvider'])
            sessions[item['train_seed']] = (session, session.get_inputs()[0].name)
    for scenario in m['scenarios']:
        initial = a.read(folder / f'{scenario}_initial.json')
        seed = m['slots'][0]['train_seed']
        canonical = OUT / ('smoke_pairing' if smoke else 'pairing') / f'{"smoke" if smoke else m["probe"]}_terrain{m["terrain_seed"]}_seed{seed}_{scenario}.json'
        if initial != a.read(canonical):
            raise ValueError('saved initial/config/physical pairing mismatch')
        with np.load(folder / f'{scenario}_physics.npz', allow_pickle=False) as data:
            physical = {k: data[k] for k in data.files}
            if oldc.array_digest(physical) != initial['physical_sha256']:
                raise ValueError('saved physical model arrays mismatch')
    rows = a.read(folder / 'episodes.json'); transitions = 0
    for row in rows:
        path = folder / row['raw_path']
        if a.sha(path) != row['raw_sha256']:
            raise ValueError('raw changed')
        with np.load(path, allow_pickle=False) as data:
            raw = {k: data[k].copy() for k in data.files}
        n = len(raw['time']); transitions += n
        fell = oldc.termination_from_raw(raw)
        np.testing.assert_allclose(raw['actor_input'][:, 9:11], raw['phase_used'], atol=1e-7)
        np.testing.assert_array_equal(raw['actor_input'][:, :9], raw['actor_unmodified'][:, :9])
        np.testing.assert_array_equal(raw['actor_input'][:, 11:], raw['actor_unmodified'][:, 11:])
        np.testing.assert_allclose(raw['actor_unmodified'][:, 9:11], raw['live_phase'], atol=1e-7)
        session, input_name = sessions[row['train_seed']]
        expected_actions = np.concatenate([session.run(None, {input_name: x[None]})[0] for x in raw['actor_input']])
        np.testing.assert_array_equal(raw['action'], expected_actions, err_msg='saved actor inference differs')
        if bank_values is not None:
            initial = np.arctan2(raw['live_phase'][0, 0], raw['live_phase'][0, 1])
            expected = np.stack([c.open_loop_phase([initial], [row['train_seed']], bank_values, i, m['method'])[0] for i in range(n)])
            np.testing.assert_allclose(raw['phase_used'], expected, atol=2e-6)
        expected_push = np.array([c.push_delta(row['scenario'], i, smoke) for i in range(n)])
        np.testing.assert_array_equal(raw['push_delta'], expected_push)
        actual_delta = raw['root_velocity_world'] - raw['push_before']
        expected_delta = np.zeros_like(actual_delta); expected_delta[:, 1] = expected_push
        np.testing.assert_allclose(actual_delta, expected_delta, atol=2e-6)
        initial = a.read(folder / f'{row["scenario"]}_initial.json')
        slot = next(i for i, s in enumerate(m['slots']) if s['eval_seed'] == row['eval_seed'] and s['train_seed'] == row['train_seed'])
        points = dict(geom=raw['points_geom'], pos=raw['points_pos'], dist=raw['points_dist'],
                      includemargin=raw['points_includemargin'], efc_address=raw['points_efc_address'])
        world = np.full(len(points['geom']), slot)
        exposure = oldc.contact_exposure(points['geom'], world, points['pos'], points['dist'],
            points['includemargin'], points['efc_address'], set(m['foot_geom_ids']), set(m['terrain_geom_ids']),
            [set(x) for x in initial['assigned']], np.asarray(initial['bounds']), len(m['slots']))
        if int(raw['foot_contact_interior'].sum()) != int(exposure['interior'].sum()):
            raise ValueError('raw assigned hfield exposure mismatch')
        offsets = raw['contact_offsets']
        if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != len(points['geom']) or np.any(np.diff(offsets) < 0):
            raise ValueError('raw contact frame offsets malformed')
        frame = np.repeat(np.arange(n), np.diff(offsets))
        valid_frame = frame[exposure['indices']]
        np.testing.assert_array_equal(raw['foot_contact_total'], np.bincount(valid_frame, minlength=n))
        np.testing.assert_array_equal(raw['foot_contact_interior'], np.bincount(valid_frame[exposure['interior_mask']], minlength=n))
        if not smoke:
            score = oldc.episode_metrics(raw, 20., 'steady', fell, True)
            for key in ('completed', 'fell', 'mae_forward_velocity_mps', 'forward_displacement_m', 'moving_fraction', 'tilt_rms_rad', 'abs_heading_change_rad'):
                if not np.isclose(float(row[key]), float(score[key]), atol=1e-10):
                    raise ValueError('raw metric mismatch: ' + key)
            if row['push_recovery'] != c.recovery(raw['time'], raw['velocity'], fell):
                raise ValueError('raw recovery mismatch')
    if transitions != m['transitions'] or len(rows) != m['episode_count']:
        raise ValueError('rollout budget mismatch')
    return rows


def rollout(split, mode, terrain_seed, seed, smoke=False, repeat=False):
    if not smoke:
        verify_seal()
    check_disk()
    method = mode
    device = 'cpu'
    tag = split
    period = .55 if mode == 'clock55' else .60
    study, torch = a.setup()
    import onnxruntime as ort
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from scripts.g1_stateful_online import learned_phase as live_learned_phase
    from src.tasks.velocity.mdp import observations
    from src.tasks.velocity.phasor.data import G1_JOINT_ORDER

    learned = mode in ('live', 'constant', 'replay')
    if not smoke and (terrain_seed not in c.MAPS[split] or seed not in c.NEW_SEEDS):
        raise ValueError('outside locked evaluation split')
    if split == 'calibration' and mode != 'live':
        raise ValueError('only live calibration')
    if split == 'validation' and mode not in ('clock55', 'clock60'):
        raise ValueError('only clock validation')
    if not smoke and split == 'test' and mode not in ('live', 'constant', 'replay', selection()):
        raise ValueError('test clock must be selected on locked validation data')
    out = rollout_folder('smoke' if smoke else split,
                         mode + ('_repeat' if repeat else ''), terrain_seed, seed)
    reserve(out)
    slots = ([dict(train_seed=seed, eval_seed=20271911, patch_col=0)] if smoke
             else c.slots(split, seed))
    batch = len(slots)
    limit = 200 if smoke else 1000
    scenarios = ('push06',) if smoke else (('steady',) if split in ('calibration', 'validation') else c.SCENARIOS)
    entries = [policy_entry(split, mode, s['train_seed'], smoke) for s in slots]
    increments = bank(smoke) if mode in ('constant', 'replay') else None
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

    cfg = eval_cfg(study, mode, terrain_seed, batch)
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
        ce.seed_all(torch, c.CONSTRUCT_SEED)
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
        record.update(foot_geom_ids=sorted(foot_ids), terrain_geom_ids=sorted(terrain_ids), hfield_geom_ids=hfield_ids.tolist())
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
                           physical_sha256=oldc.array_digest(physical),
                           initial={k: v.tolist() for k, v in initial.items()},
                           origins=origins.tolist(), slots=slots,
                           command=[.5, 0., 0.],
                           assigned=[sorted(x) for x in assigned],
                           bounds=bounds.tolist())
            map_identity(terrain_seed, physical)
            ce.pairing_guard(OUT / ('smoke_pairing' if smoke else 'pairing') /
                             f'{"smoke" if smoke else tag}_terrain{terrain_seed}_seed{seed}_{scenario}.json', pairing)
            a.write(out / f'{scenario}_initial.json', pairing)
            keys = ('time', 'velocity', 'gravity', 'joint_pos', 'joint_vel',
                    'angle', 'phase_used', 'contact', 'foot_contact_total',
                    'foot_contact_interior', 'terminated', 'timed_out', 'live_phase', 'actor_input', 'actor_unmodified', 'action', 'push_delta', 'root_velocity_world', 'push_before')
            raw = [{k: [] for k in keys} for _ in slots]
            flattened = [{k: [] for k in ('geom', 'pos', 'dist', 'includemargin',
                                          'efc_address')} for _ in slots]
            offsets = [[0] for _ in slots]
            pos, quat = [], []
            done = np.zeros(batch, bool)
            fell = np.zeros(batch, bool)
            initial_angle = None
            for step in range(limit):
                env.command_manager.get_command('twist')[:] = torch.tensor(
                    [.5, 0., 0.], device=env.device)
                push_before = robot.data.root_link_vel_w.cpu().numpy().copy()
                delta_push = c.push_delta(scenario, step, smoke)
                if delta_push:
                    ids = torch.as_tensor(np.flatnonzero(~done), device=env.device)
                    velocity = robot.data.root_link_vel_w[ids].clone()
                    velocity[:, 1] += delta_push
                    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                    env.sim.forward()
                    env.sim.sense()
                env.observation_manager._obs_buffer = None
                obs = env.observation_manager.compute()['actor'].cpu().numpy().astype(np.float32)
                live_phase = (live_learned_phase(env, str(a.ENCODER)) if learned
                              else observations.phase(env, period, 'twist'))
                live_np = live_phase.cpu().numpy().copy()
                phase_np = live_np.copy()
                np.testing.assert_allclose(obs[:, 9:11], phase_np, atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(phase_np, axis=1), 1., atol=2e-6)
                np.testing.assert_allclose(obs[:, 6:9], np.tile([.5, 0., 0.], (batch, 1)),
                                           atol=1e-7)
                original_actor = obs.copy()
                if initial_angle is None:
                    initial_angle = np.arctan2(live_np[:, 0], live_np[:, 1]).astype(float)
                if increments is not None:
                    phase_np = c.open_loop_phase(initial_angle,
                        [s['train_seed'] for s in slots], increments, step, mode)
                    obs = c.replace_phase(obs, phase_np)
                action = oldc.route_actions(obs, sessions, names)
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
                              phase_used=phase_np, live_phase=live_np, actor_input=obs.copy(), actor_unmodified=original_actor, action=action.copy(),
                              push_delta=np.full(batch, delta_push),
                              push_before=push_before,
                              root_velocity_world=robot.data.root_link_vel_w.cpu().numpy().copy(),
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
                record['step_in_flight'] = True
                result = env.step(torch.as_tensor(action, device=env.device))
                record['transitions'] += active
                record['simulator_world_steps'] = record.get('simulator_world_steps', 0) + batch
                record['step_in_flight'] = False
                raw_capacity(env)
                if step % 100 == 0:
                    a.write(out / 'progress.json', dict(at=a.now(), scenario=scenario, step=step+1, transitions=record['transitions'], simulator_world_steps=record['simulator_world_steps']))
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
                    score = oldc.episode_metrics(value, 20., 'steady', bool(fell[i]), True)
                    score['push_recovery'] = c.recovery(value['time'], value['velocity'], bool(fell[i]))
                score.update(split=split, mode=mode, probe=tag, method=method, terrain_seed=terrain_seed,
                             period=(None if learned else period), scenario=scenario,
                             raw_path=path.name, raw_sha256=a.sha(path), **slot)
                episodes.append(score)
            a.write(out / 'episodes.json', episodes)
        record.update(status='COMPLETED', ended_at=a.now(), episode_count=len(episodes), protocol_sha256=(a.sha(CONTROL / 'protocol.json') if (CONTROL / 'protocol.json').exists() else None))
    except BaseException as error:
        record.update(status='FAILED', ended_at=a.now(), error=repr(error),
                      traceback=traceback.format_exc())
        if 'raw' in locals():
            partial_entries = []
            for i, value in enumerate(raw):
                partial = {k: np.asarray(v) for k, v in value.items()}
                if 'pos' in locals() and pos:
                    partial['position'] = np.asarray(pos)[:, i]
                    partial['quaternion'] = np.asarray(quat)[:, i]
                if 'offsets' in locals():
                    partial['contact_offsets'] = np.asarray(offsets[i], dtype=np.int64)
                if 'flattened' in locals():
                    for k, v in flattened[i].items():
                        if v:
                            partial['points_' + k] = np.concatenate(v, axis=0)
                path = out / f'PARTIAL_{scenario}_slot{i}.npz'
                np.savez_compressed(path, **partial)
                partial_entries.append(dict(path=path.name, sha256=a.sha(path),
                    pre_step_states=len(value['time']), recorded_terminal_flags=len(value['terminated'])))
            record['partial_raw'] = partial_entries
            record['partial_semantics'] = 'failed/incomplete trajectory, not a successful episode; in-flight native step may have unknown outcome'
        a.write(out / 'manifest.json', record)
        try:
            failure_evidence(error)
        except Exception as evidence_error:
            record['evidence_error'] = repr(evidence_error)
        raise
    finally:
        a.write(out / 'manifest.json', record)
        if env is not None:
            env.close()


def seal():
    if a.read(CONTROL / 'cpu_preflight.json')['status'] != 'PASS' or a.read(CONTROL / 'smoke_audit.json')['status'] != 'PASS':
        raise ValueError('preflight/smoke gates not passed')
    review = a.read(CONTROL / 'implementation_review.json')
    if review['status'] != 'PASS':
        raise ValueError('fresh implementation review missing')
    for item in review['reviewed_files']:
        if a.sha(item['path']) != item['sha256']:
            raise ValueError('reviewed implementation changed')
    files = set((ROOT / 'scripts').glob('*.py')) | set((ROOT / 'src').rglob('*.py'))
    files.update([Path(c.__file__), Path(__file__), SPEC, PLAN,
                  ROOT / 'tests/test_g1_phase_supplement_20261031.py',
                  CONTROL / 'cpu_preflight.json', CONTROL / 'smoke_audit.json',
                  CONTROL / 'implementation_review.json', a.SOURCE, a.ENCODER])
    entries = [dict(path=str(p), sha256=a.sha(p)) for p in sorted(files)]
    import importlib.metadata
    import platform
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages={k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'mujoco', 'mjlab', 'warp-lang', 'rsl-rl-lib', 'onnxruntime')},
        evaluation_device='cpu', training_device='cuda:0',
        reproducibility='same-backend CPU short replay checked bitwise; GPU training is stochastic numerical continuation, not bitwise across hardware')
    value = dict(status='LOCKED_BEFORE_FORMAL_SAMPLING', at=a.now(), design=c.design(), files=entries,
                 spec_sha256=a.sha(SPEC), protocol_version=1, environment=environment,
                 hypothesis_ids=['phase_feedback', 'method_replication', 'rough_perturbation'],
                 guardrails=['budget', 'finite', 'capacity', 'physical_pairing', 'frozen_teacher_encoder'],
                 failure_action='stop on technical failure; preserve raw/budget/resources; no automatic retry',
                 scientific_action='complete fixed matrix, analyze negative/inconclusive outcomes without added budget')
    with (CONTROL / 'protocol.json').open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for item in entries:
        p = Path(item['path'])
        if p.is_relative_to(ROOT) and p.suffix in ('.py', '.md'):
            target = CONTROL / 'source_snapshot' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(p.read_bytes())


def validation_rows():
    return [row for folder in sorted((OUT / 'rollouts/validation').iterdir())
            for row in audit_rollout(folder)]


def select_clock():
    rows = validation_rows()
    expected = {(method, t, s, e, 'steady') for method in ('clock55', 'clock60')
                for t in c.MAPS['validation'] for s in c.NEW_SEEDS for e in c.STARTS['validation']}
    keys = [(r['method'], r['terrain_seed'], r['train_seed'], r['eval_seed'], r['scenario']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('clock selection validation coverage incorrect')
    selected = c.choose_clock(rows)
    value = dict(status='LOCKED_BEFORE_TEST', at=a.now(), selected=selected,
                 rule='global mean completed desc; fall asc; tie clock55', validation_episodes=len(rows),
                 inputs=[dict(path=str(p), sha256=a.sha(p)) for p in sorted((OUT / 'rollouts/validation').rglob('episodes.json'))],
                 models=[policy_entry('test', selected, s) for s in c.NEW_SEEDS])
    with (OUT / 'clock_selection.json').open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return selected


def selection():
    value = a.read(OUT / 'clock_selection.json')
    if value['status'] != 'LOCKED_BEFORE_TEST':
        raise ValueError('test clock not locked')
    for item in value['inputs']:
        if a.sha(item['path']) != item['sha256']:
            raise ValueError('validation changed after selection')
    if value['models'] != [policy_entry('test', value['selected'], s) for s in c.NEW_SEEDS]:
        raise ValueError('test models changed after selection')
    return value['selected']


def failure_evidence(error):
    import psutil
    value = dict(at=a.now(), error=repr(error), traceback=traceback.format_exc(),
        memory=psutil.virtual_memory()._asdict(), disk={str(p): shutil.disk_usage(p)._asdict() for p in (ROOT, Path('<local-data-root>/'))},
        processes=[])
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times', 'memory_info']):
        if 'python' in (proc.info['name'] or '').lower():
            info = proc.info.copy(); info['cpu_times'] = info['cpu_times']._asdict()
            info['memory_info'] = info['memory_info']._asdict(); value['processes'].append(info)
    try:
        value['gpu'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=20).stdout
    except Exception as err:
        value['gpu_error'] = repr(err)
    CONTROL.mkdir(parents=True, exist_ok=True)
    a.write(CONTROL / f'failure_evidence_pid{os.getpid()}.json', value)
    a.write(CONTROL / 'failure_evidence.json', value)
    cost_ledger()


def cost_ledger():
    entries = []
    for p in OUT.rglob('manifest.json'):
        m = a.read(p)
        if 'transitions' not in m:
            continue
        progress = p.parent / 'progress.json'
        actual = dict(m)
        if progress.exists():
            for k, value in a.read(progress).items():
                if k in ('transitions', 'simulator_world_steps', 'PPO_updates', 'Adam_steps', 'updates'):
                    actual[k] = max(actual.get(k, 0), value)
        entries.append(dict(path=str(p), status=m['status'], smoke=('smoke' in str(p.relative_to(OUT))),
            transitions=actual.get('transitions', 0), simulator_world_steps=actual.get('simulator_world_steps', 0),
            PPO_updates=actual.get('PPO_updates', actual.get('updates', 0)), Adam_steps=actual.get('Adam_steps', 0),
            caveat='last persisted progress is lower bound for running/failed process; preserve unknown tail'))
    a.write(CONTROL / 'cost_ledger.json', dict(at=a.now(), entries=entries))


def queue():
    verify_seal()
    with (OUT / 'queue_started.json').open('x', encoding='utf8') as stream:
        json.dump(dict(at=a.now(), pid=os.getpid(), protocol_sha256=a.sha(CONTROL / 'protocol.json')), stream)
    status = dict(status='RUNNING', started_at=a.now(), completed_jobs=0)
    def run(command, name, **kw):
        verify_seal(); status.update(stage=command, current=name, at=a.now())
        a.write(CONTROL / 'queue_status.json', status)
        run_child(command, OUT / 'logs' / (name + '.log'), **kw)
        status['completed_jobs'] += 1
    try:
        jobs = c.training_jobs(); random.Random(20271703).shuffle(jobs)
        for job in jobs:
            run('train', f'train_{job["method"]}_{job["seed"]}', mode=job['method'], seed=job['seed'])
        verify_training()
        for t in c.MAPS['calibration']:
            for s in c.NEW_SEEDS:
                run('rollout', f'calibration_{t}_{s}', split='calibration', mode='live', terrain=t, seed=s)
        build_bank()
        for t in c.MAPS['validation']:
            for s in c.NEW_SEEDS:
                for mode in ('clock55', 'clock60'):
                    run('rollout', f'validation_{mode}_{t}_{s}', split='validation', mode=mode, terrain=t, seed=s)
        selected = select_clock()
        verify_bank()
        with (OUT / 'test_inputs_locked.json').open('x', encoding='utf8') as stream:
            json.dump(dict(at=a.now(), bank_manifest_sha256=a.sha(OUT / 'phase_bank/manifest.json'),
                selection_sha256=a.sha(OUT / 'clock_selection.json')),
                stream, ensure_ascii=False, indent=2)
        for t in c.MAPS['test']:
            for s in c.NEW_SEEDS:
                modes = ['live', 'constant', 'replay', selected]
                random.Random(t * 100 + s).shuffle(modes)
                for mode in modes:
                    selection()
                    test_inputs = a.read(OUT / 'test_inputs_locked.json')
                    if (test_inputs['bank_manifest_sha256'] != a.sha(OUT / 'phase_bank/manifest.json') or
                            test_inputs['selection_sha256'] != a.sha(OUT / 'clock_selection.json')):
                        raise ValueError('locked derived inputs changed before test')
                    run('rollout', f'test_{mode}_{t}_{s}', split='test', mode=mode, terrain=t, seed=s)
        analyze(); audit()
        status.update(status='COMPLETED', ended_at=a.now())
    except BaseException as error:
        status.update(status='FAILED', ended_at=a.now(), error=repr(error))
        failure_evidence(error)
        raise
    finally:
        a.write(CONTROL / 'queue_status.json', status); cost_ledger()


def grouped_results(rows):
    selected = selection(); modes = ('live', 'constant', 'replay', selected)
    metrics = ('completed', 'fell', *oldc.GUARDS)
    output = {}
    for scenario in ('all', *c.SCENARIOS):
        chosen = [r for r in rows if scenario == 'all' or r['scenario'] == scenario]
        summaries = {mode: {k: float(np.mean([r[k] for r in chosen if r['method'] == mode]))
                            for k in metrics} for mode in modes}
        differences = {}
        for other in modes[1:]:
            differences[other] = {}
            for k in metrics:
                seed_diffs = [np.mean([r[k] for r in chosen if r['method'] == 'live' and r['train_seed'] == s]) -
                              np.mean([r[k] for r in chosen if r['method'] == other and r['train_seed'] == s]) for s in c.NEW_SEEDS]
                got = c.seed_stats(seed_diffs)
                map_diffs = [np.mean([r[k] for r in chosen if r['method'] == 'live' and r['terrain_seed'] == t]) -
                             np.mean([r[k] for r in chosen if r['method'] == other and r['terrain_seed'] == t]) for t in c.MAPS['test']]
                rng = np.random.default_rng(20271791)
                boot = np.asarray(map_diffs)[rng.integers(0, len(map_diffs), (10000, len(map_diffs)))].mean(axis=1)
                got.update(map_differences=map_diffs, map_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                           bootstrap_scope='maps resampled as whole groups; eight trained seeds conditioned fixed')
                differences[other][k] = got
        for family in (('constant', 'replay'), (selected,)):
            vals = [differences[m][k] for m in family for k in ('completed', 'fell')]
            for val, p in zip(vals, c.holm([v['p_exact'] for v in vals])):
                val['p_holm'] = p
        gates = {}
        for other in modes[1:]:
            d = differences[other]
            guards = {k: bool(d[k]['mean'] >= limit if limit < 0 else d[k]['mean'] <= limit)
                      for k, limit in oldc.GUARDS.items()}
            gates[other] = dict(completion=d['completed']['mean'] > 0 and d['completed']['p_holm'] < .05,
                fall=d['fell']['mean'] < 0 and d['fell']['p_holm'] < .05, guards=guards)
            gates[other]['primary_passed'] = gates[other]['completion'] and gates[other]['fall']
        output[scenario] = dict(summary=summaries, differences=differences, gates=gates)
    return output


def collect_test(audited=False):
    selected = selection(); rows = []
    for t in c.MAPS['test']:
        for s in c.NEW_SEEDS:
            for mode in ('live', 'constant', 'replay', selected):
                folder = rollout_folder('test', mode, t, s)
                rows.extend(audit_rollout(folder) if audited else a.read(folder / 'episodes.json'))
    expected = {(m, t, s, e, q) for m in ('live', 'constant', 'replay', selected)
                for t in c.MAPS['test'] for s in c.NEW_SEEDS for e in c.STARTS['test'] for q in c.SCENARIOS}
    keys = [(r['method'], r['terrain_seed'], r['train_seed'], r['eval_seed'], r['scenario']) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('final test matrix coverage incorrect')
    return rows


def analyze():
    verify_seal(); rows = collect_test()
    result = dict(at=a.now(), status='COMPLETE_PENDING_INDEPENDENT_RAW_AUDIT',
        protocol_sha256=a.sha(CONTROL / 'protocol.json'), selected_clock=selection(),
        independent_continuation_seeds=8, episodes=len(rows), grouped=grouped_results(rows),
        limitations=c.design()['limitations'],
        inference='paired continuation seeds; sign flip assumes exchangeability/symmetry under null; not independent pretraining sources',
        perturbation=perturbation_results(rows))
    a.write(CONTROL / 'analysis.json', result)


def audit():
    verify_seal(); verify_training()
    calibration = [audit_rollout(rollout_folder('calibration', 'live', t, s))
                   for t in c.MAPS['calibration'] for s in c.NEW_SEEDS]
    validation = validation_rows(); rows = collect_test(True)
    if len(validation) != 240 or sum(map(len, calibration)) != 120:
        raise ValueError('formal budget/coverage mismatch')
    if c.choose_clock(validation) != selection():
        raise ValueError('independent fixed-period selection mismatch')
    verify_bank()
    if a.read(CONTROL / 'analysis.json')['grouped'] != grouped_results(rows):
        raise ValueError('independent raw result aggregation mismatch')
    expected_perturbation = a.read(CONTROL / 'analysis.json')['perturbation']
    if a.sha(CONTROL / 'perturbation_windows.json') != expected_perturbation['raw_window_table_sha256']:
        raise ValueError('perturbation window table changed')
    if perturbation_results(rows) != expected_perturbation:
        raise ValueError('independent raw perturbation recomputation mismatch')
    all_rows = [*rows, *validation, *(r for group in calibration for r in group)]
    manifests = [a.read(p) for p in (OUT / 'rollouts').rglob('manifest.json')]
    if sum(m['simulator_world_steps'] for m in manifests) > c.design()['maximum_eval_world_steps']:
        raise ValueError('formal simulator-world budget exceeded')
    a.write(CONTROL / 'independent_audit.json', dict(status='PASS', at=a.now(),
        audited_raw=len(all_rows), formal_eval_transitions=sum(r['observed_steps'] for r in all_rows),
        simulator_world_steps=sum(m['simulator_world_steps'] for m in manifests),
        train_transitions=27721728, PPO_updates=18048, Adam_steps=360960, eval_PPO_updates=0,
        scope='same operator independent raw recomputation, not a separate human scientific review',
        analysis_sha256=a.sha(CONTROL / 'analysis.json')))


def verify_bank():
    manifest = a.read(OUT / 'phase_bank/manifest.json')
    arrays = bank(); sources = []
    for seed in c.NEW_SEEDS:
        traces = []
        for terrain in c.MAPS['calibration']:
            folder = rollout_folder('calibration', 'live', terrain, seed)
            m = a.read(folder / 'manifest.json')
            if m['policies'] != [policy_entry('calibration', 'live', seed) for _ in range(5)]:
                raise ValueError('calibration bank bound to another policy')
            for row in a.read(folder / 'episodes.json'):
                raw = folder / row['raw_path']
                if a.sha(raw) != row['raw_sha256']:
                    raise ValueError('calibration source changed')
                with np.load(raw, allow_pickle=False) as v:
                    phase = v['phase_used'].astype(float)
                    # Independent reconstruction without contract.phase_increments.
                    angle = np.arctan2(phase[:, 0], phase[:, 1])
                    inc = np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))
                    traces.append(np.clip(inc, 0, np.pi / 3))
                sources.append(dict(path=str(raw), sha256=row['raw_sha256'], train_seed=seed))
        np.testing.assert_allclose(arrays[seed], np.concatenate(traces), rtol=0, atol=1e-14)
    # Builder loops map then seed; comparison is unordered but counts are exact.
    key = lambda r: (r['train_seed'], r['path'])
    if sorted(sources, key=key) != sorted(manifest['sources'], key=key):
        raise ValueError('bank calibration provenance incomplete')


def perturbation_results(rows):
    records = []
    for row in rows:
        folder = rollout_folder('test', row['method'], row['terrain_seed'], row['train_seed'])
        with np.load(folder / row['raw_path'], allow_pickle=False) as v:
            time = v['time']; angle = np.unwrap(v['angle'].astype(float)); velocity = v['velocity']
            recovery = c.recovery(time, velocity, bool(row['fell']))
            for at, rec in zip((5., 10.), recovery):
                pre = np.flatnonzero((time >= at - 1 - 1e-8) & (time < at - 1e-8))
                post = np.flatnonzero((time >= at - 1e-8) & (time < at + 1 - 1e-8))
                complete = len(pre) == 50 and len(post) == 50
                record = {k: row[k] for k in ('method', 'train_seed', 'terrain_seed', 'eval_seed', 'scenario')}
                record.update(at=at, recovery=rec, complete_response_windows=complete,
                    phase_hz_pre=None, phase_hz_post=None, phase_hz_change=None,
                    forward_velocity_pre=None, forward_velocity_post=None, lateral_velocity_pre=None, lateral_velocity_post=None)
                if complete:
                    hz = [float((angle[i[-1]] - angle[i[0]]) / (time[i[-1]] - time[i[0]]) / (2 * np.pi)) for i in (pre, post)]
                    record.update(phase_hz_pre=hz[0], phase_hz_post=hz[1], phase_hz_change=hz[1] - hz[0],
                        forward_velocity_pre=float(velocity[pre, 0].mean()), forward_velocity_post=float(velocity[post, 0].mean()),
                        lateral_velocity_pre=float(velocity[pre, 1].mean()), lateral_velocity_post=float(velocity[post, 1].mean()))
                records.append(record)
    summary = {}
    for mode in sorted({r['method'] for r in rows}):
        summary[mode] = {}
        for scenario in c.SCENARIOS:
            group = [r for r in records if r['method'] == mode and r['scenario'] == scenario]
            exposed = [r for r in group if r['recovery']['exposed']]
            recovered = [r for r in group if r['recovery']['recovered']]
            complete = [r for r in group if r['complete_response_windows']]
            summary[mode][scenario] = dict(total_windows=len(group), exposed_windows=len(exposed),
                unexposed_windows=len(group) - len(exposed), recovered_windows=len(recovered),
                exposed_unrecovered_windows=len(exposed) - len(recovered),
                recovery_rate_all_windows=len(recovered) / len(group),
                recovery_rate_exposed=len(recovered) / len(exposed) if exposed else None,
                recovered_window_start_seconds=[r['recovery']['seconds'] for r in recovered],
                complete_response_windows=len(complete), missing_response_windows=len(group) - len(complete),
                mean_phase_hz_change=float(np.mean([r['phase_hz_change'] for r in complete])) if complete else None)
    a.write(CONTROL / 'perturbation_windows.json', dict(records=records))
    return dict(summary=summary, raw_window_table_sha256=a.sha(CONTROL / 'perturbation_windows.json'),
                meaning='descriptive paired windows with explicit coverage; steady is the sham reference; timing changes alone do not establish benefit')


def launch():
    verify_seal()
    if (OUT / 'queue_started.json').exists():
        raise ValueError('formal queue already claimed; no duplicate launch')
    with (CONTROL / 'launch_reserved.json').open('x', encoding='utf8') as stream:
        json.dump(dict(at=a.now(), protocol_sha256=a.sha(CONTROL / 'protocol.json')), stream)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(c.CONSTRUCT_SEED), PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (CONTROL / 'queue_console.log').open('x', encoding='utf8') as stream:
        process = subprocess.Popen(child_args('queue'), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    a.write(CONTROL / 'launch.json', dict(at=a.now(), pid=process.pid,
        protocol_sha256=a.sha(CONTROL / 'protocol.json'), output=str(OUT), duplicate_launch_forbidden=True))
    print(json.dumps(dict(pid=process.pid, output=str(OUT)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'initialization-preflight', 'smoke', 'train', 'rollout', 'bank', 'audit-smoke', 'seal', 'queue', 'analyze', 'audit', 'launch'))
    parser.add_argument('--mode'); parser.add_argument('--seed', type=int, default=c.NEW_SEEDS[0])
    parser.add_argument('--terrain', type=int); parser.add_argument('--split', default='test')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'preflight': preflight()
    elif args.command == 'smoke': smoke()
    elif args.command == 'train': train(args.mode, args.seed, args.smoke, args.initialize_only)
    elif args.command == 'rollout': rollout(args.split, args.mode, args.terrain, args.seed, args.smoke, args.repeat)
    elif args.command == 'bank': build_bank(args.smoke)
    elif args.command == 'audit-smoke': audit_smoke()
    else:
        globals()[args.command.replace('-', '_')]()


if __name__ == '__main__':
    main()
