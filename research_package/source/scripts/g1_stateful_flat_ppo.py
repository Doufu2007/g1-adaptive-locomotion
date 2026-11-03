"""Immutable candidate and matched flat-only PPO experiment contract."""
from dataclasses import replace
import json
from pathlib import Path

from scripts import g1_phase_adaptation_20261024 as previous

base = previous.base
ROOT = base.ROOT
OUT = ROOT/'results/g1_stateful_flat_ppo_20261025'
SNAPSHOT = ROOT/'deliveries/g1_stateful_phase_20261025/snapshot'
ENCODER_SEED = 20261025
RUNTIME = SNAPSHOT/'results/g1_stateful_phase_20261025/training/observer_sensors_mixed_seed20261025/final.pt'
RUNTIME_SHA = 'fdaad78ee4d923189f4291a0d9c8a02c0b726325d1caa2b96a2e76231c7848ca'
TRAIN_SEEDS = (20261901, 20261902, 20261903)
EVAL_SEEDS = tuple(range(20262001, 20262011))
SCENARIOS = ('steady', 'speed_change', 'push')
ITERATIONS, NUM_ENVS, STEPS = 501, 64, 24
TRANSITIONS = ITERATIONS*NUM_ENVS*STEPS
now, sha, write = base.now, base.sha, base.write
canonical, common_cfg_digest = previous.canonical, previous.common_cfg_digest


def bootstrap():
    values = base.bootstrap()
    values[1].backends.cuda.matmul.allow_tf32 = False
    values[1].backends.cudnn.allow_tf32 = False
    values[1].backends.cudnn.deterministic = True
    values[1].backends.cudnn.benchmark = False
    from scripts.g1_stateful_online import install
    install()
    return values


def verify_runtime():
    if sha(RUNTIME) != RUNTIME_SHA:
        raise ValueError('sealed E checkpoint changed')
    # The source checkpoint lacks joint_names; bind it to the sealed training definition.
    for relative in ('scripts/g1_stateful_phase.py', 'src/tasks/velocity/phasor/data.py'):
        if sha(ROOT/relative) != sha(SNAPSHOT/relative):
            raise ValueError('E model or joint-order source differs from verified snapshot: '+relative)


def dependency_ready():
    directory = ROOT/'deliveries/g1_stateful_phase_20261025'
    try:
        audit = json.loads((directory/'completion_verification.json').read_text(encoding='utf-8'))
        return (audit['status'] == 'PASS' and audit['training_runs'] == audit['evaluations'] == 15
                and audit['test_after_freeze'] and audit['files_rehashed'] == 2015)
    except (OSError, KeyError, ValueError):
        return False


def make_cfg(terrain, arm, seed, *, play, num_envs, horizon):
    if terrain != 'flat':
        raise ValueError('this experiment is flat only')
    if arm not in ('clock', 'learned'):
        raise ValueError('unknown phase source')
    cfg = previous.make_cfg('flat', 'clock', seed, play=play, num_envs=num_envs, horizon=horizon)
    if arm == 'learned':
        from mjlab.managers.event_manager import EventTermCfg
        from scripts.g1_stateful_online import learned_phase, reset_phase
        for group in ('actor', 'critic'):
            cfg.observations[group].terms['phase'] = replace(
                cfg.observations[group].terms['phase'], func=learned_phase,
                params={'encoder_path':str(RUNTIME)})
        cfg.events['reset_phasor_phase_history'] = EventTermCfg(
            func=reset_phase, mode='reset', params={'encoder_path':str(RUNTIME)})
    return cfg


def training_jobs():
    return [dict(terrain='flat', arm=arm, seed=seed) for index, seed in enumerate(TRAIN_SEEDS)
            for arm in (('clock','learned') if index % 2 == 0 else ('learned','clock'))]


def train_dir(terrain, arm, seed):
    if terrain != 'flat' or arm not in ('clock','learned') or seed not in TRAIN_SEEDS:
        raise ValueError('outside frozen flat training matrix')
    return OUT/'training'/f'flat_{arm}_seed{seed}'


def evaluation_jobs(terrain, seed):
    return [dict(label=label, arm=arm, policy=str(train_dir(terrain, trained, seed)/'policy.onnx'),
                 output=str(OUT/'evaluation'/f'flat_seed{seed}_{label}'))
            for label, trained, arm in (('clock_clock','clock','clock'),
                 ('clock_learned','clock','learned'),('learned_learned','learned','learned'))]


def choose_resume_checkpoint(directory, seed, terrain, arm):
    import torch
    rejected = []
    paths = sorted(Path(directory).glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]), reverse=True)
    for path in paths:
        try:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            state = saved['infos']['adaptation_state']
            if (state['completed_updates'] != int(saved['iter'])+1
                    or state['ppo_seed'] != seed or state['terrain'] != terrain or state['arm'] != arm
                    or state['encoder_sha256'] != RUNTIME_SHA
                    or not 0 < state['completed_updates'] <= ITERATIONS):
                raise ValueError('checkpoint identity or update mismatch')
        except Exception as error:
            rejected.append(dict(path=str(path), error=repr(error)))
            continue
        return path, rejected
    raise ValueError(f'no complete matching recovery checkpoint: {rejected}')


def paired_training_effect(rows, terrain, treatment, control, metric):
    import numpy as np
    from scipy.stats import t
    per_seed = []
    for seed in TRAIN_SEEDS:
        groups = []
        for label in (treatment, control):
            group = {}
            for row in rows:
                if (row['terrain'],row['train_seed'],row['label']) == (terrain,seed,label):
                    key = row['scenario'],row['seed']
                    if key in group:
                        raise ValueError('duplicate paired evaluation episode')
                    group[key] = float(row['fell'] if metric == 'fall_rate' else row[metric])
            groups.append(group)
        a,b = groups
        if not a or set(a) != set(b):
            raise ValueError('missing paired evaluation episodes')
        per_seed.append(dict(train_seed=seed, difference=float(np.mean([a[k]-b[k] for k in sorted(a)])),
                             paired_episode_count=len(a)))
    differences = np.asarray([r['difference'] for r in per_seed])
    effect, sd = float(differences.mean()), float(differences.std(ddof=1))
    se = sd/np.sqrt(len(differences))
    half = float(t.ppf(.975,len(differences)-1)*se)
    p = float(2*t.sf(abs(effect/se),len(differences)-1)) if se > 0 else (1. if effect == 0 else 0.)
    return dict(mean_difference=effect, training_seed_count=len(per_seed), paired_t_95pct_ci=[effect-half,effect+half],
                two_sided_p=p, per_training_seed=per_seed, metric=metric, terrain=terrain,
                treatment=treatment,control=control,
                interpretation='three matched PPO runs; conditional on the fixed flat simulation cohort')


def asset_paths():
    paths = [ROOT/p for p in (
        'scripts/g1_stateful_online.py','scripts/g1_stateful_flat_ppo.py',
        'scripts/run_g1_stateful_flat_ppo_20261025.py','scripts/g1_stateful_phase.py',
        'scripts/run_g1_phase_adaptation_20261024.py','scripts/g1_phase_adaptation_20261024.py',
        'scripts/run_g1_best_phase_control.py','scripts/g1_best_phase_batched.py',
        'scripts/summarize_g1_best_phase_control.py','scripts/g1_closed_loop_metrics.py',
        'docs/superpowers/specs/2026-10-25-g1-stateful-flat-ppo-design.md',
        'docs/superpowers/plans/2026-10-25-g1-stateful-flat-ppo.md',
        'tests/test_g1_stateful_flat_ppo.py','src/tasks/velocity/mdp/observations.py',
        'src/tasks/velocity/mdp/rewards.py','src/tasks/velocity/velocity_env_cfg.py',
        'src/tasks/velocity/config/g1/env_cfgs.py','src/tasks/velocity/config/g1/rl_cfg.py',
        'src/tasks/velocity/rl/runner.py','results/phasor_pilot_20261016/run_pilot.py')]
    paths += list((ROOT/'src/tasks/velocity/phasor').glob('*.py'))
    paths += [RUNTIME,base.POLICIES['flat']/'model_9900.pt']
    return paths


def smoke_identity():
    import hashlib
    import importlib.metadata
    import sys
    value = dict(sources={str(p.relative_to(ROOT)):sha(p) for p in asset_paths()},
                 python=sys.version,packages={name:importlib.metadata.version(name)
                 for name in ('torch','mjlab','mujoco','warp-lang','rsl-rl-lib')})
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def freeze_assets():
    verify_runtime()
    if (OUT/'asset_seal.json').exists():
        return verify_assets()
    paths = asset_paths()
    lock = dict(schema='g1.stateful_flat_ppo.asset_seal.v1', frozen_at=now(),
                authorization='2026-10-25 user: can PPO comparison start now? if so try it',
                encoder_selection='E smallest seed 20261025; no score-based seed selection',
                train_seeds=TRAIN_SEEDS, eval_seeds=EVAL_SEEDS, iterations=ITERATIONS,
                transitions_per_run=TRANSITIONS, training_jobs=training_jobs(),
                assets=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])
    for path in paths:
        target = OUT/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ('.py','.md'):
            target.write_bytes(path.read_bytes())
    write(OUT/'asset_seal.json',lock)
    return lock


def verify_assets():
    lock = json.loads((OUT/'asset_seal.json').read_text(encoding='utf-8'))
    for item in lock['assets']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('sealed flat PPO asset changed: '+item['path'])
    verify_runtime()
    return lock
