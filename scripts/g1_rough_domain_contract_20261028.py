"""Finite training-domain x phase contract; no sampling on import."""
from copy import deepcopy
import math
import random
import numpy as np
from scripts import g1_rough_transfer_contract_20261028 as r

METHODS = ('FC', 'FE', 'RC', 'RE')
TRAIN_SEEDS = (20261901, 20261902, 20261903)
TRAIN_MAPS = (20264601, 20264602, 20264603)
TERRAINS = (0, 20264701, 20264702, 20264703)
EVAL_SEEDS = tuple(range(20264811, 20264816))
SCENARIOS = ('steady', 'push')
CHECKPOINTS = (0, 1, 25, 100, 251)
UPDATES = 251
TRANSITIONS_PER_UPDATE = 64 * 24
GUARDS = dict(r.GUARDS)
FLAT_GUARDS = dict(completed=-.10, fell=.10, **GUARDS)


def source_phase(method):
    if method not in METHODS:
        raise ValueError('outside fixed S method matrix')
    return 'learned' if method.endswith('E') else 'clock'


def training_map(method, source_seed):
    source_phase(method)
    if source_seed not in TRAIN_SEEDS:
        raise ValueError('outside fixed S source seeds')
    return TRAIN_MAPS[TRAIN_SEEDS.index(source_seed)] if method.startswith('R') else 0


def training_jobs():
    jobs = [dict(method=m, source_seed=s, updates=UPDATES,
                 transitions=UPDATES*TRANSITIONS_PER_UPDATE,
                 training_map=training_map(m, s)) for m in METHODS for s in TRAIN_SEEDS]
    random.Random(20264590).shuffle(jobs)
    return jobs


def evaluation_jobs():
    jobs = [dict(method=m, terrain_seed=t) for m in METHODS for t in TERRAINS]
    random.Random(20264790).shuffle(jobs)
    return jobs


def slots():
    return [dict(train_seed=s, eval_seed=e, patch_col=i)
            for s in TRAIN_SEEDS for i, e in enumerate(EVAL_SEEDS)]


def make_training_cfg(study, method, seed):
    from mjlab.managers.event_manager import EventTermCfg
    from scripts.g1_rough_transfer_experiment_20261028 import deterministic_terrain
    phase = source_phase(method)
    terrain_seed = training_map(method, seed)
    cfg = study.make_cfg('flat', phase, seed, play=False, num_envs=64, horizon=20.)
    if terrain_seed:
        native = study.previous.make_cfg('rough', 'clock', seed, play=False, num_envs=64, horizon=20.)
        cfg.scene.terrain = deepcopy(native.scene.terrain)
        g = cfg.scene.terrain.terrain_generator
        g.seed = terrain_seed
        g.curriculum = False
        g.num_rows = 1
        g.num_cols = 5
        g.size = (32., 16.)
        rough = deepcopy(g.sub_terrains['random_rough'])
        rough.proportion = 1.
        rough.noise_range = (.02, .10)
        rough.noise_step = .02
        rough.horizontal_scale = .1
        rough.vertical_scale = .005
        rough.border_width = .25
        g.sub_terrains = {'random_rough': rough}
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.njmax = 1500
    cfg.sim.nconmax = 128
    # Native play configs have this event; training configs do not. Install
    # the same deterministic reset in all four arms, before reset_base.
    event = EventTermCfg(func=deterministic_terrain, mode='reset')
    cfg.events = {'randomize_terrain': event,
                  **{k:v for k,v in cfg.events.items() if k != 'randomize_terrain'}}
    events = list(cfg.events)
    if events.index('randomize_terrain') >= events.index('reset_base'):
        raise ValueError('terrain reset must precede robot reset')
    if list(cfg.observations['actor'].terms) != ['base_ang_vel', 'projected_gravity', 'command', 'phase', 'joint_pos', 'joint_vel', 'actions']:
        raise ValueError('actor input structure changed')
    if 'height_scan' in cfg.observations['actor'].terms:
        raise ValueError('unexpected rough perception input')
    return cfg


def _differences(means, treatment, control):
    return {k:np.asarray(means[treatment][k])-means[control][k]
            for k in ('completed', 'fell', *GUARDS)}


def _protection(diffs, limits):
    return {k:bool(diffs[k].mean() >= limit-1e-12 if limit < 0 else diffs[k].mean() <= limit+1e-12)
            for k, limit in limits.items()}


def _point_gates(diffs):
    return dict(completion_margin=bool(diffs['completed'].mean() >= .10-1e-12),
                fall_margin=bool(diffs['fell'].mean() <= -.10+1e-12),
                completion_seed_direction=bool(np.sum(diffs['completed'] > 0) >= 2),
                fall_seed_direction=bool(np.sum(diffs['fell'] < 0) >= 2))


def paired_analysis(rows):
    metrics = ('completed', 'fell', *GUARDS)
    expected = {(m, t, s, e, q) for m in METHODS for t in TERRAINS
                for s in TRAIN_SEEDS for e in EVAL_SEEDS for q in SCENARIOS}
    by = {}
    for row in rows:
        key = tuple(row[k] for k in ('method', 'terrain_seed', 'train_seed', 'eval_seed', 'scenario'))
        if key in by or not all(math.isfinite(float(row[k])) for k in metrics):
            raise ValueError('duplicate or nonfinite S episode')
        by[key] = row
    if set(by) != expected:
        raise ValueError('missing or extra S factorial episode')
    domain_means = {}
    for domain, terrains in (('rough', TERRAINS[1:]), ('flat', (0,))):
        domain_means[domain] = {m:{k:[float(np.mean([by[m,t,s,e,q][k] for t in terrains
                                                    for e in EVAL_SEEDS for q in SCENARIOS]))
                                     for s in TRAIN_SEEDS] for k in metrics} for m in METHODS}
    means = domain_means['rough']
    contrasts = {'RE_minus_RC':dict(RE=1, RC=-1), 'RE_minus_FE':dict(RE=1, FE=-1),
                 'RC_minus_FC':dict(RC=1, FC=-1), 'interaction':dict(RE=1, FE=-1, RC=-1, FC=1)}
    family = {}
    for name, weights in contrasts.items():
        for k in ('completed', 'fell'):
            family[name+':'+k] = r.seed_comparison(sum(w*np.asarray(means[m][k]) for m,w in weights.items()))
    for val, p in zip(family.values(), r.holm([v['p_exact'] for v in family.values()])):
        val['p_holm'] = p
    comparators = {}
    for other in ('RC', 'FC'):
        diffs = _differences(means, 'RE', other)
        gates = dict(**_point_gates(diffs), **_protection(diffs, GUARDS))
        comparators[other] = dict(gates=gates, passed=all(gates.values()),
                                  differences={k:r.seed_comparison(v) for k,v in diffs.items()})
    flat_diffs = _differences(domain_means['flat'], 'RE', 'FE')
    flat_guards = _protection(flat_diffs, FLAT_GUARDS)
    domains = {}
    for phase, treatment, control in (('learned','RE','FE'), ('clock','RC','FC')):
        diffs = _differences(means, treatment, control)
        gates = _point_gates(diffs)
        domains[phase] = dict(point_gates=gates, point_gate_passed=all(gates.values()),
                              differences={k:r.seed_comparison(v) for k,v in diffs.items()},
                              meaning='finite-budget training-domain point effect; not a phase benefit or confirmation')
    return dict(source_training_seeds=list(TRAIN_SEEDS), independent_units=3,
                domain_seed_means=domain_means, family=family,
                candidate=dict(comparators=comparators, flat_guards=flat_guards,
                               flat_differences={k:r.seed_comparison(v) for k,v in flat_diffs.items()},
                               passed=all(v['passed'] for v in comparators.values()) and all(flat_guards.values()),
                               meaning='new exploratory candidate only; no independent superiority or causal feedback confirmation'),
                training_domain_effect=domains,
                limitation='three reused source-seed pipelines; conditional on fixed held-out maps; exact two-sided p minimum .25')
