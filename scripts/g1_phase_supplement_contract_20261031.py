"""Finite rough-only supplement. Import has no simulator or filesystem effects."""
from itertools import product
import math
import numpy as np

OLD_SEEDS = (20261901, 20261902, 20261903)
NEW_SEEDS = tuple(range(20271301, 20271309))
MAPS = dict(calibration=tuple(range(20271101, 20271104)),
            training=tuple(range(20271401, 20271409)),
            validation=tuple(range(20271501, 20271504)),
            test=tuple(range(20271601, 20271607)))
STARTS = {role: tuple(range(first, first + 5)) for role, first in
          [('calibration', 20271111),
           ('validation', 20271511), ('test', 20271611)]}
ARMS = ('learned', 'clock55', 'clock60')
MODES = ('live', 'constant', 'replay', 'clock55')
SCENARIOS = ('steady', 'push03', 'push06')
UPDATES = 752
CONSTRUCT_SEED = 20271700


def design():
    return dict(train_transitions=24 * UPDATES * 1536,
                train_PPO_updates=24 * UPDATES, train_Adam_steps=24 * UPDATES * 20,
                formal_episodes=120 + 240 + 2880,
                maximum_eval_world_steps=3240000,
                calibration_episodes=120, mechanism_episodes=0,
                validation_episodes=240, test_episodes=2880,
                train_seeds=list(NEW_SEEDS), old_model_seeds=list(OLD_SEEDS),
                maps=MAPS, starts=STARTS, scenarios=SCENARIOS, updates=UPDATES,
                init='shared source9900 actor; paired fresh critic and empty Adam',
                primary_unit='paired continuation training seed; maps crossed; episodes nested',
                limitations=['fixed source9900 and E encoder',
                             'open-loop ablations change phase/state joint distribution',
                             'three old mechanism lineages are not new training repeats'])


def training_jobs():
    return [dict(method=m, seed=s) for s in NEW_SEEDS for m in ARMS]


def slots(split, train_seed=None):
    seeds = (train_seed,)
    if split not in STARTS or None in seeds:
        raise ValueError('undeclared split/seed')
    if train_seed not in NEW_SEEDS:
        raise ValueError('not a new training seed')
    return [dict(train_seed=s, eval_seed=e, patch_col=i)
            for s in seeds for i, e in enumerate(STARTS[split])]


def phase_increments(episodes):
    arrays = []
    for phase in episodes:
        phase = np.asarray(phase, dtype=float)
        if phase.ndim != 2 or phase.shape[1] != 2 or not np.isfinite(phase).all():
            raise ValueError('invalid calibration phase')
        np.testing.assert_allclose(np.linalg.norm(phase, axis=1), 1., atol=2e-6)
        angle = np.arctan2(phase[:, 0], phase[:, 1])
        delta = np.angle(np.exp(1j * np.diff(angle)))
        if np.any(delta < -2e-5) or np.any(delta > math.pi / 3 + 2e-5):
            raise ValueError('calibration phase discontinuity')
        arrays.append(np.clip(delta, 0., math.pi / 3))
    result = np.concatenate(arrays)
    if not len(result):
        raise ValueError('no calibration increments; do not replace maps')
    return result


def open_loop_phase(initial_angle, seeds, bank, step, mode):
    if mode not in ('constant', 'replay') or step < 0:
        raise ValueError('undeclared intervention')
    initial_angle = np.asarray(initial_angle, dtype=float)
    if initial_angle.shape != (len(seeds),) or not np.isfinite(initial_angle).all():
        raise ValueError('invalid initial phase')
    angle = initial_angle.copy()
    for i, seed in enumerate(seeds):
        inc = np.asarray(bank[int(seed)], dtype=float)
        if not len(inc) or not np.isfinite(inc).all() or np.any(inc < 0) or np.any(inc > math.pi / 3 + 1e-9):
            raise ValueError('invalid locked phase bank')
        if mode == 'constant':
            angle[i] += step * inc.mean()
        else:
            cycles, remain = divmod(step, len(inc))
            angle[i] += cycles * inc.sum() + inc[:remain].sum()
    return np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)


def replace_phase(obs, phase):
    obs, phase = np.asarray(obs), np.asarray(phase)
    if obs.ndim != 2 or obs.shape[1] != 98 or phase.shape != (len(obs), 2):
        raise ValueError('phase/actor shape mismatch')
    if not np.isfinite(obs).all() or not np.isfinite(phase).all() or not np.allclose(np.linalg.norm(phase, axis=1), 1., atol=2e-6):
        raise ValueError('nonfinite or off-circle actor/phase')
    used = obs.copy(); used[:, 9:11] = phase
    return used


def push_delta(scenario, step, smoke=False):
    if scenario not in SCENARIOS:
        raise ValueError('undeclared perturbation')
    if scenario == 'steady':
        return 0.
    amplitude = .3 if scenario == 'push03' else .6
    return {3 if smoke else 250: amplitude,
            6 if smoke else 500: -amplitude}.get(step, 0.)


def recovery(time, velocity, fell):
    time, velocity = np.asarray(time), np.asarray(velocity)
    error = np.hypot(velocity[:, 0] - .5, velocity[:, 1])
    output = []
    for start in (5., 10.):
        # Next push closes first recovery window; 20 seconds closes second.
        end = 10. if start == 5. else 20.
        indices = np.flatnonzero((time >= start - 1e-8) & (time < end))
        exposed = bool(len(indices) and abs(time[indices[0]] - start) < .011)
        success_at = None
        for i in indices:
            if i + 50 <= len(time) and time[i + 49] < end and np.all(error[i:i + 50] < .15):
                success_at = float(time[i] - start)
                break
        output.append(dict(push_time=start, exposed=exposed,
                           recovered=bool(exposed and success_at is not None),
                           seconds=success_at, fell=bool(fell)))
    return output


def choose_clock(rows):
    if not rows or any(r['split'] != 'validation' or r['method'] not in ('clock55', 'clock60') for r in rows):
        raise ValueError('only locked validation clock outcomes may select')
    scores = {}
    for method in ('clock55', 'clock60'):
        values = [r for r in rows if r['method'] == method]
        if not values:
            raise ValueError('incomplete clock validation')
        scores[method] = (np.mean([r['completed'] for r in values]),
                          -np.mean([r['fell'] for r in values]), method == 'clock55')
    return max(scores, key=scores.get)


def seed_stats(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('need finite paired seed differences')
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    radius = float(t.ppf(.975, len(values) - 1) * se)
    flips = np.asarray(list(product((-1., 1.), repeat=len(values))))
    p = float(np.mean(np.abs((flips * values).mean(axis=1)) >= abs(mean) - 1e-12))
    return dict(n_training_seeds=len(values), per_seed=values.tolist(), mean=mean,
                ci95_t=[mean - radius, mean + radius], p_exact=p,
                ci_assumption='independent paired seed differences approximately normal')


def holm(pvalues):
    order = np.argsort(pvalues); adjusted = [0.] * len(pvalues); previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., float(pvalues[i]) * (len(order) - rank)))
        adjusted[i] = previous
    return adjusted
