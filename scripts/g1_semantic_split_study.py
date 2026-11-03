"""Shared, portable contract for the G1 state/condition split experiment.

Importing this module does not construct a simulator or touch CUDA. The four
actors use RSL-RL's normalizer and state-independent Gaussian unchanged.
"""
from __future__ import annotations

import copy
import enum
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/g1_semantic_split_20261026'
ARMS = ('original', 'matched', 'semantic', 'random')
EXPECTED_TERMS = [('base_ang_vel', 3), ('projected_gravity', 3), ('command', 3),
                  ('phase', 2), ('joint_pos', 29), ('joint_vel', 29), ('actions', 29)]
STATE_TERMS = ('base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions')
CONDITION_TERMS = ('command', 'phase')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, enum.Enum):
        return {'enum': type(value).__module__ + ':' + type(value).__qualname__, 'name': value.name}
    if callable(value):
        return value.__module__ + ':' + value.__qualname__
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    raise TypeError(f'noncanonical config value: {type(value).__name__}')


def protocol_digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(canonical(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def observation_layout(terms):
    fields, offset = {}, 0
    expected = dict(EXPECTED_TERMS)
    for name, size in terms:
        size = int(size)
        if name not in expected or name in fields or size != expected[name]:
            raise ValueError(f'unsupported/duplicate observation term: {name} ({size})')
        fields[name] = {'start': offset, 'stop': offset + size, 'dimension': size}
        offset += size
    if set(fields) != set(expected) or offset != 98:
        raise ValueError('requires the complete 98-D G1 flat observation, no height scan or history')
    indices = lambda names: [i for name in names for i in range(fields[name]['start'], fields[name]['stop'])]
    return {'fields': fields, 'ordered_terms': [[n, int(d)] for n, d in terms],
            'total_dimension': offset, 'state_indices': indices(STATE_TERMS),
            'condition_indices': indices(CONDITION_TERMS)}


def runtime_layout(env):
    manager = env.observation_manager
    names = manager.active_terms['actor']
    dimensions = manager.group_obs_term_dim['actor']
    return observation_layout([(name, math.prod(shape)) for name, shape in zip(names, dimensions, strict=True)])


def random_partition(total, small, seed):
    if total != 98 or small != 5:
        raise ValueError('this experiment fixes dimensions to 93/5')
    condition = sorted(np.random.default_rng(seed).choice(total, small, replace=False).tolist())
    return [i for i in range(total) if i not in condition], condition


class SplitMLP(nn.Module):
    def __init__(self, state_indices, condition_indices, output_dim=29):
        super().__init__()
        if (len(state_indices) != 93 or len(condition_indices) != 5
                or sorted(list(state_indices) + list(condition_indices)) != list(range(98))):
            raise ValueError('split must partition all 98 observations exactly once')
        self.register_buffer('state_indices', torch.tensor(state_indices, dtype=torch.long))
        self.register_buffer('condition_indices', torch.tensor(condition_indices, dtype=torch.long))
        self.state_encoder = nn.Sequential(nn.Linear(93, 480), nn.ELU())
        self.condition_encoder = nn.Sequential(nn.Linear(5, 32), nn.ELU())
        self.head = MLP(512, output_dim, (256, 128), 'elu')

    def encode(self, x):
        return torch.cat((self.state_encoder(x.index_select(-1, self.state_indices)),
                          self.condition_encoder(x.index_select(-1, self.condition_indices))), dim=-1)

    def forward(self, x):
        return self.head(self.encode(x))


class SplitModel(MLPModel):
    """RSL-RL model retaining its distribution, normalization and export wrappers."""
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, state_indices,
                 condition_indices, hidden_dims=(512, 256, 128), activation='elu',
                 obs_normalization=True, distribution_cfg=None):
        if tuple(hidden_dims) != (512, 256, 128) or activation != 'elu' or output_dim != 29:
            raise ValueError('the frozen split architecture is 480+32 / 256 / 128 / 29 with ELU')
        if distribution_cfg is not None and distribution_cfg['class_name'] != 'GaussianDistribution':
            raise ValueError('only the shared state-independent Gaussian is supported')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, copy.deepcopy(distribution_cfg))
        if self.obs_dim != 98 or self._get_latent_dim() != 98:
            raise ValueError('split actor requires 98 scalar observations')
        self.mlp = SplitMLP(state_indices, condition_indices, output_dim)


def actor_config(arm, layout=None, partition_seed=20261026):
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    cfg = {'class_name': 'MLPModel', 'hidden_dims': [497 if arm == 'matched' else 512, 256, 128],
           'activation': 'elu', 'obs_normalization': True,
           'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}}
    if arm in ('semantic', 'random'):
        layout = layout or observation_layout(EXPECTED_TERMS)
        a, b = ((layout['state_indices'], layout['condition_indices']) if arm == 'semantic'
                else random_partition(98, 5, partition_seed))
        cfg.update(class_name='scripts.g1_semantic_split_study:SplitModel',
                   state_indices=a, condition_indices=b)
    return cfg


def build_actor(arm, obs, layout=None, partition_seed=20261026):
    config = actor_config(arm, layout, partition_seed)
    name = config.pop('class_name')
    cls = SplitModel if ':' in name else MLPModel
    return cls(obs, {'actor': ['actor']}, 'actor', 29, **config)


def model_budget(model):
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return {'linear_parameters': sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0) for m in linear),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'normalization_buffers': sum(b.numel() for _, b in model.obs_normalizer.named_buffers()),
            'all_buffers': sum(b.numel() for b in model.buffers()),
            'linear_macs_per_action': sum(m.in_features * m.out_features for m in linear),
            'mac_scope': 'linear layers only; excludes activations, normalization, gather and concatenation'}


def default_protocol():
    return {'schema': 'g1.semantic_split.protocol.v1', 'created_at': '2026-10-26',
            'status': 'LOCAL_FROZEN_PLAN_NOT_EXTERNAL_REGISTRATION',
            'question': 'At near-equal actor budgets, does state/condition grouping improve G1 velocity tracking and robustness?',
            'task': 'Unitree-G1-Flat', 'arms': list(ARMS), 'phase': 'analytic sin/cos, period 0.6s',
            'partition_seed': 20261026, 'critic': [512, 256, 128],
            'normalization': 'shared policy: running per-feature statistics fitted on each policy training rollouts only',
            'training': {'pilot_iterations': 1000, 'formal_iterations': 10001,
                         'pilot_seed': 20262100, 'seeds': [20262101, 20262102, 20262103],
                         'num_envs': 4096, 'steps_per_env': 24, 'from_scratch': True,
                         'save_interval': 100, 'hyperparameter_search_trials_per_arm': 0},
            'training_command_ranges': {'lin_vel_x': [-.5, 1.], 'lin_vel_y': [-.5, .5],
                                        'ang_vel_z': [-.5, .5]},
            'command_curriculum': 'disabled for every arm; fixed common command support',
            'checkpoint_selection': 'final_budget_checkpoint_only',
            'evaluation': {'horizon_seconds': 20., 'development_seeds': list(range(20262201, 20262211)),
                           'holdout_seeds': list(range(20262301, 20262331)),
                           'commands': [[.5, 0., 0.], [.8, 0., 0.], [.3, .2, 0.],
                                        [.3, -.2, 0.], [.3, 0., .4], [.3, 0., -.4]],
                           'scenarios': {'clean': {}, 'noise_low': {'state_noise_scale': 1.},
                                         'noise_high': {'state_noise_scale': 2.},
                                         'push_low': {'lateral_delta_v_mps': .3, 'at_seconds': 5.},
                                         'push_high': {'lateral_delta_v_mps': .6, 'at_seconds': 5.},
                                         'delay_1': {'action_delay_steps': 1},
                                         'delay_2': {'action_delay_steps': 2}},
                           'noise_uniform_half_width': {'base_ang_vel': .2, 'projected_gravity': .05,
                                                        'joint_pos': .01, 'joint_vel': 1.5},
                           'failure_tail': 'zero physical velocity through the fixed horizon; also report completion and alive-only errors',
                           'policy': 'deterministic mean, frozen normalizer, CPU float32 ONNX',
                           'holdout_gate': 'all twelve formal policies sealed before any holdout rollout'},
            'analysis': {'primary': 'clean planar velocity RMSE [m/s], equal weight per command/episode then per training seed',
                         'secondary': ['yaw RMSE [rad/s]', 'failure proportion', 'completion', 'robustness per scenario'],
                         'comparisons': [['semantic', 'matched'], ['semantic', 'random']],
                         'unit': 'independently trained policy; episodes are repeated measurements',
                         'screening_relative_improvement': .05, 'max_failure_increase': .02,
                         'threshold_status': 'implementation defaults chosen before new results; exploratory screening, not acceptance criteria for publication',
                         'uncertainty': 'paired training-seed t intervals and Holm adjustment; low precision with n=3',
                         'inconclusive_action': 'retain all runs; consider more training repeats without editing original outcomes'},
            'stopping': {'training': 'fixed updates; abort numerical or identity failures; no performance early stopping',
                         'evaluation': 'fixed horizon; first non-timeout termination marks episode failed',
                         'budget_exhaustion': 'report INCOMPLETE; never replace missing results with pilot results'},
            'latency': {'backend': 'onnxruntime CPUExecutionProvider float32', 'batch': 1,
                        'threads': 1, 'warmup': 200, 'samples': 2000,
                        'quantiles': [50, 95, 99], 'scope': 'normalization, gathers, encoders, fusion, head; excludes simulation and sensor IO'}}


def training_jobs(protocol, stage):
    if stage not in ('pilot', 'formal'):
        raise ValueError('stage must be pilot or formal')
    seeds = [protocol['training']['pilot_seed']] if stage == 'pilot' else protocol['training']['seeds']
    # Counterbalance execution order; training seed is unrelated to the partition seed.
    return [{'arm': arm, 'seed': seed, 'stage': stage} for i, seed in enumerate(seeds)
            for arm in (ARMS if i % 2 == 0 else tuple(reversed(ARMS)))]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def bootstrap():
    """Use the existing project compatibility fixes without any learned phase."""
    from src.runtime_compat import configure_windows_ascii_temp_dir, configure_warp_ascii_paths
    configure_windows_ascii_temp_dir()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('LOCAL_RANK', '0')
    import warp
    if not hasattr(warp, 'context'):
        import warp._src.context as warp_context
        warp.context = warp_context
    configure_warp_ascii_paths(warp)
    # The normal registry also constructs learned-phase tasks at import time.
    # This opt-in mode registers only the existing fixed-phase G1 baselines.
    os.environ['UNITREE_G1_FIXED_PHASE_ONLY'] = '1'
    from src.tasks.velocity.phasor.mujoco_io import install_mjlab_scene_xml_unicode_compat
    install_mjlab_scene_xml_unicode_compat()
    import src.tasks  # noqa: F401
    if os.name != 'nt':
        os.environ.setdefault('MUJOCO_GL', 'egl')
    torch.set_num_threads(2)


def make_env_config(protocol, seed, num_envs, *, evaluation=False):
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg(protocol['task'], play=evaluation)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = protocol['evaluation']['horizon_seconds'] + (1. if evaluation else 0.)
    cmd = cfg.commands['twist']
    cmd.heading_command = False
    cmd.ranges.heading = None
    cfg.curriculum.pop('command_vel', None)
    for name, values in protocol['training_command_ranges'].items():
        setattr(cmd.ranges, name, tuple(values))
    if evaluation:
        cfg.curriculum.clear()
        cfg.events.pop('push_robot', None)
        cfg.observations['actor'].enable_corruption = False
        cmd.rel_standing_envs = 0.
        cmd.resampling_time_range = (1000., 1000.)
        for name in protocol['training_command_ranges']:
            setattr(cmd.ranges, name, (0., 0.))
    for group in ('actor', 'critic'):
        term = cfg.observations[group].terms['phase']
        if term.func.__name__ != 'phase' or term.params['period'] != .6:
            raise ValueError('phase must remain the original fixed analytic clock')
    if 'height_scan' in cfg.observations['actor'].terms or 'foot_gait' not in cfg.rewards:
        raise ValueError('requires unchanged flat baseline observation and gait reward')
    observation_layout([(n, dict(EXPECTED_TERMS).get(n, -1)) for n in cfg.observations['actor'].terms])
    return cfg


class ActionDelay:
    def __init__(self, steps, batch_size, actions):
        if steps < 0:
            raise ValueError('negative delay')
        self.steps = steps
        self.shape = (batch_size, actions)
        self.reset()

    def reset(self):
        self.history = [np.zeros(self.shape, np.float32) for _ in range(self.steps)]

    def __call__(self, action):
        if action.shape != self.shape:
            raise ValueError('action shape changed')
        if not self.steps:
            return action.copy()
        self.history.append(action.copy())
        return self.history.pop(0)


def episode_metrics(commands, velocities, *, alive_steps, fell, dt):
    commands, velocities = np.asarray(commands, float), np.asarray(velocities, float)
    if (commands.ndim != 2 or commands.shape[1] != 3 or velocities.shape != commands.shape
            or not 0 <= alive_steps <= len(commands) or dt <= 0
            or not np.isfinite(commands).all() or not np.isfinite(velocities).all()):
        raise ValueError('invalid full-horizon command/velocity trajectory')
    if not fell and alive_steps != len(commands):
        raise ValueError('a surviving episode must cover the entire horizon')
    physical = velocities.copy()
    physical[alive_steps:] = 0.
    squared = (physical - commands) ** 2
    # Euclidean planar RMSE, not an average mixing m/s with rad/s.
    planar = np.sum(squared[:, :2], axis=1)
    return {'planar_rmse_mps': float(np.sqrt(planar.mean())),
            'yaw_rmse_radps': float(np.sqrt(squared[:, 2].mean())),
            'forward_rmse_mps': float(np.sqrt(squared[:, 0].mean())),
            'lateral_rmse_mps': float(np.sqrt(squared[:, 1].mean())),
            'alive_planar_rmse_mps': float(np.sqrt(planar[:alive_steps].mean())) if alive_steps else None,
            'fell': bool(fell), 'completion': int(not fell), 'alive_steps': int(alive_steps),
            'survival_fraction': alive_steps / len(commands), 'survival_seconds': alive_steps * dt}


def validate_policy_seal(seal, protocol):
    expected = {(j['arm'], j['seed']) for j in training_jobs(protocol, 'formal')}
    rows = seal['policies']
    if (len(rows) != 12 or {(r['arm'], r['seed']) for r in rows} != expected
            or seal.get('protocol_sha256') != protocol_digest(protocol)):
        raise ValueError('holdout requires exactly all 12 formal policies under the same frozen protocol')
    for row in rows:
        for name in ('policy', 'checkpoint'):
            if sha(row[name]) != row[name + '_sha256']:
                raise ValueError('sealed policy/checkpoint content changed')


def summarize_effects(rows):
    from collections import defaultdict
    from scipy.stats import t, ttest_1samp
    groups = defaultdict(list)
    for row in rows:
        if row['scenario'] == 'clean':
            groups[(row['arm'], row['training_seed'])].append(row)
    seeds = sorted({seed for arm, seed in groups})
    means = {key: {'rmse': float(np.mean([r['planar_rmse_mps'] for r in values])),
                   'failure': float(np.mean([r['fell'] for r in values]))} for key, values in groups.items()}
    comparisons = {}
    for control in ('matched', 'random'):
        paired = [s for s in seeds if ('semantic', s) in means and (control, s) in means]
        if not paired:
            raise ValueError('missing paired training repeats')
        difference = np.array([means['semantic', s]['rmse'] - means[control, s]['rmse'] for s in paired])
        relative = np.array([(means[control, s]['rmse'] - means['semantic', s]['rmse'])
                             / max(means[control, s]['rmse'], 1e-12) for s in paired])
        sd = float(difference.std(ddof=1)) if len(paired) > 1 else None
        half = float(t.ppf(.975, len(paired)-1) * sd / np.sqrt(len(paired))) if sd is not None else None
        p = (float(ttest_1samp(difference, 0.).pvalue) if sd and len(paired) > 1
             else (1. if np.all(difference == 0) else None))
        comparisons['semantic_vs_' + control] = {'paired_n': len(paired), 'training_seeds': paired,
            'mean_difference_mps': float(difference.mean()), 'seed_differences_mps': difference.tolist(),
            'paired_t_95pct_ci_mps': [float(difference.mean()-half), float(difference.mean()+half)] if half is not None else None,
            'two_sided_p': p, 'mean_relative_improvement': float(relative.mean()),
            'all_seeds_improve': bool(np.all(difference < 0)),
            'mean_failure_increase': float(np.mean([means['semantic', s]['failure']-means[control, s]['failure'] for s in paired]))}
    ordered = sorted((v['two_sided_p'], k) for k, v in comparisons.items() if v['two_sided_p'] is not None)
    running = 0.
    for i, (p, key) in enumerate(ordered):
        running = max(running, min(1., p * (len(comparisons)-i)))
        comparisons[key]['holm_adjusted_p'] = running
    return {'scope': 'screening; three training seeds do not establish generality',
            'independent_training_repeats': len(seeds), 'comparisons': comparisons,
            'per_training_seed': [{'arm': arm, 'seed': seed, **values} for (arm, seed), values in sorted(means.items())]}
