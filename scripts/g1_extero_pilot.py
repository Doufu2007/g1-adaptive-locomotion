"""Fast pilot contract for a PPO actor with a terrain-scan stream.

This module is deliberately separate from the frozen 98-D state/condition
study.  The actor keeps the 98 existing proprioceptive/task values and adds
the native terrain scan (187 values at the configured ray grid) as a second
input stream.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn

from scripts import g1_semantic_split_study as base

ARMS = base.ARMS
BASE_TERMS = list(base.EXPECTED_TERMS)
EXPECTED_TERMS = BASE_TERMS + [('height_scan', 187)]
PROPRIO_DIM = sum(size for _, size in BASE_TERMS)
EXTERO_DIM = 187
TOTAL_DIM = PROPRIO_DIM + EXTERO_DIM


def observation_layout(terms):
    expected = dict(EXPECTED_TERMS)
    fields, offset = {}, 0
    for name, size in terms:
        size = int(size)
        if name not in expected or name in fields or size != expected[name]:
            raise ValueError(f'unsupported/duplicate observation term: {name} ({size})')
        fields[name] = {'start': offset, 'stop': offset + size, 'dimension': size}
        offset += size
    if set(fields) != set(expected) or offset != TOTAL_DIM:
        raise ValueError('requires 98 base observations plus a 187-D terrain scan')
    proprio = [i for name, size in BASE_TERMS for i in range(fields[name]['start'], fields[name]['stop'])]
    extero = list(range(fields['height_scan']['start'], fields['height_scan']['stop']))
    return {'fields': fields, 'ordered_terms': [[n, int(d)] for n, d in terms],
            'total_dimension': offset, 'proprio_indices': proprio,
            'extero_indices': extero}


def runtime_layout(env):
    manager = env.observation_manager
    names = manager.active_terms['actor']
    dimensions = manager.group_obs_term_dim['actor']
    return observation_layout([(name, math.prod(shape)) for name, shape in zip(names, dimensions, strict=True)])


def random_partition(seed):
    extero = sorted(np.random.default_rng(seed).choice(TOTAL_DIM, EXTERO_DIM, replace=False).tolist())
    extero_set = set(extero)
    proprio = [i for i in range(TOTAL_DIM) if i not in extero_set]
    return proprio, extero


class ExteroSplitMLP(nn.Module):
    def __init__(self, proprio_indices, extero_indices, output_dim=29):
        super().__init__()
        if (len(proprio_indices) != PROPRIO_DIM or len(extero_indices) != EXTERO_DIM
                or sorted(list(proprio_indices) + list(extero_indices)) != list(range(TOTAL_DIM))):
            raise ValueError('split must partition all 285 observations exactly once')
        self.register_buffer('proprio_indices', torch.tensor(proprio_indices, dtype=torch.long))
        self.register_buffer('extero_indices', torch.tensor(extero_indices, dtype=torch.long))
        self.proprio_encoder = nn.Sequential(nn.Linear(PROPRIO_DIM, 480), nn.ELU())
        self.extero_encoder = nn.Sequential(nn.Linear(EXTERO_DIM, 32), nn.ELU())
        self.head = MLP(512, output_dim, (256, 128), 'elu')

    def encode(self, x):
        return torch.cat((self.proprio_encoder(x.index_select(-1, self.proprio_indices)),
                          self.extero_encoder(x.index_select(-1, self.extero_indices))), dim=-1)

    def forward(self, x):
        return self.head(self.encode(x))


class ExteroSplitModel(MLPModel):
    """RSL-RL model wrapper for the 98-D + terrain-scan actor."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, *, proprio_indices,
                 extero_indices, hidden_dims=(512, 256, 128), activation='elu',
                 obs_normalization=True, distribution_cfg=None):
        if tuple(hidden_dims) != (512, 256, 128) or activation != 'elu' or output_dim != 29:
            raise ValueError('extero split architecture is 480+32 / 256 / 128 / 29 with ELU')
        if distribution_cfg is not None and distribution_cfg['class_name'] != 'GaussianDistribution':
            raise ValueError('only the shared state-independent Gaussian is supported')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, copy.deepcopy(distribution_cfg))
        if self.obs_dim != TOTAL_DIM or self._get_latent_dim() != TOTAL_DIM:
            raise ValueError('extero split actor requires 285 scalar observations')
        self.mlp = ExteroSplitMLP(proprio_indices, extero_indices, output_dim)


def actor_config(arm, layout=None, partition_seed=20261105):
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    cfg = {'class_name': 'MLPModel', 'hidden_dims': [341 if arm == 'matched' else 512, 256, 128],
           'activation': 'elu', 'obs_normalization': True,
           'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}}
    if arm in ('semantic', 'random'):
        layout = layout or observation_layout(EXPECTED_TERMS)
        if arm == 'semantic':
            proprio, extero = layout['proprio_indices'], layout['extero_indices']
        else:
            proprio, extero = random_partition(partition_seed)
        cfg.update(class_name='scripts.g1_extero_pilot:ExteroSplitModel',
                   proprio_indices=proprio, extero_indices=extero)
    return cfg


def default_protocol():
    protocol = copy.deepcopy(base.default_protocol())
    protocol.update(schema='g1.extero_split.protocol.v1',
                    question='Does separate terrain-scan encoding improve G1 rough-terrain velocity control?',
                    task='Unitree-G1-Rough', terrain_mode='extero_rough',
                    terrain_distribution='phase_random_rough', partition_seed=20261105)
    protocol['training'].update(pilot_iterations=300, formal_iterations=300,
                                pilot_seed=20261105, seeds=[20261105], num_envs=512)
    protocol['evaluation']['horizon_seconds'] = 5.0
    protocol['evaluation']['development_seeds'] = [20262501]
    protocol['evaluation']['holdout_seeds'] = [20262501]
    protocol['pilot_amendment'] = {
        'scope': 'exploratory short pilot; not the frozen 98-D study',
        'actor_observation': '98 base terms plus native 187-D terrain_scan height map = 285',
        'streams': '98 -> 480 and 187 -> 32, concatenated to 512 before 256 -> 128 -> 29 head',
        'matched_baseline': '285 -> 341 -> 256 -> 128 -> 29',
        'budget': '512 environments x 300 updates; one training seed',
    }
    return protocol


def make_env_config(protocol, seed, num_envs, *, evaluation=False):
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg(protocol['task'], play=evaluation)
    if protocol.get('terrain_mode') != 'extero_rough':
        raise ValueError('extero pilot requires explicit extero_rough mode')
    if 'height_scan' not in cfg.observations['actor'].terms:
        raise ValueError('native rough config must expose actor height_scan')
    # Keep the native terrain sensor and use the same deterministic rough map
    # amendment as the paused blind-rough cohort.
    cfg.sim.nconmax = None
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = False
    generator.num_rows, generator.num_cols = 1, 5
    generator.size = (32., 16.)
    rough = copy.deepcopy(generator.sub_terrains['random_rough'])
    rough.proportion = 1.
    rough.noise_range, rough.noise_step = (.02, .10), .02
    rough.horizontal_scale, rough.vertical_scale, rough.border_width = .1, .005, .25
    generator.sub_terrains = {'random_rough': rough}
    cfg.curriculum.pop('terrain_levels', None)
    cfg.sim.nconmax, cfg.sim.njmax = 128, 1500
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.events.pop('randomize_terrain', None)
    from mjlab.managers.event_manager import EventTermCfg
    cfg.events = {'randomize_terrain': EventTermCfg(func=base.deterministic_rough_terrain, mode='reset'), **cfg.events}
    if evaluation:
        terrain_event = cfg.events.pop('randomize_terrain')
        cfg.events = {'randomize_terrain': terrain_event, **cfg.events}
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
    # Runtime layout is the authoritative shape check after env construction.
    return cfg
