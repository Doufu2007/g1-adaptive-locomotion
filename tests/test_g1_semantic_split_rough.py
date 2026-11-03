"""Rough terrain must preserve the 98-feature actor comparison."""
import copy
from dataclasses import asdict

import pytest

from scripts import g1_semantic_split_study as study


@pytest.mark.parametrize('evaluation', [False, True])
def test_blind_rough_retains_physical_terrain_and_fixed_phase(evaluation):
    study.bootstrap()
    from mjlab.tasks.registry import load_env_cfg

    protocol = copy.deepcopy(study.default_protocol())
    protocol.update(task='Unitree-G1-Rough', terrain_mode='blind_rough')
    baseline = load_env_cfg(protocol['task'], play=evaluation)
    cfg = study.make_env_config(protocol, 20262101, 1024, evaluation=evaluation)
    assert cfg.scene.terrain.terrain_type == 'generator'
    assert cfg.scene.terrain.terrain_generator is not None
    assert cfg.scene.terrain.terrain_generator.curriculum is (not evaluation)
    assert ('terrain_levels' in cfg.curriculum) is (not evaluation)
    assert cfg.scene.num_envs == 1024
    assert cfg.scene.terrain.terrain_generator.seed == 20262101
    assert cfg.sim.nconmax is None  # Avoid silent rough contact-buffer overflow.
    assert all(s.name != 'terrain_scan' for s in cfg.scene.sensors)
    for group in ('actor', 'critic'):
        assert 'height_scan' not in cfg.observations[group].terms
        phase = cfg.observations[group].terms['phase']
        assert phase.func.__name__ == 'phase'
        assert phase.params['period'] == .6
    assert list(cfg.observations['actor'].terms) == [n for n, _ in study.EXPECTED_TERMS]
    assert study.canonical(asdict(cfg)['rewards']) == study.canonical(asdict(baseline)['rewards'])
    assert 'command_vel' not in cfg.curriculum
    if evaluation:
        assert not cfg.observations['actor'].enable_corruption
        assert 'push_robot' not in cfg.events
        reset_terms = [name for name, term in cfg.events.items() if term.mode == 'reset']
        assert reset_terms.index('randomize_terrain') < reset_terms.index('reset_base')
    else:
        assert cfg.observations['actor'].enable_corruption
        assert 'push_robot' in cfg.events


def test_rough_requires_explicit_blind_amendment():
    study.bootstrap()
    protocol = copy.deepcopy(study.default_protocol())
    protocol['task'] = 'Unitree-G1-Rough'
    with pytest.raises(ValueError, match='blind_rough'):
        study.make_env_config(protocol, 1, 4)


def test_rough_pairing_gate_rejects_different_actual_spawn_after_other_history(tmp_path):
    from scripts.evaluate_g1_semantic_split import verify_rough_pairing
    reference = tmp_path/'episode_seed7.json'
    snapshot = {'terrain_levels': [0], 'terrain_types': [2],
                'env_origins': [[1., 2., 0.]], 'initial_root_pos_w': [[1., 2., .78]]}
    verify_rough_pairing(reference, snapshot)
    verify_rough_pairing(reference, copy.deepcopy(snapshot))
    changed = copy.deepcopy(snapshot)
    changed['initial_root_pos_w'][0][0] += 1.
    with pytest.raises(ValueError, match='rough.*pairing'):
        verify_rough_pairing(reference, changed)


@pytest.mark.parametrize('evaluation', [False, True])
def test_phase_random_rough_distribution_and_reset_match_existing_phase_study(evaluation):
    study.bootstrap()
    protocol = copy.deepcopy(study.default_protocol())
    protocol.update(task='Unitree-G1-Rough', terrain_mode='blind_rough',
                    terrain_distribution='phase_random_rough')
    cfg = study.make_env_config(protocol, 20262101, 1024, evaluation=evaluation)
    g = cfg.scene.terrain.terrain_generator
    assert (g.num_rows, g.num_cols, g.size, g.curriculum) == (1, 5, (32., 16.), False)
    assert list(g.sub_terrains) == ['random_rough']
    r = g.sub_terrains['random_rough']
    assert r.noise_range == (.02, .10)
    assert (r.noise_step, r.horizontal_scale, r.vertical_scale, r.border_width) == (.02, .1, .005, .25)
    assert 'terrain_levels' not in cfg.curriculum
    assert cfg.sim.nconmax == 128 and cfg.sim.njmax == 1500
    assert cfg.sim.mujoco.ccd_iterations == 50
    assert next(iter(cfg.events)) == 'randomize_terrain'
    assert cfg.events['randomize_terrain'].func.__name__ == 'deterministic_rough_terrain'
    assert list(cfg.observations['actor'].terms) == [n for n, _ in study.EXPECTED_TERMS]


def test_rough_physics_capacity_saturation_is_a_technical_failure():
    study.check_physics_capacity(127, 128, 1499, 1500)
    with pytest.raises(ValueError, match='capacity'):
        study.check_physics_capacity(128, 128, 100, 1500)
    with pytest.raises(ValueError, match='capacity'):
        study.check_physics_capacity(20, 128, 1500, 1500)
