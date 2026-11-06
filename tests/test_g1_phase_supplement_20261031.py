import math
import numpy as np
import pytest
from scripts import g1_phase_supplement_contract_20261031 as c


def test_finite_matrix_and_disjoint_roles():
    p = c.design()
    assert p['train_transitions'] == 27721728
    assert p['train_PPO_updates'] == 18048
    assert p['train_Adam_steps'] == 360960
    assert p['formal_episodes'] == 3240
    assert len(c.training_jobs()) == 24
    groups = list(c.MAPS.values())
    assert sum(map(len, groups)) == len(set(x for g in groups for x in g))
    assert not set(c.NEW_SEEDS) & set(c.OLD_SEEDS)


def test_open_loop_preserves_initial_angle_and_is_continuous():
    initial = np.array([3.12, -2.9])
    increments = {1: np.array([.10, .15, .08]), 2: np.array([.2, .11])}
    for mode in ('constant', 'replay'):
        prior = None
        for step in range(12):
            phase = c.open_loop_phase(initial, [1, 2], increments, step, mode)
            assert np.allclose(np.linalg.norm(phase, axis=1), 1)
            angle = np.arctan2(phase[:, 0], phase[:, 1])
            if step == 0:
                np.testing.assert_allclose(angle, initial)
            else:
                delta = np.angle(np.exp(1j * (angle - prior)))
                assert np.all(delta > 0) and np.all(delta < math.pi / 3)
            prior = angle
        # No current motion state is accepted by this function.
        np.testing.assert_array_equal(
            c.open_loop_phase(initial, [1, 2], increments, 7, mode),
            c.open_loop_phase(initial, [1, 2], increments, 7, mode))


def test_intervention_only_changes_phase_and_not_input():
    obs = np.arange(3 * 98, dtype=np.float32).reshape(3, 98)
    before = obs.copy()
    phase = np.tile([0., 1.], (3, 1))
    used = c.replace_phase(obs, phase)
    np.testing.assert_array_equal(obs, before)
    np.testing.assert_array_equal(used[:, :9], before[:, :9])
    np.testing.assert_array_equal(used[:, 11:], before[:, 11:])
    np.testing.assert_array_equal(used[:, 9:11], phase)
    with pytest.raises(ValueError):
        c.replace_phase(obs, phase * 2)


def test_bank_never_bridges_episode_boundaries():
    a = np.column_stack((np.sin([3.0, 3.1, 3.2]), np.cos([3.0, 3.1, 3.2])))
    b = np.column_stack((np.sin([-1., -.8]), np.cos([-1., -.8])))
    np.testing.assert_allclose(c.phase_increments([a, b]), [.1, .1, .2], atol=1e-12)
    with pytest.raises(ValueError):
        c.phase_increments([np.array([[0., 1.], [-1., 0.]])])


def test_selection_is_global_validation_only_and_ties_are_fixed():
    rows = [dict(method=m, completed=done, fell=fall, split='validation')
            for m, done, fall in [('clock55', 1, 0), ('clock60', 1, 0)]]
    assert c.choose_clock(rows) == 'clock55'
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError):
        c.choose_clock(rows)


def test_push_schedule_and_failed_exposure_not_recovery():
    assert c.push_delta('push03', 250) == .3
    assert c.push_delta('push06', 500) == -.6
    assert c.push_delta('steady', 250) == 0
    assert c.push_delta('push06', 20) == 0
    t = np.arange(1000) * .02
    v = np.zeros((1000, 3)); v[:, 0] = .5
    got = c.recovery(t, v, False)
    assert [x['recovered'] for x in got] == [True, True]
    got = c.recovery(t[:200], v[:200], True)
    assert all(not x['exposed'] and not x['recovered'] for x in got)
    v[:, 1] = .2
    assert all(x['exposed'] and not x['recovered'] for x in c.recovery(t, v, False))


def test_seed_statistics_do_not_count_episodes_as_training():
    got = c.seed_stats([.1] * 8)
    assert got['n_training_seeds'] == 8
    assert got['p_exact'] == 2 / 256
    assert c.holm([.01, .03, .2]) == pytest.approx([.03, .06, .2])


def test_capacity_and_nan_are_fail_closed():
    with pytest.raises(ValueError):
        c.open_loop_phase(np.zeros(1), [1], {1: np.array([np.nan])}, 1, 'constant')
    with pytest.raises(ValueError):
        c.open_loop_phase(np.zeros(1), [1], {1: np.array([2.])}, 1, 'replay')
