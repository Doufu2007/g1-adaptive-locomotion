import numpy as np
import pytest


def test_replay_uses_each_episode_index_and_rejects_missing_tail():
    from scripts.g1_phase_feedback_probe_20261026 import replay_vectors
    angles = np.array([[3., 3.2, 3.4], [-1., -0.5, 0.]])
    table = np.stack((np.sin(angles), np.cos(angles)), -1)
    np.testing.assert_allclose(replay_vectors(table, [2, 0]), table[[0, 1], [2, 0]])
    with pytest.raises(ValueError):
        replay_vectors(table, [3, 0])


def test_linearization_preserves_start_and_steady_average_rate_across_wrap():
    from scripts.g1_phase_feedback_probe_20261026 import linearize
    time = np.arange(1000) * .02
    angle = 3. + 2*np.pi*1.2*time + .1*np.sin(2*np.pi*time)
    vectors, info = linearize(time, np.angle(np.exp(1j*angle)))
    theta = np.unwrap(np.arctan2(vectors[:, 0], vectors[:, 1]))
    np.testing.assert_allclose(vectors[0], [np.sin(3.), np.cos(3.)], atol=1e-7)
    expected = (angle[-1]-angle[150])/(2*np.pi*(time[-1]-time[150]))
    assert info['frequency_hz'] == pytest.approx(expected, abs=1e-7)
    np.testing.assert_allclose(np.diff(theta), 2*np.pi*expected*.02, atol=2e-5)
    nominal, other = linearize(time, np.angle(np.exp(1j*angle)), frequency=1/.6)
    np.testing.assert_allclose(nominal[0], vectors[0], atol=1e-7)
    assert other['frequency_hz'] == pytest.approx(1/.6)


def test_linearization_refuses_incomplete_or_nonmonotone_record():
    from scripts.g1_phase_feedback_probe_20261026 import linearize
    with pytest.raises(ValueError):
        linearize(np.array([0., .02]), np.array([0., .1]))
    with pytest.raises(ValueError):
        linearize(np.arange(1000)*.02, -np.arange(1000)*.1)
