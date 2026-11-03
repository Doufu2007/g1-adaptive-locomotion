import numpy as np

from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit


def test_constant_rotation_cannot_repair_wrong_cycle_frequency():
    contacts = np.zeros((251, 2), bool)
    for start in range(10, 251, 30):
        contacts[start:start+12, 0] = True
    for start in range(25, 251, 30):
        contacts[start:start+12, 1] = True
    phase = np.arange(251) * 2*np.pi/30
    good = audit.canonical_phase_score(phase, contacts, debounce=3)
    shifted = audit.canonical_phase_score(phase+.21, contacts, debounce=3)
    slow = audit.canonical_phase_score(phase*.8+.21, contacts, debounce=3)
    assert good['completed'] == good['evaluable'] and good['evaluable'] >= 7
    assert shifted['completed'] == good['completed']
    assert slow['completed'] == 0


def test_three_frame_debounce_rejects_one_frame_contact_flicker():
    contacts = np.zeros((251, 2), bool)
    for start in range(10, 251, 30):
        contacts[start:start+12, 0] = True
    for start in range(25, 251, 30):
        contacts[start:start+12, 1] = True
    phase = np.arange(251) * 2*np.pi/30
    clean = audit.canonical_phase_score(phase, contacts, debounce=3)
    contacts[30, 0] = True
    noisy = audit.canonical_phase_score(phase, contacts, debounce=3)
    raw = audit.canonical_phase_score(phase, contacts, debounce=1)
    assert noisy['evaluable'] == clean['evaluable']
    assert noisy['completed'] == clean['completed']
    assert raw['observed'] > noisy['observed']


def test_phase_rotation_preserves_cached_output_and_sin_cos_convention():
    import torch
    cached = torch.tensor([[0., 1.], [1., 0.]])
    original = cached.clone()
    shifted = audit.rotate_phase_vectors(cached, np.pi/2)
    torch.testing.assert_close(shifted, torch.tensor([[1., 0.], [0., -1.]]), atol=1e-6, rtol=1e-6)
    assert torch.equal(cached, original)
    torch.testing.assert_close(audit.rotate_phase_vectors(shifted, -np.pi/2), original, atol=1e-6, rtol=1e-6)
