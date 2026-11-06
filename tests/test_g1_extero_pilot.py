def test_extero_layout_has_98_state_and_187_height_scan():
    from scripts import g1_extero_pilot as pilot

    layout = pilot.observation_layout(pilot.EXPECTED_TERMS)
    assert layout['total_dimension'] == 285
    assert len(layout['proprio_indices']) == 98
    assert len(layout['extero_indices']) == 187
    assert sorted(layout['proprio_indices'] + layout['extero_indices']) == list(range(285))


def test_random_extero_partition_preserves_semantic_widths():
    from scripts import g1_extero_pilot as pilot

    proprio, extero = pilot.random_partition(20261105)
    assert len(proprio) == 98
    assert len(extero) == 187
    assert sorted(proprio + extero) == list(range(285))


def test_actor_config_uses_480_32_fusion_and_budget_matched_baseline():
    from scripts import g1_extero_pilot as pilot

    semantic = pilot.actor_config('semantic')
    matched = pilot.actor_config('matched')
    assert semantic['proprio_indices'] == list(range(98))
    assert semantic['extero_indices'] == list(range(98, 285))
    assert semantic['hidden_dims'] == [512, 256, 128]
    assert matched['hidden_dims'] == [341, 256, 128]


def test_export_parity_uses_actor_observation_dimension():
    from types import SimpleNamespace
    from scripts import run_g1_semantic_split as runner

    assert runner.parity_input_dim(SimpleNamespace(obs_dim=285)) == 285


def test_extero_install_uses_matching_source_snapshot(monkeypatch):
    from scripts import run_g1_extero_pilot as wrapper
    from scripts import g1_semantic_split_study as study
    from scripts import run_g1_semantic_split as runner

    # ``install`` wires the extero pilot into the shared semantic runner by
    # direct assignment. Register every overwritten attribute with pytest's
    # undo stack before invoking it, so this test cannot affect later tests.
    for module, name in (
        (runner, 'prepare'),
        (runner, 'source_snapshot'),
        (study, 'make_env_config'),
        (study, 'runtime_layout'),
        (study, 'actor_config'),
        (study, 'default_protocol'),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))

    wrapper.install()
    assert runner.source_snapshot is wrapper.source_snapshot


def test_exploratory_noise_preserves_285_dimension():
    import numpy as np
    from scripts import evaluate_g1_extero_pilot as evaluator
    from scripts import g1_extero_pilot as pilot

    layout = pilot.observation_layout(pilot.EXPECTED_TERMS)
    protocol = pilot.default_protocol()
    noise = evaluator.observation_noise(
        layout, {'state_noise_scale': 1.}, protocol, np.random.default_rng(0), 3)
    assert noise.shape == (3, pilot.TOTAL_DIM)
    assert np.all(noise[:, 98:] == 0.)
