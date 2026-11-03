import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from scripts import g1_semantic_split_study as study


def observations():
    return TensorDict({'actor': torch.randn(4, 98)}, batch_size=[4])


def test_layout_uses_fields_not_a_contiguous_93_5_slice():
    layout = study.observation_layout(study.EXPECTED_TERMS)
    assert layout['condition_indices'] == [6, 7, 8, 9, 10]
    assert layout['state_indices'] == list(range(6)) + list(range(11, 98))
    reordered = study.observation_layout(list(reversed(study.EXPECTED_TERMS)))
    assert reordered['condition_indices'] != layout['condition_indices']
    assert len(reordered['state_indices']) == 93
    assert len(reordered['condition_indices']) == 5
    with pytest.raises(ValueError):
        study.observation_layout(study.EXPECTED_TERMS + [('height_scan', 1)])


def test_random_split_preserves_every_feature_and_is_independent_of_training_seed():
    a, b = study.random_partition(98, 5, 20261026)
    assert len(a) == 93 and len(b) == 5
    assert sorted(a + b) == list(range(98))
    assert set(a).isdisjoint(b)
    assert (a, b) == study.random_partition(98, 5, 20261026)
    assert b != [6, 7, 8, 9, 10]


@pytest.mark.parametrize('arm,expected', [
    ('original', 218653), ('matched', 213328),
    ('semantic', 213277), ('random', 213277),
])
def test_model_budget_forward_gradient_and_checkpoint(arm, expected, tmp_path):
    obs = observations()
    model = study.build_actor(arm, obs)
    report = study.model_budget(model)
    assert report['linear_parameters'] == expected
    assert report['trainable_parameters'] == expected + 29
    assert model(obs).shape == (4, 29)
    model(obs).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for name, p in model.named_parameters() if 'distribution' not in name)
    clone = study.build_actor(arm, obs)
    checkpoint = tmp_path / 'actor.pt'
    torch.save(model.state_dict(), checkpoint)
    clone.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model(obs), clone(obs))
    if arm in ('semantic', 'random'):
        x = torch.randn(3, 98)
        y = model.mlp.encode(x)
        changed = x.clone()
        changed[:, model.mlp.condition_indices] += 7
        z = model.mlp.encode(changed)
        torch.testing.assert_close(y[:, :480], z[:, :480])
        assert not torch.allclose(y[:, 480:], z[:, 480:])


def test_onnx_export_matches_cpu_torch_for_every_arm(tmp_path):
    import onnxruntime as ort
    obs = observations()
    for arm in study.ARMS:
        model = study.build_actor(arm, obs).eval()
        model.update_normalization(obs)
        wrapper = model.as_onnx(verbose=False).eval()
        path = tmp_path / f'{arm}.onnx'
        torch.onnx.export(wrapper, (obs['actor'][:1],), str(path),
                          input_names=['obs'], output_names=['actions'],
                          opset_version=18, dynamo=False)
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        for row in obs['actor']:
            actual = session.run(None, {'obs': row[None].numpy()})[0]
            with torch.no_grad():
                expected = wrapper(row[None]).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


def test_episode_score_counts_failure_and_never_drops_failed_tail():
    command = np.tile([.5, 0., .2], (10, 1))
    velocity = command.copy()
    result = study.episode_metrics(command, velocity, alive_steps=4, fell=True, dt=.02)
    assert result['completion'] == 0
    assert result['survival_fraction'] == .4
    assert result['planar_rmse_mps'] == pytest.approx(np.sqrt(.6 * .25))
    assert result['yaw_rmse_radps'] == pytest.approx(np.sqrt(.6 * .04))
    assert result['alive_planar_rmse_mps'] == 0
    with pytest.raises(ValueError):
        study.episode_metrics(command, velocity[:4], alive_steps=4, fell=True, dt=.02)


def test_action_delay_is_causal_and_reset_is_episode_local():
    delay = study.ActionDelay(2, 1, 2)
    assert np.all(delay(np.array([[1., 2.]])) == 0)
    assert np.all(delay(np.array([[3., 4.]])) == 0)
    np.testing.assert_array_equal(delay(np.array([[5., 6.]])), [[1., 2.]])
    delay.reset()
    assert np.all(delay(np.ones((1, 2))) == 0)


def test_plan_has_disjoint_seeds_and_balanced_budgets():
    protocol = study.default_protocol()
    assert len(study.training_jobs(protocol, 'formal')) == 12
    assert len(study.training_jobs(protocol, 'pilot')) == 4
    assert not set(protocol['evaluation']['development_seeds']) & set(protocol['evaluation']['holdout_seeds'])
    assert protocol['training']['formal_iterations'] == 10001
    assert protocol['training']['num_envs'] == 4096
    assert protocol['training']['steps_per_env'] == 24
    assert protocol['checkpoint_selection'] == 'final_budget_checkpoint_only'
    assert study.protocol_digest(protocol) == study.protocol_digest(json.loads(json.dumps(protocol)))


def test_matrix_rejects_incomplete_formal_seal():
    protocol = study.default_protocol()
    with pytest.raises(ValueError, match='12'):
        study.validate_policy_seal({'protocol_sha256': study.protocol_digest(protocol), 'policies': []}, protocol)


def test_provisional_analysis_does_not_treat_episodes_as_training_replicates():
    rows = []
    for seed in (1, 2, 3):
        for arm, error in [('matched', .2), ('semantic', .18), ('random', .19), ('original', .21)]:
            for episode in range(8):
                rows.append(dict(arm=arm, training_seed=seed, scenario='clean',
                                 episode_seed=episode, planar_rmse_mps=error,
                                 yaw_rmse_radps=.1, fell=False, completion=1))
    report = study.summarize_effects(rows)
    assert report['independent_training_repeats'] == 3
    assert report['comparisons']['semantic_vs_matched']['paired_n'] == 3
    assert report['comparisons']['semantic_vs_matched']['mean_relative_improvement'] == pytest.approx(.1)
    assert report['scope'] == 'screening; three training seeds do not establish generality'
