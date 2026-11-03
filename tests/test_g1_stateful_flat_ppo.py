"""Online/offline semantics and fair-comparison contracts for new flat PPO."""
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch

torch.set_num_threads(2)


def online():
    return importlib.import_module('scripts.g1_stateful_online')


def study():
    return importlib.import_module('scripts.g1_stateful_flat_ppo')


def motion(n=155):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(149)
    q = np.cumsum(rng.normal(0, .004, (n, 29)), axis=0).astype(np.float32)
    xyzw = Rotation.from_rotvec(np.cumsum(rng.normal(0, .003, (n, 3)), axis=0)).as_quat()
    return q, xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def test_online_features_match_offline_and_quaternion_sign():
    from scripts.g1_stateful_phase import causal_features
    m = online()
    q, quat = motion(25)
    expected = causal_features(q, quat)
    rows = []
    for i in range(len(q)):
        kwargs = {} if i == 0 else dict(previous_q=torch.tensor(q[i-1:i]),
                                       previous_quat=torch.tensor(quat[i-1:i]))
        rows.append(m.motion_features(torch.tensor(q[i:i+1]), torch.tensor(quat[i:i+1]), **kwargs)[0])
    np.testing.assert_allclose(torch.stack(rows).numpy(), expected, atol=2e-5, rtol=2e-5)
    a = m.motion_features(torch.tensor(q[1:2]), torch.tensor(quat[1:2]),
                         previous_q=torch.tensor(q[:1]), previous_quat=torch.tensor(quat[:1]))
    b = m.motion_features(torch.tensor(q[1:2]), -torch.tensor(quat[1:2]),
                         previous_q=torch.tensor(q[:1]), previous_quat=torch.tensor(quat[:1]))
    torch.testing.assert_close(a, b)


def test_streaming_matches_full_sequence_after_receptive_field():
    from scripts.g1_stateful_phase import PhaseNet, causal_features
    m = online()
    torch.manual_seed(123)
    model = PhaseNet(64, 'observer').eval()
    q, quat = motion()
    mean = np.linspace(-.2, .2, 64).astype(np.float32)
    std = np.linspace(.5, 1.5, 64).astype(np.float32)
    runner = m.OnlinePhase(model, mean, std, num_envs=1, device='cpu')
    with torch.inference_mode():
        reference = model(torch.tensor((causal_features(q, quat)-mean)/std)[None])['phase'][0]
        actual = torch.stack([runner.observe(torch.tensor(q[i:i+1]), torch.tensor(quat[i:i+1]), i)[0].clone()
                              for i in range(len(q))])
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
    assert not any(p.requires_grad for p in runner.encoder.parameters())
    assert not runner.encoder.training


def test_same_frame_reads_and_partial_reset_do_not_advance_other_worlds():
    from scripts.g1_stateful_phase import PhaseNet
    m = online()
    torch.manual_seed(321)
    model = PhaseNet(64, 'observer').eval()
    runner = m.OnlinePhase(model, np.zeros(64), np.ones(64), num_envs=2, device='cpu')
    fresh = m.OnlinePhase(model, np.zeros(64), np.ones(64), num_envs=1, device='cpu')
    q, quat = map(torch.tensor, motion(12))
    for t in range(8):
        current = runner.observe(q[t:t+1].repeat(2, 1), quat[t:t+1].repeat(2, 1), t).clone()
        duplicate = runner.observe(q[t:t+1].repeat(2, 1), quat[t:t+1].repeat(2, 1), t).clone()
        torch.testing.assert_close(current, duplicate, rtol=0, atol=0)
    runner.reset(torch.tensor([0]))
    after = runner.observe(q[8:9].repeat(2, 1), quat[8:9].repeat(2, 1), 7).clone()
    expected = fresh.observe(q[8:9], quat[8:9], 7)
    torch.testing.assert_close(after[0], expected[0])
    torch.testing.assert_close(after[1], current[1], rtol=0, atol=0)
    assert runner.update_counts.tolist() == [1, 8]


def test_selected_checkpoint_loading_preserves_rng_and_original_sha():
    m = online()
    s = study()
    torch.manual_seed(945)
    previous = torch.get_rng_state().clone()
    runner = m.load_online(s.RUNTIME, num_envs=1, device='cpu')
    assert torch.equal(previous, torch.get_rng_state())
    assert runner.encoder.channels == 64
    assert s.sha(s.RUNTIME) == s.RUNTIME_SHA
    assert s.ENCODER_SEED == 20261025


def test_real_frozen_candidate_matches_offline_on_recorded_inputs():
    from scripts.g1_stateful_phase import causal_features
    s, m = study(), online()
    raw = next((s.SNAPSHOT/'results/g1_stateful_phase_20261025/collection/test').rglob('*.npz'))
    with np.load(raw) as data:
        q = data['joint_pos'][:160].astype(np.float32)
        quat = data['quaternion'][:160].astype(np.float32)
    runtime = m.load_online(s.RUNTIME, num_envs=1, device='cpu')
    x = (torch.tensor(causal_features(q, quat))-runtime.mean)/runtime.std
    with torch.inference_mode():
        expected = runtime.encoder(x[None])['phase'][0]
        actual = torch.stack([runtime.observe(torch.tensor(q[i:i+1]),torch.tensor(quat[i:i+1]),i)[0].clone()
                              for i in range(len(q))])
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_gpu_guard_preserves_real_other_project_identity(monkeypatch):
    from types import SimpleNamespace
    m = importlib.import_module('scripts.run_g1_stateful_flat_ppo_20261025')
    import psutil
    monkeypatch.setattr(m.subprocess, 'run', lambda *a,**k:SimpleNamespace(stdout='12345, python.exe\n'))
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(cmdline=lambda:['python','other_project.py'],
                                                                  create_time=lambda:123.))
    jobs = m.gpu_jobs()
    assert jobs == [dict(pid=12345,name='python.exe',argv=['python','other_project.py'],created_at=123.)]


def test_evaluation_uses_same_bootstrap_precision_as_online_smoke(monkeypatch):
    from scripts import g1_best_phase_batched as batch
    from scripts import run_g1_phase_adaptation_20261024 as original
    m, s = importlib.import_module('scripts.run_g1_stateful_flat_ppo_20261025'), study()
    new_bootstrap = lambda: None
    monkeypatch.setattr(s, 'bootstrap', new_bootstrap)
    monkeypatch.setattr(m, 'verify_policies', lambda: None)
    def evaluate(*args):
        assert batch.bootstrap is new_bootstrap
    monkeypatch.setattr(original, 'evaluate', evaluate)
    m.worker('evaluate', 'clock', s.TRAIN_SEEDS[0], label='clock_clock')


def test_stale_smoke_cannot_freeze_changed_code(monkeypatch):
    m, s = importlib.import_module('scripts.run_g1_stateful_flat_ppo_20261025'), study()
    monkeypatch.setattr(s, 'smoke_identity', lambda: 'new-scientific-code', raising=False)
    record = dict(status='COMPLETED',encoder_sha256=s.RUNTIME_SHA,smoke_identity='old-scientific-code')
    with pytest.raises(ValueError, match='smoke|changed|identity'):
        m.verify_smoke(record)


def test_shared_bootstrap_fixes_torch_precision_and_cudnn(monkeypatch):
    from types import SimpleNamespace
    s, m = study(), online()
    backend = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                              cudnn=SimpleNamespace(allow_tf32=True,deterministic=False,benchmark=True))
    fake_torch = SimpleNamespace(backends=backend)
    monkeypatch.setattr(s.base,'bootstrap',lambda:(None,fake_torch))
    monkeypatch.setattr(m,'install',lambda:None)
    s.bootstrap()
    assert not backend.cuda.matmul.allow_tf32 and not backend.cudnn.allow_tf32
    assert backend.cudnn.deterministic and not backend.cudnn.benchmark


def test_new_matrix_keeps_equal_budget_and_direct_swap_policy():
    s = study()
    assert len(s.training_jobs()) == 6
    assert s.TRANSITIONS == 501 * 64 * 24
    assert s.TRAIN_SEEDS == (20261901, 20261902, 20261903)
    assert s.EVAL_SEEDS == tuple(range(20262001, 20262011))
    for seed in s.TRAIN_SEEDS:
        jobs = s.evaluation_jobs('flat', seed)
        assert len(jobs) == 3
        assert jobs[0]['policy'] == jobs[1]['policy']
        assert jobs[2]['policy'] != jobs[0]['policy']
    with pytest.raises(ValueError, match='flat'):
        s.make_cfg('rough', 'clock', s.TRAIN_SEEDS[0], play=False, num_envs=64, horizon=20.)


def test_configs_identical_outside_phase_and_old_sources_untouched():
    s = study()
    s.bootstrap()
    a = s.make_cfg('flat', 'clock', s.TRAIN_SEEDS[0], play=False, num_envs=64, horizon=20.)
    b = s.make_cfg('flat', 'learned', s.TRAIN_SEEDS[0], play=False, num_envs=64, horizon=20.)
    assert s.common_cfg_digest(a) == s.common_cfg_digest(b)
    assert 'foot_gait' not in a.rewards and 'foot_gait' not in b.rewards
    assert len(a.observations['actor'].terms) == len(b.observations['actor'].terms)
    for group in ('actor', 'critic'):
        assert b.observations[group].terms['phase'].func.__module__ == 'scripts.g1_stateful_online'
    s.verify_runtime()


def test_pairing_uses_three_training_replicates_and_rejects_missing():
    s = study()
    rows = []
    for seed in s.TRAIN_SEEDS:
        for label, value in [('clock_clock', .3), ('learned_learned', .2)]:
            for env_seed in s.EVAL_SEEDS:
                rows.append(dict(terrain='flat', train_seed=seed, label=label, scenario='steady',
                                 seed=env_seed, fell=False, mae_forward_velocity_mps=value))
    result = s.paired_training_effect(rows, 'flat', 'learned_learned', 'clock_clock',
                                      'mae_forward_velocity_mps')
    assert result['training_seed_count'] == 3
    assert result['mean_difference'] == pytest.approx(-.1)
    with pytest.raises(ValueError, match='paired|missing'):
        s.paired_training_effect(rows[:-1], 'flat', 'learned_learned', 'clock_clock',
                                 'mae_forward_velocity_mps')


def test_resume_rejects_wrong_encoder_and_preserves_files(tmp_path):
    s = study()
    good = dict(completed_updates=51, ppo_seed=s.TRAIN_SEEDS[0], terrain='flat',
                arm='learned', encoder_sha256=s.RUNTIME_SHA)
    torch.save(dict(iter=50, infos={'adaptation_state':good}), tmp_path/'model_50.pt')
    torch.save(dict(iter=100, infos={'adaptation_state':dict(good, completed_updates=101,
                                                          encoder_sha256='0'*64)}), tmp_path/'model_100.pt')
    path, rejected = s.choose_resume_checkpoint(tmp_path, s.TRAIN_SEEDS[0], 'flat', 'learned')
    assert path.name == 'model_50.pt' and len(rejected) == 1
    assert (tmp_path/'model_100.pt').exists()


def test_only_successful_finalization_and_all_pairs_can_complete():
    m = importlib.import_module('scripts.run_g1_stateful_flat_ppo_20261025')
    assert callable(m.queue) and callable(m.freeze_policies)
    assert callable(m.summarize)
    assert m.classify_effects([dict(paired_t_95pct_ci=[-.2,-.1],holm_adjusted_p=.01),
                               dict(paired_t_95pct_ci=[0.,0.],holm_adjusted_p=1.)])
    assert not m.classify_effects([dict(paired_t_95pct_ci=[-.2,.1],holm_adjusted_p=.2),
                                   dict(paired_t_95pct_ci=[0.,0.],holm_adjusted_p=1.)])
