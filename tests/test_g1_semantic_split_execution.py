import copy
import json
from pathlib import Path

import pytest

from scripts import g1_semantic_split_study as study
from scripts import run_g1_semantic_split as runner


def test_prepare_seals_protocol_sources_and_observation_layout(tmp_path):
    p = runner.prepare(tmp_path)
    assert p['task'] == 'Unitree-G1-Flat'
    assert json.loads((tmp_path/'observation_layout.json').read_text())['condition_indices'] == [6, 7, 8, 9, 10]
    assert len(json.loads((tmp_path/'training_matrix.json').read_text())['formal']) == 12
    assert runner.prepare(tmp_path) == p
    changed = json.loads((tmp_path/'protocol.json').read_text())
    changed['training']['formal_iterations'] = 1
    study.write(tmp_path/'protocol.json', changed)
    with pytest.raises(ValueError, match='protocol'):
        runner.prepare(tmp_path)


def test_training_identity_rejects_wrong_arm_seed_budget_and_source():
    identity = {'arm':'semantic','seed':123,'stage':'formal','iterations':10001,
                'num_envs':4096,'protocol_sha256':'abc','source_sha256':'def'}
    record = {'identity':identity}
    runner.check_identity(record, identity)
    for key in identity:
        wrong = dict(identity, **{key: 'wrong'})
        with pytest.raises(ValueError):
            runner.check_identity(record, wrong)


def test_seal_refuses_a_completed_label_without_checkpoint_or_export(tmp_path):
    p = runner.prepare(tmp_path)
    for job in study.training_jobs(p, 'formal'):
        study.write(runner.run_directory(tmp_path, **job)/'manifest.json', {'status':'COMPLETED'})
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        runner.seal_formal(tmp_path)


def test_gpu_admission_never_kills_or_hides_other_jobs():
    assert runner.admission_decision([{'pid':123}], free_mib=30000, required_mib=22000) == 'WAITING_FOR_GPU'
    assert runner.admission_decision([], free_mib=5000, required_mib=22000) == 'INSUFFICIENT_GPU_MEMORY'
    assert runner.admission_decision([], free_mib=30000, required_mib=22000) == 'READY'


def test_evaluation_coverage_catches_missing_and_duplicate_episodes():
    from scripts.evaluate_g1_semantic_split import validate_coverage
    p = study.default_protocol()
    expected = len(p['evaluation']['commands']) * len(p['evaluation']['scenarios']) * len(p['evaluation']['development_seeds'])
    rows = [dict(episode_seed=s, command_id=c, scenario=k)
            for s in p['evaluation']['development_seeds']
            for c in range(len(p['evaluation']['commands']))
            for k in p['evaluation']['scenarios']]
    assert validate_coverage(rows, p, 'development') == expected
    with pytest.raises(ValueError):
        validate_coverage(rows[:-1], p, 'development')
    with pytest.raises(ValueError):
        validate_coverage(rows+[rows[0]], p, 'development')


def test_shared_physical_configuration_retains_phase_reward_and_normalizer():
    study.bootstrap()
    p = study.default_protocol()
    train = study.make_env_config(p, 7, 4)
    evaluate = study.make_env_config(p, 8, 42, evaluation=True)
    assert 'foot_gait' in train.rewards
    assert train.observations['actor'].terms['phase'].func.__name__ == 'phase'
    assert 'command_vel' not in train.curriculum
    assert not evaluate.observations['actor'].enable_corruption
    assert 'push_robot' not in evaluate.events
    assert 'push_robot' in train.events
    from dataclasses import asdict
    # Real mjlab config contains enum values; their stable identity must survive
    # JSON serialization instead of dropping the physical configuration.
    encoded = study.canonical(asdict(train))
    assert study.protocol_digest(encoded) == study.protocol_digest(json.loads(json.dumps(encoded)))


def test_queue_resume_starts_new_jobs_and_validates_existing_checkpoints(tmp_path):
    import torch
    identity = {'arm': 'semantic', 'iterations': 1000}
    # The queue's global resume option must also work for its next, new job.
    assert runner.recovery_for_run(tmp_path, identity, requested=True) is None
    study.write(tmp_path/'manifest.json', {'identity': identity, 'status': 'FAILED'})
    with pytest.raises(FileExistsError):
        runner.recovery_for_run(tmp_path, identity, requested=False)
    torch.save({'iter': 99, 'infos': {'split_study': {
        'identity': identity, 'completed_updates': 100}}}, tmp_path/'model_99.pt')
    checkpoint, state, rejected = runner.recovery_for_run(tmp_path, identity, requested=True)
    assert checkpoint.name == 'model_99.pt' and state['completed_updates'] == 100
    assert rejected == []
    with pytest.raises(ValueError, match='identity'):
        runner.recovery_for_run(tmp_path, dict(identity, arm='random'), requested=True)


def test_body_lateral_impulse_rotates_with_robot_heading():
    import torch
    from scripts.evaluate_g1_semantic_split import body_lateral_velocity_delta
    # wxyz quaternion: yaw 90 degrees turns body +y into world -x.
    q = torch.tensor([[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5]])
    delta = body_lateral_velocity_delta(q, .6)
    torch.testing.assert_close(delta, torch.tensor([[0., .6, 0.], [-.6, 0., 0.]]), atol=1e-6, rtol=1e-6)


def test_cached_evaluation_recomputes_metrics_and_binds_identity(tmp_path):
    import numpy as np
    from scripts.evaluate_g1_semantic_split import load_batch
    p = copy.deepcopy(study.default_protocol())
    p['evaluation'].update(commands=[[.5, 0., 0.]], scenarios={'clean': {}}, horizon_seconds=.04)
    identity = {'arm': 'semantic', 'policy_sha256': 'one'}
    path = tmp_path/'episode_seed9.npz'
    commands = np.broadcast_to(np.array([.5, 0., 0.], dtype=np.float32), (2, 1, 3))
    np.savez_compressed(path, commands=commands, velocities=np.zeros((2, 1, 3)),
                        actions=np.zeros((2, 1, 29)), terminated=np.zeros((2, 1), bool),
                        alive_steps=np.array([2]), dt=.02,
                        identity_json=json.dumps(dict(identity, episode_seed=9), sort_keys=True))
    row = dict(episode_seed=9, command_id=0, scenario='clean', command=[.5, 0., 0.], slot=0,
               **study.episode_metrics(commands[:, 0], np.zeros((2, 3)), alive_steps=2, fell=False, dt=.02))
    record = dict(identity=dict(identity, episode_seed=9), rows=[row], trajectory_file=path.name,
                  trajectory_sha256=study.sha(path), episode_seed=9, dt=.02, horizon_steps=2)
    batch_path = tmp_path/'episode_seed9.json'
    study.write(batch_path, record)
    assert load_batch(tmp_path, 9, p, identity)['rows'] == [row]
    record['rows'][0]['planar_rmse_mps'] = 0.
    study.write(batch_path, record)
    with pytest.raises(ValueError, match='metrics'):
        load_batch(tmp_path, 9, p, identity)
    record['rows'][0] = row  # remains altered: identity failure must occur first
    with pytest.raises(ValueError, match='identity'):
        load_batch(tmp_path, 9, p, dict(identity, arm='random'))
