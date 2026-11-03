"""Post-result CPU audit of immutable best-E PPO checkpoints and event logs."""
from __future__ import annotations

import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_training_mechanism_audit_20261026'
SNAPSHOT = ROOT / 'results/g1_stateful_flat_ppo_best_e_20261026/remote_download/snapshot'
TRAIN = SNAPSHOT / 'results/g1_stateful_flat_ppo_best_e_20261026/training'
FEEDBACK = ROOT / 'results/g1_phase_feedback_probe_20261026'
SEEDS = (20261901, 20261902, 20261903)


def longpath(path):
    path = Path(path)
    text = str(path.absolute())
    return path if text.startswith('\\\\?\\') else Path('\\\\?\\' + text)


def read(path):
    return json.loads(longpath(path).read_text(encoding='utf-8'))


def sha(path):
    with longpath(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    longpath(OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                             allow_nan=False) + '\n', encoding='utf-8')


def forward(state, obs):
    x = (obs - state['obs_normalizer._mean']) / (state['obs_normalizer._std'] + .01)
    for index in (0, 2, 4, 6):
        x = F.linear(x, state[f'mlp.{index}.weight'], state[f'mlp.{index}.bias'])
        if index != 6:
            x = F.elu(x)
    return x


def pool():
    analysis = read(FEEDBACK / 'analysis.json')
    observations, sources = [], []
    for row in analysis['episodes']:
        if row['scenario'] != 'steady' or row['condition'] not in ('learned_live', 'clock_clock'):
            continue
        path = ROOT / row['raw_path']
        assert sha(path) == row['raw_sha256']
        with np.load(longpath(path), allow_pickle=False) as raw:
            observations.append(raw['observation'][150::10].copy())
        sources.append(dict(path=str(path), sha256=row['raw_sha256'],
                            condition=row['condition'], train_seed=row['train_seed']))
    result = torch.from_numpy(np.concatenate(observations))
    assert result.shape == (1530, 98)
    return result, sources


def scalar_windows(accumulator):
    windows = ((0, 10), (10, 50), (50, 100), (100, 200), (200, 251), (250, 350), (350, 450), (450, 501))
    result, all_scalars = {}, {}
    for tag in accumulator.Tags()['scalars']:
        values = accumulator.Scalars(tag)
        all_scalars[tag] = [dict(step=v.step, value=v.value) for v in values]
        result[tag] = {f'{a}_{b}': float(np.mean([v.value for v in values if a <= v.step < b]))
                       for a, b in windows if any(a <= v.step < b for v in values)}
    rates = accumulator.Scalars('Loss/learning_rate')
    return result, all_scalars, dict(
        first_at_floor=next((v.step for v in rates if v.value <= 1.00001e-5), None),
        fraction_at_floor=float(np.mean([v.value <= 1.00001e-5 for v in rates])),
        first_rate=rates[0].value, last_rate=rates[-1].value,
        min_rate=min(v.value for v in rates), max_rate=max(v.value for v in rates))


def run():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    source = ROOT / 'logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt'
    source_sha = sha(source)
    original = torch.load(longpath(source), map_location='cpu', weights_only=False)['actor_state_dict']
    obs, sources = pool()
    baseline_output = forward(original, obs)
    export = read(ROOT / 'results/g1_stateful_flat_ppo_best_e_20261026/remote_download/export_manifest.json')
    expected = {r['path']: r['sha256'] for r in export['files']}
    rows, event_rows, parity = [], [], []
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for seed in SEEDS:
        for arm in ('clock', 'learned'):
            folder = TRAIN / f'flat_{arm}_seed{seed}'
            manifest = read(folder / 'manifest.json')
            assert manifest['source_actor_sha256'] == source_sha
            event_path = next(longpath(folder).glob('events*'))
            key = event_path.as_posix().split('/snapshot/', 1)[-1]
            assert sha(event_path) == expected[key]
            acc = EventAccumulator(str(event_path), size_guidance={'scalars': 0}).Reload()
            windows, scalars, rates = scalar_windows(acc)
            event_rows.append(dict(train_seed=seed, arm=arm, source=str(event_path), sha256=sha(event_path),
                                   windows=windows, scalars=scalars, learning_rate=rates))
            for step in range(0, 501, 50):
                checkpoint = folder / f'model_{step}.pt'
                key = checkpoint.relative_to(SNAPSHOT).as_posix()
                digest = sha(checkpoint)
                assert digest == expected[key]
                saved = torch.load(longpath(checkpoint), map_location='cpu', weights_only=False)
                state, critic = saved['actor_state_dict'], saved['critic_state_dict']
                actions = forward(state, obs)
                sigma = state['distribution.std_param']
                mean = state['obs_normalizer._mean']
                std = state['obs_normalizer._std']
                normalized = (obs - mean) / (std + .01)
                shifted = obs.clone()
                c, s = np.cos(np.pi / 12), np.sin(np.pi / 12)
                shifted[:, 9] = obs[:, 9] * c + obs[:, 10] * s
                shifted[:, 10] = obs[:, 10] * c - obs[:, 9] * s
                phase_effect = forward(state, shifted) - actions
                commanded = obs.clone()
                commanded[:, 6] += .1
                command_effect = forward(state, commanded) - actions
                weights = [k for k in state if k.startswith('mlp.')]
                drift = torch.sqrt(sum(((state[k] - original[k]) ** 2).sum() for k in weights)
                                   / sum((original[k] ** 2).sum() for k in weights))
                count = int(state['obs_normalizer.count'])
                learning_rate = float(saved['infos']['adaptation_state']['learning_rate'])
                assert all(abs(g['lr'] - learning_rate) < 1e-12 for g in saved['optimizer_state_dict']['param_groups'])
                row = dict(train_seed=seed, arm=arm, iteration=step, updates=step+1,
                    checkpoint=str(checkpoint), checkpoint_sha256=digest, learning_rate=learning_rate,
                    actor_normalizer_count=count, critic_normalizer_count=int(critic['obs_normalizer.count']),
                    new_actor_normalizer_mass_fraction=(count-int(original['obs_normalizer.count']))/count,
                    actor_normalizer_max_mean_change=float((mean-original['obs_normalizer._mean']).abs().max()),
                    actor_phase_mean=mean[0,9:11].tolist(), actor_phase_std=std[0,9:11].tolist(),
                    relative_network_weight_drift=float(drift),
                    action_rms_change_from_source=float(torch.sqrt(torch.mean((actions-baseline_output)**2))),
                    action_across_pool_std_rms=float(torch.sqrt(actions.var(dim=0, unbiased=False).mean())),
                    conditional_phase15deg_action_rms=float(torch.sqrt(torch.mean(phase_effect**2))),
                    conditional_command_plus0_1_action_rms=float(torch.sqrt(torch.mean(command_effect**2))),
                    gaussian_sigma_mean=float(sigma.mean()), gaussian_sigma_min=float(sigma.min()),
                    gaussian_sigma_max=float(sigma.max()), gaussian_sigma_by_joint=sigma.tolist(),
                    normalized_observations_abs_max=float(normalized.abs().max()),
                    normalized_observations_fraction_abs_gt5=float((normalized.abs()>5).float().mean()),
                    phase_normalized_rms=float(torch.sqrt(torch.mean(normalized[:,9:11]**2))),
                    hip_pitch_action_pool_std=float(actions[:,[0,6]].std(dim=0,unbiased=False).mean()),
                    knee_action_pool_std=float(actions[:,[3,9]].std(dim=0,unbiased=False).mean()))
                rows.append(row)
                if step == 500:
                    session = ort.InferenceSession(str(longpath(folder/'policy.onnx')), sess_options=options,
                                                  providers=['CPUExecutionProvider'])
                    selected = obs[::100].numpy()
                    actual = np.concatenate([session.run(None, {'obs': x[None]})[0] for x in selected])
                    reference = forward(state, torch.from_numpy(selected)).numpy()
                    np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-5)
                    parity.append(dict(train_seed=seed,arm=arm,max_abs_error=float(np.max(np.abs(actual-reference)))))
            print(json.dumps(dict(completed=arm, seed=seed, checkpoint_rows=len(rows))), flush=True)
    result = dict(status='COMPLETED', at=datetime.now(timezone.utc).isoformat(), classification='POST_RESULTS_EXPLORATORY',
        script_sha256=sha(__file__), source_actor_sha256=source_sha,
        source_normalizer_count=int(original['obs_normalizer.count']),
        source_gaussian_sigma_mean=float(original['distribution.std_param'].mean()),
        checkpoint_count=len(rows), original_PPO_training_runs=6, new_PPO_training_runs=0,
        observation_pool_shape=list(obs.shape), observation_pool_sources=sources,
        scope='Same-state offline action responses do not establish closed-loop improvement or a causal training mechanism.',
        checkpoints=rows, events=event_rows, manual_forward_onnx_parity=parity)
    write('training_audit.json', result)
    print(json.dumps(dict(status='COMPLETED', checkpoints=len(rows),
        source_count=result['source_normalizer_count'],
        learning_rates=[dict(seed=r['train_seed'],arm=r['arm'],**r['learning_rate']) for r in event_rows],
        final=[r for r in rows if r['iteration']==500]), ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    run()
