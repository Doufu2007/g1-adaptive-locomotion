"""Exploratory rollout and CPU latency evaluation for the 285-D terrain pilot.

This evaluator intentionally does not create a formal-policy seal.  The pilot
has one training seed and a shortened budget, so its results are exploratory
evidence for deciding whether the full study is worth running.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from scripts import evaluate_g1_semantic_split as base_eval
from scripts import g1_extero_pilot as pilot
from scripts import g1_semantic_split_study as study


def observation_noise(layout, scenario, protocol, rng, n):
    """Sample state noise while preserving the 285-D actor observation width."""
    widths = np.zeros(pilot.TOTAL_DIM, np.float32)
    for name, value in protocol['evaluation']['noise_uniform_half_width'].items():
        field = layout['fields'][name]
        widths[field['start']:field['stop']] = value
    return rng.uniform(-1., 1., size=(n, pilot.TOTAL_DIM)).astype(np.float32) * widths * scenario.get(
        'state_noise_scale', 0.)


def benchmark_policy(path, protocol, *, samples=2000):
    """Measure batch-1 CPU ONNX latency for the 285-D actor."""
    import psutil

    session = base_eval.cpu_session(path)
    x = np.random.default_rng(20261105).normal(size=(1, pilot.TOTAL_DIM)).astype(np.float32)
    warmup = int(protocol['latency']['warmup'])
    for _ in range(warmup):
        session.run(None, {'obs': x})
    load = psutil.cpu_percent(interval=.1)
    measurements = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        session.run(None, {'obs': x})
        measurements.append((time.perf_counter_ns() - started) / 1000.)
    return {
        'latency_us': {f'p{q}': float(np.percentile(measurements, q))
                       for q in protocol['latency']['quantiles']},
        'raw_latency_us': measurements,
        'samples': samples,
        'warmup': warmup,
        'input_dimension': pilot.TOTAL_DIM,
        'cpu': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'logical_cpus': psutil.cpu_count(),
        'system_load_percent_before': load,
        'backend': protocol['latency']['backend'],
        'threads': protocol['latency']['threads'],
        'batch': 1,
        'scope': protocol['latency']['scope'],
        'policy_sha256': study.sha(path),
    }


def aggregate(rows):
    """Aggregate episode rows without treating one pilot seed as replication."""
    from collections import defaultdict

    groups = defaultdict(list)
    for row in rows:
        groups[row['scenario']].append(row)
    result = {}
    for scenario, values in sorted(groups.items()):
        result[scenario] = {
            'episodes': len(values),
            'completion': float(np.mean([r['completion'] for r in values])),
            'failure_rate': float(np.mean([r['fell'] for r in values])),
            'planar_rmse_mps': float(np.mean([r['planar_rmse_mps'] for r in values])),
            'forward_rmse_mps': float(np.mean([r['forward_rmse_mps'] for r in values])),
            'lateral_rmse_mps': float(np.mean([r['lateral_rmse_mps'] for r in values])),
            'yaw_rmse_radps': float(np.mean([r['yaw_rmse_radps'] for r in values])),
            'survival_fraction': float(np.mean([r['survival_fraction'] for r in values])),
        }
    return result


def run(output, *, device='cuda:0', episode_seed=None, latency_samples=2000):
    output = Path(output)
    protocol = pilot.default_protocol()
    if episode_seed is None:
        episode_seed = protocol['evaluation']['development_seeds'][0]
    protocol['evaluation']['development_seeds'] = [episode_seed]
    protocol['evaluation']['holdout_seeds'] = [episode_seed]

    # Reuse the proven paired rollout implementation, replacing only the
    # observation-width-specific noise sampler and environment contract.
    study.make_env_config = pilot.make_env_config
    study.runtime_layout = pilot.runtime_layout
    base_eval.observation_noise = observation_noise
    study.bootstrap()

    policy_root = output.parent
    result_root = output
    result_root.mkdir(parents=True, exist_ok=True)
    study.write(result_root / 'protocol.json', protocol)
    records = []
    all_rows = []
    from mjlab.envs import ManagerBasedRlEnv

    count = len(protocol['evaluation']['commands']) * len(protocol['evaluation']['scenarios'])
    env_seed = 20262500
    for arm in pilot.ARMS:
        policy_path = policy_root / 'pilot' / f'{arm}_seed{protocol["training"]["pilot_seed"]}' / 'policy.onnx'
        if not policy_path.is_file():
            raise FileNotFoundError(policy_path)
        target = result_root / arm
        identity = {
            'arm': arm,
            'training_seed': protocol['training']['pilot_seed'],
            'split': 'exploratory_development',
            'episode_seed': episode_seed,
            'policy_sha256': study.sha(policy_path),
            'protocol_sha256': study.protocol_digest(protocol),
        }
        cfg = pilot.make_env_config(protocol, env_seed, count, evaluation=True)
        raw = ManagerBasedRlEnv(cfg=cfg, device=device)
        try:
            layout = pilot.runtime_layout(raw)
            session = base_eval.cpu_session(policy_path)
            rows = base_eval.rollout_seed(raw, session, layout, protocol, episode_seed,
                                          device, target, identity)
        finally:
            raw.close()
        for row in rows:
            row.update(arm=arm, training_seed=protocol['training']['pilot_seed'])
        all_rows.extend(rows)
        record = {
            'arm': arm,
            'training_seed': protocol['training']['pilot_seed'],
            'episode_seed': episode_seed,
            'scope': 'exploratory development rollout; one pilot seed; not formal holdout',
            'policy_sha256': identity['policy_sha256'],
            'observation_dimension': pilot.TOTAL_DIM,
            'episode_count': len(rows),
            'metrics_by_scenario': aggregate(rows),
        }
        study.write(target / 'summary.json', record)
        records.append(record)

    latency = []
    for arm in pilot.ARMS:
        policy_path = policy_root / 'pilot' / f'{arm}_seed{protocol["training"]["pilot_seed"]}' / 'policy.onnx'
        latency.append({'arm': arm, **benchmark_policy(policy_path, protocol, samples=latency_samples)})
    study.write(result_root / 'latency.json', {'status': 'COMPLETED', 'results': latency})
    study.write(result_root / 'summary.json', {
        'status': 'COMPLETED',
        'scope': 'exploratory development rollout; one training seed',
        'observation_dimension': pilot.TOTAL_DIM,
        'episode_seed': episode_seed,
        'rows': all_rows,
        'arms': records,
        'latency': latency,
    })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--episode-seed', type=int, default=None)
    parser.add_argument('--latency-samples', type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.output, device=args.device, episode_seed=args.episode_seed,
                         latency_samples=args.latency_samples), ensure_ascii=False))


if __name__ == '__main__':
    main()
