"""Evaluate the long 285-D semantic-vs-matched screening runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import evaluate_g1_extero_pilot as short_eval
from scripts import evaluate_g1_semantic_split as base_eval
from scripts import g1_extero_pilot as pilot
from scripts import g1_semantic_split_study as study
from scripts import run_g1_extero_long as long


def run(output, *, device='cuda:0', episode_seed=20262601, latency_samples=2000):
    output = Path(output)
    protocol = long.long_protocol()
    protocol['evaluation']['development_seeds'] = [episode_seed]
    protocol['evaluation']['holdout_seeds'] = [episode_seed]
    study.make_env_config = pilot.make_env_config
    study.runtime_layout = pilot.runtime_layout
    base_eval.observation_noise = short_eval.observation_noise
    study.bootstrap()

    result_root = output / 'holdout'
    result_root.mkdir(parents=True, exist_ok=True)
    study.write(result_root / 'protocol.json', protocol)
    records = []
    all_rows = []
    from mjlab.envs import ManagerBasedRlEnv

    count = len(protocol['evaluation']['commands']) * len(protocol['evaluation']['scenarios'])
    env_seed = 20262600
    for seed in long.LONG_SEEDS:
        for arm in long.LONG_ARMS:
            policy_path = long.policy_directory(output, arm, seed) / 'policy.onnx'
            if not policy_path.is_file():
                raise FileNotFoundError(policy_path)
            target = result_root / f'{arm}_seed{seed}'
            identity = {
                'arm': arm,
                'training_seed': seed,
                'split': 'holdout',
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
                row.update(arm=arm, training_seed=seed)
            all_rows.extend(rows)
            record = {
                'arm': arm,
                'training_seed': seed,
                'episode_seed': episode_seed,
                'scope': 'long-screening holdout; paired training seeds',
                'policy_sha256': identity['policy_sha256'],
                'observation_dimension': pilot.TOTAL_DIM,
                'episode_count': len(rows),
                'metrics_by_scenario': short_eval.aggregate(rows),
            }
            study.write(target / 'summary.json', record)
            records.append(record)

    latency = []
    for seed in long.LONG_SEEDS:
        for arm in long.LONG_ARMS:
            policy_path = long.policy_directory(output, arm, seed) / 'policy.onnx'
            latency.append({
                'arm': arm,
                'training_seed': seed,
                **short_eval.benchmark_policy(policy_path, protocol, samples=latency_samples),
            })

    paired = []
    for seed in long.LONG_SEEDS:
        by_arm = {row['arm']: row for row in records if row['training_seed'] == seed}
        semantic = by_arm['semantic']['metrics_by_scenario']['clean']
        matched = by_arm['matched']['metrics_by_scenario']['clean']
        paired.append({
            'seed': seed,
            'semantic_rmse': semantic['planar_rmse_mps'],
            'matched_rmse': matched['planar_rmse_mps'],
            'semantic_survival': semantic['survival_fraction'],
            'matched_survival': matched['survival_fraction'],
        })
    decision = long.judge_result({'paired': paired})
    study.write(result_root / 'decision.json', decision)
    study.write(result_root / 'latency.json', {'status': 'COMPLETED', 'results': latency})
    summary = {
        'status': 'COMPLETED',
        'scope': 'long-screening holdout; three paired training seeds',
        'observation_dimension': pilot.TOTAL_DIM,
        'episode_seed': episode_seed,
        'paired': paired,
        'decision': decision,
        'records': records,
        'latency': latency,
        'rows': all_rows,
    }
    study.write(result_root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--episode-seed', type=int, default=20262601)
    parser.add_argument('--latency-samples', type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.output, device=args.device, episode_seed=args.episode_seed,
                         latency_samples=args.latency_samples), ensure_ascii=False))
