"""Train the 285-D terrain-scan comparison at a useful screening budget.

The short v4 pilot used 300 updates and all four arms.  This runner freezes the
direct comparison to the two relevant arms (parameter-matched MLP and semantic
480/32 split), pairs three seeds, and keeps the results in a new output tree.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts import g1_extero_pilot as pilot
from scripts import g1_semantic_split_study as base
from scripts import run_g1_semantic_split as runner

ROOT = Path(__file__).resolve().parents[1]
LONG_OUTPUT = ROOT / 'results/g1_extero_long_20261105'
LONG_ARMS = ('matched', 'semantic')
LONG_SEEDS = (20261111, 20261112, 20261113)
LONG_ITERATIONS = 3000
LONG_NUM_ENVS = 512


def long_protocol(*, iterations=LONG_ITERATIONS, seeds=LONG_SEEDS, num_envs=LONG_NUM_ENVS):
    protocol = copy.deepcopy(pilot.default_protocol())
    protocol.update(
        schema='g1.extero_split.long_screening.v1',
        question='At a useful training budget, does semantic terrain-scan splitting beat a parameter-matched MLP?',
        arms=list(LONG_ARMS),
        status='LONG_SCREENING_FROZEN',
    )
    protocol['training'].update(
        pilot_iterations=int(iterations),
        formal_iterations=int(iterations),
        seeds=list(seeds),
        num_envs=int(num_envs),
        save_interval=100,
    )
    protocol['long_screening'] = {
        'iterations': int(iterations),
        'num_envs': int(num_envs),
        'arms': list(LONG_ARMS),
        'seeds': list(seeds),
        'purpose': 'distinguish insufficient training from a persistent semantic-split disadvantage',
    }
    return protocol


def selected_jobs(protocol):
    return [
        {'arm': arm, 'seed': seed, 'stage': 'formal'}
        for seed in protocol['training']['seeds']
        for arm in LONG_ARMS
    ]


def policy_directory(output, arm, seed):
    return Path(output) / 'formal' / f'{arm}_seed{int(seed)}'


def source_snapshot():
    files = {}
    for name in (
        'scripts/g1_extero_pilot.py',
        'scripts/run_g1_extero_pilot.py',
        'scripts/run_g1_extero_long.py',
        'scripts/g1_semantic_split_study.py',
        'scripts/run_g1_semantic_split.py',
    ):
        files[name] = base.sha(ROOT / name)
    return {'files': files, 'sha256': base.protocol_digest(files)}


PROTOCOL = long_protocol()


def training_jobs(protocol, stage):
    if stage == 'formal':
        return selected_jobs(protocol)
    if stage == 'pilot':
        return []
    raise ValueError('stage must be pilot or formal')


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = source_snapshot()
    values = {
        'protocol.json': PROTOCOL,
        'source_manifest.json': source,
        'observation_layout.json': pilot.observation_layout(pilot.EXPECTED_TERMS),
        'training_matrix.json': {'formal': selected_jobs(PROTOCOL)},
    }
    for name, value in values.items():
        path = output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'frozen long-screening file changed: {path}')
        if not path.exists():
            base.write(path, value)
    return PROTOCOL


def install():
    runner.prepare = prepare
    runner.source_snapshot = source_snapshot
    base.make_env_config = pilot.make_env_config
    base.runtime_layout = pilot.runtime_layout
    base.actor_config = pilot.actor_config
    base.default_protocol = lambda: copy.deepcopy(PROTOCOL)
    # runner.train consults the shared study module for the frozen job matrix.
    runner.study.training_jobs = training_jobs


def run_training(output=LONG_OUTPUT, *, device='cuda:0', resume=True):
    install()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for job in selected_jobs(PROTOCOL):
        record = runner.train(output, **job, device=device, resume=resume)
        records.append({
            'arm': job['arm'],
            'seed': job['seed'],
            'status': record.get('status'),
            'completed_updates': record.get('completed_updates'),
            'completed_transitions': record.get('completed_transitions'),
            'latest_checkpoint': record.get('latest_checkpoint'),
        })
        base.write(output / 'long_progress.json', {
            'status': 'RUNNING', 'records': records,
            'completed_jobs': len(records), 'total_jobs': len(selected_jobs(PROTOCOL)),
        })
    base.write(output / 'long_progress.json', {
        'status': 'TRAINING_COMPLETED', 'records': records,
        'completed_jobs': len(records), 'total_jobs': len(selected_jobs(PROTOCOL)),
    })
    return records


def judge_result(result, *, min_mean_improvement=0.05, max_survival_drop=0.02):
    """Return a conservative post-evaluation action.

    ``result['paired']`` contains one row per training seed.  Shutdown requires
    all paired seeds to improve in RMSE, a mean improvement of at least 5%, and
    no material survival regression.  Anything else remains actionable and is
    sent to the next-experiment branch by the remote controller.
    """
    paired = list(result.get('paired', []))
    if not paired:
        return {'action': 'NEXT_EXPERIMENT', 'reason': 'no paired evaluations'}
    improvements = [
        (row['matched_rmse'] - row['semantic_rmse']) / row['matched_rmse']
        for row in paired
    ]
    survival_drops = [row['semantic_survival'] - row['matched_survival'] for row in paired]
    good = (
        len(paired) >= len(LONG_SEEDS)
        and all(value > 0.0 for value in improvements)
        and sum(improvements) / len(improvements) >= min_mean_improvement
        and min(survival_drops) >= -max_survival_drop
    )
    return {
        'action': 'SHUTDOWN' if good else 'NEXT_EXPERIMENT',
        'paired_n': len(paired),
        'rmse_improvements': improvements,
        'mean_rmse_improvement': sum(improvements) / len(improvements),
        'survival_drops': survival_drops,
        'thresholds': {
            'min_mean_improvement': min_mean_improvement,
            'max_survival_drop': max_survival_drop,
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=LONG_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--iterations', type=int, default=LONG_ITERATIONS)
    args = parser.parse_args()
    PROTOCOL = long_protocol(iterations=args.iterations)
    print(json.dumps(run_training(args.output, device=args.device, resume=not args.no_resume), ensure_ascii=False))
