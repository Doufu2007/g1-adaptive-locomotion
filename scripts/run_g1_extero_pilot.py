"""Run the short 285-D terrain-scan pilot in a separate output directory."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from scripts import g1_extero_pilot as pilot
from scripts import g1_semantic_split_study as base
from scripts import run_g1_semantic_split as runner

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/g1_extero_pilot_20261105_v4'
SEED = pilot.default_protocol()['training']['pilot_seed']


def source_snapshot():
    files = {}
    for name in ('scripts/g1_extero_pilot.py', 'scripts/run_g1_extero_pilot.py',
                 'scripts/g1_semantic_split_study.py', 'scripts/run_g1_semantic_split.py'):
        path = ROOT / name
        files[name] = base.sha(path)
    return {'files': files, 'sha256': base.protocol_digest(files)}


PROTOCOL = pilot.default_protocol()


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = source_snapshot()
    for path, value in (
        (output / 'protocol.json', PROTOCOL),
        (output / 'source_manifest.json', source),
        (output / 'observation_layout.json', pilot.observation_layout(pilot.EXPECTED_TERMS)),
        (output / 'training_matrix.json', {s: base.training_jobs(PROTOCOL, s) for s in ('pilot', 'formal')}),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'frozen pilot file changed: {path}')
        if not path.exists():
            base.write(path, value)
    return PROTOCOL


def install():
    # runner.train keeps its execution and checkpoint machinery, while the
    # patched study functions provide the new observation and actor contract.
    runner.prepare = prepare
    runner.source_snapshot = source_snapshot
    base.make_env_config = pilot.make_env_config
    base.runtime_layout = pilot.runtime_layout
    base.actor_config = pilot.actor_config
    base.default_protocol = lambda: copy.deepcopy(PROTOCOL)


def main():
    install()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in pilot.ARMS:
        record = runner.train(OUTPUT, arm, SEED, 'pilot', device='cuda:0', resume=True)
        records.append({'arm': arm, 'seed': SEED, 'status': record.get('status'),
                        'completed_updates': record.get('completed_updates'),
                        'latest_checkpoint': record.get('latest_checkpoint')})
        base.write(OUTPUT / 'pilot_progress.json', {'status': 'RUNNING', 'records': records})
    base.write(OUTPUT / 'pilot_progress.json', {'status': 'COMPLETED', 'records': records})
    print(json.dumps({'status': 'COMPLETED', 'output': str(OUTPUT), 'records': records}, ensure_ascii=False))


if __name__ == '__main__':
    main()
