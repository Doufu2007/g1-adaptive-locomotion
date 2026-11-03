"""Timestamp the pre-registration and record the identity of everything it binds."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_prereg_20261028'
S = ROOT / 'results/g1_rough_domain_adaptation_20261028'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.joinpath('lock.json').exists():
        raise SystemExit('lock.json exists; pre-registration is immutable')
    paths = [
        OUT / 'PREREGISTRATION.md',
        ROOT / 'scripts/g1_period_sweep_20261028.py',
        S / 'map_identities.json',
        S / 'prospective_protocol.json',
        S / 'analysis.json',
        ROOT / 'results/g1_comparison_summary_20261028/all_scored_episodes.csv',
        ROOT / 'results/g1_comparison_summary_20261028/rough_and_flat_method_summary.csv',
    ]
    for seed in (20261901, 20261902, 20261903):
        for method in ('RE', 'RC'):
            paths.append(S / 'training' / f'{method}_{seed}' / 'manifest.json')
    value = dict(
        status='LOCKED_BEFORE_P_AND_M_OUTCOMES_READ',
        written_at=datetime.now(timezone.utc).isoformat(),
        claim='motion-driven phase beats a fixed-frequency clock on rough terrain',
        primary_endpoint='completion-rate difference, paired McNemar, alpha=.05',
        files=[dict(path=str(p), sha256=sha(p)) for p in paths],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'lock.json').write_text(json.dumps(value, indent=1), encoding='utf-8')
    print(json.dumps(dict(files=len(paths), written_at=value['written_at'])))


if __name__ == '__main__':
    main()
