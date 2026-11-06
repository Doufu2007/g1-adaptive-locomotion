"""Wait for every pre-registered P/M run to reach a terminal manifest, then run
the CPU analysis -- but only if nothing failed and nothing is missing.

Refusing to analyse a partial grid is deliberate: the analysis script overwrites
analysis.json in place, and a partial grid would silently publish a curve with
``complete_grid: false`` as if it were the pre-registered result.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/g1_period_sweep_cpu_20261028'
PY = r'<data-root>/Program Files\Python313\python.exe'

M_MAPS = range(20269101, 20269111)
P_MAPS = (20264701, 20264702, 20264703)
PERIODS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)

EXPECTED = [('M', f'RE_live_terrain{t}') for t in M_MAPS]
EXPECTED += [('M', f'RC_clock_terrain{t}_p0.60') for t in M_MAPS]
EXPECTED += [('P', f'RC_clock_terrain{t}_p{p:.2f}') for p in PERIODS for t in P_MAPS]


def status(tag, name):
    path = OUT / tag / name / 'manifest.json'
    if not path.exists():
        return 'MISSING'
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status', 'UNKNOWN')
    except Exception:
        return 'UNREADABLE'


def main():
    while True:
        states = {name: status(tag, name) for tag, name in EXPECTED}
        pending = {k: v for k, v in states.items() if v not in ('COMPLETED', 'FAILED')}
        failed = {k: v for k, v in states.items() if v == 'FAILED'}
        if failed:
            (OUT / 'sweep_status_20261029.json').write_text(json.dumps(
                dict(state='FAILED', failed=sorted(failed)), indent=1), encoding='utf-8')
            print('FAILED jobs:', sorted(failed), flush=True)
            return 1
        if not pending:
            break
        time.sleep(120)

    record = dict(state='COMPLETE', jobs=len(EXPECTED))
    (OUT / 'sweep_status_20261029.json').write_text(
        json.dumps(record, indent=1), encoding='utf-8')
    code_a2 = subprocess.call([PY, '-X', 'utf8',
                               str(ROOT / 'scripts/g1_period_sweep_analysis_cpu_20261028.py')],
                              cwd=str(ROOT))
    print('analysis P exit', code_a2, flush=True)
    code_m = subprocess.call([PY, '-X', 'utf8',
                              str(ROOT / 'scripts/g1_period_sweep_analysis_cpu_m_20261029.py')],
                             cwd=str(ROOT))
    print('analysis M exit', code_m, flush=True)
    return code_a2 or code_m


if __name__ == '__main__':
    sys.exit(main())
