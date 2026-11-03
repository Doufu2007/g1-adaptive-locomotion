"""Exploratory (2026-10-30): the frozen-clock period sweep on the ten held-out M maps.

Why this exists
---------------
The pre-registered study (``results/g1_prereg_20261028``) swept the clock period
only on the three S evaluation maps.  The main endpoint (``analysis_M_cpu.json``)
compared ``RE_live`` against a *single* fixed period, the nominal 0.60 s, on the
ten held-out maps 20269101-20269110.  That leaves one question open:

    does the learned phase also dominate the *best* fixed period on those maps?

Nothing on disk answers it, because no non-nominal period was ever run on the M
maps.  This script runs the six missing periods (0.60 already exists and is
reused verbatim) so the claim can be stated against the whole swept family
instead of against a single nominal clock.

Execution-condition gate
------------------------
The audited runs were executed one process at a time.  This sweep runs four
concurrently to fit the wall clock, which is a *new* execution condition, and
the CPU backend's bit-reproducibility was established only for sequential
execution.  Before the sweep is allowed to continue, ``gate`` re-runs one
already-completed arm (terrain 20269102, period 0.60) under tag ``MREPRO``
*while three real jobs are running*, and demands bit-identical ``raw_sha256``
and identical metrics against the stored M run.  A failure means the parallel
condition perturbs the numerics and the sweep must be redone sequentially.

Nothing here overwrites a completed run: ``g1_period_sweep_20261028_v3.rollout``
refuses an existing COMPLETED folder, and the resumable runner skips it.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts import g1_period_sweep_20261028_v3 as v

PERIODS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
GATE_TAG = 'MREPRO'
GATE_TERRAIN = 20269102
GATE_PERIOD = 0.60
METRIC_EXCLUDE = ('probe', 'raw_path')


def jobs(periods=PERIODS, maps=None):
    periods = tuple(periods or PERIODS)
    maps = tuple(maps or v.NEW_MAPS)
    return [dict(probe='M', method='RC_clock', terrain=t, period=p)
            for p in periods for t in maps]


def completed(item):
    out = v.folder(item['probe'], item['method'], item['terrain'], item['period'])
    manifest = out / 'manifest.json'
    if not manifest.exists():
        return False
    return json.loads(manifest.read_text(encoding='utf-8')).get('status') == 'COMPLETED'


def launch(item, device):
    log = (v.OUT / 'logs' /
           f"{item['probe']}_{item['method']}_t{item['terrain']}_p{item['period']:.2f}.log")
    v.run_child(item, log, device)
    return item


def run_pool(items, device, workers):
    failures = []
    if not items:
        print('nothing to run', flush=True)
        return failures
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, item, device): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
                print(f"done   {item['terrain']} p{item['period']:.2f}", flush=True)
            except BaseException as error:  # noqa: BLE001
                failures.append((item, repr(error)))
                print(f"FAILED {item['terrain']} p{item['period']:.2f}: {error!r}", flush=True)
    return failures


def _episodes(tag, terrain, period):
    path = v.folder(tag, 'RC_clock', terrain, period) / 'episodes.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    key = lambda r: (r['train_seed'], r['eval_seed'], r['scenario'], r['patch_col'])
    return {key(r): r for r in rows}


def gate_check():
    """Demand bit-identical telemetry and identical metrics against the stored run."""
    stored = _episodes('M', GATE_TERRAIN, GATE_PERIOD)
    repro = _episodes(GATE_TAG, GATE_TERRAIN, GATE_PERIOD)
    if set(stored) != set(repro):
        print(f'GATE FAIL: episode keys differ ({len(stored)} vs {len(repro)})')
        return False
    problems = []
    for key in sorted(stored):
        left = {k: x for k, x in stored[key].items() if k not in METRIC_EXCLUDE}
        right = {k: x for k, x in repro[key].items() if k not in METRIC_EXCLUDE}
        if left['raw_sha256'] != right['raw_sha256']:
            problems.append(f'{key}: raw_sha256 {left["raw_sha256"][:12]} != {right["raw_sha256"][:12]}')
            continue
        if left != right:
            differing = sorted(k for k in left if left[k] != right.get(k))
            problems.append(f'{key}: metric mismatch {differing}')
    if problems:
        print('GATE FAIL')
        for line in problems[:10]:
            print('  ' + line)
        return False
    print(f'GATE PASS: {len(stored)} episodes bit-identical under 4-way parallel load')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'gate', 'sweep', 'status'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--periods', nargs='+', type=float, default=None)
    parser.add_argument('--maps', nargs='+', type=int, default=None)
    args = parser.parse_args(argv)

    wanted = jobs(args.periods, args.maps)

    if args.command == 'plan':
        pending = [j for j in wanted if not completed(j)]
        print(json.dumps(dict(total=len(wanted), pending=len(pending),
                              already_completed=len(wanted) - len(pending),
                              periods=sorted({j['period'] for j in wanted}),
                              maps=sorted({j['terrain'] for j in wanted}),
                              workers=args.workers), indent=1))
        return 0

    if args.command == 'status':
        pending = [j for j in wanted if not completed(j)]
        failed = [j for j in wanted
                  if (v.folder(j['probe'], j['method'], j['terrain'], j['period'])
                      / 'manifest.json').exists()
                  and not completed(j)]
        print(json.dumps(dict(total=len(wanted), completed=len(wanted) - len(pending),
                              failed=len(failed), pending=len(pending),
                              failed_jobs=[f"{j['terrain']}@{j['period']}" for j in failed]),
                         indent=1))
        return 0

    if args.command == 'gate':
        # Three real jobs plus the reproduction replay, all under the same load.
        real = [j for j in jobs((0.40,), (20269101, 20269102, 20269103)) if not completed(j)]
        replay = dict(probe=GATE_TAG, method='RC_clock', terrain=GATE_TERRAIN,
                      period=GATE_PERIOD)
        replay_out = v.folder(replay['probe'], replay['method'], replay['terrain'],
                              replay['period'])
        if (replay_out / 'manifest.json').exists():
            if not completed(replay):
                raise SystemExit('stale MREPRO run; remove ' + str(replay_out))
        else:
            real.append(replay)
        failures = run_pool(real, args.device, args.workers)
        if failures:
            print('GATE ABORTED: a job failed; see logs', file=sys.stderr)
            return 2
        return 0 if gate_check() else 1

    pending = [j for j in wanted if not completed(j)]
    failures = run_pool(pending, args.device, args.workers)
    if failures:
        print(json.dumps([f"{j['terrain']}@{j['period']}: {e}" for j, e in failures], indent=1))
        return 2
    print(f'sweep complete: {len(pending)} runs', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
