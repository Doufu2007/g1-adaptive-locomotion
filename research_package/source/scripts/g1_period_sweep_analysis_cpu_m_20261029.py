"""Pre-registered analysis of the M probe (A1 main endpoint), CPU backend.

The P-probe analysis (`g1_period_sweep_analysis_cpu_20261028.py`) covers A2 only.
This file covers the one part of the pre-registration that carries an inference
test: learned phase vs fixed clock on the ten pre-registered new maps.

Everything that decides the verdict is fixed in
`results/g1_prereg_20261028/PREREGISTRATION.md` and is imported here verbatim:
McNemar exact two-sided, paired 95% CI from the discordant pairs, map as the
clustering unit, and the four-row decision table of section 5.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_period_sweep_cpu_20261028'
METRICS = ('forward_displacement_m', 'mae_forward_velocity_mps', 'tilt_rms_rad',
           'abs_heading_change_rad', 'survival_seconds', 'moving_fraction')


def key(row):
    return (int(row['terrain_seed']), int(row['train_seed']), int(row['eval_seed']),
            row['scenario'], int(row['patch_col']))


def dedupe(rows):
    """Drop rows written twice for the same episode.

    Map 20269101 was run both as one combined folder and as two per-scenario
    folders (``..._push`` / ``..._steady``) for both arms, so 45 episodes appear
    in two places.  They were verified bit-identical (same ``raw_sha256``); a
    duplicate that is not identical is a real contradiction and raises.

    The arm identity has to be part of the key: ``key()`` alone is shared by the
    learned and the clock arm of the same episode, which is the whole point of
    the pairing.
    """
    out, dropped = {}, []
    for row in rows:
        k = (row.get('method'),
             None if row.get('period') is None else round(float(row['period']), 3)) + key(row)
        if k in out:
            prev = out[k]
            if (bool(prev['completed']), bool(prev['fell']), prev['raw_sha256']) != \
                    (bool(row['completed']), bool(row['fell']), row['raw_sha256']):
                raise ValueError(f'conflicting duplicate episode {k}: '
                                 f'{prev["raw_sha256"]} vs {row["raw_sha256"]}')
            dropped.append(k)
            continue
        out[k] = row
    return list(out.values()), dropped


def load_probe(tag):
    rows = []
    base = OUT / tag
    if not base.exists():
        return rows, []
    for manifest in sorted(base.glob('*/manifest.json')):
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record.get('status') != 'COMPLETED':
            continue
        for row in json.loads((manifest.parent / 'episodes.json').read_text(encoding='utf-8')):
            rows.append(row)
    return dedupe(rows)


def index(rows):
    out = {}
    for row in rows:
        k = key(row)
        if k in out:
            raise ValueError(f'duplicate episode {k}')
        out[k] = row
    return out


def mcnemar(pairs, field):
    a_only = sum(1 for x, y in pairs if x[field] and not y[field])
    b_only = sum(1 for x, y in pairs if y[field] and not x[field])
    n = len(pairs)
    na = sum(1 for x, _ in pairs if x[field])
    nb = sum(1 for _, y in pairs if y[field])
    nd = a_only + b_only
    if nd == 0:
        return dict(n=n, a=na, b=nb, diff_pp=0., a_only=0, b_only=0, p=1.0,
                    ci95_pp=[0., 0.])
    p_exact = min(1.0, 2 * sum(math.comb(nd, i)
                               for i in range(min(a_only, b_only) + 1)) / 2 ** nd)
    # The difference in proportions is (a_only - b_only)/n = (2q - 1) * nd/n with
    # q = a_only/nd, so its standard error is 2 * (nd/n) * sqrt(q(1-q)/nd).  An
    # earlier version dropped the leading 2 and therefore reported intervals
    # exactly half as wide as they should be; point estimates and the exact
    # p-values never depended on this and were correct.
    share = a_only / nd
    se = 2 * math.sqrt(share * (1 - share) / nd) * nd / n
    point = (a_only - b_only) / n
    return dict(n=n, a=na, b=nb, diff_pp=100 * point, a_only=a_only, b_only=b_only,
                p=float(p_exact),
                ci95_pp=[100 * (point - 1.96 * se), 100 * (point + 1.96 * se)])


def map_breakdown(pairs):
    groups = defaultdict(list)
    for x, y in pairs:
        groups[int(x['terrain_seed'])].append((x, y))
    result = {}
    for terrain, items in sorted(groups.items()):
        ca = sum(1 for x, _ in items if x['completed'])
        cb = sum(1 for _, y in items if y['completed'])
        result[int(terrain)] = dict(
            n=len(items), completed_a=ca, completed_b=cb, delta_completed=ca - cb,
            fell_a=sum(1 for x, _ in items if x['fell']),
            fell_b=sum(1 for _, y in items if y['fell']))
    deltas = np.array([v['delta_completed'] / v['n'] for v in result.values()])
    n = len(deltas)
    half = (float(stats.t.ppf(.975, n - 1) * deltas.std(ddof=1) / np.sqrt(n))
            if n > 1 else float('nan'))
    return dict(per_map=result, maps=n,
                positive_maps=int((deltas > 0).sum()),
                zero_maps=int((deltas == 0).sum()),
                negative_maps=int((deltas < 0).sum()),
                map_mean_diff_pp=100 * float(deltas.mean()),
                map_ci95_pp=[100 * (float(deltas.mean()) - half),
                             100 * (float(deltas.mean()) + half)])


def metric_pairs(pairs):
    out = {}
    for metric in METRICS:
        diffs = np.asarray([x[metric] - y[metric] for x, y in pairs], float)
        if not diffs.size:
            continue
        mean = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        half = float(stats.t.ppf(.975, len(diffs) - 1) * se)
        out[metric] = dict(mean=mean, ci95=[mean - half, mean + half],
                           p=float(stats.ttest_1samp(diffs, 0).pvalue))
    return out


def verdict(completed, maps, learned_total, fixed_total):
    p = completed['p']
    point = completed['diff_pp']
    positive = maps['positive_maps']
    if p < .05 and point >= 10 and positive >= 7:
        return 'SUPPORTS_A1', '在新地图上完成率显著更高（+%.2f pp，p=%.4g）' % (point, p)
    if p < .05:
        return 'WEAK_A1', '差异统计显著但幅度小（+%.2f pp），不足以支撑"改善鲁棒性"' % point
    return ('NOT_SUPPORTED_A1',
            '未见显著差异（%+.2f pp，95%% CI [%+.2f, %+.2f]，p=%.4g）'
            % (point, completed['ci95_pp'][0], completed['ci95_pp'][1], p))


def main():
    rows, dropped = load_probe('M')
    learned = index([r for r in rows if r.get('period') is None])
    fixed = index([r for r in rows if r.get('period') is not None])
    shared = sorted(set(learned) & set(fixed))
    pairs = [(learned[k], fixed[k]) for k in shared]
    maps = map_breakdown(pairs)
    completed = mcnemar(pairs, 'completed')
    fell = mcnemar(pairs, 'fell')
    code, wording = verdict(completed, maps, len(learned), len(fixed))
    report = dict(
        backend='cpu', probe='M',
        preregistration_sha=__import__('hashlib').sha256(
            (ROOT / 'results/g1_prereg_20261028/PREREGISTRATION.md')
            .read_bytes()).hexdigest(),
        counts=dict(pairs=len(pairs), learned_only=len(set(fixed) - set(learned)),
                    fixed_only=len(set(learned) - set(fixed)),
                    maps=len({k[0] for k in shared}),
                    dropped_bit_identical_duplicates=len(dropped)),
        primary=completed, fell=fell, maps=maps,
        descriptive=metric_pairs(pairs),
        verdict=code, wording=wording)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis_M_cpu.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(verdict=code, wording=wording,
                          primary=completed, maps=dict(
                              positive=maps['positive_maps'], zero=maps['zero_maps'],
                              negative=maps['negative_maps'])), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
