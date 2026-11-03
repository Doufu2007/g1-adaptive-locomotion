"""Pre-registered analysis of the P probe, re-pointed at the CPU backend.

The GPU backend is not reproducible (see results/g1_nondeterminism_20261028/
NOTES.md), so no GPU number can be used as a reference for a paired comparison.
This analysis therefore takes **both** arms from the same CPU probe:

  fixed arm    RC_clock at the seven pre-registered periods, S maps
  learned arm  RE_live, S maps, same slots

and applies the pre-registered selection rule (maximise completion, ties toward
0.60) and the pre-registered paired tests.  The GPU reference table is no longer
read: comparing a CPU rollout to a GPU rollout would reintroduce a backend
difference on top of the arm difference.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_period_sweep_cpu_20261028'
NOMINAL = .6
METRICS = ('forward_displacement_m', 'mae_forward_velocity_mps', 'tilt_rms_rad',
           'abs_heading_change_rad', 'survival_seconds', 'moving_fraction')


def key(row):
    return (int(row['terrain_seed']), int(row['train_seed']), int(row['eval_seed']),
            row['scenario'], int(row['patch_col']))


def dedupe(rows):
    """Drop rows that were written twice for the same episode.

    The determinism check wrote maps 20264701 and 20264703 at period 0.60 both as
    one combined folder and as two per-scenario folders (``..._p0.60_push`` /
    ``..._p0.60_steady``).  Counting both would give that one period n=150 while
    every other period has n=90, and ``selection`` maximises the raw completion
    **count** -- so the duplicated period would win the p\\* draw on sample size
    alone.  The duplicates were verified bit-identical (same ``raw_sha256``); a
    duplicate that is not identical is a real contradiction and raises.
    """
    out, dropped = {}, []
    for row in rows:
        # The pairing key alone is *not* unique here: the seven period arms all
        # share the same (terrain, train, eval, scenario, patch_col).  Fold the
        # arm identity into the dedupe key.
        k = (row.get('method'), None if row.get('period') is None else round(float(row['period']), 3)) + key(row)
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


def normalise(rows):
    out = {}
    for row in rows:
        fixed = dict(row)
        fixed['completed'] = bool(fixed['completed'])
        fixed['fell'] = bool(fixed['fell'])
        k = key(fixed)
        if k in out:
            raise ValueError(f'duplicate episode {k}')
        out[k] = fixed
    return out


def mcnemar(pairs, field):
    import math
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
        fa = sum(1 for x, _ in items if x['fell'])
        fb = sum(1 for _, y in items if y['fell'])
        result[int(terrain)] = dict(n=len(items), completed_a=ca, completed_b=cb,
                                    delta_completed=ca - cb, fell_a=fa, fell_b=fb,
                                    delta_fell=fa - fb)
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
                             100 * (float(deltas.mean()) + half)],
                map_level_p=(float(stats.ttest_1samp(deltas, 0).pvalue)
                             if n > 1 else float('nan')))


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


def period_curve(rows):
    by_period = defaultdict(list)
    for row in rows:
        if row.get('period') is None:
            continue
        by_period[round(float(row['period']), 3)].append(row)
    curve = {}
    for period, items in sorted(by_period.items()):
        per_seed = {}
        for seed in sorted({r['train_seed'] for r in items}):
            sub = [r for r in items if r['train_seed'] == seed]
            per_seed[int(seed)] = dict(n=len(sub),
                                       completed=sum(1 for r in sub if r['completed']),
                                       fell=sum(1 for r in sub if r['fell']))
        curve[period] = dict(
            n=len(items),
            completed=sum(1 for r in items if r['completed']),
            fell=sum(1 for r in items if r['fell']),
            displacement_m=float(np.mean([r['forward_displacement_m'] for r in items])),
            mae_mps=float(np.mean([r['mae_forward_velocity_mps'] for r in items])),
            tilt_rad=float(np.mean([r['tilt_rms_rad'] for r in items])),
            per_train_seed=per_seed)
    return curve


def selection(curve):
    if not curve:
        return None
    best = max(v['completed'] for v in curve.values())
    tied = [p for p, v in curve.items() if v['completed'] == best]
    return dict(p_star=min(tied, key=lambda p: (abs(p - NOMINAL), p)),
                best_completed=best, tied=tied, complete_grid=(len(curve) == 7))


def paired(a_rows, b_rows):
    a = normalise(a_rows)
    b = normalise(b_rows)
    shared = sorted(set(a) & set(b))
    pairs = [(a[k], b[k]) for k in shared]
    return dict(pairs=len(pairs),
                missing_a=len(set(b) - set(a)), missing_b=len(set(a) - set(b)),
                completed=mcnemar(pairs, 'completed'), fell=mcnemar(pairs, 'fell'),
                metrics=metric_pairs(pairs), maps=map_breakdown(pairs))


def main():
    rows, dropped = load_probe('P')
    fixed = [r for r in rows if r.get('period') is not None]
    learned = [r for r in rows if r.get('period') is None]
    report = dict(backend='cpu',
                  preregistration_sha=__import__('hashlib').sha256(
                      (ROOT / 'results/g1_prereg_20261028/PREREGISTRATION.md')
                      .read_bytes()).hexdigest(),
                  counts=dict(fixed=len(fixed), learned=len(learned),
                              dropped_bit_identical_duplicates=len(dropped)))
    curve = period_curve(fixed)
    report['period_curve'] = curve
    report['selection'] = selection(curve)

    if report['selection'] and learned:
        p_star = report['selection']['p_star']
        best = [r for r in fixed if abs(float(r['period']) - p_star) < 1e-9]
        if best:
            report['A2_learned_vs_best_fixed'] = paired(learned, best)
        nominal = [r for r in fixed if abs(float(r['period']) - NOMINAL) < 1e-9]
        if nominal:
            report['A2_learned_vs_nominal'] = paired(learned, nominal)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'],
                          selection=report['selection'],
                          curve={str(k): (v['completed'], v['n'])
                                 for k, v in curve.items()}), indent=1))


if __name__ == '__main__':
    main()
