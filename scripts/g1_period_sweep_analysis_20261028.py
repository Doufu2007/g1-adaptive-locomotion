"""Pre-registered analysis of the P (period sweep) and M (new-map) probes.

Implements exactly what results/g1_prereg_20261028/PREREGISTRATION.md fixed:
one primary endpoint, one pre-declared selection rule for p*, map as the
clustering unit, and no post-hoc endpoint swaps.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_period_sweep_20261028'
PREREG = ROOT / 'results/g1_prereg_20261028/PREREGISTRATION.md'
SUMMARY_CSV = ROOT / 'results/g1_comparison_summary_20261028/all_scored_episodes.csv'
NOMINAL = .6
METRICS = ('forward_displacement_m', 'mae_forward_velocity_mps', 'tilt_rms_rad',
           'abs_heading_change_rad', 'survival_seconds', 'moving_fraction')


def load_probe(tag):
    rows = []
    base = OUT / tag
    if not base.exists():
        return rows
    for manifest in sorted(base.glob('*/manifest.json')):
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record.get('status') != 'COMPLETED':
            continue
        for row in json.loads((manifest.parent / 'episodes.json').read_text(encoding='utf-8')):
            rows.append(row)
    return rows


def load_reference():
    if not SUMMARY_CSV.exists():
        return []
    rows = []
    with open(SUMMARY_CSV, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['stage'] != 'S' or row['terrain_seed'] == '0':
                continue
            if row['method'] not in ('RE', 'RC'):
                continue
            rows.append(dict(method=row['method'], terrain_seed=int(row['terrain_seed']),
                             train_seed=int(row['train_seed']),
                             eval_seed=int(row['eval_seed']),
                             scenario=row['scenario'], patch_col=int(row['patch_col']),
                             completed=row['completed'] == 'True',
                             fell=row['fell'] == 'True',
                             **{m: float(row[m]) for m in METRICS}))
    return rows


def key(row):
    return (int(row['terrain_seed']), int(row['train_seed']), int(row['eval_seed']),
            row['scenario'], int(row['patch_col']))


def normalise(rows, method=None):
    out = {}
    for row in rows:
        fixed = dict(row)
        if method:
            fixed['method'] = method
        fixed['completed'] = bool(fixed['completed'])
        fixed['fell'] = bool(fixed['fell'])
        k = key(fixed)
        if k in out:
            raise ValueError(f'duplicate episode {k}')
        out[k] = fixed
    return out


def mcnemar(pairs, field):
    a_only = sum(1 for x, y in pairs if x[field] and not y[field])
    b_only = sum(1 for x, y in pairs if y[field] and not x[field])
    n = len(pairs)
    na = sum(1 for x, _ in pairs if x[field])
    nb = sum(1 for _, y in pairs if y[field])
    nd = a_only + b_only
    if nd == 0:
        return dict(n=n, a=na, b=nb, diff_pp=0., a_only=0, b_only=0,
                    p=1.0, ci95_pp=[0., 0.])
    p_exact = min(1.0, 2 * sum(math.comb(nd, i) for i in range(min(a_only, b_only) + 1)) / 2 ** nd)
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
    half = float(stats.t.ppf(.975, n - 1) * deltas.std(ddof=1) / math.sqrt(n)) if n > 1 else float('nan')
    return dict(per_map=result, maps=n,
                positive_maps=int((deltas > 0).sum()),
                zero_maps=int((deltas == 0).sum()),
                map_mean_diff_pp=100 * float(deltas.mean()),
                map_ci95_pp=[100 * (float(deltas.mean()) - half),
                             100 * (float(deltas.mean()) + half)],
                map_level_p=float(stats.ttest_1samp(deltas, 0).pvalue) if n > 1 else float('nan'))


def metric_pairs(pairs):
    out = {}
    for metric in METRICS:
        diffs = [x[metric] - y[metric] for x, y in pairs]
        if not diffs:
            continue
        diffs = np.asarray(diffs, float)
        mean = float(diffs.mean())
        se = float(diffs.std(ddof=1) / math.sqrt(len(diffs)))
        half = float(stats.t.ppf(.975, len(diffs) - 1) * se)
        out[metric] = dict(mean=mean, ci95=[mean - half, mean + half],
                           p=float(stats.ttest_1samp(diffs, 0).pvalue))
    return out


def period_curve(p_rows):
    by_period = defaultdict(list)
    for row in p_rows:
        by_period[round(float(row['period']), 3)].append(row)
    curve = {}
    for period, rows in sorted(by_period.items()):
        per_seed = {}
        for seed in sorted({r['train_seed'] for r in rows}):
            sub = [r for r in rows if r['train_seed'] == seed]
            per_seed[int(seed)] = dict(n=len(sub),
                                       completed=sum(1 for r in sub if r['completed']),
                                       fell=sum(1 for r in sub if r['fell']))
        curve[period] = dict(
            n=len(rows),
            completed=sum(1 for r in rows if r['completed']),
            fell=sum(1 for r in rows if r['fell']),
            displacement_m=float(np.mean([r['forward_displacement_m'] for r in rows])),
            mae_mps=float(np.mean([r['mae_forward_velocity_mps'] for r in rows])),
            tilt_rad=float(np.mean([r['tilt_rms_rad'] for r in rows])),
            per_train_seed=per_seed)
    return curve


def selection(curve):
    """Pre-declared: maximise completion, break ties toward the nominal period."""
    if not curve:
        return None
    best = max(v['completed'] for v in curve.values())
    tied = [p for p, v in curve.items() if v['completed'] == best]
    return dict(p_star=min(tied, key=lambda p: (abs(p - NOMINAL), p)),
                best_completed=best, tied=tied,
                complete_grid=(len(curve) == 7))


def paired_between(a_rows, b_rows):
    """Rows must already be single-method; duplicates are a hard error."""
    a = normalise(a_rows)
    b = normalise(b_rows)
    shared = sorted(set(a) & set(b))
    pairs = [(a[k], b[k]) for k in shared]
    return dict(pairs=len(pairs), missing_a=len(set(b) - set(a)),
                missing_b=len(set(a) - set(b)),
                completed=mcnemar(pairs, 'completed'), fell=mcnemar(pairs, 'fell'),
                metrics=metric_pairs(pairs), maps=map_breakdown(pairs))


def main():
    p_rows = load_probe('P')
    m_rows = load_probe('M')
    reference = load_reference()
    report = dict(preregistration_sha=__import__('hashlib').sha256(
        PREREG.read_bytes()).hexdigest() if PREREG.exists() else None,
        counts=dict(P=len(p_rows), M=len(m_rows), reference_S_rough=len(reference)))

    # --- reproduction gate: period .60 must equal the audited S numbers -----
    gate_rows = [r for r in p_rows if abs(float(r['period']) - NOMINAL) < 1e-9]
    gate = None
    if gate_rows:
        completed = sum(1 for r in gate_rows if r['completed'])
        fell = sum(1 for r in gate_rows if r['fell'])
        disp = float(np.mean([r['forward_displacement_m'] for r in gate_rows]))
        gate = dict(n=len(gate_rows), completed=completed, fell=fell,
                    displacement_m=disp,
                    expected=dict(completed=22, fell=46, displacement_m=4.2437),
                    passed=bool(completed == 22 and fell == 46
                                and abs(disp - 4.2437) <= .01))
    report['reproduction_gate'] = gate

    curve = period_curve(p_rows)
    report['period_curve'] = curve
    report['selection'] = selection(curve)

    # --- A2: learned phase versus the swept-best fixed period on S maps -----
    if isinstance(report['selection'], dict):
        p_star = report['selection']['p_star']
        best_rows = [r for r in p_rows if abs(float(r['period']) - p_star) < 1e-9]
        reference_re = [r for r in reference if r['method'] == 'RE']
        if reference_re and best_rows:
            report['A2_learned_vs_best_fixed_S'] = paired_between(
                reference_re, best_rows)
        nominal_rows = [r for r in p_rows if abs(float(r['period']) - NOMINAL) < 1e-9]
        if reference_re and nominal_rows:
            report['consistency_learned_vs_nominal_S'] = paired_between(
                reference_re, nominal_rows)

    # --- A1: primary endpoint on new maps -----------------------------------
    if m_rows:
        learned = [r for r in m_rows if r['method'] == 'RE_live']
        fixed = [r for r in m_rows if r['method'] == 'RC_clock']
        if learned and fixed:
            report['A1_primary_new_maps'] = paired_between(learned, fixed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'], gate=gate,
                          selection=report['selection']), indent=1))


if __name__ == '__main__':
    main()
