"""Exploratory analysis: fixed-clock period sweep on the ten held-out M maps.

Reads the completed M-probe runs and answers one question that the
pre-registered study left open:

    on the held-out maps, does RE_live also exceed the *best* fixed period, or
    only the nominal 0.60 s clock it was pre-registered against?

Everything here is descriptive. The decision rules were fixed before the outcome
was read, in
`results/g1_prereg_20261028/AMENDMENT_20261030_M_period_sweep.md`:

  E1  dominance requires c_learned > c_p strictly, for every swept period p.
  E2  if some c_p >= c_learned, the argmax period is compared by paired McNemar
      and reported as selection-inflated: it is chosen on the same 300 episodes
      used to test it.
  E3  no confirmatory language; A1's verdict is unchanged by this file.
  E4  a partial grid is reported as incomplete, with no per-period aggregate.

The paired tests reuse the pre-registered machinery (exact McNemar, paired 95%
CI from the discordant pairs, map as the clustering unit) by importing
`g1_period_sweep_analysis_cpu_m_20261029`, so the sweep cannot drift away from
the definition used for the main endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import g1_period_sweep_analysis_cpu_m_20261029 as m

OUT = m.OUT
PERIODS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
LEARNED_COMPLETED = 156          # analysis_M_cpu.json, unchanged; re-derived below
TOTAL = 300


def main():
    rows, dropped = m.load_probe('M')
    learned = {}
    by_period = {}
    for row in rows:
        period = row.get('period')
        if period is None:
            learned[m.key(row)] = row
        else:
            by_period.setdefault(round(float(period), 3), {})[m.key(row)] = row

    if len(learned) != TOTAL:
        raise ValueError(f'expected {TOTAL} learned episodes, got {len(learned)}')

    per_period = {}
    for period in PERIODS:
        cells = by_period.get(round(period, 3), {})
        if len(cells) != TOTAL:
            per_period[f'{period:.2f}'] = dict(period=period, n=len(cells), complete=False)
            continue
        shared = sorted(set(learned) & set(cells))
        pairs = [(learned[k], cells[k]) for k in shared]
        per_period[f'{period:.2f}'] = dict(
            period=period, n=len(shared), complete=len(shared) == TOTAL,
            completed=sum(1 for _, y in pairs if y['completed']),
            fell=sum(1 for _, y in pairs if y['fell']),
            vs_learned_completed=m.mcnemar(pairs, 'completed'),
            vs_learned_fell=m.mcnemar(pairs, 'fell'),
            vs_learned_maps=m.map_breakdown(pairs))

    complete = {k: v for k, v in per_period.items() if v['complete']}
    missing = sorted(f'{p:.2f}' for p in PERIODS if f'{p:.2f}' not in complete)

    learned_completed = sum(1 for row in learned.values() if row['completed'])
    learned_fell = sum(1 for row in learned.values() if row['fell'])

    report = dict(
        backend='cpu', probe='M', scope='EXPLORATORY',
        amendment='results/g1_prereg_20261028/AMENDMENT_20261030_M_period_sweep.md',
        grid_complete=not missing, missing_periods=missing,
        dropped_bit_identical_duplicates=len(dropped),
        learned=dict(n=len(learned), completed=learned_completed, fell=learned_fell),
        per_period=per_period)

    if missing:
        report.update(verdict='INCOMPLETE_GRID',
                      wording=f'周期网格不完整，缺失 {missing}；不报告逐周期汇总（E4）。')
        write(report)
        return

    counts = {k: v['completed'] for k, v in complete.items()}
    best_key = max(counts, key=lambda k: (counts[k], -abs(float(k) - .6)))
    best = complete[best_key]
    strictly_dominated = all(learned_completed > c for c in counts.values())
    ties = [k for k, c in counts.items() if c == learned_completed]

    if strictly_dominated:
        report.update(verdict='E1_DOMINATES_SWEPT_FAMILY',
                      best_period=float(best_key), best_completed=best['completed'],
                      ties=ties,
                      wording=('在 10 张留出图上，学习相位 156/300 严格高于扫过的全部 7 个固定周期'
                               '（E1 成立）。'))
    else:
        vs = best['vs_learned_completed']
        report.update(verdict='E2_BEST_PERIOD_NOT_BEATEN',
                      best_period=float(best_key), best_completed=best['completed'],
                      ties=ties, vs_best=vs,
                      wording=('学习相位 156/300 未超过扫过的全部固定周期：最优为 %.2f s 的 '
                               '%d/300，配对 McNemar p=%.4g。该比较是探索性的，且该周期是在同一批 '
                               '300 回合上选出的，存在选择偏差（E2）。'
                               % (float(best_key), best['completed'], vs['p'])))
    write(report)


def write(report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis_M_sweep_20261030.json').write_text(
        json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(verdict=report['verdict'], wording=report['wording'],
                          learned=report['learned'],
                          per_period={k: dict(completed=v.get('completed'),
                                              fell=v.get('fell'), n=v.get('n'))
                                      for k, v in report['per_period'].items()}),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
