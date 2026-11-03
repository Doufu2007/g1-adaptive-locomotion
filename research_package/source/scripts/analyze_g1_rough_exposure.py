"""Post-result sensitivity: paired prefix before either root leaves the patch.

All pairs remain included. Falls inside the shared horizon retain zero-speed
padding. This supplements, and never replaces, the original 20-second score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.g1_closed_loop_metrics import command_schedule, fixed_horizon_mae
from scripts.summarize_g1_best_phase_control import OUT, file_sha, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    directory = OUT/args.label
    # The main comparison must have completed its pairing checks first.
    comparison = json.loads((directory/'comparison.json').read_text(encoding='utf-8'))
    assert any(r['terrain'] == 'rough' for r in comparison['pairing_checks'])
    arms = {arm: score(directory/('rough_'+arm)) for arm in ('clock', 'learned')}
    assert arms['clock'][2].keys() == arms['learned'][2].keys()
    dt = .02
    rows = []
    for key in sorted(arms['clock'][2]):
        raw = {arm: arms[arm][2][key] for arm in arms}
        metadata = {arm: next(r for r in arms[arm][1]
                             if (r['terrain'], r['scenario'], r['seed']) == key) for arm in arms}
        horizon = round(arms['clock'][0]['horizon']/dt)
        first_exit = {}
        for arm in arms:
            local = raw[arm]['position'][:, :2] - np.asarray(metadata[arm]['initial_env_origin'])[:2]
            outside = (np.abs(local[:, 0]) >= 15.75) | (np.abs(local[:, 1]) >= 3.75)
            indices = np.flatnonzero(outside)
            first_exit[arm] = int(indices[0]) if len(indices) else horizon
        steps = min(horizon, *first_exit.values())
        if not steps:
            raise ValueError('a paired rollout started outside the rough interior')
        commands = command_schedule(horizon, dt, key[1])[:steps]
        result = dict(scenario=key[1], seed=key[2], shared_horizon_seconds=steps*dt,
                      first_exit_seconds={arm: n*dt if n < horizon else None for arm, n in first_exit.items()})
        for arm in arms:
            result[arm] = dict(
                mae_forward_velocity_mps=fixed_horizon_mae(raw[arm]['velocity'][:steps, 0], commands),
                fell_within_shared_horizon=bool(metadata[arm]['fell'] and len(raw[arm]['time']) <= steps),
                raw_path=metadata[arm]['raw_path'], raw_sha256=metadata[arm]['raw_sha256'])
        rows.append(result)
    aggregates = {arm: dict(
        falls=sum(r[arm]['fell_within_shared_horizon'] for r in rows),
        mae_forward_velocity_mps=float(np.mean([r[arm]['mae_forward_velocity_mps'] for r in rows])))
        for arm in arms}
    data = dict(label=args.label, analysis_type='POST_RESULTS_SUPPLEMENTARY_SENSITIVITY',
        rule='all pairs; prefix until first root exit by either arm; fallen remainder padded with zero speed',
        patch_root_bounds_relative_to_origin_m=[15.75, 3.75],
        interpretation='root-interior proxy, not foot-surface verification; not a new primary endpoint',
        original_comparison_sha256=file_sha(directory/'comparison.json'),
        analysis_code_sha256=file_sha(__file__), pairs=len(rows),
        mean_shared_horizon_seconds=float(np.mean([r['shared_horizon_seconds'] for r in rows])),
        minimum_shared_horizon_seconds=min(r['shared_horizon_seconds'] for r in rows),
        maximum_shared_horizon_seconds=max(r['shared_horizon_seconds'] for r in rows),
        aggregates=aggregates, rows=rows)
    (directory/'rough_exposure_sensitivity.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
