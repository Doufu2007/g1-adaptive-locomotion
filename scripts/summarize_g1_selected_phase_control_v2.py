"""Pair and score the 30-seed selected-model vs fixed-clock rollouts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from scripts.summarize_g1_best_phase_control import aggregate, score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/g1_phase_model_comparison_20261023/control_v1'
RUNTIME = ROOT / 'results/g1_phase_model_comparison_20261023/runtime_selected/phasor_runtime.pt'
SELECTION = ROOT / 'results/g1_phase_model_comparison_20261023/tctst_residual_v4/selected_model.json'
SCENARIOS = ('steady', 'speed_change', 'push')
SEEDS = tuple(range(20261201, 20261231))
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 20000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def batch_directories(terrain: str, arm: str) -> list[Path]:
    rollouts = OUT / 'rollouts'
    return sorted(rollouts.glob(f'{terrain}_{arm}_batch[0-9][0-9]'))


def collect(terrain: str, arm: str):
    manifests, rows, telemetry = [], [], {}
    for directory in batch_directories(terrain, arm):
        manifest, current_rows, current_raw = score(directory)
        manifests.append(manifest)
        rows.extend(current_rows)
        overlap = set(telemetry) & set(current_raw)
        if overlap:
            raise ValueError(f'duplicate rollout keys: {sorted(overlap)[:3]}')
        telemetry.update(current_raw)
    if len(rows) != len(SEEDS) * len(SCENARIOS):
        raise ValueError(f'{terrain}/{arm}: expected 90 rows, got {len(rows)}')
    if set(telemetry) != {(terrain, scenario, seed) for scenario in SCENARIOS for seed in SEEDS}:
        raise ValueError(f'{terrain}/{arm}: incomplete seed/scenario cohort')
    return manifests, rows, telemetry


def cluster_bootstrap_differences(clock_rows, learned_rows, metric: str):
    clock = {(row['scenario'], row['seed']): row for row in clock_rows}
    learned = {(row['scenario'], row['seed']): row for row in learned_rows}
    per_seed = []
    for seed in SEEDS:
        diffs = []
        for scenario in SCENARIOS:
            a, b = clock[(scenario, seed)], learned[(scenario, seed)]
            av = float(a['fell']) if metric == 'fall_rate' else float(a[metric])
            bv = float(b['fell']) if metric == 'fall_rate' else float(b[metric])
            diffs.append(bv - av)
        per_seed.append(float(np.mean(diffs)))
    per_seed = np.asarray(per_seed)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(per_seed), size=(BOOTSTRAP_REPLICATES, len(per_seed)))
    means = per_seed[samples].mean(axis=1)
    return {
        'learned_minus_clock_mean': float(per_seed.mean()),
        'cluster_seed_count': len(per_seed),
        'bootstrap_95pct_ci': [float(x) for x in np.quantile(means, [.025, .975])],
        'seed_block_differences': per_seed.tolist(),
    }


def main() -> int:
    import torch
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    runtime = torch.load(RUNTIME, map_location='cpu', weights_only=False)
    if (runtime.get('selected_model_family') != selection['selected_family']
            or int(runtime.get('selected_seed', -1)) != int(selection['selected_seed_for_control'])):
        raise ValueError('runtime checkpoint does not match the frozen model selection')
    runtime_sha = sha(RUNTIME)
    results, pairing_checks, paired, aggregates = [], [], [], []
    by_terrain = {}
    for terrain in ('flat', 'rough'):
        arms = {arm: collect(terrain, arm) for arm in ('clock', 'learned')}
        clock_m, clock_rows, clock_raw = arms['clock']
        learned_m, learned_rows, learned_raw = arms['learned']
        for manifests in (clock_m, learned_m):
            for key in ('seeds', 'scenarios', 'horizon', 'parallel_worlds'):
                expected = list(SEEDS) if key == 'seeds' else (
                    list(SCENARIOS) if key == 'scenarios' else (20. if key == 'horizon' else 10)
                )
                if any(manifest[key] != expected for manifest in manifests):
                    raise ValueError(f'{terrain}: batch manifest mismatch for {key}')
        if len({m['policy_sha256'] for m in clock_m + learned_m}) != 1:
            raise ValueError(f'{terrain}: fixed actor weights differ across arms')
        if {m['encoder_sha256'] for m in clock_m + learned_m} != {runtime_sha}:
            raise ValueError(f'{terrain}: rollout encoder hash differs from the selected runtime')
        if len({m['startup_model_hashes']['body_mass'] for m in clock_m + learned_m}) != 1:
            raise ValueError(f'{terrain}: randomized body mass differs')
        clock_by_key = {(r['scenario'], r['seed']): r for r in clock_rows}
        learned_by_key = {(r['scenario'], r['seed']): r for r in learned_rows}
        for scenario, seed in sorted(clock_by_key):
            key = (terrain, scenario, seed)
            c, l = clock_raw[key], learned_raw[key]
            errors = {name: float(np.max(np.abs(c[name][0].astype(float) - l[name][0].astype(float))))
                      for name in ('position', 'quaternion', 'joint_pos', 'velocity')}
            if max(errors.values()) > 1e-6:
                raise ValueError(f'unmatched reset state {key}: {errors}')
            n = min(len(c['command']), len(l['command']))
            np.testing.assert_array_equal(c['command'][:n], l['command'][:n])
            np.testing.assert_array_equal(clock_by_key[(scenario, seed)]['initial_env_origin'],
                                          learned_by_key[(scenario, seed)]['initial_env_origin'])
            pairing_checks.append({'terrain': terrain, 'scenario': scenario, 'seed': seed,
                                   'initial_state_max_errors': errors})
            a, b = clock_by_key[(scenario, seed)], learned_by_key[(scenario, seed)]
            paired.append({'terrain': terrain, 'scenario': scenario, 'seed': seed,
                           'learned_minus_clock': {
                               key: float(b[key] - a[key]) for key in
                               ('survival_seconds', 'forward_displacement_m',
                                'mae_forward_velocity_mps', 'lateral_velocity_rmse_mps',
                                'tilt_rms_rad')},
                           'learned_fell_minus_clock': int(b['fell']) - int(a['fell'])})
        for arm, rows in (('clock', clock_rows), ('learned', learned_rows)):
            results.extend(rows)
            for scenario in ('all', *SCENARIOS):
                selected = [r for r in rows if scenario == 'all' or r['scenario'] == scenario]
                aggregates.append({'terrain': terrain, 'arm': arm, 'scenario': scenario,
                                   **aggregate(selected)})
        by_terrain[terrain] = {
            'rough_fall_rate': cluster_bootstrap_differences(clock_rows, learned_rows, 'fall_rate'),
            'rough_speed_mae': cluster_bootstrap_differences(
                clock_rows, learned_rows, 'mae_forward_velocity_mps'),
        }
    flat = {arm: [r for r in results if r['terrain'] == 'flat' and r['arm'] == arm]
            for arm in ('clock', 'learned')}
    by_terrain['flat_speed_mae'] = cluster_bootstrap_differences(
        flat['clock'], flat['learned'], 'mae_forward_velocity_mps')
    decision = bool(
        by_terrain['rough_fall_rate']['bootstrap_95pct_ci'][1] < 0
        and by_terrain['rough_speed_mae']['bootstrap_95pct_ci'][1] < 0
        and by_terrain['flat_speed_mae']['bootstrap_95pct_ci'][1] <= 0
    )
    result = {
        'schema': 'g1.selected_phase_control_comparison.v2',
        'interpretation': 'paired simulation study; seed-block bootstrap over 30 environment seeds; no hardware claim',
        'selected_model_family': selection['selected_family'],
        'selected_seed': int(selection['selected_seed_for_control']),
        'runtime_sha256': runtime_sha,
        'episodes_per_arm_per_terrain': len(SEEDS) * len(SCENARIOS),
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'primary_inference': by_terrain,
        'predeclared_claim_rule_passed': decision,
        'claim_rule': 'rough fall-rate and full-horizon speed-MAE paired 95% CI upper bounds < 0; flat speed-MAE upper bound <= 0',
        'aggregates': aggregates, 'pairing_checks': pairing_checks,
        'paired_episode_differences': paired, 'episodes': results,
    }
    (OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                         encoding='utf-8')
    lines = [
        '# 入选相位模型与固定 clock：成对闭环仿真', '',
        '每个地形 30 个环境种子 × 3 个情景；相同 clock actor、命令、随机种子、启动状态和物理模型。',
        '区间按环境 seed 分块 bootstrap，涵盖同一 seed 的三个情景。速度 MAE 使用完整 20 秒，跌倒后的剩余时间计零速。', '',
        '| 地形 | 相位 | 回合数 | 跌倒 | 生存秒 | 前进(m) | 速度MAE(m/s) | 横向RMS(m/s) | 倾斜RMS(rad) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in aggregates:
        if row['scenario'] == 'all':
            lines.append(f"| {row['terrain']} | {row['arm']} | {row['episodes']} | {row['falls']} | {row['survival_seconds']:.2f} | {row['forward_displacement_m']:.3f} | {row['mae_forward_velocity_mps']:.4f} | {row['lateral_velocity_rmse_mps']:.4f} | {row['tilt_rms_rad']:.4f} |")
    lines += ['', '## 成对主指标', '',
              f"粗糙地形跌倒率差（learned−clock）= {by_terrain['rough_fall_rate']['learned_minus_clock_mean']:.4f}，95% CI {by_terrain['rough_fall_rate']['bootstrap_95pct_ci']}。",
              f"粗糙地形速度 MAE 差 = {by_terrain['rough_speed_mae']['learned_minus_clock_mean']:.4f} m/s，95% CI {by_terrain['rough_speed_mae']['bootstrap_95pct_ci']}。",
              f"平地速度 MAE 差 = {by_terrain['flat_speed_mae']['learned_minus_clock_mean']:.4f} m/s，95% CI {by_terrain['flat_speed_mae']['bootstrap_95pct_ci']}。",
              f"预注册改善判据：{'通过' if decision else '未通过'}。", '',
              '该判据只适用于这一检查点与 MuJoCo 场景，不等于真机稳定性或部署验收。', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'primary_inference': by_terrain,
                      'predeclared_claim_rule_passed': decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
