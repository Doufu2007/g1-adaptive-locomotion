"""Recompute exploratory control results from paired, immutable rollout telemetry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.g1_closed_loop_metrics import command_schedule, summarize_episode

OUT = ROOT/'results/g1_best_phase_control_20261022'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def score(directory):
    manifest = read(directory/'manifest.json')
    if manifest['status'] != 'COMPLETED':
        raise ValueError(f'incomplete rollout: {directory}')
    rows, raw_data = [], {}
    for original in read(directory/'episodes.json'):
        path = directory/original['raw_path']
        if file_sha(path) != original['raw_sha256']:
            raise ValueError(f'telemetry hash mismatch: {path}')
        with np.load(path) as archive:
            raw = {key: archive[key].copy() for key in archive.files}
        raw['scenario'] = original['scenario']
        dt = .02
        commands = command_schedule(round(manifest['horizon']/dt), dt, original['scenario'])
        row = summarize_episode(raw, commands, dt, original['fell'])
        row.update(terrain=manifest['terrain'], arm=manifest['arm'], scenario=original['scenario'],
                   seed=original['seed'], raw_path=str(path.relative_to(ROOT)), raw_sha256=file_sha(path),
                   initial_env_origin=original['initial_env_origin'])
        row['lateral_velocity_rmse_mps'] = float(np.sqrt(np.mean(raw['velocity'][:, 1]**2)))
        tilt = np.arccos(np.clip(-raw['gravity'][:, 2], -1, 1))
        row['tilt_rms_rad'] = float(np.sqrt(np.mean(tilt**2)))
        row['max_tilt_rad'] = float(tilt.max())
        row['phase']['frequency_rmse_hz'] = mean([
            (c['predicted_frequency_hz']-c['measured_frequency_hz'])**2
            for c in row['phase']['cycles']])
        if row['phase']['frequency_rmse_hz'] is not None:
            row['phase']['frequency_rmse_hz'] **= .5
        q = raw['quaternion']
        yaw = np.unwrap(np.arctan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                                  1-2*(q[:, 2]**2+q[:, 3]**2)))
        row['heading_change_rad'] = float(yaw[-1]-yaw[0])
        row['phase_segments'] = []
        if row['scenario'] == 'speed_change':
            for start, end, target in [(0., 5., .3), (5., 12., .8), (12., manifest['horizon'], .4)]:
                cycles = [c for c in row['phase']['cycles']
                          if raw['time'][c['start']] >= start and raw['time'][c['end']] < end]
                mask = (raw['time'] >= start) & (raw['time'] < end)
                row['phase_segments'].append(dict(start_seconds=start, end_seconds=end, target_mps=target,
                    complete_cycles=len(cycles),
                    measured_frequency_hz=mean([c['measured_frequency_hz'] for c in cycles]),
                    predicted_frequency_hz=mean([c['predicted_frequency_hz'] for c in cycles]),
                    observed_velocity_mps=mean(raw['velocity'][mask, 0].tolist())))
        if row['terrain'] == 'rough':
            local = raw['position'][:, :2]-np.asarray(row['initial_env_origin'])[:2]
            inside = (np.abs(local[:, 0]) < 15.75) & (np.abs(local[:, 1]) < 3.75)
            row['rough_patch_interior_fraction'] = float(inside.mean())
        key = (row['terrain'], row['scenario'], row['seed'])
        raw_data[key] = raw
        rows.append(row)
    return manifest, rows, raw_data


def aggregate(rows):
    cycles = [c for row in rows for c in row['phase']['cycles']]
    pushes = [p for row in rows for p in row['push_recovery'] if p['exposed']]
    recovered = [p for p in pushes if p['recovered_seconds'] is not None]
    return dict(episodes=len(rows), falls=sum(r['fell'] for r in rows),
        locomotion_episodes=sum(r['locomotion_observed'] for r in rows),
        survival_seconds=mean([r['survival_seconds'] for r in rows]),
        forward_displacement_m=mean([r['forward_displacement_m'] for r in rows]),
        mae_forward_velocity_mps=mean([r['mae_forward_velocity_mps'] for r in rows]),
        lateral_velocity_rmse_mps=mean([r['lateral_velocity_rmse_mps'] for r in rows]),
        tilt_rms_rad=mean([r['tilt_rms_rad'] for r in rows]),
        complete_cycles=len(cycles),
        one_turn_fraction=mean([float(c['one_turn']) for c in cycles]),
        touchdown_phase_concentration=mean([r['phase']['touchdown_phase_concentration'] for r in rows]),
        frequency_rmse_hz=mean([r['phase']['frequency_rmse_hz'] for r in rows]),
        frequency_correlation=mean([r['phase']['frequency_correlation'] for r in rows]),
        push_exposures=len(pushes), recovered_pushes=len(recovered),
        recovered_only_mean_seconds=mean([p['recovered_seconds'] for p in recovered]),
        rough_patch_interior_fraction=mean([r.get('rough_patch_interior_fraction') for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='direct_swap_v1')
    parser.add_argument('--terrains', nargs='+', choices=['flat', 'rough'], default=['flat', 'rough'])
    args = parser.parse_args()
    directory = OUT/args.label
    results, checks, paired, aggregates, training_checks = [], [], [], [], []
    for terrain in args.terrains:
        arms = {arm: score(directory/(terrain+'_'+arm)) for arm in ('clock', 'learned')}
        clock, learned = arms['clock'], arms['learned']
        for key in ('seeds', 'scenarios', 'horizon'):
            if clock[0][key] != learned[0][key]:
                raise ValueError(f'unmatched evaluation protocol: {terrain}, {key}')
        if clock[0].get('parallel_worlds', 1) != learned[0].get('parallel_worlds', 1):
            raise ValueError('different world-slot mapping')
        if args.label.startswith('direct_swap') and clock[0]['policy_sha256'] != learned[0]['policy_sha256']:
            raise ValueError(f'direct-swap policy weights differ: {terrain}')
        if args.label.startswith('adapted'):
            training = {arm: read(Path(arms[arm][0]['policy']).parent/'manifest.json') for arm in arms}
            for arm in arms:
                if training[arm]['status'] != 'COMPLETED' or training[arm]['policy_sha256'] != arms[arm][0]['policy_sha256']:
                    raise ValueError('adaptation incomplete or policy artifact changed')
            fields = ['source_actor_sha256', 'initial_actor_state_sha256', 'initial_critic_state_sha256',
                      'iterations', 'num_envs', 'environment_transition_budget', 'ppo_seed',
                      'completed_environment_transitions', 'completed_learning_iterations']
            for key in fields:
                if training['clock'][key] != training['learned'][key]:
                    raise ValueError(f'unmatched training: {terrain}, {key}')
            training_checks.append(dict(terrain=terrain, **{key:training['clock'][key] for key in fields}))
        if clock[0]['startup_model_hashes'] != learned[0]['startup_model_hashes']:
            raise ValueError(f'unmatched startup physics: {terrain}')
        if clock[2].keys() != learned[2].keys():
            raise ValueError(f'unmatched episode keys: {terrain}')
        for key in clock[2]:
            a, b = clock[2][key], learned[2][key]
            errors = {name: float(np.max(np.abs(a[name][0].astype(float)-b[name][0].astype(float))))
                      for name in ('position', 'quaternion', 'joint_pos', 'velocity')}
            if max(errors.values()) > 1e-6:
                raise ValueError(f'unmatched initial state: {key}, {errors}')
            np.testing.assert_array_equal(a['command'], b['command'][:len(a['command'])]) if len(a['command']) <= len(b['command']) else np.testing.assert_array_equal(a['command'][:len(b['command'])], b['command'])
            checks.append(dict(terrain=key[0], scenario=key[1], seed=key[2], initial_state_max_errors=errors))
        by_key = {arm: {(r['scenario'], r['seed']): r for r in arms[arm][1]} for arm in arms}
        for key, c in by_key['clock'].items():
            l = by_key['learned'][key]
            np.testing.assert_array_equal(c['initial_env_origin'], l['initial_env_origin'])
            paired.append(dict(terrain=terrain, scenario=key[0], seed=key[1],
                learned_minus_clock={name:l[name]-c[name] for name in
                    ('survival_seconds', 'forward_displacement_m', 'mae_forward_velocity_mps',
                     'lateral_velocity_rmse_mps', 'tilt_rms_rad')}))
        for arm in arms:
            results.extend(arms[arm][1])
            for scenario in ('all', 'steady', 'speed_change', 'push'):
                selected = [r for r in arms[arm][1] if scenario == 'all' or r['scenario'] == scenario]
                if selected:
                    aggregates.append(dict(terrain=terrain, arm=arm, scenario=scenario, **aggregate(selected)))
    data = dict(label=args.label, scoring_script_sha256=file_sha(__file__),
        metrics_script_sha256=file_sha(ROOT/'scripts/g1_closed_loop_metrics.py'),
        interpretation='exploratory; episode repeats are not independent policy-training seeds',
        pairing_checks=checks, training_pair_checks=training_checks,
        aggregates=aggregates, paired_differences=paired, episodes=results)
    (directory/'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 当前最佳相位模型与固定 clock 的闭环探索对照', '',
        f'结果目录：`{args.label}`。每组 3 个初态种子 × 恒速/切速/横推，20 秒、50 Hz。', '',
        '下表的速度 MAE 按完整 20 秒评分，跌倒后剩余时间以零速度计；距离为固定初始朝向的前向位移。', '',
        '| 地形 | 相位 | 回合数 | 跌倒 | 实际行走 | 平均存活(s) | 前进(m) | 速度MAE(m/s) | 倾斜RMS(rad) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['scenario'] == 'all':
            lines.append(f"| {r['terrain']} | {r['arm']} | {r['episodes']} | {r['falls']} | {r['locomotion_episodes']} | {r['survival_seconds']:.2f} | {r['forward_displacement_m']:.3f} | {r['mae_forward_velocity_mps']:.4f} | {r['tilt_rms_rad']:.4f} |")
    lines += ['', '## 分场景结果', '',
        '| 地形 | 场景 | 相位 | 跌倒/n | 前进(m) | 速度MAE(m/s) | 横向速度RMS(m/s) |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in aggregates:
        if r['scenario'] != 'all':
            lines.append(f"| {r['terrain']} | {r['scenario']} | {r['arm']} | {r['falls']}/{r['episodes']} | {r['forward_displacement_m']:.3f} | {r['mae_forward_velocity_mps']:.4f} | {r['lateral_velocity_rmse_mps']:.4f} |")
    lines += ['', '## 相位与实际触地', '',
        '| 地形 | 相位 | 完整周期 | 一周期一圈比例 | 触地相位集中度 | 周期频率RMSE(Hz) |',
        '|---|---|---:|---:|---:|---:|']
    for r in aggregates:
        if r['scenario'] == 'all':
            values = [f'{r[k]:.4f}' if r[k] is not None else '不足' for k in
                      ('one_turn_fraction', 'touchdown_phase_concentration', 'frequency_rmse_hz')]
            lines.append(f"| {r['terrain']} | {r['arm']} | {r['complete_cycles']} | {' | '.join(values)} |")
    lines += ['', '## 解释范围', '',
        '- `direct_swap_v1` 使用相同的既有 clock 策略权重，仅切换相位输入，回答直接替换兼容性。推理时改奖励不会训练策略；它不能替代分别充分训练的方法对照。',
        '- `adapted_v1`（若存在）使用相同 actor 初始化和同预算短 PPO 适配；单一 PPO 种子不能冒充跨训练种子结论。',
        '- 粗糙条件仅为固定生成种子下的随机高度地形，噪声范围 0.02–0.10 m，32×8 m patch。不代表坡地、楼梯或全部粗糙地形。',
        '- 相位指标使用仿真接触测量，未经相位重标定。一周期一圈比例只是本次描述指标，不是原离线门禁验收。',
        '- 配对检查核对启动物理参数和每回合初始位置、姿态、关节与速度。完整周期、切速暴露、恢复删失和 patch 内覆盖见 comparison.json。',
        '- 本次没有硬件机器人试验，不能表述为真机稳定性或安全性确认。', '']
    rough = [r for r in aggregates if r['terrain'] == 'rough' and r['scenario'] == 'all']
    if rough:
        lines += ['## 指定粗糙区域覆盖', '',
            '| 相位 | 根部在指定 patch 内部的时间比例均值 |', '|---|---:|']
        for r in rough:
            lines.append(f"| {r['arm']} | {100*r['rough_patch_interior_fraction']:.2f}% |")
        lines += ['',
            '该比例以根部相对初始 patch 原点 |x|<15.75 m、|y|<3.75 m 为代理，不能验证每次脚底接触面。20 秒结果保留离开 patch 后的观测，存在不同地形暴露；应结合该覆盖解释。',
            '结果后共同前缀补充分析可运行 scripts/analyze_g1_rough_exposure.py；它保留全部配对、计入跌倒剩余时间，改变了原时间窗，不能代替主结果。', '']
    (directory/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps([r for r in aggregates if r['scenario'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
