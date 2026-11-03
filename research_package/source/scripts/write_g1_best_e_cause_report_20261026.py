"""Build the reviewable diagnostic report after both intervention queues finish."""
import json
from pathlib import Path

from scripts import diagnose_g1_best_e_closed_loop_20261026 as audit


def read(name):
    return json.loads((audit.OUT/name).read_text(encoding='utf-8'))


def build():
    phase, training = read('phase_intervention_analysis.json'), read('training_intervention_analysis.json')
    assert phase['status'] == training['status'] == 'COMPLETED'
    assert phase['verified_raw_episodes'] == 630 and training['verified_raw_episodes'] == 54
    assert training['training_integrity']['completed_training_runs'] == 12
    saved, reward = read('saved_analysis.json'), read('reward_state_replay.json')
    probe = training['training_integrity']['sampling_phase_check']
    assert saved['all_270_raw_files_verified']
    conclusions = read('finding_status.json')
    assert conclusions['status'] == 'COMPLETED', 'Finish evidence-bound interpretation first'
    lines = ['# 最佳E闭环表现逐项排查', '',
        '固定模型：observer_sensors_mixed，seed20261026；SHA256：`'+audit.ENCODER_SHA+'`。', '',
        '复核原270回合；新增27条开发轨迹、630条相位/检查点评测；12项训练各追加100次更新，随后54条独立恒速评测。所有固定最终检查点均保留，不按成绩挑选模型或种子。', '',
        '本轮已完成：适配器数值、缓存与重置检查通过；此前“周期完成率降至54%–68%”的解释因接触统计口径不同而撤回。统一口径后为91.21%–97.55%。相位零点、有限续训和pose奖励均出现影响线索，但没有锁定唯一根因，也未建立E整体优于clock的结论。', '',
        '新增计算使用本机RTX 4050 Laptop、PyTorch 2.6.0+cu124；原远端为PyTorch 2.8，mjlab/MuJoCo/Warp/RSL-RL版本一致。以下是本地配对探索性诊断，统计单位为三个原PPO训练种子；每个策略的多条回合不是独立训练重复。', '',
        '技术恢复：390条正式评测完成后，旧Windows启动器重复添加盘符映射触发WinError 122；故障发生在下一项创建环境之前。失败状态已保存于failed_attempts/windows_bootstrap_20261026T200832。独立恢复启动器复用相同映射并缓存同一bootstrap模块，重复调用检查通过；既有结果不重跑，实验脚本、模型、校准、种子和预算不变。', '',
        '## 逐项结论', '', '| 原因 | 判断 | 证据与边界 |', '|---|---|---|']
    for item in conclusions['findings']:
        lines.append('| '+item['title']+' | '+item['conclusion']+' | '+item['detail']+' |')
    lines += ['', '## 适配器与周期口径', '',
        '原270条原始NPZ全部核对SHA；其中180条使用E的在线轨迹与离线模型重放，最大角误差为2.25543e-5 rad。此前局部reset、重复读取缓存、关节顺序、WXYZ、差分与归一化检查通过。11份控制、模型、奖励和评测源码与原封存SHA一致。', '',
        f"真实PPO冒烟采样另检查{probe['steps']}步×64环境，共{probe['compared_environment_frames']}个环境帧，actor/critic相位与当步E缓存输出一致；最大误差分别为{probe['max_abs_error']['actor']:.3g}/{probe['max_abs_error']['critic']:.3g}，期间发生{probe['reset_transitions']}次终止/截断重置。该读取探针未额外调用E、采样动作或推进物理步。", '',
        '原控制统计使用原始接触与最短间隔过滤，离线验收使用接触去抖3帧与半开周期区间。将54%–68%与离线约97%直接比较，会夸大退化；这项解释已纠正。全部1/3/5帧敏感性结果保留，主口径复用原离线验收函数，未按结果调阈值。', '',
        '| PPO seed | clock步态上重放E完成率 | E步态上E完成率 | E步态触地相位MAE(rad) |', '|---:|---:|---:|---:|']
    for seed in (20261901,20261902,20261903):
        a = next(r for r in saved['groups'] if r['train_seed']==seed and r['label']=='clock_clock')
        b = next(r for r in saved['groups'] if r['train_seed']==seed and r['label']=='learned_learned')
        lines.append(f"| {seed} | {a['E_macro_completion_by_debounce']['3']:.2%} | {b['recorded_phase_canonical_macro_completion']:.2%} | {b['E_macro_contact_phase_mae_rad']:.4f} |")
    lines += ['', '全部轨迹未发现归一化特征绝对值大于5；这只排除极端越界，不能排除中等程度的分布变化。停止指令下clock置零而E不置零，但本轮指令始终大于0.1 m/s，此差异未被触发。', '',
        '补充启动敏感性：原第三seed恒速轨迹去掉前3秒后，E速度仍约0.378 m/s、MAE约0.1224；clock约0.485 m/s、MAE约0.0420。因此仅有缓存启动阶段不能解释持续慢走。该补充窗口不替换完整20秒主指标。', '',
        '## 相位零点与中间检查点', '',
        '在27条开发clock轨迹上估计单一全局偏移，锁定为+0.250400573 rad（14.35°，按0.6秒周期约23.91 ms）；之后未使用holdout调节。正式holdout为3训练seed×10初态seed×3场景。校准只旋转输入角度，不更改周期净进展，也没有重新训练策略。', '',
        '原clock轨迹上的E−clock偏移随已训练clock策略变化，三个seed分别约−0.082、−0.449、−0.151 rad。因此该单一全局旋转只检验一种可部署校准，不能排除所有逐策略或动态对齐方案。', '',
        '| PPO seed | 条件 | 速度MAE(m/s，越小越好) | 根部倾斜RMS(rad) | 初始方向位移(m) | 航向变化绝对值(rad) | 跌倒/n |', '|---:|---|---:|---:|---:|---:|---:|']
    names = {'holdout_clock_clock':'clock策略→clock', 'holdout_clock_raw':'clock策略→原E',
             'holdout_clock_aligned':'clock策略→校准E', 'holdout_learned_raw':'E策略→原E',
             'holdout_learned_aligned':'E策略→校准E', 'midpoint_clock':'251次clock→clock',
             'midpoint_learned':'251次E→E'}
    for row in phase['aggregates']:
        if row['scenario']=='all':
            lines.append(f"| {row['train_seed']} | {names[row['condition']]} | {row['mae_forward_velocity_mps']:.5f} | {row['tilt_rms_rad']:.5f} | {row['forward_displacement_m']:.3f} | {row['abs_heading_change_rad']:.3f} | {row['falls']}/{row['episodes']} |")
    lines += ['', '相位评测的所有启动物理参数与配对初态核查通过。251次检查点固定使用model_250.pt，完成更新数为251；六个检查点均通过64输入的PyTorch/ONNX一致性检查，最大误差1.13249e-6。', '',
        '## 预算和姿势奖励干预', '',
        '原501次最终检查点恢复actor、critic和optimizer；每个条件重新初始化物理状态，原PPO seed+5000作为追加训练随机种子。三seed×clock/E×原奖励/去pose共12项，每项追加100更新、64环境×24步，共153600 transitions；最终总更新数601。去pose仅将pose权重1改为0。源状态、初始学习率、优化器、配置唯一差异、冻结E与实际步数均检查通过。', '',
        '评测使用新初态seed20262711–20262713，仅0.5 m/s恒速20秒。下表不能外推到切速、横推或更长训练。速度指令与MAE使用身体坐标系，初始方向位移与航向变化另列，不将速度误差下降表述为整体控制更好。', '',
        '| 相位来源 | 原501更新MAE | 原奖励续训至601更新MAE | 去pose续训至601更新MAE |',
        '|---|---:|---:|---:|']
    for arm, label in [('clock', 'clock'), ('learned', 'E')]:
        values = [next(r['mae_forward_velocity_mps'] for r in training['aggregates']
                       if r['condition'] == f'training_holdout_{arm}_{mode}'
                       and r['train_seed'] == 'all' and r['scenario'] == 'all')
                  for mode in ('baseline501', 'original_reward', 'pose_removed')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['',
        '均为m/s，越小越好。去pose后的两相位组均值接近，其中包含clock变差和E改善两部分，不能把相对差距缩小全归因于E改善。E组去pose后平均速度由0.43750提高到0.48600 m/s，但root倾斜RMS由0.04752升至0.05616 rad，航向绝对变化由0.46886升至0.70220 rad。第一seed速度达到0.54281 m/s且MAE略增，第三seed仍仅0.43008 m/s。所有54回合均无跌倒。', '',
        'E组追加100更新的原奖励效果三seed为2改善、1变差；去pose效果也是2改善、1变差。相位与去pose的交互三seed同方向，均值为−0.020642 m/s，未经校正p=0.0262，但完整五项比较BH q=0.1308；因此仍按探索性线索解释。', '',
        '| PPO seed | 相位/训练条件 | 速度MAE(m/s) | 平均实际速度(m/s) | 根部倾斜RMS(rad) | 初始方向位移(m) | 航向变化绝对值(rad) | 跌倒/n |', '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in training['aggregates']:
        if row['scenario']=='all':
            lines.append(f"| {row['train_seed']} | {row['condition'].removeprefix('training_holdout_')} | {row['mae_forward_velocity_mps']:.5f} | {row['mean_observed_forward_velocity_mps']:.5f} | {row['tilt_rms_rad']:.5f} | {row['forward_displacement_m']:.3f} | {row['abs_heading_change_rad']:.3f} | {row['falls']}/{row['episodes']} |")
    lines += ['', '## 全部速度差值与不确定性', '',
        '差值按每个训练seed先对配对评测回合取均值，再汇总三个seed；负值表示误差降低。95% t区间假设策略级效应近似正态，n=3无法检验该假设，区间与q仅作为探索性参考。相位报告六项比较、训练报告五项比较，分别进行完整family的BH校正，不将每个回合当独立样本。相位的条件与校准在holdout前锁定；六个报告比较在首个seed部分结果可见后明确，属于探索性分析。训练五项比较在新训练前已锁定。', '',
        '| 比较 | 三seed差值(m/s) | 平均差(m/s) | 95% t区间 | BH q |', '|---|---|---:|---|---:|']
    for record in (phase, training):
        for row in record['speed_contrasts']:
            values=', '.join(f'{v:+.5f}' for v in row['per_training_seed'].values())
            lo, hi = row['paired_t_95pct_ci_mps']
            lines.append(f"| {row['name']} | {values} | {row['mean_difference_mps']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {row['q_bh_full_family']:.4f} |")
    lines += ['', '## 奖励核算的含义', '',
        '训练的速度项同时包含前向、横向和竖向速度，评测MAE只包含前向；姿态项使用torso_link，报告中的tilt使用root。因此训练回报增加并不保证前向MAE或root tilt同时变好。', '',
        '原第三seed轨迹去前3秒后，E相对clock的速度奖励差−0.05211；pose、torso直立、动作平滑三项合计约+0.05029，抵消约96.5%。这只是不同策略访问状态上的四项可辨认奖励核算，不是完整训练回报，也不是奖励权重的因果证据。torso姿态重算已与MuJoCo原生运动学的20个状态验证，最大重力分量误差2.55e-9。因果判断以等预算pose去除试验为依据。', '',
        '## 证据文件', '',
        '- `saved_analysis.json`：原270回合SHA、E重放、原验收口径与接触敏感性。',
        '- `phase_intervention_analysis.json`：新增630回合、全部配对与六项差值。',
        '- `training_intervention_analysis.json`：12项训练恢复/配置/预算核查、54回合与五项差值。',
        '- `calibration.json`、两份`prospective_protocol.json`：固定校准与实验条件。',
        '- `reward_state_replay.json`、`reward_training_audit.json`：奖励描述性证据。',
        '- `original_source_integrity.json`、`semantic_interface_audit.json`、`midpoint_export_verification.json`：来源与接口。',
        '- `training_interventions/smoke_sampling_phase_consistency.json`：真实PPO采样中actor/critic相位一致性。',
        '- `phase_origin_semantics.json`、`startup_sensitivity.json`：相位坐标与启动窗口补充核查。',
        '- `rollouts/`、`training_interventions/training/`、`logs/`：原始轨迹、全部检查点和日志。',
        '- `failed_attempts/`：技术失败记录；未因成绩删除或重选条件。', '']
    (audit.OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    audit.write('completion_verification.json',dict(status='PASS',at=audit.now(),
        original_raw_replayed=270,phase_holdout_raw_verified=630,training_holdout_raw_verified=54,
        training_runs=12,extra_updates_per_run=100,all_E_frozen=True,
        report_sha256=audit.sha(audit.OUT/'diagnosis.md'),
        analysis_sources={p:audit.sha(audit.OUT/p) for p in
                         ('saved_analysis.json','phase_intervention_analysis.json',
                          'training_intervention_analysis.json','finding_status.json')}))
    print(audit.OUT/'diagnosis.md')


if __name__ == '__main__':
    build()
