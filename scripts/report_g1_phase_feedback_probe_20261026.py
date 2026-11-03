"""Render the complete feedback diagnosis after evidence-bound findings are written."""
import json
from pathlib import Path
import numpy as np

from scripts import g1_phase_feedback_probe_20261026 as exp


def figures(analysis, baseline_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True,layout='constrained')
    for i,seed in enumerate(exp.SEEDS):
        for arm,mode,color in [('clock','clock','#3873a6'),('learned','live','#bd6235')]:
            samples=[]
            for eval_seed in exp.EVAL:
                path=exp.OUT/'rollouts'/f'{arm}_{mode}_{seed}'/f'steady_{eval_seed}.npz'
                with np.load(path) as raw:
                    samples.append((raw['time'].copy(),np.diff(raw['shadow_phase'].astype(float))/(2*np.pi*.02),raw['velocity'][:,0].copy()))
            for column in (0,1):
                window=25
                curves=[np.convolve(s[1 if column==0 else 2],np.ones(window)/window,mode='valid') for s in samples]
                length=min(map(len,curves)); curve=np.mean([v[:length] for v in curves],axis=0)
                times=samples[0][0][window//2:window//2+length]
                axes[i,column].plot(times,curve,color=color,label=arm,linewidth=1.1)
        axes[i,0].axhline(1/.6,color='0.5',linestyle=':',linewidth=.8)
        axes[i,1].axhline(.5,color='0.5',linestyle=':',linewidth=.8)
        axes[i,0].set_ylabel(f'Seed {str(seed)[-2:]}\nE phase rate (Hz)')
        axes[i,1].set_ylabel('Body forward speed (m/s)')
        for ax in axes[i]:
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
        axes[i,0].set_ylim(0,3.2); axes[i,1].set_ylim(0,.8)
    axes[0,0].legend(frameon=False)
    axes[-1,0].set_xlabel('Time (s)'); axes[-1,1].set_xlabel('Time (s)')
    fig.suptitle('Shadow E rate and speed on fixed-policy steady trajectories\nMean over three fixed physics worlds; 0.5 s moving average',fontsize=11)
    fig.savefig(exp.OUT/'phase_rate_and_speed.png',dpi=170); plt.close(fig)
    if baseline_only:
        return
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    modes=list(exp.E_MODES)
    for j,seed in enumerate(exp.SEEDS):
        values=[next(r['mae_forward_velocity_mps'] for r in analysis['aggregates']
                     if r['condition']=='learned_'+mode and r['train_seed']==seed and r['scenario']=='steady') for mode in modes]
        ax.plot(np.arange(len(modes))+(j-1)*.045,values,'o-',linewidth=.8,markersize=4,label=f'Seed {str(seed)[-2:]}')
    clock=next(r['mae_forward_velocity_mps'] for r in analysis['aggregates']
               if r['condition']=='clock_clock' and r['train_seed']=='all' and r['scenario']=='steady')
    ax.axhline(clock,color='0.25',linestyle='--',linewidth=1,label='Clock-policy baseline mean')
    ax.set_xticks(np.arange(len(modes)),['Live E','Self replay','Clock donor\nE replay','Linear\nown rate','Linear\n1.667 Hz','Clock\nzero origin','Frozen'])
    ax.set_ylabel('Forward speed MAE (m/s; lower is better)'); ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
    ax.legend(frameon=False,ncol=2,fontsize=8)
    ax.set_title('Fixed E-policy interventions: every training seed retained')
    fig.savefig(exp.OUT/'phase_interventions.png',dpi=170); plt.close(fig)


def build():
    analysis=exp.read(exp.OUT/'analysis.json')
    sensitivity=exp.read(exp.OUT/'policy_phase_sensitivity.json')
    gait=exp.read(exp.OUT/'reward_gait_supplement.json')
    reward_cfg=exp.read(exp.OUT/'reward_configuration_supplement.json')
    cards=exp.read(exp.OUT/'statistical_cards/status.json')
    conclusions=exp.read(exp.OUT/'finding_status.json')
    assert analysis['status']==sensitivity['status']==conclusions['status']=='COMPLETED'
    assert analysis['verified_episodes']==180 and len(analysis['contrasts'])==9
    assert len(sensitivity['rows'])==48
    assert gait['status']=='COMPLETED' and gait['verified_episodes']==54
    assert reward_cfg['status']=='VERIFIED' and reward_cfg['training_configurations']==6
    assert reward_cfg['all_training_rewards_equal'] and reward_cfg['no_clock_gait_reward']
    assert cards['status']=='COMPLETED' and len(cards['results'])==9
    for row in reward_cfg['rows']:
        assert exp.prior.sha(row['path'])==row['sha256']
    assert analysis['exact_replay_phase_input_verified']
    assert analysis['numerical_repeatability']['status']=='COMPLETED'
    numerical=analysis['numerical_repeatability']
    figures(analysis)
    lines=['# 最佳E：反馈、相位推进与策略依赖诊断','',
        '固定原最佳E（seed20261026，SHA256 `'+exp.prior.ENCODER_SHA+'`），使用三个原PPO种子的501更新最终策略。新增180回合固定对照，另18回合数值重复性检查：每条件三个初态、恒速与横推各20秒。不增加PPO训练，不按结果选择轨迹、种子或检查点。','',
        conclusions['lead'],'',
        '![固定策略相位干预](<'+(exp.OUT/'phase_interventions.png').as_posix()+'>)','',
        '## 逐项判断','','| 假设 | 判断 | 证据与边界 |','|---|---|---|']
    for row in conclusions['findings']:
        lines.append('| '+row['title']+' | '+row['conclusion']+' | '+row['detail']+' |')
    lines+=['','## 对照含义','',
        '- live：实时E，包含动作→传感→E→动作的反馈。',
        '- self_replay：重放同策略、同初态无扰动轨迹的E相位。无扰动应复现，横推时切断对扰动的相位响应。',
        '- clock_replay：重放同初态clock策略轨迹上估计的E相位。来源同时改变频率与波形，不能单独识别反馈机制。',
        '- linear_self：保留自身E初始相位和3秒后平均频率，整段改为匀速相位。它同时移除启动阶段的变速、周期内波动和长期漂移，并切断反馈；不能将改善单独归因于周期内波形。',
        '- linear_nominal：在linear_self基础上，只将频率改成1/0.6 Hz。',
        '- clock：固定1/0.6 Hz，初始相位为0。与linear_nominal只相差相位起点。',
        '- frozen：保持初始相位，消除周期信息。属于强干预，退化不等于原模型错误。','',
        '所有记录通过原始SHA、相位与actor输入一致性、相位单位范数、每环境episode步索引核查。全部条件与clock基线已记录的初始位置、姿态、关节位置和身体速度配对一致；body_mass/body_ipos/geom_friction三项启动模型数组及相位以外配置也一致。','',
        f"self_replay在steady全20秒以及push前5秒，共18组轨迹/窗口与无扰动live原件核对。相位输入逐点一致，但物理/动作轨迹未全部通过预设最大误差1e-5（通过{analysis['strict_identity_passes']}/18），保留原失败判据。因此不将其称为精确恒等反事实，也不凭微小重放差异认定反馈机制。",'',
        f"在首个自重放结果可见后，另行锁定全部三个seed、两个场景各重复live一次（18回合），估计数值漂移，未重选模型/种子或更改九项比较family。重复live的最大逐回合MAE差为{numerical['max_abs_mae_repeat_difference_mps']:.8f} m/s；严格1e-5轨迹检查通过{numerical['strict_reproducibility_passes']}/18。重复检查用于限定结论，不能抹去原恒等检查失败。",'',
        '没有使用重放序列的未来值估计控制器状态；预录重放是一种离线诊断，不能直接当作在线部署方案。','',
        '## 全部运行时干预结果','',
        'MAE单位m/s、倾斜及航向单位rad。距离沿初始朝向计算。每行9回合包含3训练种子，不能当作9次独立训练。','',
        '| 策略/相位 | 场景 | MAE↓ | 实际速度 | 倾斜RMS↓ | 航向变化绝对值↓ | 初始方向位移↑ | 跌倒/n |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in analysis['aggregates']:
        if row['train_seed']=='all':
            lines.append(f"| {row['condition']} | {row['scenario']} | {row['mae_forward_velocity_mps']:.5f} | {row['mean_observed_forward_velocity_mps']:.5f} | {row['tilt_rms_rad']:.5f} | {row['abs_heading_change_rad']:.3f} | {row['initial_direction_displacement_m']:.3f} | {row['falls']}/{row['episodes']} |")
    lines+=['','## 三个训练种子的全部速度对比','',
        '以下为结果前锁定的完整九项family。差值=干预−对照，负数为误差下降。先在每个训练seed内部对3个初态配对取均值，再对3个训练seed计算95% t区间及BH q。n=3无法验证seed级正态性，推断限于探索性；q≥0.05或区间跨0不声称方向已确立。','',
        '| 比较 | 三seed差值(m/s) | 平均差 | 95% t区间 | BH q |','|---|---|---:|---|---:|']
    for row in analysis['contrasts']:
        values=', '.join(f'{v:+.5f}' for v in row['per_training_seed'].values())
        lo,hi=row['paired_t_95pct_ci_mps']
        lines.append(f"| {row['name']} | {values} | {row['mean_difference_mps']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {row['q_bh_nine_contrasts']:.4f} |")
    lines+=['','## 相位频率与步幅','',
        '![相位频率与速度](<'+(exp.OUT/'phase_rate_and_speed.png').as_posix()+'>)','',
        '统计3秒后样本；停顿阈值<0.2 Hz、快进阈值>3 Hz预先固定。触地用原3帧去抖；每周期前向行程为身体x速度积分，方向可能随航向变化。频率/速度滞后相关保存在JSON，只作描述，不作为反馈因果证据。','',
        '| 策略/输入 | 训练seed | 场景 | 输入平均Hz | shadow E平均Hz | E瞬时Hz标准差 | E停顿占比 | E快进占比 | 接触周期Hz | 每周期身体前向行程(m) |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in analysis['aggregates']:
        if row['train_seed']!='all' and row['condition'] in ('learned_live','clock_clock'):
            p=row['phase_diagnostics']; g=row['gait_diagnostics']
            fmt=lambda v:'—' if v is None else f'{v:.4f}'
            lines.append(f"| {row['condition']} | {row['train_seed']} | {row['scenario']} | {p['input']['mean_hz']:.4f} | {p['shadow_E']['mean_hz']:.4f} | {p['shadow_E']['std_hz']:.4f} | {p['shadow_E']['fraction_below_0_2Hz']:.2%} | {p['shadow_E']['fraction_above_3Hz']:.2%} | {fmt(g['mean_cycle_frequency_hz'])} | {fmt(g['mean_body_forward_travel_per_cycle_m'])} |")
    lines+=['','## 同状态相位依赖','',
        '对两种基线策略的全部steady原始状态，取3秒后每0.2秒样本。非相位输入保持逐值相同，将相位±15°/90°旋转、冻结或换clock，比较动作RMS差。覆盖每seed的clock/E在251、501及601原奖励/去pose检查点，共48个策略×状态来源组合。','',
        '下表按三个训练seed均值，动作RMS变化除以原输出的每关节跨样本波动RMS（样本拼接了三个世界的时间序列，包含世界间差异）。它衡量条件动作敏感度，不代表动作变化有益；不同历史策略在这些状态上属于离线评估，不能代替从头训练的因果对照。','',
        '| 检查点 | 状态来源 | +15°动作变化/原波动 | 冻结相位动作变化/原波动 | 换clock动作变化/原波动 |','|---|---|---:|---:|---:|']
    labels=list(dict.fromkeys(r['policy'] for r in sensitivity['rows']))
    for label in labels:
        for source in ('clock','learned'):
            rs=[r for r in sensitivity['rows'] if r['policy']==label and r['state_source']==source]
            values=[np.mean([r['changes'][change]['relative_to_action_temporal_rms'] for r in rs])
                    for change in ('rotate_plus15','frozen','clock')]
            lines.append('| '+label+' | '+source+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## 已完成奖励试验的步态补充','',
        '本节在奖励成绩已知后重算全部54条旧轨迹，属于事后描述，不新增假设检验，不作为因果中介分析。以下只列E三种奖励/预算条件；完整clock/E条件均保存在reward_gait_supplement.json。关节范围为3秒后各关节5%–95%分位范围，再对左右腿平均，不是关节极限。','',
        '| PPO seed | 条件 | 去前3秒速度(m/s) | 接触周期Hz | 每周期前进量(m) | 髋pitch范围(rad) | 膝范围(rad) |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for row in gait['aggregates']:
        if row['arm']=='learned':
            lines.append(f"| {row['train_seed']} | {row['mode']} | {row['after3_mean_velocity_mps']:.4f} | {row['mean_contact_cycle_frequency_hz']:.4f} | {row['mean_body_forward_travel_per_cycle_m']:.4f} | {row['hip_pitch_90pct_range_rad']:.4f} | {row['knee_90pct_range_rad']:.4f} |")
    lines+=['',
        '第三seed原E每周期前进量约0.222 m，clock约0.291 m；E频率反而略高。等预算601更新中，去pose后的E髋/膝范围及每周期行程均比原奖励大，但平均速度仍未完全达到目标。三个E种子去pose后髋/膝范围均增加，第一个种子出现超速和MAE略增。因此这支持“动作幅度/奖励取舍参与其中”的解释，不支持简单删除pose即可全面改好。上一轮MAE比较及交互的完整校正结果仍适用，不能由本节描述将其升级为可靠的跨seed改善。','',
        '## 任务目标及边界','',
        '六份原训练env.yaml的奖励配置逐项一致，且全部移除了foot_gait固定clock步态奖励。因此没有发现该奖励项仍按clock给E策略打分的冲突；这不排除其他奖励项与相位策略的取舍。静态核查记录于reward_configuration_supplement.json。','',
        '已核查heading_command=false、角速度指令为0；actor没有初始航向误差项，奖励惩罚当下角速度而不是累计偏航。因而航向偏差会累积，身体前向MAE小不保证沿最初世界方向前进得远。该目标差异能够解释指标分歧，不能独自解释E为何速度误差更大。','',
        '新增运行使用本机RTX4050、PyTorch2.6，原远端为PyTorch2.8。所有本轮条件在同一配置下配对。评测的三个world slots使用固定startup物理参数；新初态seed改变平地xy起点，初始关节/姿态/速度与上一轮三世界cohort相同。因此这是固定物理cohort上的配对机制诊断，不能称独立新物理cohort泛化验证。运行时干预只识别固定策略对相位输入的响应；不能排除校准后重训、更长预算、其他奖励或从头训练会改变结果。原实验、失败记录及模型均保留。','',
        '## 证据','',
        '- `prospective_protocol.json`：结果前条件、九项比较family和源码SHA。',
        '- `analysis.json`：180条原始SHA、全部初态/物理配对、恒等检查、每seed每场景指标与区间。',
        '- `policy_phase_sensitivity.json`：48个策略/状态来源组合及逐关节动作变化。',
        '- `reward_gait_supplement.json`：全部54条旧奖励轨迹的事后节律、每周期行程和关节范围分解。',
        '- `objective_audit.json`：航向控制配置、actor输入和奖励源码。',
        '- `reward_configuration_supplement.json`：六份原训练配置的奖励一致性、固定clock步态奖励缺席及来源SHA。',
        '- `numerical_control_amendment.json`、`numerical_repeatability.json`：事后技术误差检查协议及18条重复原始SHA/误差。',
        '- `finding_status.json`：逐项判断及证据边界。',
        '- `statistical_cards/`：九项结果卡、完整family的BH/证据强度核查；探索性诊断不作为强结论的写作交接。',
        '- `rollouts/`、`logs/`：原始观测、相位、动作、状态及日志。','']
    path=exp.OUT/'diagnosis.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    snapshot=exp.OUT/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ('g1_phase_feedback_probe_20261026.py','analyze_g1_phase_feedback_probe_20261026.py',
                 'g1_phase_feedback_controls_20261026.py','report_g1_phase_feedback_probe_20261026.py',
                 'analyze_g1_gait_tradeoff_20261026.py','cards_g1_phase_feedback_probe_20261026.py',
                 'write_g1_phase_feedback_findings_20261026.py'):
        (snapshot/name).write_bytes((exp.ROOT/'scripts'/name).read_bytes())
    assert exp.prior.sha(snapshot/'analysis_at_sensitivity.py')==sensitivity['analysis_script_sha256']
    exp.write('completion_verification.json',dict(status='COMPLETED',artifact_integrity='PASS',at=exp.prior.now(),
        verified_episodes=180,numerical_repeat_episodes=18,training_runs=0,training_seeds=3,contrast_family_size=9,
        self_replay_identity_checks=18,strict_identity_passes=analysis['strict_identity_passes'],
        scientific_identity_status='STRICT_IDENTITY_WARNING_PRESERVED',
        source_E_sha256=exp.prior.sha(exp.prior.ENCODER),
        report_sha256=exp.prior.sha(path),figure_sha256={p:exp.prior.sha(exp.OUT/p) for p in ('phase_interventions.png','phase_rate_and_speed.png')},
        sources={p:exp.prior.sha(exp.OUT/p) for p in (
            'prospective_protocol.json','analysis.json','policy_phase_sensitivity.json','objective_audit.json',
            'numerical_control_amendment.json','numerical_repeatability.json','reward_gait_supplement.json',
            'reward_configuration_supplement.json','finding_status.json','statistical_cards/status.json',
            'statistical_cards/stat_rigor_gate.json','statistical_cards/evidence_strength.json')}))
    print(path)


if __name__=='__main__':
    build()
