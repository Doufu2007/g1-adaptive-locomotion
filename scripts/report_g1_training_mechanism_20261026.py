"""Write a Chinese evidence report after all bounded diagnostic work finishes."""
from datetime import datetime,timezone
import json
import numpy as np
from scripts.audit_g1_training_mechanism_20261026 import ROOT,OUT,read,sha,write


def run():
    training=read(OUT/'training_audit.json')
    drift=read(OUT/'policy_drift_supplement.json')
    gait=read(OUT/'gait_chronology.json')
    sample=read(OUT/'sampling_analysis.json')
    noise=read(OUT/'action_noise_analysis.json')
    speed=read(OUT/'speed_priority/analysis.json')
    smoke=read(OUT/'speed_priority/smoke_repeat_verification.json')
    assert all(x['status']=='COMPLETED' for x in (training,drift,gait,sample,noise,speed))
    seedlist=(20261901,20261902,20261903)
    lines=['# 最佳 E 经 PPO 适配后慢走：新增机制诊断', '',
        '本轮完成：六组原训练的66个检查点与六份完整事件流审计、120条既有步态时间线复算、36回合固定策略动作采样对照，以及三种子各100更新的速度奖励对照和9回合新评测。全部为探索性诊断，保留原实验；服务器未重启，本机不关机。', '',
        '目前能定位到的是：两个 E 种子在后半段 PPO 中形成较小的髋膝幅度和每周期行程；并非 E 相位普遍停转。去掉评测中的随机动作，不能充分解释这个现象。训练设置存在适配受限的线索，奖励取舍的具体作用见下方单变量对照。尚未识别唯一根因。', '',
        '## 1. 学习率很早触底，但这是六组共有设置', '',
        '初始学习率为0.001。六组保存的首次更新结束检查点已经是0.00001；后续绝大多数更新也在下限。原日志没有保存每个训练小批次的 KL，不能把离线状态池上的 KL 当成训练 KL，也不能据此断言发生了某一次具体失控。', '',
        '| PPO种子 | clock在下限的更新占比 | E在下限的更新占比 | 首次更新clock动作RMS变化 | 首次更新E动作RMS变化 |',
        '|---|---:|---:|---:|---:|']
    for seed in seedlist:
        fractions=[];changes=[]
        for arm in ('clock','learned'):
            e=next(x for x in training['events'] if (x['train_seed'],x['arm'])==(seed,arm))
            values=[v['value'] for v in e['scalars']['Loss/learning_rate']]
            fractions.append(float(np.mean(np.asarray(values)<=1.00001e-5)))
            c=next(x for x in training['checkpoints'] if (x['train_seed'],x['arm'],x['updates'])==(seed,arm,1))
            changes.append(c['action_rms_change_from_source'])
        lines.append(f'| {seed} | {fractions[0]:.1%} | {fractions[1]:.1%} | {changes[0]:.4f} | {changes[1]:.4f} |')
    lines.extend(['',
        '解释边界：首次保存时 E 的动作变化在三个种子中均大于 clock；早期变化与长期低学习率值得继续检验。但 E 第一种子到251更新时仍接近目标速度，因此“第一轮改变直接决定最终慢走”不被现有时间线充分支持。没有改变学习率/初始更新策略的正式对照，不能称其已确诊。', '',
        '## 2. 归一化漂移和共同梯度裁剪的简单解释不成立', '',
        f"原actor归一化计数为{training['source_normalizer_count']:,}。501更新新增样本只占约0.079%；统计基本沿用原策略。固定最终网络、换回原归一化统计，输出RMS变化仅约0.00009–0.00018，而总网络输出变化约0.17–0.21。因此归一化统计漂移本身无法解释大的动作变化。继承的统计是否适合新步态仍未用部署对照检验，相位sin/cos本身尺度匹配。", '',
        '源码中的actor与critic梯度分别裁剪，因此“critic大梯度通过共同裁剪直接压小actor更新”的简单机制不成立。fresh critic仍可能影响优势估计；现有证据没有识别这条路径。', '',
        '统一状态池上，三个最终E策略将相位改动15°后的动作RMS变化为0.0540–0.0575；速度指令增加0.1 m/s后的变化为0.0253–0.0271，与clock同量级。因此“网络完全忽略E/速度指令”的简单解释不符合这些固定状态检查；这不等于策略在真实步态上有效利用了它们。', '',
        '## 3. 小幅度步态是在后续训练中形成，而且有种子差异', '',
        '以下是全部对应恒速轨迹，每格为10个初态的平均。速度去掉前3秒；MAE按完整20秒。周期行程是左脚触地到下一次左脚触地的身体前向速度积分，不是单步长度或沿世界直线的位移。', '',
        '| E种子 | 更新数 | MAE(m/s) | 平均速度(m/s) | 每周期行程(m) | 髋pitch范围(rad) | 膝范围(rad) |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for r in gait['aggregates']:
        if r['arm']=='learned':
            lines.append(f"| {r['train_seed']} | {r['updates']} | {r['full20_mae_mps']:.4f} | {r['after3_mean_velocity_mps']:.4f} | {r['cycle_forward_travel_m']:.3f} | {r['hip_pitch_90pct_range_rad']:.3f} | {r['knee_90pct_range_rad']:.3f} |")
    lines.extend(['',
        '第一、第三种子从251到501更新同时出现行程缩短、关节幅度减小和速度变差；第二种子相反。三个clock种子的MAE均改善。这里的共变定位了慢走的动作表现，不能单凭相关认定哪个奖励或优化步骤造成它。', '',
        '## 4. 加回训练时的随机动作，不能一致修复 E', '',
        '固定501策略，三个PPO种子×clock/E×平均动作/原高斯采样×三个初态，共36回合。标准差取各检查点，配对使用同一标准高斯序列。环境初始状态、物理配置、动作注入、相位输入和全部原始SHA均已核对。此对照恢复条件动作分布，没有完整复现训练观测噪声及命令/物理分布。', '',
        '| E种子 | 平均动作MAE | 加回随机动作MAE | 平均动作速度 | 随机动作速度 |',
        '|---|---:|---:|---:|---:|'])
    def sgroup(seed,arm,mode):
        return next(r for r in sample['aggregates'] if (r['train_seed'],r['arm'],r['mode'])==(seed,arm,mode))
    for seed in seedlist:
        a,b=sgroup(seed,'learned','deterministic'),sgroup(seed,'learned','sampled')
        lines.append(f"| {seed} | {a['mae_forward_velocity_mps']:.4f} | {b['mae_forward_velocity_mps']:.4f} | {a['mean_observed_forward_velocity_mps']:.4f} | {b['mean_observed_forward_velocity_mps']:.4f} |")
    a,b=sgroup('all','learned','deterministic'),sgroup('all','learned','sampled')
    lines.extend(['',f"E平均MAE {a['mae_forward_velocity_mps']:.4f}→{b['mae_forward_velocity_mps']:.4f}；两个种子变差，最慢的第三种子略有改善。clock平均MAE 0.0508→0.1038。36回合无跌倒。E的速度改善不一致，也没有修复总体偏航；不得称随机动作有益或评测方式是主要根因。", '',
        '| 预定比较（MAE差） | 均值(m/s) | 95% t区间 | BH q（三项family） |',
        '|---|---:|---|---:|'])
    for r in sample['contrasts']:
        ci=r['paired_t_95pct_ci_mps']
        lines.append(f"| {r['name']} | {r['mean_difference_mps']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r['q_bh_three_contrasts']:.4g} |")
    lines.extend(['','统计单位只有三个训练种子；t区间所需的种子效应分布假设无法检验。回合数、时间点和检查点均不增加独立训练样本量。', '',
        '## 5. 平滑奖励主要受探索噪声影响，但不是 E 特有证据', '',
        '对36回合完整分解 Δ动作=Δ平均动作+Δ噪声，逐项验证平方和恒等式，保留交叉项。以下均为去前3秒、每时刻的加权action_rate奖励，不是完整训练回报。', '',
        '| 策略 | 部署动作 | 平均动作项 | 噪声项 | 交叉项 | 合计 |',
        '|---|---|---:|---:|---:|---:|'])
    for r in noise['aggregates']:
        if r['train_seed']=='all':
            lines.append(f"| {r['arm']} | {r['mode']} | {r['mean']:.4f} | {r['noise']:.4f} | {r['cross']:.4f} | {r['total']:.4f} |")
    lines.extend(['',
        '采样时总惩罚约−0.52，与原训练末期−0.55量级接近；确定性评测约−0.025。平均动作会随过去噪声反馈而改变，所以交叉项不为零，不能把噪声项比例直接当成独立因果贡献。该现象解释训练和评测的平滑奖励为何差很多，尚不能证明它使E比clock更差。', '',
        '## 6. 单变量对照：只提高速度奖励相对权重', '',
        '在看到原结果后明确锁定一次新探索：每个E种子从原501检查点恢复全部网络、优化器和学习率，续训100更新，只把速度奖励权重1改2。与已知的原奖励601缓存作等预算对照。两次2更新smoke仅作技术预检，不作科学样本；新正式训练总transitions=460800。三种子的起始状态、YAML唯一差异、encoder冻结和ONNX导出均核对。新旧评测配置完全一致。', '',
        '| 种子 | 原奖励601 MAE | 速度权重2 MAE | 原奖励速度 | 权重2速度 | 原奖励周期行程 | 权重2周期行程 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    def wgroup(seed,arm):
        return next(r for r in speed['aggregates'] if (r['train_seed'],r['condition'])==(seed,arm))
    for seed in seedlist:
        a,b=wgroup(seed,'original_reward'),wgroup(seed,'speed_priority')
        lines.append(f"| {seed} | {a['mae_forward_velocity_mps']:.4f} | {b['mae_forward_velocity_mps']:.4f} | {a['after3_mean_velocity_mps']:.4f} | {b['after3_mean_velocity_mps']:.4f} | {a['cycle_forward_travel_m']:.3f} | {b['cycle_forward_travel_m']:.3f} |")
    r=speed['contrast'];ci=r['paired_t_95pct_ci_mps']
    lines.extend(['',f"唯一预定新比较：MAE差={r['mean_difference_mps']:.4f} m/s，95% t区间=[{ci[0]:.4f}, {ci[1]:.4f}]，p={r['two_sided_p']:.4g}（此新family仅1项；此前实验后的选择性探索背景仍需保留）。", '',
        '全部保护指标：', '',
        '| 条件（三种子共9回合均值） | 跌倒 | 倾斜RMS(rad) | 航向绝对变化(rad) | 初始方向位移(m) |',
        '|---|---:|---:|---:|---:|'])
    for arm in ('original_reward','speed_priority'):
        r=wgroup('all',arm)
        lines.append(f"| {arm} | {r['falls']} | {r['tilt_rms_rad']:.4f} | {r['abs_heading_change_rad']:.4f} | {r['initial_direction_displacement_m']:.3f} |")
    translations=dict(SUPPORTED_CONDITIONAL_SENSITIVITY='本次三个种子均降低MAE，区间在0以下且保护指标通过，支持这些策略对速度奖励相对权重敏感。此结果不能证明原501慢走的唯一原因，也没有建立E优于clock。',
        SPEED_GUARDRAIL_TRADEOFF='平均速度误差降低，但至少一项保护指标越过预定边界；结果表明存在速度与姿态/方向之间的取舍，不能称整体修复。',
        INCONCLUSIVE_MIXED_OR_IMPRECISE='平均值有下降，但种子方向或区间不满足预定支持标准；未见一致修复证据。不能把相对奖励权重确诊为唯一原因。',
        NO_REPAIR_AT_THIS_WEIGHT='该权重档位本次没有降低平均速度误差，未支持它能一致修复当前慢走。它不否定所有奖励机制；三个种子方向混合且区间较宽，不能把一次未成功外推为普遍无效。')
    lines.extend(['',translations[speed['decision_branch']], '',
        '## 当前归因及边界', '',
        '1. 已定位的直接表现：两个种子在后续适配中走成小幅度、短周期行程，导致速度偏低；另一个种子（20261902）能学到接近目标速度，说明不是E接口必然限制速度。',
        '2. 下一步优先核查PPO的起始更新及适配路径。学习率首轮触底、首轮E策略漂移更大，以及同一奖励修改下的种子反应相反，都是线索；尚无改变更新策略或初始化critic的控制实验，不能确诊。奖励取舍仍可能参与，但单纯提高速度权重并未一致修复。',
        '3. 当前不支持作为充分解释：单纯关闭部署动作噪声、归一化统计漂移、actor/critic共同梯度裁剪。适配器数值正确性结论仍限于既有测试。',
        '4. 所有新比较为三种子、固定预算及一次选定权重；缓存对照已知且本机物理不保证逐位复现。原先相位重放的严格物理恒等失败仍保留，不将新结果写成精确物理反事实。',
        '5. 本轮到此停止，不据成绩试更多权重/挑选检查点。若继续定位起点机制，最有区分力的是从同一原策略比较受限首轮更新与原更新，并独立检查critic初始化；须另立预算和协议，避免与本次权重试验混在一起。', '',
        '静态审查：代码扫描无critical项；seed_audit对继承训练路径提示PYTHONHASHSEED和cuDNN确定性配置未完整覆盖，exit=1。为保持与缓存对照一致，本轮没有事后修改这些设置，也不把这一门写成通过。因此结论限于已有固定种子与匹配初始状态下的探索性结果，严格训练复现和论文级因果归因尚未通过。', '',
        f"两次既有2更新短预检进一步显示：起始条件一致，但最终actor/critic并非逐位相同；同一状态池动作RMS差={smoke['same_pool_action_rms_difference']:.6f}，最大绝对差={smoke['same_pool_action_max_abs_difference']:.6f}。这只量化了同进程短预检，不能估计100更新结果的波动，也不能说新的权重效果完全由数值噪声造成。", '',
        '统计机检：比较覆盖和多重校正检查通过；三种子效应方向翻转、n=3分布假设未知，以及缓存结果早于新方案，使证据卡保持REVISION_REQUIRED。报告仅交付诊断观察，不作为确认性或论文级因果结论；这些失败记录全部保留，不据此自动扩跑。Rscript不可用，未宣称R交叉核验通过。', '',
        '## 文件证据', '',
        '- training_audit.json：全部原检查点和事件流、归一化、学习率及ONNX一致性。',
        '- policy_drift_supplement.json：统一状态池上的动作/KL与归一化替换诊断。',
        '- gait_chronology.json：完整120条251/501轨迹。',
        '- sampling_analysis.json / action_noise_analysis.json：完整36回合、三项比较及奖励分解。',
        '- speed_priority/prospective_protocol.json / analysis.json：运行前锁定来源、九条新轨迹、等预算对照及保护指标。',
        '- statistical_cards/：比较family、证据边界和统计机检。', ''])
    (OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    files=['training_audit.json','policy_drift_supplement.json','gait_chronology.json','sampling_analysis.json',
           'action_noise_analysis.json','speed_priority/analysis.json','speed_priority/smoke_repeat_verification.json',
           'diagnosis.md','statistical_cards/status.json']
    write('completion_verification.json',dict(status='COMPLETED',at=datetime.now(timezone.utc).isoformat(),
        report_sha256=sha(OUT/'diagnosis.md'),original_training_checkpoints=66,old_gait_episodes=120,
        new_sampling_episodes=36,new_speed_priority_training_runs=3,new_speed_priority_eval_episodes=9,
        no_budget_extension_or_outcome_selection=True,no_remote_restart_or_local_shutdown=True,
        strict_reproducibility_gate_passed=False,publication_causal_claims_ready=False,
        artifacts=[dict(path=f,sha256=sha(OUT/f)) for f in files]))
    print(json.dumps(dict(status='COMPLETED',report=str(OUT/'diagnosis.md'),speed_branch=speed['decision_branch']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    run()
