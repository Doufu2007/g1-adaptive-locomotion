"""Write UTF-8 findings from the completed, verified diagnostic outputs."""
from scripts import g1_phase_feedback_probe_20261026 as e


def run():
    x=e.read(e.OUT/'analysis.json')
    assert x['status']=='COMPLETED' and x['verified_episodes']==180
    numerical=x['numerical_repeatability']
    assert numerical['status']=='COMPLETED' and numerical['episodes']==18
    c={r['name']:r for r in x['contrasts']}
    def row(condition,seed='all'):
        return next(r for r in x['aggregates'] if (r['condition'],r['train_seed'],r['scenario'])==(condition,seed,'steady'))
    a,b=row('learned_live',20261903),row('clock_clock',20261903)
    findings=[
        dict(title='E变慢或停滞拖慢步态',conclusion='简单解释不获支持',
             detail=f"第三seed的E平均{a['phase_diagnostics']['shadow_E']['mean_hz']:.3f} Hz，高于clock输入1.667 Hz；无低于0.2 Hz的停顿。每个完整接触周期的身体前向行程约{a['gait_diagnostics']['mean_body_forward_travel_per_cycle_m']:.3f} m，clock约{b['gait_diagnostics']['mean_body_forward_travel_per_cycle_m']:.3f} m。低速主要表现为每周期行程偏小，不能仅由平均频率解释。"),
        dict(title='当前实时E反馈有害',conclusion='横推对照未见一致支持；训练期反馈仍未识别',
             detail=f"自身相位预录重放在push的三seed MAE差为−0.00554/+0.00018/+0.00218 m/s，均值{c['feedback_under_push']['mean_difference_mps']:+.5f}，BH q={c['feedback_under_push']['q_bh_nine_contrasts']:.3f}。相位输入逐点一致，但18项严格物理恒等均未通过；重复live最大单回合MAE差为{numerical['max_abs_mae_repeat_difference_mps']:.6f} m/s。不能宣称精确反事实或彻底排除反馈。"),
        dict(title='周期内推进、平均频率与相位起点',conclusion='有种子依赖影响，均未单独修复慢走',
             detail=f"匀速化平均MAE差{c['phase_shape']['mean_difference_mps']:+.5f}（q={c['phase_shape']['q_bh_nine_contrasts']:.3f}）；再改为1.667 Hz的差为{c['phase_rate']['mean_difference_mps']:+.5f}（q={c['phase_rate']['q_bh_nine_contrasts']:.3f}）；起点改0的差为{c['phase_origin']['mean_difference_mps']:+.5f}（q={c['phase_origin']['q_bh_nine_contrasts']:.3f}）。匀速化同时改变启动变速、漂移与反馈，不能称纯波形因果；第三seed直接接clock后仍仅{row('learned_clock',20261903)['mean_observed_forward_velocity_mps']:.3f} m/s、MAE {row('learned_clock',20261903)['mae_forward_velocity_mps']:.4f}。"),
        dict(title='策略完全忽略相位',conclusion='同状态动作及冻结干预均不支持',
             detail='E501在E状态上旋转15°，动作RMS差为原输出跨样本波动的13.3%/13.2%/15.2%。冻结相位后三seed MAE均上升（约0.135/0.214/0.187 m/s），小样本完整family q=0.150，仍按描述性结果。策略也能用关节与动作反馈维持运动，敏感度不等于相位使用有效。'),
        dict(title='学习后的动作幅度与奖励取舍',conclusion='目前最有证据的解释线索，尚无唯一根因',
             detail='第三seed原E髋pitch/膝活动范围约0.241/0.455 rad，clock约0.346/0.615；每周期行程约0.222对0.291 m。上一轮等预算601更新去pose后，第三seed行程从原奖励0.241增至0.258 m，髋膝范围同步增大。三个E种子关节范围均增大，但第一seed超速、MAE略增；相位×pose交互BH q=0.131，不能将奖励解释升级为已确立的跨seed因果。'),
        dict(title='clock预训练习惯与适配路径',conclusion='仍可能参与，当前未识别其独立因果',
             detail=f"同一实时E接到clock组策略时平均MAE {row('clock_live')['mae_forward_velocity_mps']:.4f}，E组策略则为{row('learned_live')['mae_forward_velocity_mps']:.4f}。第三seed切回clock仍慢，提示差异已存在于学习后的控制行为；没有从头训练与相同初始化史的独立对照，不能单独归因于clock习惯、PPO算法或E模型本身。"),
        dict(title='航向漂移与评测目标',conclusion='训练目标差异已核实，可解释部分指标分歧',
             detail=f"heading_command=false，actor无初始航向误差，奖励追踪即时角速度而非累计偏航。第二E seed MAE {row('learned_live',20261902)['mae_forward_velocity_mps']:.4f}，但航向变化{row('learned_live',20261902)['abs_heading_change_rad']:.2f} rad、20秒初始方向位移{row('learned_live',20261902)['initial_direction_displacement_m']:.2f} m。身体前向速度接近目标不能保证沿初始世界方向走得远，该目标差异不能独自解释第三seed的速度缺口。"),
        dict(title='适配器数值或clock步态奖励冲突',conclusion='已测范围内未发现',
             detail='上一轮在线/离线、特征顺序、重置与缓存及真实PPO采样3072帧检查已通过。本轮六份原训练奖励配置（含函数tag）逐项相等，全部无foot_gait固定clock步态奖励。这些检查不排除所有接口设计问题，也不抹去本轮严格轨迹恒等失败。'),
    ]
    lead='本轮证据更支持把排查重点放在PPO适配后形成的步态幅度及奖励/目标取舍：最差seed的E相位不慢，但每个完整落脚周期前进量偏小；直接切回clock仍慢走。重放与匀速化未形成一致的跨seed修复。没有锁定唯一上游根因，固定策略干预也不能排除训练期间的反馈耦合或clock预训练依赖。'
    assert sum('\u4e00'<=v<='\u9fff' for v in lead)>50 and '?' not in lead
    e.write('finding_status.json',dict(status='COMPLETED',at=e.prior.now(),lead=lead,findings=findings,
        analysis_sha256=e.prior.sha(e.OUT/'analysis.json'),
        numerical_repeatability_sha256=e.prior.sha(e.OUT/'numerical_repeatability.json'),
        generator_sha256=e.prior.sha(__file__),
        main_contrast_snapshot=[dict(name=r['name'],mean_difference_mps=r['mean_difference_mps'],q=r['q_bh_nine_contrasts']) for r in x['contrasts']],
        numerical_limitation='18项严格自重放及18项重复live均为0/18通过1e-5门槛。相位输入逐点一致，但不能宣称精确物理反事实。'))
    print('UTF-8 findings written and checked.')


if __name__=='__main__':
    run()
