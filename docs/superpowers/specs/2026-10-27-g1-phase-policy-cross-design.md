# 阶段C：冻结策略×在线相位交叉，定位直接输入与训练后权重的作用

用户已授权四步原因排查，A/B均未形成优越候选后检查输入/训练路径。本阶段不新增PPO更新，不改变A/B冻结来源，不以再加训练seed碰成绩。所有新选择在新回合前封存。

## 已有结果与目标
B新增3×501和135回合完成，A同预算clock_low/E_low重评分且初态/物理/原始SHA审计通过。steady MAE：clock .0466504、E .0550103、reference .0477986。reference-clock均差+.0011482，seed差[-.013615,+.026447,-.009387]，95%tCI[-.05353,.05583]、Holm1；reference-E -.0072117，CI跨0。reference航向增+.2404rad、初始方向位移-.8294m，保护门不通过。
B只能说明这一种参考未稳定赢clock，不能断言全部参考无用/根因就是训练。E/参考均有seed反转，单圈完成率不足以解释行走。
补充旧checkpoint静态核查（0采样）：九个低LR训练的动作std均由均值.419降至约.400..408，无std爆炸；actor归一化count约9.73亿且新增769536，整体均值/方差仅小幅变化。这些是描述信号，不等于已因果排除。
源奖励代码证实command是body速度且heading_command False；track_angular_velocity只跟踪瞬时转速，未跟踪初始方向误差。小yawrate可能累计偏航，但必须与相位/策略原因分开，不贸然加reward再训练。

## 固定交叉矩阵
四个策略来源：source9900（0更新）、A clock_low501、A E_low501、B reference_low501；各配三种输入clock/E/reference。12groups，各9个独立并行world。9slot固定为3个原训练seed20261901/02/03各3个eval初态，eval种子20263611/12/13；slot=index(trainseed)*3+index(eval)。source同一策略权重在三seed槽重复，只用于配对起点，不能称3独立source训练。
每组20秒steady .5m/s、50Hz，108回合、最多108000环境transition（按终止实际列数记）；0PPO更新。新物理9world cohort，各12组完全相同cfg物理和初态，核对body_mass/body_ipos/geom_friction与每slot起点position/quaternion/joint_pos/velocity≤1e-6。不能把9world结果与A/B3或10world的分数绝对值直接等同。
策略seed只决定该slot读取哪个已封存501checkpoint导出的ONNX，绝不重新初始化；source每slot相同model0 ONNX。输入采用A/B原实现，E checkpoint不变，reference目标不变；仅phase两列变化。其余观测、奖励（评测无优化）、物理、命令、reset规则保持共同。初始yaw0、标准play无观测噪声，关闭randompush。fall后只记录到终止，完整20秒MAE余段按0速度；不补回合。
所有组以同一固定seed20263600构造，评测初态按slot依次public reset。无obs历史缓存污染，actor phase读取与日志重复调用需allclose且不推进。每world使用属于slot的actor，ONNX session单线程；不按batch改变策略输入shape，逐world [1,98]推理。
每次group独立进程，按Random(20263690)固定打乱12组顺序；一个GPU顺序运行，不重复启动。不在中间结果出现后改矩阵或跳过差组。

## 预定比较与解释
主指标完整20秒body-forward MAE，最终终点固定已有501，无择优检查点。每3初态先按trainseed聚合，展示三seed差值及均值，探索性不做显著优越结论。
固定六项描述性比较：E_policy(clock)-E_policy(E)为E输入更换效应；E_policy(clock)-clock_policy(clock)为同clock输入下权重差；ref_policy(clock)-ref_policy(ref)和ref_policy(clock)-clock_policy(clock)同理；source(E)-source(clock)、source(ref)-source(clock)为起点兼容性。完整12矩阵全部报告，所有保护指标同A/B。
若更换clock一致减少E误差且保护条件不恶化，支持当前在线输入是可干预因素；不直接证明训练E无用，因为部署输入改变了。若同clock输入E策略仍落后且source(E)好于E_policy(E)，支持策略更新路径有损失。可能两者同时存在；交互不能归于单一纯主效应。参考输入的同类结果提供构造对照，不能称真值。
候选修正需对应单因素预期、先封存后同预算clock/E成对训练；不按本阶段单次最高分直接宣布稳定优越。若本阶段仍不区分，报告未识别，不无限新增任意干预追赢。整体目标尚未完成，不暂停跟进。

## 来源与审计
锁A/B protocol、analysis、源actor/E模型、参考校准raw及代码、12组各policy SHA；保存新脚本/tests/spec/plan及包版本。新结果raw NPZ逐文件SHA；score重算，不抄未审计数字。训练seed是聚类单位；108回合非n108独立算法重复。技术smoke只用合成obs路由/CPU单元，不另新增GPU回合；实际group本身的shape/phase/finite/physics断言遇错立刻保留失败，按技术修订恢复，不重复有效回合。

工程记录：raw文件及score用唯一episode_id=20263700+slot编号避免三个trainseed重复eval_seed导致字典覆盖；初态public reset仍用设计中的20263611/12/13，metadata显式保存train_seed/eval_seed/slot/episode_id。
