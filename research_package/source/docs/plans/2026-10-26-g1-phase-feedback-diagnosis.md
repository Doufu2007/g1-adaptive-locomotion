# 最佳E反馈、相位形状和策略依赖诊断

用户已授权按剩余假设实际尝试并判断原因。顺序执行，复用原最佳E和三个原PPO种子，不追加PPO训练。

## 固定设计

- 使用原501更新最终策略：clock/E各三seed；原检查点与历史结果不修改。
- 新初态seed20262811/12/13；0.5 m/s、20秒，steady与5/10秒世界横向±0.3 m/s扰动。
- E策略：live、self_replay、clock_replay、linear_self、linear_nominal、clock、frozen，共7条件。
- clock策略：clock、live、frozen，共3条件。总180回合。三训练seed为统计单位，不把回合当独立训练重复。
- self_replay使用同策略、同初态的steady预录相位。steady为实现恒等检查；push才切断相对于扰动的反馈。相同初态无扰动的自身重放不能独自检验反馈危害。
- clock_replay使用同seed的clock策略steady轨迹上重放E获得的相位，不按成绩选择来源；它同时改变相位波形/频率/来源，不能单独证明反馈机制。
- linear_self保留自身steady E相位起点及3秒后的平均净进展频率；linear_nominal仅将频率改为1/0.6 Hz；clock再将起点改为0。分别检查相位形状、频率和起点。
- frozen保持初始相位向量，作为去除周期信息的强干预；可能超出策略训练分布，不把其退化单独解释为原E有错。
- 所有相位均传给actor/critic；同时记录实际动作输入、E shadow、相位净进展。保存逐回合原始SHA与初态/物理配对。

## 判据与完整比较family

主指标为固定20秒身体x速度MAE；同时报告root倾斜、横向速度、航向变化、沿初始方向位移、跌倒、扰动后窗口。速度改善伴随其他指标恶化，不能称整体控制改善。

预先固定九项MAE比较：E self_replay−live（push）；E clock_replay−live（steady）；E linear_self−live（steady）；E linear_nominal−linear_self（steady）；E clock−linear_nominal（steady）；E frozen−live（steady）；clock frozen−clock（steady）；E live−clock策略clock（steady）；clock策略live−clock（steady）。全部按三个训练seed配对均值、95%t区间和同一family BH校正，探索性解释。

记录频率分位数、停顿（<0.2Hz）、快进（>3Hz）、连续停顿长度、触地周期和速度滞后相关；相关只作描述。离线对同一批实际98维actor输入替换相位/小角旋转，比较251/501/601检查点的动作变化，评估相位依赖而非以动作敏感度直接证明控制质量。

## 执行

- [x] 合成测试：重放按环境episode步索引、圆周跨界、匀速化保持频率与起点；停止与缺失来源拒绝。3项通过。
- [x] 独立脚本与协议封存，完成短仿真和重放恒等检查；原锁定计划及源码另存source_snapshot。
- [x] 串行完成180回合，保留原记录；未按成绩重选种子/轨迹。
- [x] 验证SHA、记录的初态与物理数组配对、steady自重放及push前一致性。相位输入逐点一致，严格物理恒等0/18通过，保留失败判据。
- [x] 完成48项同状态策略相位敏感度、全部九项比较和逐项结论，输出报告及两图。

2026-10-26完成。首个严格重放问题可见后，追加固定的18回合技术重复检查，全部完成；无PPO追加训练，模型、原始九项family与训练种子不变。重复live的严格1e-5检查亦为0/18通过，最大单回合MAE差0.000715162 m/s。180条干预和18条重复原始SHA校验通过；完成状态与科学限制分开记录，未将精确相位输入重放当作精确物理反事实。三训练seed的全部九项BH q≥0.150，结果为探索性诊断。报告：results/g1_phase_feedback_probe_20261026/diagnosis.md。

边界：固定策略的运行时相位干预不等同于从头训练；不会单独识别clock预训练历史的因果效应。本地GPU/PyTorch与原远端不同，采用本地配对。
