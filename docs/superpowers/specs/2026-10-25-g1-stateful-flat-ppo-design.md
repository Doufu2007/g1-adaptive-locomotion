# G1 新E相位模型：平地PPO三组对照

授权：2026-10-25用户“那现在是否可以开PPO进行对照？可以的话尝试一下”。执行者沿用此前已经要求的三组、等预算、冻结相位模型方案。以下数值为执行者在本轮结果产生前固定的实现决定。本轮是新平地有限预算适配，不更改此前540回合控制研究或本日离线相位实验。

## 问题、范围与候选

检验在相同旧clock actor起点上，用新E相位模型训练适配策略，是否降低平地速度误差且不增加跌倒；与直接替换比较适配是否有帮助。90%不是本轮探索性PPO的启动条件，也不声称旧相位质量门已通过。平地97.22%是已查看的三seed离线均值，不能当作本轮控制成功或盲选依据。

候选固定为E（observer_sensors_mixed）最小编号seed20261025的最终2000更新检查点，选择按编号，不按离线成绩。来源为已下载校验的 `deliveries/g1_stateful_phase_20261025/snapshot/results/g1_stateful_phase_20261025/training/observer_sensors_mixed_seed20261025/final.pt`，SHA256 `fdaad78ee4d923189f4291a0d9c8a02c0b726325d1caa2b96a2e76231c7848ca`。不再预训练、调参或重拟合标准化。所有PPO种子使用同一模型。

仅平地，不外推rough或真机。两臂均从 `logs/rsl_rl/g1_velocity/2026-08-10_09-35-45_baseline_full/model_9900.pt` 的同一actor及actor normalizer开始；同seed的critic和optimizer均为相同新初始化。测试clock→clock、clock→learned、learned→learned，直接替换使用本轮clock策略。

## 固定矩阵、预算和环境

- PPO种子20261901、20261902、20261903；各训练clock和learned，共6项。seed1/3先clock，seed2先learned，不按中途成绩改变。
- 每项64环境×24步×501更新＝769536环境transition；501更新最终模型统一评测，每50次更新保存完整恢复点。没有因低回报、跌倒或结果不理想的提前停止、追加预算或重启。
- 复用上一轮PPO超参、旧actor加载、动作/观测维数和仿真配置。学习率1e-3；两臂均移除foot_gait，其余奖励、命令0.3–0.8m/s前向行走、噪声、随机化、终止条件一致。actor和critic只改变2维phase来源，E额外运动输入不直接拼接到actor。
- 测试环境seed20262001–20262010，三场景steady/speed_change/push，20秒固定上限；每训练seed×组合30回合，9项评测、共270回合。场景和推扰幅度沿用封存的 `g1_closed_loop_metrics.py`。测试初态、物理随机化和命令按环境seed配对。训练全部完成并封存后才评测，不根据评测挑策略。
- 本地GPU串行运行。启动每个GPU子进程前核实nvidia-smi计算进程及真实命令行；其他项目GPU作业活跃时等待，不终止它们。不重新开机租用服务器，不关本地计算机。

## 在线相位接口与必要验证

E输入为29关节角、29后向差分速度、当前body重力、后向四元数body角速度，50Hz；严格使用原joint_names和checkpoint mean/std。接触、参考相位、命令和未来帧不进入E。每次reset后首帧速度和角速度置0；各TCN层历史清0，积分相位用首帧anchor初始化。

运行时以层内因果卷积缓存递推，必须与冻结 `PhaseNet` 全序列预测在CPU/GPU分别通过数值容差验证（CPU相位rtol=2e-4、atol=2e-4；GPU同2e-3，记录实际误差）。不修改权重或前向数学语义。actor、critic及日志在同一控制帧共用一次状态更新；部分环境reset不得改变其他环境。在线64维特征与离线SciPy实现逐帧对齐，四元数q/-q等价。统一验证有限值、50Hz和关节顺序。

encoder为eval、requires_grad=False，不进入PPO optimizer；初始/保存/结束检查状态SHA及源文件SHA。构造encoder时保留CPU RNG，防止改变配对策略初始化。旧源码不修改，只在新进程安装限定到该checkpoint的观察函数和加载器适配。

两臂共用float32，禁用matmul/cuDNN的TF32、启用cuDNN deterministic并关闭benchmark；在线验证与正式评测必须通过同一bootstrap保留这些设置。MuJoCo/Warp仿真仍不宣称跨硬件或中断前后bitwise复现。GPU smoke记录绑定全部科学源码、候选和包版本；代码变化后不能静默复用旧通过记录。

## 判据与失败树

主指标为完整20秒前向速度MAE，跌倒后剩余时间按零速度计入；跌倒率为共同主保护指标，同时报告存活、位移、推扰恢复（包括未暴露/删失）、相位/触地描述。不排除跌倒、站立、无周期或未恢复回合，不只报成功回合。

每一对比族：learned→learned对clock→clock，以及对clock→learned。每族两项：平地MAE与跌倒率；以3个配对PPO训练差值为重复，报告差值、95% t区间、双侧p和族内Holm校正。评测回合不能冒充独立训练。速度改善的预定分支为MAE差值95%区间上界<0且Holm p<.05，同时跌倒率差值95%区间上界<=0。跌倒率也满足上界<0且Holm p<.05才可单列跌倒改善。只有三次训练，区间不跨零仍只支持有限队列/预算的仿真比较。

- 两项满足：报告在固定平地预算下建立相应对照改善，不外推rough。
- MAE均值下降但区间跨零：描述性信号，尚未建立；不加预算补种子。
- MAE下降但跌倒保护失败：tradeoff，不宣布联合改善。
- 相位变化但控制不改善：如实报告，不以离线97.22%替代闭环结果。
- 技术smoke失败：保留日志和目录，定位后留痕修复再跑smoke；正式矩阵未启动前可以修技术接口。
- 正式技术失败：保留manifest/检查点/日志，仅从最新同身份完整恢复点恢复；无恢复点则标记阻塞并请求决定，不按成绩0步重启。仿真状态不连续恢复必须声明；丢弃计算单列开销。
- 配对、预算、SHA、原始轨迹或encoder冻结审计失败：不给完整实验结论；不删除失败回合或静默换模型。

## 工件与执行顺序

新增独立脚本 `scripts/g1_stateful_flat_ppo.py`（协议/矩阵/配置）、`scripts/g1_stateful_online.py`（在线特征/缓存）、`scripts/run_g1_stateful_flat_ppo_20261025.py`（smoke/训练/评测/汇总/队列），测试 `tests/test_g1_stateful_flat_ppo.py`。结果在 `results/g1_stateful_flat_ppo_20261025/`，源SHA、环境、命令、失败、逐次预算、最终policy、逐回合原始npz及全部比较保留。

CPU gold/状态/配置/矩阵测试→独立设计与代码检查→在线GPU数值对齐和两臂各2更新smoke→封存源码/配置/源权重→6训练→所有最终策略封存→9评测→配对和预算/冻结审计→完整报告。新测试seed已固定，不使用旧24回合离线test挑策略。

隔离决定：本仓库必要源码和数据大量未跟踪；沿用新脚本/结果目录隔离和SHA快照，不重置用户改动、不批量提交、不另建缺少这些资产的worktree。后续仅完成、故障或需要新决定时通知；长任务以本地队列和自动跟进持续执行。
