# G1 语义分流：粗糙地形修订

2026-11-03，用户明确要求将主实验从平地切换至粗糙地形。本修订独立于 2026-10-26 平地协议及其 1024 环境资源修订；原结果和冻结源码保留。

研究问题、四种 Actor、随机分组、指令范围、PPO、Critic、归一化、奖励和最终 checkpoint 选择规则沿用已有协议。所有新正式策略从头训练；不使用平地 pilot 或中断正式 run 的权重。

## 冻结范围

- 任务：`Unitree-G1-Rough`，显式 `terrain_mode=blind_rough`。
- 物理地形：原生 G1 rough generator，训练保留 terrain curriculum；固定指令范围并关闭 command curriculum。
- Actor 仍为 98 维：身体状态 93；指令 3＋固定 0.6 秒解析相位 2。移除 Actor/Critic 的 height_scan 和无用地形扫描传感器。Critic 保留其他特权观测。
- 保留原生 gait reward；Actor 和 Critic 使用相同固定 clock。学习相位不属于本次 12-run 队列的因子。
- `nconmax=None` 使用仿真器自动分配接触容量，避免原生训练的 48 接触上限；其余参数通过 GPU smoke 核查。
- 原始 MLP、等预算 MLP、语义分流、固定随机分流；正式种子 20262101/20262102/20262103。
- 每策略 1024 环境 × 24 步 × 10001 更新 = 245,784,576 transitions；12 次合计 2,949,414,912。

## 执行和评估

独立部署根目录 `/root/autodl-tmp/g1_semantic_split_rough_20261103`，源码子目录 `g1_semantic_split`，正式输出 `formal_1024`，工程输出 `rough_smoke_1024`。复用既有虚拟环境，不修改其他 GPU 服务。

先让四种 Actor 各执行 1024 环境 × 2 次 PPO 更新，核查真实观测布局、有限梯度与动作、checkpoint、ONNX parity 和资源共存。这只替代切换地形后的工程启动检查，不是完整 pilot 或方法性能证据。平地的四组 1000 更新 pilot 保留。

工程检查全部通过后，队列执行 12 次正式训练，再封存全部策略、运行 holdout、CPU 延迟和统计分析。每个 worker 使用同一个 rough adapter；协议包含 adapter 与执行引擎 SHA，源码包含逐文件 SHA。

粗糙训练的 generator seed 显式等于该 run 的种子；评估使用既有公共 construction seed，配对策略共用地图和 reset 序列。30 个 holdout episode seeds 是重复测量，不代表 30 张独立生成地图。play 模式关闭 terrain curriculum，使用原生粗糙评估地形。

固定时长 20 秒，六组指令、七个场景，失败尾部计入误差；独立统计单位仍为训练种子。完整训练前不产生优势结论。

## 与学习相位研究的关系

2026-10-31 最新 E 实验采用特定 random_rough 地图分布、source9900 续训、仅前向指令，并删除 foot_gait；它与本轮原生 rough 多地形、全速度指令、从零训练属于不同批次。不能拼接为完整的“相位 × Actor”析因结果。后续联合矩阵需统一相位奖励、初始化、指令与地形，并统计 encoder 成本。

## 平地停止记录

仅停止本任务 supervisor PID 5634 和 child PID 7574，先核对 PID 创建时间与命令。保留 model_0 至 model_600、日志及失败诊断，另以 `SUPERSEDED_BY_USER_TERRAIN_SWITCH` 说明人工切换。它不计为完成的正式 run，也不计为方法失败。
