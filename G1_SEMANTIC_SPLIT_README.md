# G1 身体状态／任务条件分流实验

> 本文件保留最初 4,096 环境方案的使用说明。2026-11-03 最新首轮已获批改为 1,024 环境 × 10,001 更新；当前状态与实际部署入口以根目录 README.md、cloud/runtime/ 和 results/g1_semantic_split_cloud_20261103/ 为准。

2026-10-26。本目录是隔离工作区，分支 `codex/g1-semantic-split`，以 `1425b15` 和原工作区现有兼容代码为基础。没有改动原工作区中的训练任务。新实验的实现、协议、检查与查新材料均在本目录。

## 当前交付与运行范围

已经实现四种 Actor、真实字段索引、训练／恢复／导出、七场景配对评估、CPU 推理计时、训练种子层面的统计与结果报告。最终验证证据见 `results/g1_semantic_split_20261026/IMPLEMENTATION_STATUS.md`；21 项检查的记录见同目录 `verification/`。

**工程 smoke 与论文实验严格区分。** smoke 每组仅 2 次 PPO 更新、4 个环境，不能判断方法优劣。每组 1,000 更新 pilot 和 12 次完整训练尚未执行。代码不会用 smoke 数字填补正式结果。本机 RTX 4050 6 GB 曾成功运行 1,024 环境；本机预算候选见 `docs/superpowers/specs/2026-10-26-g1-semantic-split-local-options.md`。当前还有其他任务占用 GPU；此前三台云实例端口拒绝连接。

## 四种模型与观测

| arm | 结构 | 线性层参数 | 线性层 MAC／动作 |
|---|---|---:|---:|
| original | `98→512→256→128→29` | 218,653 | 217,728 |
| matched | `98→497→256→128→29` | 213,328 | 212,418 |
| semantic | `93→480` 与 `5→32`，拼接后 `512→256→128→29` | 213,277 | 212,352 |
| random | 固定随机 93/5 分组，与 semantic 同结构 | 213,277 | 212,352 |

另外每组有 29 个 Gaussian 标准差参数。归一化缓冲区、索引缓冲区和真实总量均写入训练 manifest。matched 与 semantic 的线性层参数差为 51，约 0.024%；分流相对原 MLP 只少约 2.46%，没有“减少 30% 参数”的依据。MAC 只计线性层，真实推理时间另行测量。

实际字段为 `base_ang_vel(3), projected_gravity(3), command(3), phase(2), joint_pos(29), joint_vel(29), actions(29)`。条件分支索引是 `[6,7,8,9,10]`，身体分支是其补集；5 维是指令和解析相位。运行时再次核对 ObservationManager，按真实字段顺序计算索引；缺失、新增或维数变化即拒绝运行。随机分组 seed 为 20261026，完整索引存入 `random_partition.json`。

## 环境准备

已实际验证 Windows／Python 3.13／torch 2.6.0+cu124 的 CPU 仿真路径；具体软件版本记录在每次运行的 `environment.json`。云端 Linux GPU 尚待实际验证。建议使用独立 Python 环境，先安装适配目标 GPU 的 PyTorch，再执行：

```bash
python -m pip install -r requirements-semantic-split.txt
python -m pip install -e .
python -m pytest tests/test_g1_semantic_split.py tests/test_g1_semantic_split_execution.py -q
```

RTX 5090 需要支持其架构的 PyTorch/CUDA 版本，应重新跑工程检查并记录版本；不要直接套用本机 cu124 软件栈。无需 learned-phase 权重或 PHASOR 数据集。

## 执行全部实验

所有命令均在本目录执行。运行包内已含本次工程证据；正式训练输出沿用固定目录，不修改封存的源文件和协议：

```bash
python -m scripts.run_g1_semantic_split --stage prepare --output results/g1_semantic_split_20261026
python -u -m scripts.run_g1_semantic_split --stage queue --device cuda:0 --resume --wait-for-gpu --output results/g1_semantic_split_20261026
```

队列依次执行：四组 smoke → 四组 pilot → 四组×三个种子的正式训练 → 封存全部 12 个最终策略 → holdout 评估 → CPU 推理计时 → 分析报告。已完成并通过身份校验的任务自动跳过；`--resume` 对尚未开始的任务正常从头初始化，对中断任务验证 checkpoint 后恢复。它不会停止其他进程。

原定 4,096 环境方案的 GPU 准入阈值是 22,000 MiB **可用**显存，属于当前保守资源配置，并非已经测得的最小显存，也不意味着本机无法训练。GPU 忙时队列每 30 秒重新检查；显存本身不足则退出并记录。当前命令仍对应原预算；本机 1,024 环境候选需要独立冻结协议和输出目录，不直接修改已有结果。

```bash
python -m scripts.run_g1_semantic_split --stage status --output results/g1_semantic_split_20261026
```

查看 `queue_status.json` 及其中给出的日志位置。Linux 上可用任务管理器或终端复用器保持队列运行；代码没有购买或创建云实例的功能。

如需手动执行单项，例如语义组 pilot：

```bash
python -m scripts.run_g1_semantic_split --stage pilot --arm semantic --seed 20262100 --device cuda:0 --output results/g1_semantic_split_20261026
```

正式种子为 20262101、20262102、20262103；每次 10,001 更新，4,096 环境×24 步。pilot 每组 1,000 更新，从头训练且不用于正式初始化。每个阶段的墙钟时间、真实 transitions 和配置均有记录；先以 GPU pilot 实测重新估算训练时间，不沿用 67 GPU 小时的历史外推作为实测值。

如果源文件或协议被修改，封存校验会拒绝混入已有实验；保留原输出，使用新的 `--output` 目录并说明变更。无 checkpoint 的失败启动不会自动清理，修复后使用新输出目录。恢复 checkpoint 会重置物理状态，不能声称是未中断训练的逐位复现。

## 评估与结果规则

G1 平地、98 维观测、0.6 秒解析步态周期，共用奖励、Critic、归一化算法、PPO 超参数和训练预算。关闭四组 command curriculum，固定相同指令范围。Critic 独立初始化并验证同种子 hash，记录启动物理参数 hash。

评估固定 20 秒，六个指令×七个场景×30 个 holdout episode seeds，即每策略 1,260 回合，共 15,120 回合。七场景为 clean、两档身体观测噪声、两档身体 y 轴速度脉冲推扰、1/2 步动作延迟。身体 y 轴通过施扰时姿态旋转到世界坐标；这是指定速度增量的仿真实验，不是牛顿单位的实机力。评估显式设置噪声与推扰，不依赖 play 默认值。

主指标是 clean 平面速度 RMSE（m/s）；转向角速度 RMSE（rad/s）单独报告。首次非 timeout termination 记失败，剩余时域按实际速度 0 计误差，同时报告完成率、存活时间及存活片段误差。保留完整指令／速度／动作／终止轨迹。缓存恢复和最终分析均从原始轨迹重算指标，并核对身份、覆盖和 SHA。

独立统计单位为训练种子，不把 1,260 个回合视作 1,260 次独立训练。报告配对差值、区间、Holm 校正以及每场景结果。三种子仅用于筛选；工程筛选门槛与失败树见协议，不代表论文录用标准。

全部正式训练和评估完成后，队列生成 `analysis.json`、`raw_evaluation_rows.json`、`REPORT.md`、`latency.json`。缺少正式数据时分析程序拒绝生成主结论。

## 文件入口

- `scripts/g1_semantic_split_study.py`：观测、Actor、协议、指标、统计。
- `scripts/run_g1_semantic_split.py`：准备、训练、恢复、封存、队列。
- `scripts/evaluate_g1_semantic_split.py`：原始轨迹、鲁棒性和 CPU 计时。
- `scripts/analyze_g1_semantic_split.py`：完整性核验和结果报告。
- `docs/superpowers/specs/2026-10-26-g1-semantic-split-protocol.md`：完整实验协议。
- `results/g1_semantic_split_20261026/literature/REVIEW.md`：查新与最相近前作；附原始检索回执及全文。
- `scripts/package_g1_semantic_split.py`：生成源码与本实验材料 ZIP，并核验 CRC。

本次新增逻辑之外的 `src` 和部分脚本继承自用户当前工作区，是运行依赖，不应把它们都当成本次原创修改。对旧注册的唯一新增行为由 `UNITREE_G1_FIXED_PHASE_ONLY=1` 显式启用，避免固定相位实验在导入时读取无关 PHASOR 权重；默认注册行为保留。
