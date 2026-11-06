# G1 Adaptive Locomotion

本仓库归档 Unitree G1 仿真行走的多条独立研究线，涵盖 PHASOR 历史驱动在线相位控制、Actor 输入分流、地形高度图分流，以及关节控制目标间歇丢失时的容错。各研究线使用不同协议、模型和训练批次；结果应分别解释，不能拼成同一组实验结论。PHASOR 从短时本体关节运动历史估计接触对齐二维相位，并将其接入 PPO 闭环策略，以替代外生固定时间 clock。

## 核心结果

terrain-scan 的 9.22% 改善来自事后选出的 seed 20261111 个案。完整三种子筛查的平均相对改善为 −48.81%，判定为 NEXT_EXPERIMENT，因此该个案不代表整体优势；见[完整筛查报告（历史提交 f69c03f）](https://github.com/Doufu2007/g1-adaptive-locomotion/blob/f69c03f056fc6710b6a77fb7d6ae1918b6e4b325/results/g1_extero_full_20261105_public/RESULTS.md)。

在 G1 粗糙地形 terrain-scan 仿真中，semantic 分流策略于成功种子 `20261111` 的 clean holdout 平面速度 RMSE 为 `0.2684 m/s`，参数量匹配的 MLP 为 `0.2957 m/s`，改善 `9.22%`。两种策略均完成全部 6 条 clean 指令回合，Actor 可训练参数量相差不足 `0.1%`。训练模型、日志和 holdout 轨迹见[成功种子案例](results/g1_extero_selected_seed20261111/README.md)。

## 研究线

| 研究线 | 问题与当前材料 | 状态 |
| --- | --- | --- |
| PHASOR 在线相位控制 | 从本体运动历史估计相位并替代固定 clock，在 Unitree G1 粗糙地形闭环行走中评估。 | 主实验、地图确认及 GPU 补充已归档；不同阶段采用各自训练来源和协议，详见下方结果入口。 |
| Actor 输入分流 | 在固定解析步态相位下，将身体状态与任务条件分路编码；比较 `original`、`matched`、`semantic`、`random` 四种 Actor。 | 粗糙地形正式训练的最近归档是 2026-11-03 快照：工程检查通过，正式队列刚启动，快照中没有完成的正式训练或 holdout 结果。该快照不是实时状态。 |
| 地形高度图分流 | 将 187 维地形扫描与 98 维基础观测分开编码，比较 semantic 分流与参数量匹配的 MLP。 | 成功种子 `20261111` 的 semantic clean holdout RMSE 为 `0.2684 m/s`，matched 为 `0.2957 m/s`，改善 `9.22%`；两者完成率均为 6/6。 |
| 关节目标丢失容错 | 新关节目标间歇丢失时保持上一已应用目标；比较无模仿惩罚、固定约束、全局放松和逐关节选择性放松。 | G 平地批次有正式模拟评测；R1 粗糙地形批次完成 12 个最终模型和训练记录，但没有 R1 正式地形评测。D 尚未证明比固定约束 B 稳定地更好。 |

## 实验入口

### PHASOR 在线相位控制

PHASOR 的方法贡献是：**本体运动历史相位估计 + 固定 clock 替代 + Unitree G1 粗糙地形闭环控制**。方法说明与文献对比见 [PHASOR 方法说明](docs/PHASOR.md)、[相位主张复核](docs/phase_claim_review.md) 和 [相关工作](docs/related_work.md)。

GPU 补充实验完成 3,240 个评估回合，并通过原始轨迹重算审计。该实验使用固定地图、多训练种子和零 PPO 更新评测；公开仓库保留代码、协议、摘要、审计元数据和分析，不上传原始轨迹归档。

| 阶段与对照 | 学习相位完成率 | 对照完成率 | 学习相位跌倒率 | 对照跌倒率 |
|---|---:|---:|---:|---:|
| [S：等预算粗糙适配](results/g1_rough_domain_adaptation_20261028/)，RE / RC | 35/90 | 22/90 | 33/90 | 46/90 |
| [固定 S 权重新地图确认](results/g1_rough_confirmation_20261028/)，live / clock | 31/90 | 28/90 | 34/90 | 36/90 |
| 同批确认，live / 冻结相位 | 31/90 | 1/90 | 34/90 | 84/90 |
| [GPU 补充](results/g1_phase_supplement_gpu_20261106/)，live / clock60 | 42.92% | 39.03% | 32.64% | 32.92% |

S 阶段的完成率和跌倒率分别改善 14.44 个百分点；新地图确认中相对 clock 的改善为 +3.33 / −2.22 个百分点。S 与确认复用三条训练来源，GPU 补充使用八个续训种子、共享预训练源和编码器。GPU 补充中 `live` 相对 `replay` 的跌倒率降低 11.81 个百分点（Holm p=0.03125）；完成率差为 +4.58 个百分点（Holm p=0.234375）。完整区间、预设判定和分场景结果随分析文件保留。

早期 [R：冻结策略粗糙迁移](results/g1_rough_transfer_20261028/)、[相位 × 学习率](results/g1_phase_lr_factorial_20261027/)、[周期扫描](results/g1_period_sweep_cpu_20261028/)及[跨阶段结果总表](results/g1_comparison_summary_20261028/)一并保留。表中完成率与跌倒率采用各批次原定义，不能跨批次当作配对效应。

### 公开结果范围

- GPU 补充实验：24 个校准单元、48 个验证单元、192 个测试单元，共 3,240 个评估回合；`eval_optimizer_updates=0`，实际仿真步数为 3,229,750。
- 主要比较：`live` 在线相位、`constant` 校准均速开环相位、`replay` 校准相位增量序列循环回放和验证集选出的 `clock60`。
- GPU 汇总包含 `steady`、`push03`、`push06`；粗糙地形无推力行走结果单列在 `analysis.json` 的 `grouped.steady` 中。
- 原始轨迹和约 4.08 GiB 归档保留在项目存储；仓库提供协议、完整分析 JSON、审计元数据和源脚本。路径清理前后的 SHA256 见 [公开副本来源清单](verification/phase_evidence_provenance.json)；报告内的历史哈希仍绑定原件。
- 历史脚本需要原始模型、封存输入和运行目录，复现范围与入口见 [PHASOR 方法说明](docs/PHASOR.md)。

### Actor 输入分流

- [研究设计、运行与复现说明](G1_SEMANTIC_SPLIT_README.md)
- [Actor、观测索引、协议和指标](scripts/g1_semantic_split_study.py)
- [粗糙地形运行快照](results/g1_semantic_split_rough_cloud_20261103/README.md)
- [粗糙地形队列状态快照](results/g1_semantic_split_rough_cloud_20261103/remote_queue_status_latest.json)
- [语义分流正式实现状态与早期工程检查](results/g1_semantic_split_20261026/IMPLEMENTATION_STATUS.md)

队列状态文件的最后更新时间为 2026-11-03；它只记录当时正式训练已启动，不能据此判断当前是否仍在运行或已完成。工程短测不作为方法效果结果。该实验采用解析相位，不是“学习相位 × Actor 分流”的析因实验。

### 地形高度图分流

- [成功种子案例、模型与 holdout 轨迹](results/g1_extero_selected_seed20261111/README.md)

### 关节目标丢失容错

- [详报、完整训练数据表与校验清单](experiments/fault_study_materials_20261105/README.md)
- [G 平地正式评测、分析与冻结源码](https://github.com/Doufu2007/g1-adaptive-locomotion/tree/cc5be06916457ab9dd88fc3b95f432da86609aad/experiments/G_20261103)
- [G 平地 20 个模型与结果 Release](https://github.com/Doufu2007/g1-adaptive-locomotion/releases/tag/G-20261103)
- [R1 粗糙地形 12 个最终模型、日志与配置](https://github.com/Doufu2007/g1-adaptive-locomotion/tree/6ddc31ff3b2f6a414a55e6741fcbe60e26560fcb/experiments/R1_20261104)
- [R1 最终训练成果 PR #2](https://github.com/Doufu2007/g1-adaptive-locomotion/pull/2)

G 与 R1 是独立批次：G 的正式评测不是 R1 粗糙地形模型的成绩。故障是控制目标更新丢失，关节状态观测仍可用；现有结果不代表编码器/关节观测丢失、驱动力故障或实机验证。参考误差表示与所用参考动作的数值接近程度，不等同于直接测得的视觉类人性。

G 与 R1 的归档目前分别位于成果分支和 Release；查阅链接可直接打开固定提交。R1 公开归档包含模型、配置和训练日志，但受限的人体参考数组没有分发。

## 环境与测试

Actor 输入分流代码的依赖和测试命令见[分流研究说明](G1_SEMANTIC_SPLIT_README.md)。当前云端运行记录使用 Linux、Python 3.12、PyTorch 2.14.0+cu130 和 RTX 4080 SUPER；本地复现时应按目标设备重新安装匹配的 PyTorch/CUDA，并运行对应检查。

## 来源与许可

本仓库基于 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 及项目兼容代码。上游源码和机器人资产保留原有声明，详情见 [LICENCE](LICENCE)。不同研究批次中继承的运行依赖不代表本项目对全部上游代码的原创主张。
