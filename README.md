# G1 Adaptive Locomotion

本仓库归档 Unitree G1 仿真行走的多条独立研究线，关注策略结构、粗糙地形感知，以及关节控制目标间歇丢失时的容错。各研究线使用不同协议、模型和训练批次；结果应分别解释，不能拼成同一组实验结论。

## 研究线

| 研究线 | 问题与当前材料 | 状态 |
| --- | --- | --- |
| Actor 输入分流 | 在固定解析步态相位下，将身体状态与任务条件分路编码；比较 `original`、`matched`、`semantic`、`random` 四种 Actor。 | 粗糙地形正式训练的最近归档是 2026-11-03 快照：工程检查通过，正式队列刚启动，快照中没有完成的正式训练或 holdout 结果。该快照不是实时状态。 |
| 地形高度图分流 | 将 187 维地形扫描与 98 维基础观测分开编码，比较四种 Actor。 | 单训练种子、每组 300 次更新的探索性 pilot。记录的 5 秒粗糙地形 rollout 中四组完成率均为 0，不支持分流方法优越的结论。 |
| 关节目标丢失容错 | 新关节目标间歇丢失时保持上一已应用目标；比较无模仿惩罚、固定约束、全局放松和逐关节选择性放松。 | G 平地批次有正式模拟评测；R1 粗糙地形批次完成 12 个最终模型和训练记录，但没有 R1 正式地形评测。D 尚未证明比固定约束 B 稳定地更好。 |

## 实验入口

### Actor 输入分流

- [研究设计、运行与复现说明](G1_SEMANTIC_SPLIT_README.md)
- [Actor、观测索引、协议和指标](scripts/g1_semantic_split_study.py)
- [粗糙地形运行快照](results/g1_semantic_split_rough_cloud_20261103/README.md)
- [粗糙地形队列状态快照](results/g1_semantic_split_rough_cloud_20261103/remote_queue_status_latest.json)
- [语义分流正式实现状态与早期工程检查](results/g1_semantic_split_20261026/IMPLEMENTATION_STATUS.md)

队列状态文件的最后更新时间为 2026-11-03；它只记录当时正式训练已启动，不能据此判断当前是否仍在运行或已完成。工程短测不作为方法效果结果。该实验采用解析相位，不是“学习相位 × Actor 分流”的析因实验。

### 地形高度图分流

- [探索性 rollout 结果](results/g1_extero_pilot_20261105_v4/exploratory_rollout_20261105/RESULTS.md)
- [冻结协议与 pilot 预算](results/g1_extero_pilot_20261105_v4/protocol.json)
- [四组训练进度](results/g1_extero_pilot_20261105_v4/pilot_progress.json)

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
