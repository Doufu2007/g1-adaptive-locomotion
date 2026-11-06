# 关节目标更新丢失容错研究详报与数据表

本目录补充归档 2026 年 11 月 5 日整理的详报与训练数据表原件。两份附件保持原始字节，可直接下载；既有模型、源码和日志通过下表访问。

| 资料 | 内容与入口 |
| --- | --- |
| [研究与训练数据详报](G1攀登计划前期研究与训练数据详报_20261105.docx) | 方法、四组对照、R1 训练结果、G 正式评价及解释边界 |
| [完整数据表 ZIP](G1攀登计划训练数据表_20261105.zip) | 50 个 CSV、JSON、TXT 及压缩计时文件，包括 R1 的 24,000 行逐更新宽表、模型及组窗口统计和 G 的 240 个评价案例 |
| [G 平地正式评价与冻结源码](https://github.com/Doufu2007/g1-adaptive-locomotion/tree/cc5be06916457ab9dd88fc3b95f432da86609aad/experiments/G_20261103) | 四方案、五训练种子、240 个评价案例、分析图及源码；对应分支 `results/G-20261103-public` |
| [G 模型与结果 Release](https://github.com/Doufu2007/g1-adaptive-locomotion/releases/tag/G-20261103) | 20 个模型的公开归档 `G_20261103_public_results.zip` 及校验文件 |
| [R1 粗糙地形最终训练成果](https://github.com/Doufu2007/g1-adaptive-locomotion/tree/6ddc31ff3b2f6a414a55e6741fcbe60e26560fcb/experiments/R1_20261104) | 12 个最终模型、12 份 TensorBoard 日志、12 份计时日志、24 份配置及清单；对应分支 `results/R1-20261104-final12` |
| [目标更新丢失与保持机制源码](https://github.com/Doufu2007/g1-adaptive-locomotion/blob/cc5be06916457ab9dd88fc3b95f432da86609aad/experiments/G_20261103/source/src/tasks/velocity/mdp/action_dropout.py) | 新目标丢失时保持上一已应用目标 |
| [逐关节参考约束源码](https://github.com/Doufu2007/g1-adaptive-locomotion/blob/cc5be06916457ab9dd88fc3b95f432da86609aad/experiments/G_20261103/source/src/tasks/velocity/mdp/style_reference.py) | 固定、全局及选择性参考权重与恢复逻辑 |

数据表 ZIP 中，R1 的 24 份实际配置、12 份计时日志、发布清单，以及 G 的 7 份数据文件，共 44 个文件已在上述成果分支中按相同字节归档。本次补齐详报及完整表格包，其中另有 R1 逐更新宽表、两份窗口统计、完成与文件身份表、来源清单和数据表说明。

## 两批实验的解释

G 为独立平地批次，四方案各五个训练种子，累计训练至 3000 次更新，并完成四场景、三个评价种子的正式评价。R1 为从头训练的粗糙地形批次，四方案各三个训练种子，每模型 2000 次更新，合计 589,824,000 个环境步。本次表格中的 R1 记录是训练日志，G 的评价结果不作为 R1 测试成绩。

故障定义为关节目标更新丢失后保持上一已应用目标，关节观测仍然存在。附件未证明传感器观测缺失容错或实机效果；选择性方案相对固定约束的稳定性优势尚未确立。参考误差衡量对所用参考的数值接近程度，不直接代表视觉类人性。

人体参考的来源与许可见 [R1 reference_provenance.json](https://github.com/Doufu2007/g1-adaptive-locomotion/blob/6ddc31ff3b2f6a414a55e6741fcbe60e26560fcb/experiments/R1_20261104/reference_provenance.json)。原始及适配后的受限动作数组未在本次附件中分发。

## 文件校验

两份原件的 SHA256 见 [SHA256SUMS](SHA256SUMS)。2026 年 11 月 6 日补充归档时，已核对复制前后的 SHA256 一致；该校验说明附件身份，不构成效果或实机验证。
