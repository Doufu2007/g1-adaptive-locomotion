# G1 Adaptive Locomotion

宇树 G1 平地速度跟踪实验：在接近相同的 Actor 参数预算下，比较身体状态／任务条件分流、普通 MLP 和随机分流。任务条件为速度指令与解析步态相位，Actor 输入共 98 维，输出为 29 维动作。

## 当前状态

截至 2026-11-03：实验实现完成，云端 26 项检查通过。原始 MLP、等预算 MLP、语义分流的 1,024 环境 × 1,000 更新工程 pilot 已完成，随机分流 pilot 正在运行。正式调度器已启动，等待四组 pilot 验证后，自动执行 **四模型 × 三种子，每次 1,024 环境 × 10,001 更新**。

**当前结果用于验证工程流程。12 次正式训练、holdout 和性能比较尚未全部完成，因此本仓库暂不报告语义分流的性能优势。** 最新可核验状态见 [云端结果快照](results/cloud_snapshot_20261103/PUBLICATION_SNAPSHOT.json) 和 [运行记录](results/g1_semantic_split_cloud_20261103/README.md)。快照不会随远端训练自动变化。

## 对照模型

| 模型 | Actor 结构 | 线性层参数 |
|---|---|---:|
| original | `98→512→256→128→29` | 218,653 |
| matched | `98→497→256→128→29` | 213,328 |
| semantic | `93→480` 与 `5→32`，拼接后 `512→256→128→29` | 213,277 |
| random | 固定随机 93/5 分组，网络与 semantic 一致 | 213,277 |

93 维身体状态包含角速度、投影重力、关节位置／速度、上一动作；5 维条件包含速度指令和解析相位。条件索引为 `[6,7,8,9,10]`，按运行时观测字段核对，不能直接切成“前 93／后 5 维”。各组使用同一 Critic、奖励、归一化算法和 PPO 超参数。

原协议采用 4,096 环境。用户已明确批准资源修订为 1,024 环境，更新数保持 10,001，每次正式训练为 **245,784,576 transitions**，共 12 次；这是原计划采样量的 1/4。详情见 [资源修订](results/g1_semantic_split_cloud_20261103/resource_amendment.md)。

## 安装与验证

先安装适合当前 CUDA/GPU 的 PyTorch，再执行：

```bash
python -m pip install -r requirements-semantic-split.txt
python -m pip install -e .
MUJOCO_GL=disable python -m pytest tests/test_g1_semantic_split.py tests/test_g1_semantic_split_execution.py cloud/runtime/test_formal_1024.py -q
```

当前云端验证环境为 Linux、Python 3.12、PyTorch 2.14.0+cu130、RTX 4080 SUPER。软件清单和原始检查输出保存在结果快照中。无渲染训练使用 `MUJOCO_GL=disable`。

## 代码与资料

- [观测索引、Actor、冻结协议、指标](scripts/g1_semantic_split_study.py)
- [训练、恢复、导出与策略封存](scripts/run_g1_semantic_split.py)
- [七场景评估、原始轨迹与 CPU 推理计时](scripts/evaluate_g1_semantic_split.py)
- [完整性核验和训练种子层面的分析](scripts/analyze_g1_semantic_split.py)
- [当前 1,024 环境正式调度器](cloud/runtime/formal_1024.py)
- [完整实验协议](docs/superpowers/specs/2026-10-26-g1-semantic-split-protocol.md)
- [查新记录与相近工作](results/g1_semantic_split_20261026/literature/REVIEW.md)
- [早期 smoke、导出、原始评估轨迹和检查记录](results/g1_semantic_split_20261026)
- [最新云端 checkpoint、ONNX、训练日志与状态快照](results/cloud_snapshot_20261103)

`cloud/runtime/` 保留真实云端启动脚本。当前部署布局为“工作目录中放置脚本，`g1_semantic_split/` 放置项目源码”，工程 pilot 与正式输出分别位于 `engineering_pilot_1024/`、`formal_1024/`。脚本中的旧实例 PID 和已有探测路径属于运行记录，迁移至其他实例时需要重新测量资源。

只使用最终预算 checkpoint；全部 12 个正式策略封存后才运行 holdout。评价包括速度跟踪、跌倒／完成率、两档观测噪声、两档推扰、1／2 步控制延迟和 CPU ONNX batch=1 延迟。独立统计单位是训练种子，三种子作为首轮筛选。

## 来源与许可

基于 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 及既有项目兼容代码。上游源码和机器人资产保留原有声明，许可见 [LICENCE](LICENCE)。本轮新增实验逻辑主要位于 `scripts/*g1_semantic_split*.py`、对应测试及 `cloud/runtime/formal_1024.py`；保留的兼容模块不全部属于本轮原创。
