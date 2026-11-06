# G1 地形扫描语义分流：成功种子案例

在这组参数量近似的配对仿真中，terrain-scan semantic actor 的 clean holdout 平面速度 RMSE 为 `0.2684 m/s`，matched MLP 为 `0.2957 m/s`，前者降低 `9.22%`；两种策略各自在 6 条 clean 指令回合中全部完成。两者 Actor 可训练参数量相差不足 `0.1%`。

本目录收录训练种子 `20261111` 的 semantic 策略及 matched 对照，包括最终 checkpoint、ONNX 策略、训练日志、holdout 轨迹、运行配置和哈希清单。这是根据已完成结果选出的单种子案例，呈现该配对实验的效果。

## 策略结构

任务为仿真双足机器人 `Unitree-G1-Rough`，Actor 输出 29 个关节动作。285 维观测由 98 维本体与任务信息、187 维 terrain height scan 组成。

| 策略 | Actor 结构 | 可训练参数 | 最终 checkpoint |
| --- | --- | ---: | --- |
| Semantic split | `98 -> 480` 与 `187 -> 32`；拼接后 `512 -> 256 -> 128 -> 29` | 221,530 | `formal/semantic_seed20261111/model_10000.pt` |
| Matched MLP | `285 -> 341 -> 256 -> 128 -> 29` | 221,744 | `formal/matched_seed20261111/model_10000.pt` |

两种策略均训练 10,001 次更新，每次使用 512 个并行环境、每环境 24 步。Semantic 组在更新 4,701 时从 checkpoint 恢复；恢复时仿真物理状态重置，记录见各自 `run_manifest.json`。

## Holdout 与延迟

每种策略评测 42 个回合：7 种场景、每场景 6 条速度指令，每回合 5 秒。主要结果如下：

| 策略 | Clean planar RMSE | 完成率 | CPU ONNX p50 | CPU ONNX p95 |
| --- | ---: | ---: | ---: | ---: |
| Semantic split | 0.2684 m/s | 6/6 | 21.212 μs | 22.123 μs |
| Matched MLP | 0.2957 m/s | 6/6 | 17.794 μs | 19.177 μs |

延迟使用 ONNX Runtime CPU、float32、batch 1、单线程测量，包含观测预处理与策略推理，不含仿真和传感器 I/O。两个导出策略均通过记录的 PyTorch 与 ONNX 数值一致性检查。

Push 场景的预设触发时刻为 `5.0 s`，与回合时长相同；rollout 在执行第 250 步前结束，因此该评测没有施加推力。Push 鲁棒性尚无实测结果。

## 使用与复现

ONNX 输入为 `float32[1, 285]`，输出为 `float32[1, 29]`。从仓库根目录运行以下 CPU 推理形状检查：

```python
import numpy as np
import onnxruntime as ort

model = "results/g1_extero_selected_seed20261111/formal/semantic_seed20261111/policy.onnx"
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
obs = np.zeros((1, 285), dtype=np.float32)
actions = session.run(None, {session.get_inputs()[0].name: obs})[0]
assert actions.shape == (1, 29)
```

模型定义与环境代码见 [`scripts/g1_extero_pilot.py`](../../scripts/g1_extero_pilot.py)、[`scripts/g1_semantic_split_study.py`](../../scripts/g1_semantic_split_study.py) 和 `src/tasks/velocity/`。协议设置摘录、运行环境、逐场景结果及文件哈希分别见本目录的 `protocol_excerpt.json`、`runtime.json`、`holdout/` 和 `SHA256SUMS`。所有评测均在仿真中完成。
