# R1 粗糙地形最终训练成果（2026-11-04）

本目录归档 A/B/C/D × 训练种子 42、43、44 的 **12 个最终模型**。每个模型完成 **2000 次 PPO 更新**，最终检查点为 `model_1999.pt`。

## 训练范围

| 项目 | 数值 |
|---|---|
| 训练模型 | A / B / C / D × 42 / 43 / 44 |
| 并行环境 / rollout | 1024 / 24 |
| 每模型更新 / 环境步 | 2000 / 49,152,000 |
| 合计环境步 | 589,824,000 |
| Actor / Critic 输入 | 285 / 298 |
| 动作维度 | 29 |
| 控制间隔 / episode | 0.02 秒 / 20 秒 |
| 初始化 | 各模型从新网络开始 |

A、B、C、D 分别采用 none、fixed、global、selective 的关节参考权重模式。完整 PPO 参数、地形、故障课程和奖励配置见各模型的 `configs/`；动作顺序及软件版本见 `manifest.json`。R1 模型按本目录的 285 维 Actor 接口使用，仓库中其他批次各自保留其接口定义。

## 文件

- `models/<组>_seed<种子>/model_1999.pt`：最终 PyTorch 训练检查点，保留原始字节，含 Actor、Critic、归一化和优化器状态。
- `configs/<模型>/agent.json`、`env.json`：从实际运行配置记录提取的配置快照；可调用对象以名称记录，参考文件路径使用项目相对路径。
- `logs/<模型>/events.out.tfevents.gz`：原始 TensorBoard 标量日志的无损 gzip 压缩，每模型 45 个标签，各有 2000 条记录。43 个更新轴标签的绝对 step 为 0–1999，另 2 个 `/time` 标签使用原始累计时间轴，其范围逐模型记录在 `manifest.json`。
- `logs/<模型>/iteration_timing.csv.gz`：每模型 2000 行更新用时，记录更新数 1–2000、控制步 24–48000、UTC 时间和采集/学习秒数。
- `training_summary.csv`：全部 12 个模型的训练预算和最终检查点 SHA256。
- `manifest.json`：公开文件大小、SHA256、来源映射及处理说明。
- `SHA256SUMS`：本目录交付文件校验清单。

查看 TensorBoard 时，先将日志解压到独立目录。例如在本目录执行：

```python
from pathlib import Path
import gzip

for source in Path("logs").glob("*/events.out.tfevents.gz"):
    target = Path("tensorboard_view") / source.parent.name / "events.out.tfevents"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.decompress(source.read_bytes()))
```

然后运行 `tensorboard --logdir tensorboard_view`。检查点载入设备可使用 CPU 映射；模型层结构、观测布局和动作顺序以本目录实际配置为准。

## 日志解释与来源

本次交付用于保存最终训练模型及训练过程。标量日志描述优化过程：`Metrics/twist/error_vel_xy` 使用配置中的 command resampling 时间归一化，其数值尺度不能作为 m/s 速度 RMSE。训练奖励和标量代理各自按配置解释，模型间任务效果和实机迁移需要对应的独立测量依据。

人体参考的来源、固定版本、内容哈希及许可见 `reference_provenance.json`；适配后的参考数组和原始/重定向动作数组按来源限制保留在私有训练资料中。本目录保留最终检查点、必要训练日志、配置与来源说明；中间检查点、机器资源记录、调度记录和安装环境由私有归档管理。
