# 粗糙地形学习相位 GPU 补充实验

完成于 2026-11-06（UTC+08:00）。本目录保存正式补充实验的协议、完整统计分析、审计记录和执行源码。仿真使用 RTX 5090D 的 `cuda:0`，ONNX 策略推理使用 CPU；复用 24 个已完成的续训单元，在 GPU 上执行全部校准、验证和测试。

| 阶段 | 完成单元 | 评估回合 |
|---|---:|---:|
| calibration | 24/24 | 120 |
| validation | 48/48 | 240 |
| test | 192/192 | 2,880 |
| 合计 | 264/264 | 3,240 |

实际仿真 world steps 为 3,229,750，预算上限为 3,240,000；11 个测试单元在全部场景结束后提前终止。有效观测 transitions 为 2,605,221，评估期间 PPO/Adam 更新为 0。续训总计 27,721,728 transitions、18,048 PPO updates、360,960 Adam steps。八个续训种子共享预训练源 `source9900` 和冻结相位编码器。

## 对照定义

- `live`：依据当前关节运动历史在线估计相位。
- `constant`：使用校准相位增量的均值，按恒定速度推进的开环相位。
- `replay`：循环回放校准相位增量序列的开环相位。
- `clock60`：按验证集规则选定的 0.60 秒固定 clock。

`constant` 和 `replay` 保留每回合初始相位；二者的后续相位推进不读取当前运动状态。旧确认实验的 `RE_frozen` 是保持相位初值，属于另一种对照。

## 测试结果

下表合并预设的 `steady`、`push03`、`push06` 三场景，每种方法 720 回合。

| 方法 | 完成率 | 跌倒率 | 速度 MAE (m/s) | 位移 (m) | 倾斜 RMS (rad) | 航向变化 (rad) |
|---|---:|---:|---:|---:|---:|---:|
| live | 0.4292 | 0.3264 | 0.3222 | 5.7747 | 0.1383 | 0.4141 |
| constant | 0.4333 | 0.3694 | 0.3245 | 5.9410 | 0.1495 | 0.4217 |
| replay | 0.3833 | 0.4444 | 0.3401 | 5.9723 | 0.1586 | 0.4460 |
| clock60 | 0.3903 | 0.3292 | 0.3304 | 5.2482 | 0.1383 | 0.3804 |

无推力粗糙地形行走单列于 `analysis.json → grouped.steady`，每种方法 240 回合：

| 方法 | 完成率 | 跌倒率 |
|---|---:|---:|
| live | 0.3958 | 0.3333 |
| constant | 0.4292 | 0.3833 |
| replay | 0.3583 | 0.4583 |
| clock60 | 0.3708 | 0.3417 |

以八个配对续训种子为统计单位，合并场景下 `live − replay` 跌倒率差为 −11.81 个百分点，95% t 区间 [−16.24, −7.37]，Holm p=0.03125；完成率差为 +4.58 个百分点，区间 [−0.27, +9.43]，Holm p=0.234375。`live − clock60` 的完成率差为 +3.89、跌倒率差为 −0.28 个百分点。完整分场景统计、地图重采样区间与预设联合判定均保存在 [analysis.json](analysis.json)；本批联合 `primary_passed` 均为 `false`。

## 审计与文件来源

[原始轨迹独立重算](independent_audit.json)和[交付验收](delivery_validation.json)均为 `PASS`，复核了 528 个 rollout JSON、全部 3,240 回合引用、292 个封存输入和 144 条模型路径映射。这里的“独立重算”指同一操作者使用单独计算实现复核原始轨迹。

- [protocol.json](protocol.json)：封存设计、种子、地图及预算。
- [clock_selection.json](clock_selection.json)：验证集周期选择。
- [map_identities.json](map_identities.json)及[phase_bank/manifest.json](phase_bank/manifest.json)：地图与校准库身份。
- [gpu_complete.json](gpu_complete.json)：最终完成标记。分析内的 `COMPLETE_PENDING_INDEPENDENT_RAW_AUDIT` 是生成分析时的状态；后续审计与完成标记记录验收结果。
- [source/](source/)：实际执行源码、适配器和冻结设计。
- [raw_archive_SHA256SUMS.txt](raw_archive_SHA256SUMS.txt)：约 4.08 GiB 原始归档哈希；原始轨迹、模型和归档未上传 Git。

公开副本已清理个人及实例路径。原审计绑定的分析 SHA256 为 `71a9e01b5b1a42d6f556380acd91a9058c79fe23436c0fdab0b421f102308156`，该值继续指向原始封存分析。[公开副本来源清单](../../verification/phase_evidence_provenance.json)逐文件记录原始 SHA256、公开 SHA256 和转换范围，科学数值与历史判定保持原样。
