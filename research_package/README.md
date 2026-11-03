# G1 自适应相位粗糙地形研究包

本包对应 `Doufu2007/g1-adaptive-locomotion` 的研究代码与可审计结果整理。

## 内容

- `source/`：G1 相位编码器、在线状态相位、粗糙地形环境配置、训练/评测/审计脚本，以及 Linux 远程执行适配器。
- `source/docs/`：相关实验设计与执行计划。
- `results/`：S、R、D 和周期扫描的协议、预检、独立审计、分析与证据状态文件。原始 `npz` 轨迹、训练日志和大批量检查点未纳入。
- `artifacts/models/`：源策略 `model_9900.pt` 与冻结的学习相位编码器 `final.pt`。

## 当前状态

- 已有粗糙地形实验结果与独立复算记录，周期扫描结果也已整理。
- `g1_phase_supplement_20261031` 的原始正式队列曾在采样前因本地磁盘门停止；恢复包只表示“准备就绪”，不表示正式补充训练已完成。
- 新远程节点适配器只改变路径、启动方式和 MuJoCo/CUDA bootstrap，不改变实验 seed、预算、模型、指标或科学协议。

## 复现提示

远程正式运行需要 `mjlab==1.2.0`、`mujoco-warp==3.5.0`、`onnxruntime` 和 CUDA。请先阅读相应结果目录中的 `operator_handoff.md`、`prospective_protocol.json` 和 `independent_audit_plan.json`。
