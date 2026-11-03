# 云端 G1 分流实验：2026-11-03

用户已批准首轮使用 **1,024 环境 × 10,001 更新、四模型 × 三种子**。每次正式训练为 245,784,576 transitions，12 次共 2,949,414,912；这是原 4,096 环境方案采样量的 1/4。正式训练从头开始，工程 pilot 权重不复用。

## 当前连接与恢复

- 当前 SSH：`ssh -p 16963 root@connect.weste.seetacloud.com`。
- 先前地址：`connect.westc.seetacloud.com:35063`，仅保留作历史来源。
- 远程目录：`/root/autodl-tmp/g1_semantic_split_20261103`，源码、venv、已有输出均保留。
- GPU：RTX 4080 SUPER，32,760 MiB。新端点首次检查没有 GPU 作业，旧两个 vLLM 服务及原 pilot 进程均已不存在；本任务没有停止、修改或启动这些服务。
- 原始与等预算 MLP 的 1,000 更新 pilot 已完成。语义分流从第 401 更新 checkpoint 恢复，随后运行随机分流。恢复会重置物理状态，因此该 pilot 仅用于工程稳定性，不形成方法结论。
- 源码 SHA 仍为 `473b1ac73da939c4f069143471d3a06e26d1f9b1fccbba5670d97b6c485613f0`。连接变更后原 21 项检查再次通过，新预算适配器另有 5 项测试通过。

## 正式队列

正式队列由 `formal_1024.py` 驱动，使用独立的 `formal_1024/` 输出目录。状态以远程 JSON 及本目录同步快照为准，等待 pilot 时不写成正式训练已经开始。

执行顺序：四组 pilot 验证 → 12 次正式训练 → 核对同种子环境、Critic 初始化与最终 checkpoint → 封存全部策略 → 12 个 holdout 评估 → CPU ONNX 延迟 → 统计分析 → 打包交付。

协议新增 `resource_amendment`，记录用户授权、原协议 SHA、适配器 SHA、预算缩减和解释范围；原始协议文件保持封存。所有 worker 均通过同一适配器进入，训练、封存、评估与分析使用同一修订协议。正式种子：20262101、20262102、20262103；只使用最终预算 checkpoint。

已识别的既有 vLLM 作业可以共存；遇到未知 GPU 作业或容量不足时等待。不会停止其他进程。当前实例没有这些服务，未来外部新增任务会触发资源重新检查。

```bash
cd /root/autodl-tmp/g1_semantic_split_20261103
cat engineering_pilot_1024/pilot_queue_status.json
cat formal_1024/queue_status.json
```

SSH 断开不影响已启动的队列。若实例再次停止，先核对远程状态；恢复 pilot 后可显式恢复正式队列：

```bash
MUJOCO_GL=disable venv/bin/python -u formal_1024.py --mode queue --resume-queue
```

此命令会拒绝与活跃正式队列重复运行。已完成 run 验证后跳过，未完成的正式 run 只从身份匹配的正式 checkpoint 恢复。打包中断时按封存文件清单和内容 hash 验证并复用已有归档。

最终可迁移结果为 `g1_semantic_split_formal_1024_results.tar.gz`，SHA 和大小记在 `formal_1024_delivery.json`。包内保留原始评估轨迹、指标、全部配置和日志、ONNX、最终 checkpoint；中间 checkpoint 留在远端。正式训练、评估与本地下载尚未全部完成时，不据 pilot 宣称语义分流有效。

密码没有写入源码、启动脚本或本目录。训练与检查使用 `MUJOCO_GL=disable`，以兼容镜像缺少 EGL 的条件。
