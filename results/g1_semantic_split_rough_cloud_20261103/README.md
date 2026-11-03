# G1 分流：粗糙地形运行记录

2026-11-03，按用户“切换一下吧”指令，将主队列改为粗糙地形。四种 Actor × 三个正式种子；每次仍为 1024 环境 × 24 步 × 10001 更新，从零训练。

## 部署

- 独立远端根目录：`/root/autodl-tmp/g1_semantic_split_rough_20261103`。
- Python：`/root/autodl-tmp/g1_semantic_split_20261103/venv/bin/python`。
- 入口：`rough_1024.py`；复用冻结 `formal_1024.py` 调度引擎，身份由两个适配器 SHA 和源码逐文件 SHA 绑定。
- `autorun.py` 已后台启动：36 项检查 → prepare → 四种 Actor 各 1024 环境 × 2 更新与 0.2 秒七场景评估 → 正式队列 → 全部封存 → holdout → CPU 延迟 → 分析打包。
- 其他 GPU 服务继续运行；GPU 短测和正式队列均有排他锁及逐任务资源准入。

## 科学范围

物理地形使用已有 E 相位实验的 1×5 `random_rough` 分布（32×16 m patch、噪声 .02–.10 m）。关闭 terrain 和 command curriculum，固定共同指令范围。Actor 保持 98 维，Critic 保留原有其他特权观测，双方都移除 height_scan。原生 gait reward 和固定 0.6 秒 clock 对四组保持一致。

正式 seed 为 20262101/20262102/20262103。Actor 分别是 original、matched、semantic、random，条件分支仍为指令与解析相位，不是地形感知。

粗糙评估先选择地形 patch，再重置机器人。每个 episode seed 保存实际地形索引、origin、初始 root 位置／姿态／速度，并与其他策略的公共参考逐值核验。单纯相同随机种子不作为地形配对已经成立的证据。

本轮与已有学习相位工作属于不同批次；尚未训练“学习相位＋语义分流”组合。本轮完成后也不能直接宣称已完成相位 × Actor 析因实验。具体边界见 [粗糙修订](../../docs/superpowers/specs/2026-11-03-g1-semantic-split-rough-amendment.md)。

## 当前证据

本机 36 项检查通过，云端 36 项检查通过。粗糙地形四组 GPU 短测与短评估已通过，正式队列正在运行。GPU 工程结果、最新队列及实际进度以同目录快照和远端状态为准。两次 PPO 更新和短评估只证明工程链路，不用于方法效果结论。

数值修订为 CCD iterations=50、nconmax=128、njmax=1500。500 iterations 在同资源下会因 EPA scratch OOM；诊断见 [resource_diagnostic_500_50.md](resource_diagnostic_500_50.md)。

平地四组 1000 更新 pilot 均已完成，旧正式队列已按用户要求停止；首个正式 run 的 model_0 至 model_600 与日志保留。人工切换记录为 `SUPERSEDED_BY_USER_TERRAIN_SWITCH`，不计为完成的正式训练或方法失败。

## 状态与恢复

```bash
cd /root/autodl-tmp/g1_semantic_split_rough_20261103
cat autorun_status.json
cat rough_smoke_1024/smoke_status.json
cat formal_1024/queue_status.json
```

SSH 断开不影响队列。先检查进程及锁，不能重复启动。若正式队列中断且工程门已完成，可显式恢复：

```bash
MUJOCO_GL=disable /root/autodl-tmp/g1_semantic_split_20261103/venv/bin/python -u rough_1024.py --mode queue --resume-queue
```

保留所有失败／中断诊断；只有通过身份和完整性核验的同 run checkpoint 可恢复。恢复重置物理状态，不声称轨迹逐位连续。
