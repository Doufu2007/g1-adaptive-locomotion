# G1 Stateful Flat PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 执行已获授权的新E冻结相位模型平地三组等预算PPO对照，保留完整失败和原始结果。

**Architecture:** 独立在线特征/逐层缓存模块，复用已审查的旧PPO训练与batched评测函数，但在新进程注入新协议和限定checkpoint适配；新队列负责GPU等待、6训练、策略封存、9评测和汇总。

**Tech Stack:** 现有Python/PyTorch/mjlab/rsl-rl/MuJoCo/ONNX Runtime，Windows本地CUDA。

**Spec:** `docs/superpowers/specs/2026-10-25-g1-stateful-flat-ppo-design.md`

## Global Constraints

- 固定E seed20261025 final.pt，SHA fdaad78ee4d923189f4291a0d9c8a02c0b726325d1caa2b96a2e76231c7848ca；不重选或重拟合。
- 3训练seed20261901/02/03×2arm，501×64×24=769536 transition/run；9评测、270回合，env seed20262001–10。
- 平地；两臂相同actor起点/critic初始化/优化器、奖励和预算；encoder冻结，源文件不改。
- 其他GPU项目活跃时等待；不终止其他项目，不唤醒租用服务器，不关本地机器。
- 失败保留；正式训练从完整同身份检查点恢复，所有结果后选择留痕；不按分数追加预算。

## Review Focus

- 在线缓存对比真实PhaseNet及64维离线特征；覆盖首帧、q/-q、长于感受野。
- 同帧actor/critic多次读取不重复积分，部分reset与未reset环境相互隔离。
- RNG不被encoder构造改变，clock与learned初态及PPO预算一致。
- GPU真实计算进程/旧锁身份、队列child退出和失败不被误判为完成。
- 三训练seed而非90回合为推断重复；跌倒后固定时域指标、全部三组配对完整。

### Task 1: 在线数学接口

**Files:** Create `scripts/g1_stateful_online.py`; Test `tests/test_g1_stateful_flat_ppo.py`.

**Interfaces:** `OnlinePhase(model, mean, std, num_envs, device)`，`features(q, quat)`，`observe(q, quat, stamp)`，`reset(ids)`；加载器bundle提供冻结encoder。

- [x] 写特征与SciPy、完整PhaseNet预测、重复帧/部分reset、RNG保留测试，运行确认缺实现失败。
- [x] 实现torch四元数特征及逐层卷积缓存，状态按env隔离；限定路径观察适配。
- [x] 运行CPU测试，记录误差和全部通过结果。

### Task 2: 协议与执行队列

**Files:** Create `scripts/g1_stateful_flat_ppo.py`, `scripts/run_g1_stateful_flat_ppo_20261025.py`; extend same test file.

**Interfaces:** 新study模块满足旧train/evaluate所需常量/配置/工件接口；runner提供 `--stage queue|smoke|train|evaluate|summarize`。

- [x] 写矩阵、相位以外配置相同、恢复身份、GPU等待/锁、三seed统计及缺失配对失败测试，确认RED。
- [x] 实现固定协议、来源验证、封存、受控复用旧runner和新GPU等待队列；正式评测前封存6策略。
- [x] CPU suite通过；独立检查，实际阻塞修复用回归测试验证。

### Task 3: Smoke与正式运行

**Files:** Output `results/g1_stateful_flat_ppo_20261025/`。

- [ ] 核验GPU空闲后在线GPU数值对齐和两臂2更新smoke；保存失败和日志。
- [ ] smoke成功后封存全部新源码/原权重/配置并串行运行6训练，固定最终策略封存。
- [ ] 9评测全部完成后审计270回合配对、encoder冻结、预算，输出两族完整效应和报告。
- [ ] 长任务启动后更新官方heartbeat持续跟进；结果就绪后交付并停止跟进。

## 执行账本

- 设计/预算/seed为执行者依据已获授权旧三组设计固定的决定；不是声称用户逐参数选择。
- 无新worktree：本checkout含必要未跟踪科学资产，按独立目录和SHA隔离；不修改旧封存源码。
- 实验当前从准备开始；未启动GPU，与其他项目不重叠。
- Tasks 1/2 complete：40项CPU合并回归通过。独立审查0 Critical/1 Important/1 Minor；评测bootstrap精度修复、旧smoke复用保护均有RED→GREEN回归；额外cuDNN显式设置同样有RED→GREEN。
- 静态门Ruling：原review_gate将PyTorch Module.eval()误判为Python eval；原输出保留，保留正确推理语义。seed_audit仅剩未识别子进程启动前dict(os.environ,PYTHONHASHSEED=...)的模式；真实环境由队列设置，不伪报扫描全绿。见verification/manual_gate_adjudication.json。
- 本计划作为启动时源码封存的一部分；GPU工作开始后不再勾改本文件，后续进度写结果目录progress.md。
