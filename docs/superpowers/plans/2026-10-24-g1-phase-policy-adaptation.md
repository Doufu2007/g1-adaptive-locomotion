# G1 Phase Policy Adaptation Implementation Plan

> **For agentic workers:** Use executing-plans to implement this authorized experiment inline. Track checks and decisions in the experiment progress ledger.

**Goal:** 当前闭环研究结束后，自动运行冻结 TCTST 的等预算控制策略适配和三组评测。

**Architecture:** 复用现有 MuJoCo bootstrap、PPO actor 加载和逐回合评分。新增独立实验入口与共同配置工厂，使用有状态串行队列；所有来源资产哈希封存。

**Tech Stack:** Python 3.13、PyTorch、mjlab 1.2、RSL-RL、MuJoCo/Warp、pytest。

**Spec:** docs/superpowers/specs/2026-10-24-g1-phase-policy-adaptation-design.md

## Global Constraints

- 当前评估完成之前不启动新 GPU 作业；冻结 encoder SHA 不变；不得改旧研究封存文件。
- 64 env × 24 steps × 501 updates，各臂 769536 transitions；3 paired PPO seeds；2 terrains；3 evaluation combinations。
- 两臂共同移除 clock foot_gait，其他奖励逐字段一致；禁止按结果挑选检查点、种子或加预算。

## Review Focus

- actor 与 critic 相位是否都切换，是否留有隐蔽 clock 奖励。
- 后台进程退出不等于研究完成；需要全部 360 个有效 episode 及封存 hash 通过。
- 相同 checkpoint/critic/optimizer/训练 transition；重启不重复计算已完成更新。
- 评测必须使用每个 seed 的对应策略，第二组与第一组同 actor。
- 训练 seed 为算法重复，评测 episode 不能扩大算法独立样本数。

## Tasks

- [x] 1. 写有意义的配置、公平性、队列依赖和三组配对测试，观察缺实现失败。
- [x] 2. 实现共同配置工厂与协议/资产封存；通过 CPU 测试。
- [x] 3. 实现短程 PPO 训练、冻结验证、50-update checkpoint 和可追踪失败记录。
- [x] 4. 实现三组合评测及按训练 seed 分层的汇总。
- [x] 5. 用新汇总器修复旧汇总脚本的 batch seed 分片/结果索引问题，保留旧文件及 hash。
- [x] 6. 实现串行依赖队列：等待旧评估完成→旧结果汇总→GPU smoke→资产封存→正式训练→三组评测→汇总。
- [x] 7. fresh-context 审查、修复恢复/导出问题；7 项 CPU 测试通过，旧 79 项封存资产通过；交给等待队列和后续检查。

GPU smoke 及正式数值结果仍待现有评估完成后产生；上述勾选只表示实现任务完成。运行状态记录在 results/g1_phase_controller_adaptation_20261024/queue_status.json 与 progress.md，不在封存后改写本计划。
