# G1 Phase LR Factorial Implementation Plan

> **For agentic workers:** Use executing-plans for native execution; user already authorized the above experimental design and continuation.

**Goal:** 执行配对2×2完整适配，对首轮修正的长期效果及E相对clock收益作可复算判断。
**Architecture:** 新增独立脚本，复用冻结任务配置、Windows启动器、原生PPO更新和原评测器；原源码不改。每项独立进程，固定顺序，日志/检查点/输入SHA齐全。
**Tech Stack:** Python3.13 / torch2.6 CUDA / mjlab / rsl-rl / MuJoCo / NumPy。
**Spec:** docs/superpowers/specs/2026-10-27-g1-phase-lr-factorial-design.md

## Global Constraints
- 三seed、四组、501更新、64×24采样；不改奖励、E、源actor或主终点。
- 新目录results/g1_phase_lr_factorial_20261027，不修改旧结果，不提交，不关机。
- 当前工作区含大量既有未提交资产；本次仅新增文件，原位复用大模型/仿真资产。

## Review Focus
- checkpoint编号是已完成更新数，0是更新前，501是最终。
- 学习率只有初始值改变，自适应仍启用；同seed四组初始化一致。
- 训练seed是算法重复单位，回合嵌套聚合；缺配对或重复键应拒绝。
- 监测不能消耗额外随机数改变训练；输出和预算需真实核验。
- 失败不覆盖，恢复状态与额外采样分别记录。

### Task 1: 冻结协议及统计契约
**Files:** 新增scripts/g1_phase_lr_factorial_20261027.py、tests/test_g1_phase_lr_factorial_20261027.py。
**Interfaces:** build_jobs()；factorial_contrasts(rows,metric)；candidate_gate(per_seed,guards)。
- [ ] 先写缺功能的人工可算配对/交互/重复/保护条件测试并见失败。
- [ ] 实现最小纯逻辑，测试通过。
- [ ] 冻结种子、顺序、指标、比较、预算、源文件和环境，记录协议SHA。

### Task 2: 训练与检查点
**Interfaces:** train(arm,lr_mode,seed,smoke=False)；evaluate(job,completed_updates)。
- [ ] 复用原生runner和同初态构建，设置唯一学习率差异。
- [ ] 每更新记录KL、LR、损失、动作变化，保存固定检查点与RNG；监测不额外随机采样。
- [ ] 四组各2更新技术smoke，核对同seed初始actor/critic与配置、预算、encoder冻结和ONNX。
- [ ] 原生PPO与监测PPO冻结batch相同输出；优先复用已封存的验证，并新增实际首批一致性核对。

### Task 3: 正式执行、原始轨迹和核验
- [ ] 按冻结顺序执行12项501更新；各0/1/25/100/251三初态恒速，501十初态三场景。
- [ ] 原始轨迹重算指标、初态与启动物理配对，逐文件hash并保存所有失败。
- [ ] 整理四项比较与中间时间线；按预定分支执行后续，未支持不选好seed。

### Task 4: 结论与后续分支
- [ ] 判断训练修正、E差距及首个差异区间，各自列证据强度。
- [ ] 未形成候选时完成参考相位分支的事前实现锁定并执行；有候选则锁独立确认矩阵。
- [ ] 保存诊断、状态及证据清单，明确完成与未解决内容。
