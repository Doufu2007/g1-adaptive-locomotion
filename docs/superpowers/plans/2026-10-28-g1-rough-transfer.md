# G1 Rough Transfer Implementation Plan

> **For agentic workers:** Execute inline using executing-plans; a fresh builder reads the design and a fresh reviewer checks the final implementation. 用户已授权持续探索，本机只新增实验文件、不提交/清理，不需要重复许可。

**Goal:** 给最新最佳E与匹配clock补充有限、前瞻、可审计的rough迁移证据，0新增PPO。
**Architecture:** 三文件分工：契约/评分纯CPU模块，fresh-child GPU评测与顺序queue入口，必要统计/路由/暴露测试。复用既有50Hz bootstrap和最新E运行时，四组全保留，禁止改冻结源码。
**Tech Stack:** Python3.13, NumPy/SciPy, torch/MuJoCo-Warp, ONNX CPU。
**Spec:** docs/superpowers/specs/2026-10-28-g1-rough-transfer-design.md

## Global Constraints
- 16 method×terrain组，15world，steady/push各20s，480回合最多480000正式eval transitions，0PPO；四smoke最多40eval另账。
- AE/AC/DE/DC全部501、原三训练seed；新三rough seed×5patch固定，flat5初态保护。
- 98维actor，无height_scan；共同rough求解配置、实际foot接触geometry暴露、父协议/来源SHA。
- 不改A/B/C/D/site-packages、不提交/清理、不连接旧服务器、不关闭本机、不扩预算追赢。

## Review Focus
- native nacon计数/跨world接触不能错配；无效efc/address与邻patch不能算rough。
- 地形UUID不能影响物理hash，hfield数组或几何位置变化必须被发现。
- 跌倒后的自动reset记录不可拼接，完整MAE和完成率必须计入失败。
- 来源缺失/已存在未完成目录不能被当作完成后跳过或自动覆盖。
- n3零方差不能产生虚假p0；统一8项Holm，不把地图/回合当独立训练。

### Task 1: 纯CPU契约与评分
Create scripts/g1_rough_transfer_contract_20261028.py; tests/test_g1_rough_transfer_20261028.py。
Interfaces: build_jobs()16项；world_slots()15项；route_actions(obs,sessions)；contact_exposure(geom,world,positions,dist,margin,efc,foot_ids,terrain_ids,assigned,bounds,num_envs)；episode_metrics(raw,horizon,scenario,fell,rough)；paired_analysis(rows)。
- [ ] 先写gold测试：跨world足地接触、邻patch/非约束/腾空、早跌倒MAE补零与站立不成功、n3精确p下界.25、缺失矩阵拒绝、地图不作训练n。
- [ ] 运行RED并保留日志。
- [ ] 实现最小函数、GREEN；维护0/40/480k分账。

### Task 2: 环境/原始记录/冻结queue
Create scripts/g1_rough_transfer_experiment_20261028.py。
Interfaces: make_cfg(study,arm,terrain_seed,smoke=False)；policy_entry(method,seed)；lock()；rollout(method,terrain_seed,smoke=False)；queue()；analyze()。
- [ ] 先写配置结构与heightfieldhash、fresh-child计划/防覆盖测试，并RED。
- [ ] 写98维flat基底+rough地形，确定性origins；按slot路由同训练seed的ONNX；保存raw基本字段及实际contactpoint记录、hfield物理数组；每50step落progress和真实采样计数。
- [ ] 完整来源lock、四smoke单进程各10steps、正式16组单workerfresh child、完成自动analyze，失败保留且停止。
- [ ] CPU preflight核所有公共cfg/actor结构、训练与policySHA、16job矩阵；GREEN相关回归及三文件语法。

### Task 3: 审查、前瞻封存、采样、复核
- [ ] fresh独立review，修Critical/Important并补回归。
- [ ] 源码/设计/计划/测试/12模型对/编码器/原生地形源/父结果冻结；真实四smoke40eval、0PPO核hash/配对通过。
- [ ] 正式queue隐藏启动一次，核实际step推进而非仅派发；更新交接/heartbeat事实源。
- [ ] 16组全部完成后独立复算480raw SHA、物理/初态/command/hfield配对、primary/guard/exposure与8项family；如实给候选/失败/无结论，不暂停于“已启动”。

## Execution rulings
- 用户已授权“继续探索剩余没有证明的主张”，按其偏好自主执行可逆有限评测；不重复问是否继续。模型/地图/终点/阈值为操作者前瞻选择，非声称用户逐项指定。
- 工作路径复用已在进行的共享实验workspace，只新建独立文件，冻结旧来源不改；不创建会丢失现有大模型依赖的新checkout，不提交/清理。
- 此处未达新seed确认；严格n3探索，不能在最终报告称统计显著优势。未来新seed确认需先过本轮候选再锁方案。
