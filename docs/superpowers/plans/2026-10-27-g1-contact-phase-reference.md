# 阶段B：触地参考相位执行计划
Spec: docs/superpowers/specs/2026-10-27-g1-contact-phase-reference-design.md

## 全局约束
用户授权继续全链路排查；当前工作目录已有冻结实验，新增文件隔离，无提交/清理/旧服务器/关机。本计划和spec先封存；代码完成后技术smoke通过再正式采样。

## Task 1: 状态器与训练批次校准
Interfaces: scripts/g1_contact_phase_reference.py 导出 ContactPhaseReference(num_envs,device,targets)、update(stamp,episode_steps,contact)->[B,2]、reset(ids)；校准函数消费训练batch，输出每脚事件数/圆均值/集中度与SHA。
先写行为测试并确认失败，再实现：无事件时nominalclock；3帧确认/初始触地不算事件；相同时间戳幂等；部分重置隔离；合法周期更新/非法拒绝；相位单调及速度边界；未来前缀不变；同时事件处理；数据非有限/跳号拒绝；校准终止边界和来源。
Verification: python -m unittest tests.test_g1_contact_phase_reference_20261027。先锁校准规则再运行真实训练批次校准，过门后记录校准SHA。

## Task 2: 配对训练评测适配器
Interfaces: 新脚本进程内复用A.train；重绑定OUT/lock/setup，make_cfg只替换reference phase和专属reset。评测复用batch原始轨迹链，仅本进程重绑定clock phase读取至reference以检查actor/记录一致。其他代码不改。
先测试配置摘要与对照拒缺失/源SHA变化，再实现封存、训练、评测与分析。冻结源A全部协议链和新文件。技术smoke2更新+短评测，与A同seed初始化actor/critic/common_cfg/common_agent/physics匹配；失败保留记录和实际消耗。正式用2worker固定不重叠seed，状态原子写入；0/1/25/100/251/501均评测。
Verification: 既有A八项单元+新核心单元，语法；smoke确切3072；评测phase记录与actor一致；初始化匹配。

## Task 3: 正式训练、分析与后续分支
Interfaces: 三seed501更新与135原始回合，analysis.json/diagnosis.md/queue_status.json。先核来源/原始SHA/预算/同检查点初态物理，再用冻结两项family、seed级CI/Holm、保护条件决定候选。中间点只定位机制。
Verification: 三项769536、总2308608，135回合完整、对应A低LR初态/physics逐条一致。只有候选才事前锁独立验证。否则按spec因果分支继续，不把A/B探索性完成虚报全部目标完成。
