# 连续相位实验执行计划

设计：../specs/2026-10-25-g1-stateful-phase-design.md。用户已授权直接执行，使用新脚本/结果目录隔离原封存资产。

1. 编写tests/test_g1_stateful_phase.py，先验证缺实现时失败：因果输入、四元数符号、接触去抖/周期、在线状态分块等价、完整周期损失反传、train-only归一化、队列依赖与矩阵/预算。
2. scripts/g1_stateful_phase.py：纯数据/特征/模型/训练损失/评分接口；CPU小型两次训练复现，输出hash证据。
3. scripts/run_g1_stateful_phase_20261025.py：原motion数据构造、固定来源新rollout采集、train/dev/test隔离、等预算训练/完整评分、恢复与不可覆盖输出。
4. 新results/g1_stateful_phase_20261025/保存预注册、数据血缘/配置/hash和progress；核查当前封存资产，排队等待，不提前启动GPU。
5. 独立复核实现，修复关键问题并保留测试；启动后台队列，验证真实进程和等待状态；将新实验交接纳入既有静默跟进。

重点核查：参考周期不得成为推理重置；后向特征不得读未来；新test不得拟合/挑选；失败回合不得删除；当前GPU队列未完成不得启动。报告点误差与完成率tradeoff，避免结构约束刷高门禁。
