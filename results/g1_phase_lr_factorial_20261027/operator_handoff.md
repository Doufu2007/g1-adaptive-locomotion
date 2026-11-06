# 本聊天最新任务交接（2026-10-27）

用户授权：按四步完成最佳E对clock原因排查和修正验证；禁止把修好首轮更新等同于E优越。回复简体中文。

当前有效实验：<project-root>/results/g1_phase_lr_factorial_20261027
原始服务器实验已经备份并关机。不要连接/重启旧服务器，不要关闭本机。

阶段A：四组phase(clock/learned)×初始LR(default .001 / low .00001)，3训练seed20261901/02/03，每项501×64×24=769536，共12项9234432 transitions；自适应保持。E seed20261026固定SHA 8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346；源actor9900固定SHA 949fae33632b9a3592e6346a6155eea059ee2fd39487207a0a0cea5d35edd8b4。正式540回合；0/1/25/100/251各三初态恒速，501十初态×三场景。正式终点固定501，不选最好checkpoint/seed。

执行：parallel_launch.json记录控制器PID19936，两个worker按冻结jobs偶/奇索引处理不交叉的cells。启动时worker0接管首项clock_default_20261902训练子PID31404，worker1从clock_low_20261902开始。2026-10-27T02:34Z复核：这两项均已完整完成501更新和ONNX导出，当前两个worker继续它们的固定检查点评测；不要再等待旧训练PID。以最新queue/worker/progress为事实源。queue_status.json、parallel_worker_0/1.json、training/<cell>/progress.json、parallel_queue_console.log是当前事实源；最早的launch.json/active_child.json属旧串行控制器（交接记录controller_handoff.json）。不要重复启动队列。

脚本：scripts/g1_phase_lr_factorial_20261027.py（训练/评测/统计契约）、run_g1_phase_lr_factorial_parallel_20261027.py（当前控制器）、analyze_g1_phase_lr_factorial_20261027.py（全部重算，控制器完成后自动运行）、analyze_g1_phase_lr_mechanism_20261027.py（额外固定状态漂移；阶段A完成后人工运行一次）。原源码不改；新增脚本和协议已封存，改动前保留版本和明确技术amendment。

已完成：8项单元测试、4模块语法、四组各2更新smoke共12288transitions及4个短评测；四组同seed初始化/物理一致；新更新记录器与原生PPO在冻结batch上actor/critic和RNG逐位相同。独立代码审查无初始化/计数/统计实质错误；已补extended_source_seal，分析入口逐文件复核。还有一个已记录恢复窗口：底层评测可能已COMPLETED但来源元数据未附加，若真的触发先核对checkpoint与raw SHA，仅幂等补来源字段，不重跑或覆盖原数据。

保留失败：technical_failure_01/02均在0采样/0optimizer时因自定义Velocity保存器的文件名/日志初始化依赖失败。有效实现使用MjlabOnPolicyRunner父类保存模型，训练完再导出所有ONNX。每个失败前做了2次技术冻结回放，有效版本另2次，所以技术回放总6次；科学budget另计，无隐藏追加采样。technical_amendment_01/02有证据。

阶段A完成后：核对12×501和540 raw SHA、配对初态及来源；运行mechanism补充；按prospective_protocol/spec冻结的四项family计算训练seed级配对t区间+Holm；n=3探索性。候选门：E_low−clock_low平均≤−.005m/s、至少2/3seed改善、无新跌倒、倾斜增量≤.02rad/航向≤.2rad/初始方向位移≥−.5m。仅形成候选，不能声称稳定更好。

后续按用户批准四步：候选成立→先锁唯一方案和新训练seed的独立确认，依据配对方差规划样本量与资源；候选未成立→先按中间检查点、训练KL、动作漂移、步幅、相位误差定位差异，再事前锁定只用当前/过去触地信息的参考相位构造并做同预算训练对照。参考不是完美真值，不能仅凭它获益就认定E数值bug；后续clock基础节奏+E修正属于新方法，另立比较。不要无限加预算碰运气，不提高reward再挑好seed，不把单个好结果当完成目标。

当前任务尚未完成，继续执行条件分支；最终需如实区分已定位机制、修正是否有效、是否独立验证稳定优于clock。若无优势，报告实测限制，不保证必定胜出。正常运行无可操作变化时保持安静。

工具：PowerShell曾挂起，优先隐藏Node REPL tools.mcp__node_repl__js（ALL_TOOLS可发现）。fs读写；Python为<data-root>/Program Files/Python313/python.exe，入口需site.addsitedir('<user-profile>/AppData/Roaming/Python/Python313/site-packages')。子进程使用env启动PYTHONHASHSEED=20263300、PYTHONIOENCODING=utf-8，windowsHide=true。不要复用系统HOME变量。源码已冻结；进度写progress.md，不改plan。不要因原始父目录旧FAILED误判有效版本。

补充2026-10-27T09:01Z：12项训练均COMPLETED，剩余评测仍在跑。机制CPU分析已启动，mechanism_launch.json/PID28804和mechanism_analysis.log为执行记录；优先检查mechanism_supplement.json是否COMPLETED，勿重复启动。未新增任何训练或评测采样。

补充2026-10-27T09:36Z：阶段A已STAGE_A_COMPLETED（09:12:33Z），两个旧worker全部完成，勿重启。12×501与540原始评测已审计，analysis/diagnosis和mechanism_supplement已COMPLETED。E_low .05501 vs clock_low .04665m/s，只赢1/3seed且位移少1.127m，候选门不通过。机制解释见mechanism_interpretation.md。当前转入阶段B，事实源为 results/g1_contact_phase_reference_20261027/operator_handoff.md。
