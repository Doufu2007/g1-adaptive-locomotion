# S实施台账

计划：docs/superpowers/plans/2026-10-28-g1-rough-domain-adaptation.md

2026-10-27：设计fresh builder-check通过；六D501完整状态CPU审计PASS，0采样。首批7契约测试RED→GREEN。

Ruling: 原生randomize_terrain只在play配置创建，训练配置缺失；S统一四臂在reset_base前插入同一deterministic事件，已有时替换 — 原设计“替换”假设不适用于训练 — 若错误将影响origin配对，因此CPU顺序及后续真实hfield接触门保留。零采样配置测试复现缺失异常，未构造环境/未PPO；不构成采样技术失败成本。

尚无S协议封存、真实smoke、训练或队列。Task1真实配置回归与Task2恢复实现进行中。

Task1 CPU真实12配置gold与六D501原生CPU完整restore/真实ONNX预检PASS（0环境步/PPO）。Task2/3训练和独立namespace评测实现完成，初批14测试/4语法PASS。

Fresh实现review发现6项Important、无Critical：实体DOF API不存在/字符串names被旧numeric digest拒绝、mjlab load恢复common counter、rough构造提前消耗RNG影响startup配对、analysis完成先于S绑定核验、运行期磁盘门覆盖。前三项真实native/MuJoCo CPU RED→GREEN；其余构造随机流、完成落盘、磁盘回归RED→GREEN。

Ruling: 在新S Env子类load_managers入口重设四臂相同source seed构造随机流 — 地形origin初始化即使curriculum=False仍抽torch randint，不能让其改变startup摩擦/COM和interval/command初态 — 若接口调用顺序变化，真实0step物理配对门将拒绝；此为全部四臂共同控制，不改原生随机化分布/PPO或地形因子。之后仍在全部构造后恢复D源RNG。

Ruling: names数组用支持字符串的shape/dtype/字节digest，DOF用原生速度地址与joint[axis]命名包含浮动根；common_step_counter在mjlab load之后归零 — 不把warm start说成原仿真续接。

研究计划target/failure均PASS，完整package WARN仅H2未隔离状态反馈；已降为完整管线探索、n3最小p=.25。不是采样故障；所有开发回归均0环境步/PPO，未产生训练技术损耗。

尚待最终fresh review定点通过、修复后零采样preflight和最终协议封存；未运行任何S真实smoke/训练/评测/队列。

Final CPU: 41相关测试/4语法PASS；fresh实现审查最终7Important全部关闭/Minor关闭。修复后CPU preflight PASS，277来源协议SHA8d06e16ef96741333596080ed687c1831143b2f2ba18e81ab04904251020786a。21:22:47Z固定smoke隐藏PID24236，尚未核实采样推进；formal queue未启动，不标整体完成。

Technical failure01：首smoke在0step拒绝原生35个空名geom，已完整归档0train/eval/PPO/Adam。technical_amendment_01前锁修复，原生G1 CPU gold全部68geom/35DOF覆盖、41测试/4语法/fresh定点review/preflight再PASS，S协议修订为0a1ad3b7d6b8023db05a567d108d737e17236c386b08d07ec0136ea1413eadc2/280来源。21:30:22Z重新隐藏smoke PID32864，已核FC2更新/3072train/40Adam真实完成，非整套smoke成功，formal queue未开始。

2026-10-27T21:42Z：修订smoke四训练+八评测全部完成，12288train/80eval、8PPO160Adam；自动工程审计及补充独立CPU审计均PASS。唯一正式queue于21:40:51Z隐藏PID15892，首项FC20261903实际12/251更新、18432train/240Adam，runtime_confirmation.json；不是只派发、不是性能结论或整体完成。保持12训练+16评测预锁预算与停止规则，禁止重复启动。
