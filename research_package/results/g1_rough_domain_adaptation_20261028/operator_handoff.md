# S交接：smoke双审计PASS，正式有界队列真实推进

截至2026-10-27T21:42:13Z，S正式queue于21:40:51Z隐藏启动PID15892；queue_started.json排他标记已创建，禁止重复启动。首项FC/source20261903已真实12/251更新、18432train、12PPO/240Adam（runtime_confirmation.json），完整D501 RNG相等、CUDA RNG恢复及源Adam10020步已核。队列是单worker，12训练cell后16评测组，各fresh child；读取training各progress，不以尚未最终刷新的manifest判断停滞。queue_status与queue_console.log及train_*.log为当前运行证据。整体任务未完成，不能仅启动队列算完成。

当前协议SHA0a1ad3b7d6b8023db05a567d108d737e17236c386b08d07ec0136ea1413eadc2，280来源封存。41相关CPU测试/4语法、12配置/六D501真实mjlab CPU完整restore/6ONNX零采样preflight PASS；fresh设计builder与实现review通过（7Important、1Minor全关闭）。计划target/failure PASS，完整package WARN仅反馈未隔离，S仅完整管线探索。

修订smoke PID32864已2026-10-27T21:39:15Z全部完成：FC/FE/RC/RE各2更新=12288train/8PPO160Adam；四模型各flat/首新rough八项各10帧=80eval/0评测PPO。smoke_audit.json工程PASS；smoke_independent_audit_01.json独立CPU复算完整源恢复/12checkpoint/预算/teacher及encoder冻结/8raw SHA/真实指定hfield接触/物理初态/模型绑定PASS。补充工程审计在封存后写成，不冒称结果前科学审计、第二review或GPU长期稳定证明。formal已开始，不重复smoke。

S四臂：FC/FE=flat额外训练clock/learned，RC/RE=rough额外训练clock/learned；来源20261901/02/03为同3个D501，非新独立训练seed。每cell额外251×64×24=385536train，12cell4626432train/3012PPO60240Adam。完整载入actor/critic/Adam/normalizer/原source9900 teacher/实际LR/RNG，beta.1与最新encoder冻结；clock02源LR2.25e-5，其余约1e-5，不重置优化器。源Adam10020、最终累计15040；一次native learn251，不循环learn1。相对checkpoint0/1/25/100/251，主比较只最终251。新仿真非精确轨迹续接。

构造：S Env子类scene/terrain后load_managers入口四臂共同按source seed重新对齐Python/NumPy/CPU-CUDA RNG，避免rough origin randint影响startup物理/command/timer，然后保持原生规则；全部构造后恢复D源RNG。common_step_counter在mjlab load后归零。同phase跨域完整状态/RNG及机器人命名动力学配对，rough须真实指定hfield接触。每step有限/容量、每更新20旧KL与20sourceKL、更新边界暴露覆盖明确非全帧，教师/encoder冻结。磁盘10GiB门。共98actor/原critic与reward、solver500/500/1500/128。

train maps20264601..03；eval flat0及20264701..03、初态20264811..15；R地图20264201..03不回流，完整hfield数组隔离。正式480eval最多480000有效步/0评测PPO。第一native终止不拼接、早倒完整20sMAE补零、实际指定patch足接触与边界门；完成要求无倒20s、前向位移>=5m、moving>=.8及rough内部接触>=.95。16评测组完成自动analysis后，必须运行采样前封存independent_audit.py，复算480raw/模型/物理起点/接触/原生终止/8family/候选、12训练恢复与预算及60checkpoint。这是同操作者独立计算，不冒称第二review或新seed。

候选：RE同时对RC与FC完成+.10/跌倒-.10、各至少2/3seed方向，rough四保护及RE-FE flat六保护全过；8seed级family统一Holm，n3最小精确双侧p=.25，不能把480回合当480独立训练。域门有利不等于phase候选。仅候选通过才能另锁全新seed/map确认及匹配反馈干预。失败停止S分支，不自动改beta/奖励/编码器/地图/难度/seed/checkpoint/主指标/阈值或扩预算追赢。

技术失败01：原smoke在首次控制步前因原生35个未命名geom被检查拒绝，0train/eval/PPO/Adam；旧协议8d06e16ef96741333596080ed687c1831143b2f2ba18e81ab04904251020786a与snapshot/失败manifest/日志/资源证据完整保存technical_failure_01。technical_amendment_01预锁修复，保留全部geom，以父body+局部序号标识空名，未改动力学。native G1 CPU gold30body/68geom/35DOF、回归/fresh定点review/preflight再PASS后当前协议重新封存；不是自动重试或性能调参。NVML既有Unknown Error，真实四训练smokeCUDA成功不保证长期稳定。

R已2026-10-27T20:21:36Z全16组480回合完成、独立审计PASS。rough各90：AC完成4/倒83、AE1/86、DC9/71、DE13/59。DE-DC完成+4.44pp（CI-8.20..17.09）、跌倒-13.33pp（CI-57.15..30.48），p=.5、8Holm1；对DC完成+.10门失败，联合候选false。push完成DE=DC=5/45，rough phase/contact关联未一致；源12环境均plane只证明域差异，不证明原因。R有效289880/world465600、0PPO，smoke40eval另账，无R采样故障。D已全完成独立audit PASS，E平地MAE.0477710对同修正clock.0388324，速度候选失败；两phase bankKL都下降仅说明source偏离约束生效。A/B/C及旧TCN20rough负结果保留。

成本分账：S正式与成功smoke/失败01分开；历史D失败train531456..1542144/PPO346..1004/Adam6920..20080精确未知，C旧eval6750..9000未知，D成功smoke6144train20eval/4PPO80Adam另账。新故障先取进程CPU/GPU/系统/日志和保留目录/协议/源码/真实采样，再评估及预锁同有效预算恢复；无自动循环、无完整仿真state不称精确续接。旧服务器不连接/重启，本机不关闭，不提交/清理，不改冻结A/B/C/D/R/S或site-packages。另有其他任务Python，不干预。

原主张是状态相位改善稳定/鲁棒尤其rough；目前仍未建立确认性优势，不用平地速度替代全部主张。健康无可操作变化安静，仅实质结果/故障/需操作通知；heartbeat继续ACTIVE，整体真正结束才报告并暂停。

最近健康核查 2026-10-27T22:58:32.375Z：4/12训练cell完整完成（FC03、RE03、RE01、RC01），当前RC02 51/251更新；0/16正式评测。PID15892及fresh子进程存活，280封存来源verify_protocol PASS，rough完成项均有真实指定patch接触，teacher/encoder冻结及一次native learn标记齐全；磁盘约36.2GB。队列正常，禁止重复启动；当前仅工程进度，尚无S性能结果。

健康核查 2026-10-27T23:48:04.095Z：7/12训练完整完成，当前RE02 8/251更新；0/16正式评测。PID15892及fresh子进程存活，280来源校验PASS，已完成项teacher/encoder冻结和5固定checkpoint齐全，rough真实接触存在；磁盘约35.6GB。成本账按当前progress刷新，继续原固定队列，无新故障/无S性能结论，禁止重复启动。

健康核查 2026-10-28T00:33:43.795Z：8/12训练完整完成，当前FE03 91/251更新（progress时间2026-10-28T00:33:41.874034+00:00）；0/16正式评测。PID15892与fresh子进程存活、CPU有实际计算、280来源verify_protocol PASS，已完成项5固定checkpoint/教师/编码器冻结齐全。磁盘约36.0GB，运行继续推进，未见新故障，成本账按progress刷新；不能由训练日志推断主张成立，不重复启动。

健康核查 2026-10-28T00:48:22.636Z：9/12训练完整完成，当前FC02 72/251更新；0/16正式评测。PID15892及fresh子进程存活，来源verify_protocol PASS，已完成项5 checkpoint/教师/编码器冻结齐全；磁盘约36.0GB，运行正常，无新故障或性能结论，成本账已刷新，禁止重复启动。

健康核查 2026-10-28T01:03:26.279Z：10/12训练完整完成，当前FE02 87/251更新；0/16正式评测。PID15892及fresh子进程存活，来源verify_protocol PASS；磁盘约35.9GB，运行正常，无新故障或性能结论，成本账已刷新，禁止重复启动。

健康核查 2026-10-28T01:19:06.418Z：11/12训练完整完成，当前FC01 110/251更新；0/16正式评测。PID15892及fresh子进程存活，来源verify_protocol PASS；磁盘约41.5GB，内存较低但进程仍有CPU推进，未见新故障或性能结论，成本账已刷新，禁止重复启动。

健康核查 2026-10-28T01:34:18.123Z：12/12训练完整完成，正式评测已开始，当前已完成2/16评测项（rollout目录以实际manifest/progress为准）；队列运行正常，来源校验PASS，磁盘约41.3GB，无新故障或性能结论，成本账已刷新，禁止重复启动。

健康核查 2026-10-28T01:35:48.367Z：12/12训练完成，正式评测完成3/16，当前RE/terrain20264701按progress推进（manifest未最终刷新）；PID15892与fresh子进程存活，来源verify_protocol PASS，评测均PPO_updates=0，成本账已刷新。尚无性能结论，继续固定队列，禁止重复启动。

健康核查 2026-10-28T01:49:48.935Z：12/12训练完成，正式评测完成7/16；当前RC/terrain20264702按progress推进，已完成回合使用PPO_updates=0。PID15892与fresh子进程存活，来源verify_protocol PASS，成本账已刷新，累计已知有效评测约154639；无新故障或性能结论，禁止重复启动。

健康核查 2026-10-28T02:04:41.979Z：12/12训练完成，正式评测完成12/16；当前FE/terrain20264701按progress推进，已完成评测PPO_updates均为0。PID15892与fresh子进程存活，来源verify_protocol PASS，累计已知有效评测约275888，成本账已刷新；无故障/性能结论，禁止重复启动。

S收尾 2026-10-28T02:37:21.332Z：queue已12/12训练、16/16评测完成并写analysis，独立CPU审计PASS（analysis SHA 901ef23f...，raw480/effective_eval358726/simulator480000；正式train4626432/PPO3012/Adam60240）。探索性候选门：RE相对RC与FC均PASS，flat六保护均PASS；rough RE−RC completion均值+0.1444 CI[-0.1217,0.4106]、fall均值-0.1444 CI[-0.1923,-0.0966]，RE−FC completion+0.2889、fall-0.3778；3复用source lineages、精确p最小.25、Holm=1。不能据此宣布独立优越或反馈因果确认。S不再追加预算；下一阶段须先锁全新eval seed/map与匹配live-vs-frozen/replay反馈消融，必要时再锁全新source lineages。
