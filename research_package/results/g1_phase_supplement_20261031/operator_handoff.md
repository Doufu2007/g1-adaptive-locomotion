# 粗糙地形学习相位补充实验交接

用户已授权三项补充并要求执行，不要重新问是否继续；当前真正阻断是磁盘空间。

原主张：近期运动学习相位替代固定CPG，使节奏随实际状态调整，改善稳定/鲁棒，限定粗糙地形。用户不要求平地或任意固定周期普遍支配。误贴目标检测解冻任务已撤回，勿运行。

新spec顶部审查修订为准：8新续训种子20271301..08×learned/clock55/clock60，每cell752×64×24；24cell正式27,721,728train/18048PPO/360960Adam。共同source9900 actor+normalizer、seed配对新critic、空Adam、LR1e-5自适应、冻结source教师beta.1和原E编码器。是条件于同一预训练源的新续训，不是8独立预训练来源，不恢复旧D/S Adam/RNG。

校准地图20271101..03/初态20271111..15共120回合；两个重训clock验证20271501..03/初态20271511..15共240，按全局完成率/跌倒/0.55顺序锁一个周期；最终20271601..06/初态20271611..15、live/constant/replay/selected_clock、steady/push03/push06，共2880。总3240，最大3,240,000world步；评测0PPO。常速/回放同一个learned actor，仅替换观测9:11，保持转动，不使用后续状态；校准串从所有存活增量按固定次序建立，首步相位对齐。

所有新源码和审查/测试/真实smoke已完成；29相关测试PASS，新contract8PASS，fresh设计/实现review均PASS。CPU零采样、GPU三臂各2更新9216train/6PPO/120Adam、CPU6×200=1200eval（含重复）及独立raw工程审计PASS。三臂额外零采样初态配对PASS。不要重跑smoke。

原正式协议SHA31535c957252fd9a31d33db630ecb21a53e88941fb77ee4bebec9498750cb1d8；源files和snapshot冻结，勿编辑这些源码。唯一原正式launch PID14064，queue_started锁存在，但在首个子训练进程生成前因C盘低于10GiB停止；正式0train/eval/PPO/Adam，0jobs，没有正式training/rollouts manifests。失败目录/锁/原协议禁止清理/重命名/重启。

当前C仅约1.4GiB空闲，E约58.6GiB。read-only检查页面文件约29.7GB；缺少原尺寸，不作确定因果归因。所有新大结果本来就在E。不得清理别的任务/旧结果、改系统页面文件、关本机、连接旧服务器，不能降低预锁资源门追启动。

恢复已于2026-10-31T07:08:17Z prepare完成：独立scripts/run_g1_phase_supplement_recovery_01_20261031.py经过technical_recovery_01_review.json PASS（wrapper SHA6adf6151e027f9e742d12c085281cc8ec7b76d0b2a2b12c525c100c322ae3ff4；旧289输入全部复核）。它验证旧协议和零正式采样，创建全新E:/phasor_phase_supplement_20261031/formal_recovery_01与当前控制包/formal_recovery_01；继承相同科学协议/源码/预算，fresh仿真从零，并非精确续接。ready状态PREPARED_WAITING_FOR_DISK，没有新launch/queue锁或进程。实际launch仍检查C≥10GiB/E≥15GiB后才预留launch锁。无自动重试循环。恢复若失败再停，不改beta/奖励/难度/map/seed/checkpoint/指标/阈值/训练budget追赢。

原launcher/status helper仅供查看旧故障，不得再launch。恢复准备后只用新wrapper的status/launch；新队列仍是单worker fresh child，24训练→校准bank→240验证锁周期→四臂正式→analysis+独立raw审计自动执行。正式完成前不得声称主张已验证或实验整体完成。所有旧A/B/C/D/R/S/确认/扫描及负结果保留，原模型不改。

Windows默认PowerShell工具命令本轮存在解析挂起；使用shell=cmd.exe与无引号短路径C:\Progra~1\Python313\python.exe运行saved launcher，不依赖shell profile。不要用cmd的分号串命令。无关AIC项目Python/SSH进程不要触碰。

当前阻断已通过异步输入卡告知用户：可自行释放C盘空间，或要求先列可迁移大文件供确认。尚无清理/迁移新授权。不要因时间过去而默认取得授权，也不要启动循环重试。下一步先看用户答复和实际磁盘，若磁盘恢复到门槛且旧状态仍0，使用恢复wrapper launch一次并核真实progress。整体科学实验未完成，不报告任何新的优势结论。
