# 粗糙地形补充实验进度

用户2026-10-31授权执行反馈消融、新种子训练复现、分级扰动验证。规格顶部“采样前审查修订”为权威版本；旧3来源正式机制矩阵已取消，最终四模式统一用8个新续训种子。

- 设计fresh builder审查通过，代码fresh实现审查修复6项后PASS；review JSON含精确SHA。
- 新contract 8测试PASS；加旧rough domain共29测试PASS。另跑旧confirmation测试时1个历史状态测试因旧queue_started已存在而失败，未改旧结果/测试绕过，不属于新实现故障。
- CPU真实零采样三配置preflight PASS，actor98/critic113，foot_gait奖励不存在。
- 首次路径检查在任何env/采样前因repr转义误判（0train/eval/PPO/Adam），已修为Path.resolve及内容hash；原失败日志和资源证据保留。没有按性能调参。
- GPU三臂2更新全部完成，9216train、6PPO、120Adam；CPU六次短rollout各200步，共1200eval，含live重复。新增三臂初始化配对preflight均零采样完成，初始root/joint pose/vel摘要相同。
- 当前版本audit-smoke已PASS：模型、物理/初态、逐帧ONNX动作、逐帧接触、开环干预、推力和CPU bitwise重复均核对；1200eval步、9216train/6PPO/120Adam分账。
- 正式协议已封存，SHA31535c957252fd9a31d33db630ecb21a53e88941fb77ee4bebec9498750cb1d8。
- 唯一正式launch PID14064在2026-10-31T06:59:26Z启动；随后首个run_child启动前因C盘低于10GiB门停止。0已完成job，没有任何正式training/rollouts目录/manifest，正式train/eval/PPO/Adam均严格0。队列/launch锁和源snapshot全部保留，无重复启动。
- 失败资源与日志：failure_evidence_pid14064.json、queue_status.json、queue_console.log、disk_failure_diagnosis.json。C约1.3–1.4GiB，E约58.6GiB；页面文件当前29.7GB。无此前页文件大小，不宣称已确定其导致骤降；已扫描cache/Temp/AIC近期写入，没有相应9GB增量。
- 新独立恢复wrapper仅改变formal_recovery_01输出/控制/child入口，不改原封存采样实现/设计/预算。技术恢复review已PASS，prepare已完成，状态PREPARED_WAITING_FOR_DISK、0新采样；实际启动依旧要求C≥10GiB/E≥15GiB，不清理、不降低阈值、不自动重试。

冻结plan实际执行台账：Task1–6已完成；Task7首次启动因资源门在0采样停止、恢复入口已review/prepare待磁盘；Task8正式实验与科学分析尚未完成。plan源文件本身保持封存SHA，不勾选修改。

正式固定预算：24cell×752×64×24=27,721,728训练步，18,048 PPO/360,960 Adam；校准120+固定周期验证240+四模式测试2880=3240回合，最多3,240,000 simulator-world，有效步另计，评测0PPO。8续训种子条件于共同source9900和固定E编码器，不冒称独立预训练来源。

大产物E:/phasor_phase_supplement_20261031。小控制包当前目录。单worker fresh child，技术失败保留并停，无自动重试。旧结果不改、不提交/清理、不连接旧服务器。
