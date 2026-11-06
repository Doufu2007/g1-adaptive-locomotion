# PHASOR：历史驱动在线相位条件控制

PHASOR 利用短时本体关节运动历史估计接触对齐的二维当前步态相位，并将该相位接入 Unitree G1 的 PPO 策略及相位相关反馈，以替代外生固定时间 clock。输入、标签时刻和控制接口见 [相位控制预注册](PHASE_CONTROL_PREREGISTRATION.md)。

方法贡献是“历史驱动的在线当前相位估计、固定 clock 替代、G1 粗糙地形闭环控制”这一组合。[相位主张复核](phase_claim_review.md)与[相关工作](related_work.md)比较了运动相位表示、CPG 调节、时间 clock 与运动历史估相位之间的具体差异。

## 实验资料

- [S：等预算粗糙适配](../results/g1_rough_domain_adaptation_20261028/)：学习相位相对同样粗糙训练的 clock，完成率提高 14.44 个百分点、跌倒率降低 14.44 个百分点。
- [新地图确认](../results/g1_rough_confirmation_20261028/)：固定 S 权重，学习相位相对 clock 完成率 +3.33、跌倒率 −2.22 个百分点；相对每回合冻结相位的差异为 +33.33 / −55.56 个百分点。冻结对照同时停止节律推进。
- [GPU 补充](../results/g1_phase_supplement_gpu_20261106/)：八个续训种子、固定地图、校准均速及循环回放对照；无推力行走与两档推扰分别报告。
- [历史结果总表](../results/g1_comparison_summary_20261028/)、[R 粗糙迁移](../results/g1_rough_transfer_20261028/)和[周期扫描](../results/g1_period_sweep_cpu_20261028/)提供前序实验与分析修订记录。

S 与确认复用三条训练来源；GPU 八个种子条件于同一个预训练源与冻结编码器。各批次保留原始统计区间、预设判定和审计记录。轨迹复算 `PASS` 表示计算及文件绑定验收通过，性能比较的统计结果保存在各自的分析文件中。

## 代码与复现

核心相位估计与运行时在 [src/tasks/velocity/phasor](../src/tasks/velocity/phasor/)。`scripts/` 提供历史实验所需的脚本依赖；GPU 实际执行入口和适配器的快照在 [source/](../results/g1_phase_supplement_gpu_20261106/source/)。[源码来源清单](../verification/phase_source_provenance.json)记录原件与公开副本哈希。

这些脚本记录实际实验实现。完整仿真重跑还需要安装与记录相符的 mjlab、MuJoCo Warp、PyTorch/CUDA、ONNX Runtime，以及恢复封存模型、相位库和输入目录。模型和约 4.08 GiB 原始归档未纳入 Git；脚本中的路径占位符需要映射回实际目录。历史协议包含原件 SHA256，路径清理后的文件不能直接冒充已封存原件通过 seal 校验。

公开副本只规范化路径及文本格式，数值、种子和历史科学判定保留。[结果来源清单](../verification/phase_evidence_provenance.json)同时给出原始与公开哈希，原报告内的哈希继续指向原始封存文件。
