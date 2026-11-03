# External Phase Models Implementation Plan

> **For agentic workers:** Use executing-plans inline; verify data boundaries, model adapters, then execute the three stages in the requested order.

**Goal:** 在冻结 G1 开发划分上实测 ModernTCN 10 帧、TCTST 100 帧、FLD 51 帧，保存可复算预测与 v4.2 门禁结果。

**Architecture:** 新增独立模型适配模块与实验运行器。复用既有标准化数据和门禁；官方 ModernTCN block、FLD 编码器保留来源快照；TCTST 按论文正文与图 3 实现。

**Tech Stack:** Python 3.13、PyTorch 2.6 CUDA、NumPy、pytest，本地 RTX 4050 6 GB。

**Spec:** 用户指定顺序；results/g1_external_methods_search_20261022/external_methods_review_20261022.md。

## Global Constraints

- 仅读取固定 train/validation 数组，不读取 test 数组或新跑步数据；训练端不读取验证标签。
- 原始训练标准化参数直接复用；55820 个一致性训练端点保持不变。
- 完整 45236 个验证端点输出预测；跨周期输入允许，禁止按周期截断长历史；序列开端可因果重复最早可用帧并计数。
- 保留完整 603 周期、检测器一致性 504 周期、计数不一致 99 周期的独立统计。
- v4.2 容差及既有长度切点 [50.66666666666666,75.0] 固定；门禁全周期及分层覆盖一起报。
- 三配对种子 20261021/20261022/20261023；监督训练固定 32 epoch，最终 checkpoint 评分，不按验证分数挑 epoch。
- ModernTCN 与 TCN10：Adam 1e-3、有效 batch512；TCTST/TCN100 与 FLD 读出：Adam 6e-4、有效 batch512；统一裁剪梯度范数1。
- TCTST：384维、8头、8层；每通道共享 Conv1d(1,4,3,stride3)→ReLU→Conv1d(4,4,3,stride3)→ReLU→MaxPool2→Flatten→Linear384；通道位置嵌入；3个相位占位通道推理始终置零。
- TCTST 预训练32 epoch、遮蔽比例0.3、端点stride5（50Hz下对应论文100Hz的stride10）；仅训练集传感器和一致性相位标签参与重建。另跑无预训练版本隔离预训练收益。
- FLD：51帧、8潜通道、卷积宽64/64；32 epoch五步重建推进预训练，stride5、Adam1e-4；冻结编码器后训练相位/相位速度读出32 epoch。
- FLD fft 防零谱数值保护及 Hz 频率单位显式适配；读出使用片段末端相位；同一模型比较直接读出与固定alpha0.1的因果相位推进跟踪，不用触地重置。
- 不改部署契约，不重启PPO；所有输出为已有开发集上的适配实测，不称新受试者最终确认或原论文完整复现。
- 模型/优化配置在正式训练前写入 protocol.json 并hash；先做无评分数据的形状和资源 smoke；OOM可减microbatch并增加累积保持有效batch，记入运行配置。
- GPU训练及推理使用bf16 autocast，损失/相位归一化/FFT为float32；不使用GradScaler。数值边界测试保留float32。microbatch在无验证评分的资源smoke后确定。
- 现有分支 codex/g1-phase-runtime 有大量既有改动。Ruling: 仅增加本实验专属文件，沿用当前分支及未跟踪数据，不迁移/修改已有实验；避免丢失当前授权上下文。

## Review Focus

- 长历史跨周期但绝不跨序列/真实帧缺口；左侧填充只使用已发生的最早输入。
- 训练阶段索引只取训练端点；预训练目标缺失位置不能进入重建损失。
- 相位向量顺序始终 sin/cos；FLD中心到末端、cycles/s与rad/s转换正确。
- FLD跟踪不读参考触地或未来帧；训练/验证状态分别初始化。
- 原始门禁先跑全序列，再按周期事件数分组，不能通过删帧改变周期边界。

## Tasks

- [ ] 1. Tests: gold causal windows、prefix invariance、wrapped phase rate、FLD末端相位、跟踪跨序列重置、恒定输入有限梯度。先红后绿。
- [ ] 2. Models/data: 官方代码来源快照、三模型适配、GPU按批索引窗口、冻结划分/原标准化复用。
- [ ] 3. Runner: provenance/config/log/checkpoint/predictions/逐周期原始结果、失败留档、同种子短训练复现。
- [ ] 4. Stage 1: TCN10 vs ModernTCN10，三种子全部完成后再进入Stage 2。
- [ ] 5. Stage 2: TCN100相位/速度、TCTST无预训练、TCTST遮蔽预训练，三种子全部完成后再进入Stage 3。
- [ ] 6. Stage 3: FLD预训练+冻结读出；直接/状态推进同checkpoint对照。
- [ ] 7. Review/report: 全部成功/失败run、配对差、平均±种子SD、完整门禁与子组覆盖，核算结论。

## Execution ledger

- 2026-10-22: 读取既有训练代码、门禁、数据元数据和论文图3。GPU可用；系统可用RAM约1.2GB。决定使用稠密帧库+按批窗口，避免构造GB级窗口数组。
- 2026-10-22: 用户已明确授权三阶段实测，直接执行上述落地选择，不增加确认轮次。
- 2026-10-22: 恒定输入测试定位官方FLD的Atan2Backward(0,0)产生NaN；在适配层给零幅相位对设置固定非零回退，保留官方源文件不变。测试先红，随后验证修复。
- 2026-10-22: 独立审查发现100帧TCTST下采样丢弃最新10帧；末端影响测试已先红。Ruling: 仅在左侧重复首帧补至108帧，使18倍下采样与当前帧对齐；这是50Hz输入适配，代价是较论文增加8个重复历史采样点，所有TCTST臂统一应用，正式训练前冻结。
- 2026-10-22: 续跑每次重新核对两份实际源数据hash，阶段2/3强制检查前序三种子及Modern同种子完整复训完成。独立审查未发现数据泄漏、FLD单位/末端方向或评分先删帧问题。
