> 档案说明：文中以原文件名引用的检索账本、原始网页、查询响应及其他文献清单属于本地研究档案，本仓库未附。公开引用与参考文献见本文链接和 [phase_references.bib](phase_references.bib)。

# Unitree G1 自适应与容错行走控制的国内外同类研究

检索与复核日期：2026年11月5日。

研究对象：本项目的三项设计——从关节运动历史学习相位的 PHASOR、身体状态与任务条件分流的双分支 Actor、训练期关节目标 dropout。

**主要发现：三项设计都有相关研究基础，其中身体与任务分别编码、随机关节遮罩训练已经存在直接前作；学习相位的相关方法则横跨机器人节律控制和图形学运动表示。项目更值得研究的问题，是这些设计在 G1 的统一控制框架下是否带来可重复的收益，以及收益分别来自相位信息、语义分组还是故障训练。**

本报告按研究机制、国内外团队和与项目的接近程度组织证据，同时提供可用于项目总结的研究现状文字。它是广泛的整合性调研，不是完成全部数据库全文筛选的系统综述；“未找到完全相同组合”不构成首次性证明。

## 一 检索范围与证据口径

覆盖英文和中文检索词、arXiv 元数据与开放全文、Crossref 题录、OpenAlex、作者项目页、开源仓库以及《机器人》期刊网站。近期工作重点关注2023—2026年，同时补充2013—2022年的损伤恢复、周期表示和适应控制基础。

三条线分别收录14、18和16项核心研究，合并去重后为**47篇核心文献**；另补2篇辅助论文和1篇中文背景论文，共整理**50项相关研究**。另有2项待补阅读候选单列，未混入50项统计。保存266份原始响应及SHA-256清单。这里统计的是本次整理的条目，不是数据库中全部相关论文的数量。

国内研究按论文的机构署名识别，包含中国大陆和香港团队；国际合作单独标注，不根据作者姓名判断地域。英文发表的国内成果纳入国内研究。

| 检索方向 | 核心问题 | 主要检索词及扩展 |
|---|---|---|
| 学习相位 | 相位从时间推进、振荡器生成，还是由身体运动估计 | learned phase、motion phase manifold、periodic autoencoder、state-based phase、adaptive gait timing、CPG-RL、步态相位、中枢模式发生器 |
| 双分支 | 哪些输入被分开编码，何时融合，训练与部署是否使用同一信息 | command encoder、history encoder、goal-conditioned policy、late fusion、dual branch、proprioception、身体状态、任务指令、双分支 |
| 故障容错 | 哪种故障被模拟，策略是否知道故障，是否真正验证受损硬件 | random joint masking、target/action/torque dropout、locked joint、power loss、actuator degradation、malfunction、关节故障、容错、恢复 |
| 工程与替代路线 | 现有平台、教师—学生、环境适应和结构化网络提供什么对照 | humanoid locomotion、sim-to-real、RMA、CTS、Humanoid-Gym、G1、PPO |

证据分为开放全文、官方摘要或项目说明、题录三类。题录只能确认论文存在、作者、时间和部分机构，不能支撑未读取的故障方程、真机平台或性能数值。正文所称“已核验”限于相应证据能够支持的内容。预印本按已读取版本标注，不自动写成已被会议或期刊接收。

检索过程包含有效查询、失败查询和定向复核，保留在本目录的 JSON 日志与 `raw/` 响应中。分项记录包含近邻筛选和机制定位；总文献表按 DOI、arXiv ID 和规范化标题合并，跨创新线出现的同一论文不会被当成多篇独立成果。

## 二 研究脉络与主要技术路线

早期损伤恢复研究通过离线行为库、自模型和在线试错寻找补偿动作，例如 [Robots that can adapt like animals](https://arxiv.org/abs/1407.3501)。之后，强化学习将环境随机化、历史适应和统一策略引入腿式控制，[RMA](https://arxiv.org/abs/2107.04034) 是实时环境适应的代表。它们提供了鲁棒控制背景，但并不等于关节目标 dropout。

节律控制同时沿两条路线演进：一条通过显式 CPG、周期奖励和可调步频组织步态，另一条从运动数据学习周期潜变量。前者可见 [CPG-RL](https://arxiv.org/abs/2211.00458) 与 [Periodic Reward Composition](https://arxiv.org/abs/2011.01387)，后者可见 PFNN 与 DeepPhase。两条路线在“利用周期结构”上相通，在“相位如何得到、如何进入控制回路”上不同。

2023年以来，研究进一步面向人形机器人复杂地形、多步态、全身动作和执行器故障。国内的 Humanoid-Gym、CTS、DreamPolicy、RGMT、Tac4Loco、UMC，以及国外的 Gait-Conditioned RL、Random Joint Masking、DreamFLEX、TOLEBI，形成了本项目最应比较的研究集合。

| 方法族 | 信息或干预来源 | 代表研究 | 与本项目的关系 |
|---|---|---|---|
| 显式节律与周期奖励 | 时钟、CPG状态、步频、周期奖励 | CPG-RL、SYNLOCO、Periodic Reward Composition | PHASOR应比较的节律基线 |
| 数据驱动周期表示 | 运动轨迹中的低维周期结构 | PFNN、DeepPhase、VAE-Loco | PHASOR的表示学习基础与跨领域近邻 |
| 本体与目标分别表示 | 身体历史、当前观测、指令或未来目标 | RGMT、DreamPolicy | 双分支最直接的机制近邻 |
| 环境和身体适应 | 特权环境参数、历史状态、形态潜变量 | RMA、DreamWaQ、DMAP、CTS | 相关表征路线，分支含义与本项目不同 |
| 受损训练和在线故障适应 | 动作或力矩遮罩、故障状态、动力学衰减 | Random Joint Masking、UMC、FT-Net、TOLEBI | 第三项设计的直接对照 |

## 三 国内同类研究

### 3.1 身体与任务表示已经出现直接的人形前作

**北京理工大学与 Humanoid Robotics（Shanghai）的 RGMT 是当前最直接的架构近邻。** [Robust and Generalized Humanoid Motion Tracking](https://arxiv.org/abs/2601.23080)（2026）用本体历史编码器提取动力学表示，再以该表示条件化任务命令的 cross-attention 编码，融合当前观测后供 PPO 策略决策，并在29自由度 G1 真机上验证。全文 Fig.2、§II-B1、§II-C1/2 明确给出身体历史与命令的分工。

它与本项目的接近程度超过“都用了两个网络”：每步本体观测同为93维，平台同为 G1，算法同属 PPO。区别是 RGMT 面向全身参考动作跟踪，使用身体历史 Transformer 和38维参考命令上下文；本项目采用单帧93维身体状态与5维速度/相位条件的两条 MLP 分支。可检验差异在于轻量结构、分组依据和等预算比较，而不是“首次分开处理身体与任务”。

**上海科技大学与 InstAdapt 的 [DreamPolicy](https://arxiv.org/abs/2505.18780)**（2025预印本，2026修订）用世界模型预测未来状态，控制器分别编码历史和未来目标，再与当前本体信息融合，并包含 G1 真机地形实验。这说明目标条件与身体状态的专门编码已是人形控制的重要路线。其未来目标序列和世界模型，比本项目的小型任务条件分支复杂得多。

**香港科技大学（广州）的 [Tac4Loco](https://arxiv.org/abs/2608.15766)**（2026）在 G1 真机上使用足底压力空间分支和时间分支，融合后支持复杂支撑条件下的行走。它确实采用双分支，但分流对象是足底压力的空间和时间特征。项目报告应解释分支语义，避免仅凭“双分支”名称判断方法相同。

### 3.2 国内已有掩码容错和多关节故障研究

**四川大学、上海人工智能实验室、武汉大学等参与的 [UMC](https://arxiv.org/abs/2502.03035)**（2025，国内外合作）采用正常预训练和受损微调两阶段流程，以故障检测和遮罩机制减少策略对失效关节信息的依赖，提供 Transformer 与 MLP 版本。全文 §4.1 明确包含 A1、H1 和 **G1 仿真任务**。因此，将掩码训练用于 G1 已有直接先例；本轮未确认其 G1 真机容错，不扩大其验证范围。

**香港大学 ArcLab 的 [FT-Net](https://doi.org/10.1109/LRA.2023.3329766)**（IEEE RA-L，2023）研究四足机器人部分和完全电机故障，通过隐式神经适配器实现故障恢复与四足到三足行走切换。题录、机构和摘要已经核验；具体硬件型号和完整故障方程尚需原文补充。它是项目海报引用的真实研究，应采用该 DOI，不能误连到另一篇论文。

**东北大学的 [FT-CPG](https://doi.org/10.1109/LRA.2025.3572772)**（IEEE RA-L，2025）题名明确研究多关节失效下的 CPG 容错行走。它连接了项目第一项节律控制和第三项故障容错，提示“相位/节律与容错结合”也需要具体比较。当前核实了题录和作者机构，详细训练与硬件设置未读到，不在本报告中补写。

### 3.3 国内节律控制与开源平台基础

[SYNLOCO](https://arxiv.org/abs/2310.06606)（2023，中山大学与英国、荷兰团队合作）将 CPG 与强化学习融合，并有 Go1 真机验证；[Terrain-adaptive Central Pattern Generators with Reinforcement Learning for Hexapod Locomotion](https://arxiv.org/abs/2310.07744)（2023，上海交通大学）研究六足地形适应。它们属于显式节律生成与参数调节，不能直接等同于从 G1 关节历史估计相位。

中文期刊中，本轮从《机器人》官方网站核到 [侧向冲击下脊柱关节机器人D-SLIP与CPG融合控制策略](https://robot.sia.cn/article/doi/10.13973/j.cnki.robot.250023)（2026，艾青林等），以 D-SLIP、CPG耦合和关节抗扰控制处理四足机器人的侧向冲击。它提供国内模型控制与节律融合的背景，研究对象和控制机制与本项目不同，不能用其抗扰实验代替人形关节故障证据。

工程基础方面，[Humanoid-Gym](https://arxiv.org/abs/2404.05695)（2024，上海期智、RobotEra、清华等）提供人形强化学习与零样本仿真到真实迁移框架；[CTS](https://arxiv.org/abs/2405.10830)（2024，南方科技大学、逐际动力）并行训练教师和学生，解决特权信息与本体信息之间的适应问题。它们有助于选择训练和部署基线，但教师—学生不能被写成本项目的两条并行 Actor 输入分支。

## 四 国外同类研究

### 4.1 从周期先验到学习运动相位

[Phase-Functioned Neural Networks for Character Control](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control)（PFNN，2017）将网络权重表示为相位的周期函数，用于交互式虚拟角色运动。其核心是相位条件化的动作生成，不是机器人动力学中的闭环相位估计。

[DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds](https://doi.org/10.1145/3528223.3530178)（2022，Edinburgh、Electronic Arts与香港大学合作）利用周期自编码器学习运动的相位流形，是 PHASOR 需要明确引用的表示学习基础。它为“从运动数据提取周期结构”提供方法来源，但虚拟运动数据的周期表示并不自动证明机器人在接触冲击、速度变化和故障下仍能稳定闭环运行。

[VAE-Loco](https://arxiv.org/abs/2205.01179)（2022，Oxford等）学习可解耦步态潜变量，在 ANYmal 真机上实现多样运动。它与 PHASOR 同属紧凑步态表示，但其潜变量、控制生成方式和显式二维相位估计应分别分析。

在机器人节律控制中，[CPG-RL](https://arxiv.org/abs/2211.00458)（2022，EPFL）由强化学习调节 CPG 振荡器参数，并在 A1 验证；[Visual CPG-RL](https://arxiv.org/abs/2212.14400)（2022，EPFL）结合视觉和节律控制，在 Go1 验证。它们已经超出单一固定时钟：项目若只与一个固定周期比较，尚不足以说明对已有自适应节律方法的优势。

[Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387)（2020预印本，Oregon State）用周期奖励组织 Cassie 双足步态。[Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion](https://arxiv.org/abs/2505.20619)（2025，UCL）把速度指令、sin/cos相位和步态标识输入统一循环策略，在 G1 真机上验证站立、行走及走停转换。后一项与项目的平台和条件输入高度接近，但不是身体/任务两条轻量 MLP 分支，也没有证明使用了从关节短历史估计的 PHASOR 相位。

### 4.2 分离信息通路与适应策略

[RMA](https://arxiv.org/abs/2107.04034)（2021，Berkeley、CMU、Facebook）将当前状态与环境动力学潜变量结合，部署时从历史观测估计潜变量；[Rapid Locomotion](https://arxiv.org/abs/2205.02824)（2022，MIT）和 [DreamWaQ](https://arxiv.org/abs/2301.10602)（2023，KAIST）进一步利用历史适应与环境上下文估计。这些工作说明分离表示和融合已有成熟基础，但拆分的是环境/历史适应信息，与本项目的身体/任务条件分流不同。

[DMAP](https://arxiv.org/abs/2209.14218)（2022，EPFL）使用独立传感处理、形态注意力和分布式关节控制，适应身体结构变化；[Neural Circuit Architectural Priors for Quadruped Locomotion](https://arxiv.org/abs/2410.07174)（2024，NYU）探索结构化网络先验。它们支持将“分组方式是否有用”作为实验问题，而不是只比较参数更多或层数不同的网络。

[Real-World Humanoid Locomotion with Reinforcement Learning](https://arxiv.org/abs/2303.03381)（2023）和 [Learning Humanoid Locomotion over Challenging Terrain](https://arxiv.org/abs/2410.03654)（2024，均为 Berkeley 方向工作）则提供真实人形行走和复杂地形的重要背景。它们是任务和泛化能力的参考，不能仅凭人形平台相同便视作三项机制的直接复现。

### 4.3 从四足受损训练走向双足真机容错

**[Learning Quadrupedal Locomotion with Impaired Joints Using Random Joint Masking](https://arxiv.org/abs/2403.00398)**（ICRA 2024，韩国首尔科学技术大学与 CMU）是与项目 dropout 形式最接近的工作。全文 §III-B 同时遮罩动作与力矩，配合渐进课程和关节状态估计，并验证实验室与室外真实受损四足行走。它比“训练时随机扰动动作”的泛泛描述更接近项目，必须逐项比较。

[Saving the Limping](https://arxiv.org/abs/2210.00474)（2022预印本）通过教师—学生策略处理 A1 的关节锁止等故障；[DreamFLEX](https://arxiv.org/abs/2502.05817)（2025，KAIST）在 Go1 粗糙地形中结合故障估计和调制，处理关节锁死与电机弱化；[AcL](https://arxiv.org/abs/2503.21401)（2025，NUS）将多故障教师蒸馏为统一控制器，并在 Go2 真机验证。它们分别代表受损迁移、显式故障感知和多策略蒸馏路线。

**[TOLEBI](https://arxiv.org/abs/2602.05596)**（2026，首尔大学）已将学习式容错推进到 TOCABI 双足真机，研究关节锁死、动力丢失、状态估计和恢复奖励。项目不能把“人形双足故障容错”整体表述为尚无研究的空白。

**[Learning Fault-Tolerant Locomotion with Adaptive Gait Timing](https://arxiv.org/abs/2608.07328)**（2026，IIT）进一步结合执行器失效和自适应步态时序，在 Kyon 四足平台验证。它是“节律适应与故障恢复相互关联”的直接交叉前作；学习步频与从身体运动估计相位仍然是不同机制。该点值得进入三模块联合实验设计。

## 五 最接近前作与项目差异

| 项目设计 | 最应精读的前作 | 已有共同点 | 项目应证明的具体差异 |
|---|---|---|---|
| PHASOR 学习相位 | DeepPhase；CPG-RL；Gait-Conditioned RL | 周期表示、节律学习、相位条件控制均已有研究 | 由短关节位置历史得到的二维相位，在变速、接触变化和扰动下是否有效；相对无相位和自适应CPG有何收益 |
| 身体/任务双分支 | RGMT；DreamPolicy | 身体与任务/目标分别编码再融合已有明确先例，且有G1真机 | 单帧93/5轻量MLP分流的收益，是否来自语义分组而非预算、层数或随机划分 |
| 关节目标 dropout | Random Joint Masking；UMC | 随机关节遮罩、两阶段故障训练已有先例，UMC包含G1仿真 | 目标层干预的物理含义；故障持续时间和类型；无需额外估计器时的性能与开销 |
| 三模块组合 | FT-CPG；Adaptive Gait Timing；TOLEBI | 节律、故障与状态估计已经出现交叉研究 | 相位是否在故障后仍可信；分支结构是否改善恢复；联合收益是否超过各单项之和 |

**相位术语需要精确。** 固定时钟相位、可调频CPG相位、由关节状态估计的相位、动画相位流形和神经网络潜变量不是同一对象。论文标题中出现 phase 并不意味着方法相同。

**双分支术语需要说明分支输入。** Actor/Critic、教师/学生、身体/环境、身体/目标、压力空间/时间，具有不同训练目标与部署结构。应将本项目写成“身体状态与速度及相位条件分别编码的轻量 Actor”，而非笼统称为首创双分支网络。

**目标置零不等于关节失效。** 若动作是关节目标位置偏移，dropout 后往往回到默认姿态目标；PD 控制器仍可能输出力矩。关节锁死约束关节位置，自由摆动或断电涉及力矩消失，执行器弱化改变可用力矩或增益。第三项设计必须按实际代码定义命名，不能直接从目标缺失推到真实断电容错。

## 六 对项目定位和验证的建议

推荐项目定位为：**面向 Unitree G1 的身体与任务语义分流、运动状态相位估计及关节目标遮罩训练的统一行走控制研究。** 这句话交代研究内容，不预先宣称性能或首次性。

三项工作的可检验贡献分别是：

1. **相位信息是否有用。** 对比无相位、固定时钟、可调频/接触重置基线与学习相位；保留相位打乱或冻结干预，辨别有效时序信息与单纯增加输入通道的作用。相位跟踪准确性与闭环任务收益分别报告。
2. **语义分组是否有用。** 同时比较普通MLP、近似等参数MLP、语义93/5分流和固定随机93/5分流；保持训练预算一致，报告速度误差、跌倒、推理延迟和总参数。维数和隐藏层宽度的组合本身不能证明创新。
3. **目标遮罩能处理哪些受损情形。** 分别测试瞬时/持续遮罩、单/多关节和未见关节组合；若声称物理故障，应额外测试力矩衰减、断电或锁死，记录正常工况代价及故障发生后的恢复时间。

如果最终对外声称三项设计形成统一系统，应比较基线、三个单模块和组合，条件允许时使用三因素消融。固定种子与场景配对，按独立训练种子汇总不确定性，避免把同一策略的多个回合当成多次独立训练。联合设计尤其需要检查：故障改变步态后，相位编码器的输出是否仍然反映有效周期。

这些是依据前作提出的验证建议，不是已经完成的项目实验。本轮研究现状调研不重新认证项目海报数字，也不把本项目现有分项日志扩写为正式性能结论。

## 七 可直接用于项目总结的研究现状文字

### 国内研究现状

国内人形机器人运动控制研究已从通用强化学习训练框架，逐步发展到结构化观测编码、任务条件控制与受损状态适应。Humanoid-Gym为人形机器人仿真训练及零样本迁移提供了开源实现，南方科技大学与逐际动力提出的CTS通过并行教师—学生训练学习可部署的本体适应策略。在输入表征方面，北京理工大学等提出的Robust and Generalized Humanoid Motion Tracking分别编码本体历史与任务命令，并在Unitree G1上实现全身运动跟踪；上海科技大学等提出的DreamPolicy进一步将历史状态和未来目标编码引入统一人形行走控制；香港科技大学（广州）的Tac4Loco则通过足底压力的空间和时间分支改善复杂支撑条件下的运动适应。上述研究说明，按信息含义组织策略输入已有明确基础，轻量语义分流的价值需要通过等预算和随机分组对照加以验证。[Humanoid-Gym](https://arxiv.org/abs/2404.05695)、[CTS](https://arxiv.org/abs/2405.10830)、[RGMT](https://arxiv.org/abs/2601.23080)、[DreamPolicy](https://arxiv.org/abs/2505.18780)、[Tac4Loco](https://arxiv.org/abs/2608.15766)。

国内团队也已探索节律生成与故障容错。中山大学等参与的SYNLOCO、上海交通大学的地形自适应CPG研究，以及东北大学的FT-CPG，分别涉及CPG与强化学习融合、地形适应和多关节故障下的步态生成。香港大学的FT-Net通过隐式适配应对四足电机故障；四川大学、上海人工智能实验室等参与的UMC利用两阶段遮罩训练处理多种损伤，并包含H1和G1仿真任务。本项目在此基础上，围绕G1的低维身体与任务输入、由关节历史估计的相位，以及目标层随机遮罩展开统一研究，重点考察各项机制对行走跟踪、恢复能力和计算成本的独立与联合影响。[SYNLOCO](https://arxiv.org/abs/2310.06606)、[地形自适应CPG](https://arxiv.org/abs/2310.07744)、[FT-CPG](https://doi.org/10.1109/LRA.2025.3572772)、[FT-Net](https://doi.org/10.1109/LRA.2023.3329766)、[UMC](https://arxiv.org/abs/2502.03035)。

### 国外研究现状

国外相关研究主要沿节律表示、结构化适应策略和受损行走三条路线发展。PFNN与DeepPhase从运动数据学习或利用周期结构，为相位条件控制提供了表示基础；CPG-RL通过强化学习调节显式振荡器，周期奖励组合则在Cassie上组织多种双足步态。近期Gait-Conditioned Reinforcement Learning进一步将速度指令、相位和步态标识输入统一策略，并在G1上验证部分真实运动模式。这些工作表明，周期信号与步态条件已经广泛用于运动控制，关键区别在于相位来自预设节律、可调振荡器还是实时身体运动，以及它在闭环中的有效性。[PFNN](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control)、[DeepPhase](https://doi.org/10.1145/3528223.3530178)、[CPG-RL](https://arxiv.org/abs/2211.00458)、[周期奖励](https://arxiv.org/abs/2011.01387)、[Gait-Conditioned RL](https://arxiv.org/abs/2505.20619)。

在适应与容错方面，RMA、DreamWaQ和DMAP通过环境潜变量、上下文估计或形态注意力组织控制信息。Random Joint Masking已在训练中联合遮罩关节动作与力矩，并验证真实受损四足运动；DreamFLEX结合故障估计与调制处理粗糙地形和执行器异常；TOLEBI进一步在TOCABI双足真机上研究锁死与动力丢失。2026年的Adaptive Gait Timing又将可学习节律与故障恢复结合。相对于这些路线，本项目应明确限定身体/任务分支、运动状态相位估计和关节目标遮罩的作用范围，通过统一预算、多故障工况及模块消融验证其适用条件。[RMA](https://arxiv.org/abs/2107.04034)、[DreamWaQ](https://arxiv.org/abs/2301.10602)、[DMAP](https://arxiv.org/abs/2209.14218)、[Random Joint Masking](https://arxiv.org/abs/2403.00398)、[DreamFLEX](https://arxiv.org/abs/2502.05817)、[TOLEBI](https://arxiv.org/abs/2602.05596)、[Adaptive Gait Timing](https://arxiv.org/abs/2608.07328)。

## 八 覆盖程度与尚未解决的取数缺口

| 来源 | 本轮实际情况 | 对结论的影响 |
|---|---|---|
| arXiv | 三条线均获得有效检索响应，并保存多篇官方摘要与开放全文；架构线20篇候选的摘要及HTML共40次200 | 对公开预印本及具体机制核验支持较强；不代表所有论文均由arXiv收录 |
| Crossref | 多次200，部分429；核实PFNN、DeepPhase、FT-Net、FT-CPG及Random Joint Masking等题录 | 可核对DOI、正式题名和部分机构；不能用题录替代全文 |
| OpenAlex | 部分200，部分429/503 | 用于辅助发现；失败查询不记为零文献 |
| 作者主页与GitHub | 核到PFNN、AI4Animation、RMA、CTS、Humanoid-Gym等页面；部分仓库连接重置或超时 | 支持机制和可复现入口；README中的效果自述不当独立验证 |
| 中文公开网站 | 《机器人》首页与一篇CPG研究页200；站内查询页面未返回可用题录 | 获得真实中文研究入口，但不能宣称充分覆盖中文期刊 |
| CNKI、万方、维普 | CNKI请求403；万方只获得动态站点外壳；维普未完成有效题录检索 | 中文学位论文、非开放期刊和国内未公开成果仍有缺口 |
| 通用搜索与Firecrawl | Firecrawl免登录额度受限；Google连接失败；Bing多词检索存在跑题或空结果 | 这些来源的失败不用于判断“没有前作” |

本次从最相关论文的相关工作、参考文献和项目链接补充经典及相邻路线，属于有记录的定向扩展，未声称完整遍历全部前向被引网络。正式投稿前，优先补读FT-Net、FT-CPG等尚缺主论文全文的直接近邻，并通过机构入口补检中文数据库。新颖性判断还需要项目最终实现与最近邻逐项核对。

## 九 随附材料

- `phase_sources.md` 与 `phase_sources.json`：相位线注释、完整标题、平台及证据边界。
- `architecture_sources.md` 与 `architecture_sources.json`：18篇核心架构文献、2篇辅助条目及正文定位。
- `fault_sources_zh.md` 与 `fault_sources.json`：16项故障容错文献；英文详细记录另见 `fault_sources.md`。
- `文献总表.csv` 与 `参考文献.bib`：合并后的题录、机制和证据字段；BibTeX统一保留已核实的信息。
- `检索覆盖与校验.json`：实际文献计数、重复合并、源状态和人工核查重点。
- `raw/` 与分项查询日志：本轮原始响应，可用于复核题名、版本和方法描述。

引用纠错已经单独保留：`arXiv:1703.05972`并非PFNN；`arXiv:2210.00474`是Saving the Limping而非FT-Net；UMC的API与HTML题名存在版本差异。它们不作为多篇论文重复计数，也不沿用错误链接。
