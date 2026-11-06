> 档案说明：文中以原文件名引用的检索账本、原始网页、查询响应及其他文献清单属于本地研究档案，本仓库未附。公开引用与参考文献见本文链接和 [phase_references.bib](phase_references.bib)。

# 学习相位控制（PHASOR）：带注释文献与检索记录

检索日期：2026-11-05（Asia/Shanghai）。本文件为第一创新线的文献证据清单，不作首次性/新颖性判决。

## 检索范围与核验
- 重点覆盖：显式CPG+RL、状态驱动相位、周期奖励、相位/步态潜在表示、图形学可迁移周期自编码器；年份包括经典2017/2020和2022—2025前沿。
- 公开来源：Crossref、arXiv、作者官方项目页/GitHub；保存原始HTTP响应。已读8篇机器人开放全文与多个可读摘要，图形学核心从官方作者页+Crossref核对。
- 原始记录：`phase_search_log.json`、`phase_target_search_log.json`、`phase_web_search_log.json`、`phase_fulltext_manifest.json`、`phase_deepphase_receipts.json`及`raw/phase_*`。
- 修订：初轮6条arXiv检索存在%2522双重编码，0条结果作废；正确原始字符串查询已重跑，见`raw/phase_arxiv_corrected_*.xml`。`DeepPhase`首次长题名检索漏检，后用短词检索+作者仓库补回；不能据漏检写“无可信论文”。
- 可用性：arXiv/作者GitHub/多数Crossref为200；OpenAlex503/429，S2 429，部分出版社403/202，Firecrawl免登录额度受限。未把不可用来源当作无前例。

## 正式核心条目（14篇，按路线分组）

### P01 Phase-Functioned Neural Networks for Character Control（2017）
- 作者：Daniel Holden; Taku Komura; Jun Saito
- 状态：ACM TOG / SIGGRAPH 2017
- 机构/归属：官方作者页；Crossref作者元数据；国外
- 链接：https://theorangeduck.com/page/phase-functioned-neural-networks-character-control；DOI `10.1145/3072959.3073663`
- 机制：PFNN 将网络权重表示为相位的周期函数，并把用户控制、过去状态与场景几何共同输入，端到端生成走/跑/跳/攀爬；官方页明确报告崎岖地形拟合和毫秒级推理。
- 机器人真机：否；虚拟角色/游戏控制
- 与PHASOR的关系：与本项目共享“从运动数据得到相位、用相位调制控制器”的抽象，但本项目对象是G1关节位置历史和PPO策略；相位不是场景动画权重插值。
- 证据：官方作者项目页+Crossref元数据；`调研_国内外同类研究_20261105\raw\phase_target_pfnn_official.html`
- 边界：项目页没有机器人硬件证据，不能把图形学结果直接等同于机器人行走性能。

### P02 DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds（2022）
- 作者：Sebastian Starke; Ian Mason; Taku Komura
- 状态：ACM TOG 41(4), Article136 / SIGGRAPH2022
- 机构/归属：University of Edinburgh / Electronic Arts / The University of Hong Kong；中国香港与国外合作
- 链接：https://doi.org/10.1145/3528223.3530178；DOI `10.1145/3528223.3530178`
- 机制：周期自编码器以无监督方式从非结构化全身动作数据学习多通道周期特征，提取多维phase space，改善时空对齐和神经动作合成。
- 机器人真机：无机器人真机；虚拟角色locomotion/舞蹈/足球动作等
- 与PHASOR的关系：与PHASOR最接近的表示学习源：都是从运动数据学习周期结构。差异是DeepPhase面向图形学的多通道全身动作相位流形；PHASOR拟以短G1关节历史输出二维相位并接入PPO闭环控制。
- 证据：官方作者项目页+Crossref元数据/摘要；`raw/phase_deepphase_crossref.txt; raw/phase_ai4animation_github.txt`
- 边界：本条机制描述仅限已读取的摘要/项目页；正式比较宜逐项核对控制接口与评测协议。

### P03 Local Motion Phases for Learning Multi-Contact Character Movements（2020）
- 作者：Sebastian Starke; Yiwei Zhao; Taku Komura; Kazi Zaman
- 状态：ACM TOG39(4), Article54 / SIGGRAPH2020
- 机构/归属：University of Edinburgh / Electronic Arts（Crossref作者机构）；国外
- 链接：https://doi.org/10.1145/3386569.3392450；DOI `10.1145/3386569.3392450`
- 机制：引入局部motion phases，在不手工标注统一全局相位的前提下，从非结构化动作数据学习多接触/多技能角色运动。
- 机器人真机：无机器人真机；图形学角色控制
- 与PHASOR的关系：提示复杂多接触运动未必适合一个全局相位。PHASOR单一二维相位需以消融验证适用范围；不能把局部相位学习视为空白。
- 证据：官方作者项目页+Crossref元数据/摘要；`raw/phase_metadata_fix_0.json; raw/phase_ai4animation_github.txt`
- 边界：本条机制描述仅限已读取的摘要/项目页；正式比较宜逐项核对控制接口与评测协议。

### P04 Motion In-Betweening with Phase Manifolds（2023）
- 作者：Paul Starke; Sebastian Starke; Taku Komura; Frank Steinicke
- 状态：ACM TOG /2023
- 机构/归属：Electronic Arts Germany / Universität Hamburg / Meta Reality Labs / The University of Hong Kong（Crossref）；中国香港与国外合作
- 链接：https://doi.org/10.1145/3606921；DOI `10.1145/3606921`
- 机制：将phase manifolds用于动作中间帧生成/动作时序对齐（作者开源项目PhaseBetweener与题名核对）。
- 机器人真机：无机器人真机；图形学
- 与PHASOR的关系：可迁移的相位流形来源；任务是动作生成，不是G1闭环力矩/关节控制。
- 证据：官方作者项目页+Crossref元数据/摘要；`raw/phase_metadata_fix_2.json; raw/phase_ai4animation_github.txt`
- 边界：本条机制描述仅限已读取的摘要/项目页；正式比较宜逐项核对控制接口与评测协议。

### P05 CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion（2022）
- 作者：Bellegarda, Guillaume; Ijspeert, Auke
- 状态：IEEE RA-L 2022；全文首页Accepted September2022
- 机构/归属：EPFL BioRobotics Laboratory，瑞士（全文首页）；国外
- 链接：https://arxiv.org/abs/2211.00458；DOI `10.1109/LRA.2022.3218167`
- 机制：DRL直接调制耦合振荡器的幅值和频率设定值，并协调各振荡器节律；研究感知反馈与耦合的作用。
- 机器人真机：是；真实Unitree A1，仿真训练后sim-to-real
- 与PHASOR的关系：与本项目同属“学习节律/相位用于腿式运动”，差异在于本项目直接学习运动相位表示并输入策略，且对象为人形G1；CPG-RL通常学习显式振荡器或其参数。
- 证据：开放全文+摘要+元数据；`raw/phase_full_cpgrl.txt`
- 边界：学习的是显式CPG振荡器设定值，不应写成从关节短历史估计二维相位。

### P06 Visual CPG-RL: Learning Central Pattern Generators for Visually-Guided Quadruped Locomotion（2022（预印本首版））
- 作者：Bellegarda, Guillaume; Shafiee, Milad; Ijspeert, Auke
- 状态：已核预印本；正式会议版本另核
- 机构/归属：EPFL BioRobotics Laboratory，瑞士（全文首页）；国外
- 链接：https://arxiv.org/abs/2212.14400
- 机制：结合外感知/本体感知和耦合CPG，以DRL调节振荡行为跟踪速度并避障；比较记忆网络、耦合和感知延迟。
- 机器人真机：是；真实Unitree Go1
- 与PHASOR的关系：提供视觉环境下的节律适应对照；相位载体仍是 CPG 与 RL 的耦合，不是短关节位置历史编码器。
- 证据：开放全文+摘要+元数据；`raw/phase_full_visualcpg.txt`
- 边界：相位由显式振荡器产生；非短关节历史phase encoder。

### P07 SYNLOCO: Synthesizing Central Pattern Generator and Reinforcement Learning for Quadruped Locomotion（2023）
- 作者：Zhang, Xinyu; Xiao, Zhiyuan; Zhang, Qingrui; Pan, Wei
- 状态：arXiv预印本，2023-10-10
- 机构/归属：中山大学航空航天学院（中国深圳）/ University of Manchester / Delft University of Technology（全文首页）；国内外合作
- 链接：https://arxiv.org/abs/2310.06606
- 机制：CPG提供稳定节律与足端净空先验，RL提高环境适应；使用性能驱动奖励和两阶段训练。
- 机器人真机：是；真实Unitree Go1，多速度/地形/载荷
- 与PHASOR的关系：属于固定/显式节律先验与 RL 融合路线，可作为本项目“学习相位替代固定时钟”的竞争基线；不是二维相位编码器。
- 证据：开放全文+摘要+元数据；`raw/phase_full_synloco.txt`
- 边界：不是本项目的相位编码器；不能以姓氏判定国内团队，机构已全文核对。

### P08 Terrain-adaptive Central Pattern Generators with Reinforcement Learning for Hexapod Locomotion（2023）
- 作者：Yang, Qiyue; Gao, Yue; Li, Shaoyuan
- 状态：arXiv预印本，2023-10-11
- 机构/归属：上海交通大学自动化系/人工智能教育部重点实验室与AI Institute（全文首页）；国内
- 链接：https://arxiv.org/abs/2310.07744
- 机制：CPG生成同步节律作为基础步态，DRL调整CPG映射函数参数以适应不平地形。
- 机器人真机：否；文中摘要明确为Isaac Gym六足仿真实验
- 与PHASOR的关系：与本项目粗糙地形迁移问题相关；其适应量通常是振荡器参数/步态，不等同于从G1关节历史回归二维相位。
- 证据：开放全文+摘要+元数据；`raw/phase_full_terraincpg.txt`
- 边界：六足仿真不能等同人形G1真机；相位源仍为CPG。

### P09 VAE-Loco: Versatile Quadruped Locomotion by Learning a Disentangled Gait Representation（2023（TRO）；2022预印本）
- 作者：Mitchell, Alexander L.; Merkt, Wolfgang; Geisert, Mathieu; Gangapurwala, Siddhant; Engelcke, Martin; Jones, Oiwi Parker; Havoutis, Ioannis; Posner, Ingmar
- 状态：IEEE TRO2023；全文版权与首版元数据核到
- 机构/归属：Oxford Robotics Institute / Agility Robotics / DeepMind（全文首页）；国外
- 链接：https://arxiv.org/abs/2205.01179；DOI `10.1109/TRO.2023.3297015`
- 机制：用VAE学习捕获支撑相位的解耦latent space；驱动一个潜变量可连续控制步频、摆足高度和全支撑时长，并用生成模型检测/抑制扰动。
- 机器人真机：是；真实ANYmal B和ANYmal C
- 与PHASOR的关系：与本项目的低维步态表示最接近之一，但 representation 未必是显式周期相位，也未必使用短关节位置历史作为监督/输入。
- 证据：开放全文+摘要+元数据；`raw/phase_full_vaeloco.txt`
- 边界：是潜在步态规划/驱动结构，非本项目固定二维在线相位估计。

### P10 Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion（2025）
- 作者：Peng, Tianhu; Bao, Lingfan; Zhou, Chengxu
- 状态：arXiv预印本v3；Crossref另核Humanoids2025 DOI
- 机构/归属：University College London，英国（全文首页）；国外
- 链接：https://arxiv.org/abs/2505.20619；DOI `10.1109/Humanoids65713.2025.11203058`
- 机制：单一recurrent policy输入本体状态、速度、sin-cos gait phase和one-hot gait ID；gait ID同时路由奖励，配多阶段课程。
- 机器人真机：是；真实Unitree G1站立、行走和walk-to-stand；跑步/部分转换为仿真
- 与PHASOR的关系：这是本项目人形方向的重要近邻：说明步态条件与相位信号并非空白。差异是本项目以可学习相位编码器替代固定时钟，并评估G1粗糙地形迁移。
- 证据：开放全文§III+摘要+元数据；`raw/phase_full_gaitconditioned.txt`
- 边界：该文的Multi-Phase首先指课程阶段，不能只据题名等同于学习相位。全文已核到sine-cosine gait phase输入，但未核到由短关节历史学习phase encoder。

### P11 PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion（2025）
- 作者：Ntagkas, Alexandros; Kiourt, Chairi; Chatzilygeroudis, Konstantinos
- 状态：arXiv预印本，2025-10-21
- 机构/归属：University of Patras / Athena RC / Archimedes，希腊（全文首页）；国外
- 链接：https://arxiv.org/abs/2510.18348
- 机制：用三次Hermite样条表达每腿phase，基于局部heightmap调整摆腿高度，用摆动期接触惩罚等reward shaping引入节律；策略直接在关节空间输出。
- 机器人真机：是；真实Unitree Go2；另报告ANYmal-C初步结果
- 与PHASOR的关系：与本项目“相位—粗糙地形”连接最接近的2025条目之一；差异在于其可能用预定义/规划相位与感知，而本项目相位来自G1关节短历史。
- 证据：开放全文+摘要+元数据；`raw/phase_full_pgtt.txt`
- 边界：主要是phase-guided奖励塑形与地形感知，不是学习关节历史相位编码器。

### P12 Learning Velocity-based Humanoid Locomotion: Massively Parallel Learning with Brax and MJX（2024）
- 作者：Thibault, William; Melek, William; Mombaur, Katja
- 状态：arXiv预印本，2024-07-06
- 机构/归属：当前仅摘要元数据，机构未核；机构待核（不据姓名猜地域）
- 链接：https://arxiv.org/abs/2407.05148
- 机制：Brax/MJX大规模并行学习REEM-C速度控制，采用periodic reward formulation。
- 机器人真机：否；摘要明确REEM-C仿真，真机工作在进行
- 与PHASOR的关系：作为训练平台与速度条件基线；相位学习不是其标题主贡献，不能用来证明学习相位效果。
- 证据：可读摘要+元数据；`调研_国内外同类研究_20261105\raw\phase_target_humanoid_mjx.html`
- 边界：对象为REEM-C，不是G1；仅做训练/周期奖励基线。

### P13 Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition（2021（ICRA）；2020预印本）
- 作者：Siekmann, Jonah; Godse, Yesh; Fern, Alan; Hurst, Jonathan
- 状态：ICRA2021；Crossref DOI核到
- 机构/归属：Oregon State University，CRIS Institute，美国（全文首页）；国外
- 链接：https://arxiv.org/abs/2011.01387；DOI `10.1109/ICRA48506.2021.9561814`
- 机制：以概率周期cost组合力/速度奖励，参数化常见步态；LSTM策略输入clock、phase ratio、速度指令，输出PD目标和增益。
- 机器人真机：是；真实Cassie双足机器人
- 与PHASOR的关系：说明周期结构可通过奖励/训练课程注入，而不是时钟输入；可作为固定时钟之外的周期控制对照。
- 证据：开放全文+摘要+Crossref元数据；`raw/phase_full_periodic.txt`
- 边界：本方法也用clock输入；不能误写为完全没有时钟的奖励路线。它提供与学习相位不同的周期结构对照。

### P14 Phase portraits as movement primitives for fast humanoid robot control（2020）
- 作者：Guilherme Maeda; Okan Koç; Jun Morimoto
- 状态：Neural Networks2020
- 机构/归属：Crossref作者元数据，无机构字段；机构待核
- 链接：https://doi.org/10.1016/j.neunet.2020.04.007；DOI `10.1016/j.neunet.2020.04.007`
- 机制：以phase portraits作为快速人形机器人控制的movement primitives（题名和元数据可核；未读到全文，避免扩写具体算法）。
- 机器人真机：真机待核
- 与PHASOR的关系：仅作为phase-space运动基元邻域，不能将标题级证据写成在线相位估计。
- 证据：官方作者项目页+Crossref元数据/摘要；`raw/phase_metadata_fix_1.json`
- 边界：本条机制描述仅限已读取的摘要/项目页；正式比较宜逐项核对控制接口与评测协议。

## 最接近的三项前作信号
1. **DeepPhase (2022)**：与“由运动数据学习周期表示”最直接；多维周期自编码器/图形学动作合成 vs G1短关节历史/二维相位/PPO闭环。
2. **Gait-Conditioned RL (2025)**：同为真实G1、PPO/recurrent、sin-cos相位与速度条件；其主要贡献是gait ID路由奖励和课程，并非短历史学习相位。
3. **CPG-RL (2022)及VAE-Loco (TRO2023)**：前者通过RL学习显式振荡器幅频，后者学习支撑相位latent并连续调节步频/摆足参数；都要求PHASOR说清“学习了什么、在哪个控制接口起作用”。

## 国内/国际覆盖要点
- 国内：上海交通大学的Terrain-adaptive CPG（2023），六足Isaac Gym仿真，DRL调节CPG映射参数。
- 国内外合作：中山大学—曼彻斯特—代尔夫特SYNLOCO（2023），Go1真机；中国香港大学参与DeepPhase（2022）和Motion In-Betweening with Phase Manifolds（2023），属于图形学方法合作。
- 国外：EPFL的CPG-RL/Visual CPG-RL；Oxford等的VAE-Loco；UCL的G1 gait-conditioned控制；Oregon State的Cassie周期奖励。上述归属依据论文机构，未依据姓名/国籍猜测。
- 中文数据库CNKI/万方/维普未通过机构订阅覆盖；公开中文网页检索噪声大，故不能声称国内成果已查全。

## 待核候选（不计核心条目）

- **DeepCPG Policies for Robot Locomotion（2023）**：https://doi.org/10.1109/TCDS.2023.3250393。当前仅元数据/HTTP202，需补摘要和全文后再比较机制。
- **FT-CPG: Learning Central Pattern Generators for Fault-Tolerant Quadruped Locomotion Under Multi-Joint Failures（2025）**：https://doi.org/10.1109/LRA.2025.3572772。当前仅元数据/HTTP202，需补摘要和全文后再比较机制。
- **Learning Free Gait Transition for Quadruped Robots via Phase-Guided Controller（2022）**：https://doi.org/10.1109/LRA.2021.3136645。当前仅元数据/HTTP202，需补摘要和全文后再比较机制。

## 排除与错配（不计核心条目）
- https://arxiv.org/abs/1703.05972：返回Giant spin splitting and 0-π Josephson transitions…，与PFNN无关；不得当作PFNN引文。
- https://arxiv.org/abs/1807.10565：Surgical Phase Recognition in CATARACTS Videos属于医学视频分期，不是运动相位流形。

## 可交给根任务的证据边界
已找到学习周期表示、CPG与RL、人形相位条件、相位引导地形奖励等真实路线。PHASOR适宜比较“相位来源（时钟/振荡器/运动状态）—表示维度—接入策略位置—闭环地形/扰动表现”，不适宜以“首次学习步态相位”概括。所有他文指标必须保留其平台/场景；本文件未把它们直接与项目现有跌倒率横向排名。

附：全文抓取中PFNN作者PDF与DeepPhase仓库PDF未完成有效下载，已停止等待；其论文条目依据可读官方项目页、GitHub作者总README及Crossref元数据/摘要。成功读取的8篇机器人PDF列于`phase_fulltext_manifest.json`，全部原始材料SHA256列于`phase_raw_manifest.json`。
