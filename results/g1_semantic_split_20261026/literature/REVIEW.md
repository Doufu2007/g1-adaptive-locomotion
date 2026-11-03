# 状态／指令／相位分流：首轮查新记录

检索日期：2026-10-26。8 次 arXiv/OpenAlex 查询共返回 61 条记录（未去重），核对了 7 篇公开全文。查询、时间、URL、响应和 SHA 均保存在本目录。Firecrawl 首先检查但未认证，真实调用返回需要 API key；因此改用公开 API，不把服务不可用计作没有前例。

**当前结论：已发现身体状态与指令分别编码后参与控制的相近前作；未确认与本次 G1、93/5、480/32、等预算及随机分流对照完全一致的公开组合。确切组合的新颖性仍为 UNKNOWN。** 本轮关键词检索、摘要索引及有限全文核对不能证明“无人做过”。

| 真实前作 | 核对到的内容 | 与本实验的关系 |
|---|---|---|
| [Robust and Generalized Humanoid Motion Tracking](https://arxiv.org/abs/2601.23080) | 全文 Fig. 2、§III 的 history encoder 从本体历史提取动力学表示；command encoder 用动力学条件 cross-attention 汇聚上下文指令，并与当前观测融合供 Actor-Critic 使用。 | 目前最直接的“本体信息／指令编码”近邻。任务是全身动作跟踪、带历史与上下文窗口，未在核对内容中确认本实验 93/5、无历史前馈结构和预算控制实验。不能把“首次分开编码身体与指令”作为贡献。 |
| [Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion](https://arxiv.org/abs/2505.20619) | 人形 locomotion，输入含身体状态、速度指令、sin/cos 相位和 gait 条件，使用 LSTM 与 MLP。 | 最接近本实验的条件集合之一；核对全文未确认 93/5 独立分支或 matched/random 对照。它表明条件输入本身并非新概念。 |
| [DMAP: a Distributed Morphological Attention Policy for Learning to Locomote with a Changing Body](https://arxiv.org/abs/2209.14218) | 分布式关节控制与形态注意力，使用本体信息和上下文适应身体变化。 | 控制结构和语义归纳偏置的相关工作，不能直接等同于本次状态／指令分流。 |
| [RANDPOL: Parameter-Efficient End-to-End Quadruped Locomotion via Randomized Policy Learning](https://arxiv.org/abs/2505.19054) | Go2 locomotion，冻结随机隐层特征、训练输出层，讨论参数效率与运行成本。 | 参数效率和延迟对照的相关工作。其“可训练参数”减少不能直接等同于部署时总参数/MAC 减少。 |
| [D2RL: Deep Dense Architectures in Reinforcement Learning](https://arxiv.org/abs/2010.09163) | 在深层网络中反复拼接状态，研究 RL 网络结构。 | 支持加入合理网络结构对照；不是本次两类语义分支的直接复现。 |
| [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) | DDPG 的 Q 网络将动作在后层合入状态表征。 | 是 Critic 的状态／动作融合，不能当作 Actor 的身体／任务条件分流完全相同的前例。 |
| [When Is a Learned Command Adapter Worth It? Closed-Loop Identification and Counterfactual Auditing of Frozen Locomotion Policies](https://arxiv.org/abs/2607.21867) | 冻结 locomotion 策略，学习指令适配器，并进行闭环和反事实审计。 | 指令接口与控制归因的邻近问题；不同于重训双分支 Actor。 |

重点检索词包括 locomotion + command encoder/state encoder/task-conditioned，humanoid + late fusion/conditioning/dual stream，locomotion + parameter efficient/compact policy，以及 G1 + command embedding/dual stream/state encoder。最后一组返回 0 条，只说明该次索引查询结果为空。

最适合当前实验的可检验贡献表述是：**在 G1 平地速度跟踪、固定解析相位与近似相同 Actor 预算下，系统比较语义分流、普通 MLP 和固定随机分流的跟踪、失败率、扰动表现与推理成本。** 此时尚无正式结果，不能提前写成“语义分流形成更优 Pareto 前沿”。

若随后获得明确收益，再扩展到多个预算、地形、随机分组种子和机器人；这些尚未进入本次首轮设计。是否可发表需要与实际结果及前作差异一起判断，不能由某组维数未检索到相同条目直接推出。
