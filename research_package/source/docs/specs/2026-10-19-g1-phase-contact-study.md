# G1 接触事件相位研究设计（2026-10-19）

## 研究主张

核心问题固定为：

> 仅根据最近 10 帧、每帧 29 个关节位置，冻结的 1D-CNN 能否估计由足部接触事件定义的步态相位？该表征接入 G1 控制器后，是否改善预先指定的运动指标？

相位标签是按接触事件定义的操作性相位，不称为无争议的生物力学真值。只有相位表征门和控制实验主比较同时通过，摘要才允许写控制性能增益。

## 数据核查结论

当前仓库文件 `src/assets/motions/g1/dance1_subject2.csv` 的核查结果：

- 可追溯到 Hugging Face `lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/dance1_subject2.csv`，数据集目录提交 `07ff8845a94d59d4195a8ac04fe918d4e6338239`；本地文件 SHA-256 为 `1793edcd8345fa4736676c06008186d1a3ecaaf99df0c380218eaa4e0de100ec`。
- 上游 LAFAN1 是 Ubisoft La Forge Animation Dataset；原始数据为 5 名受试者、77 条序列、496,672 帧、30 FPS。
- 原始许可为 Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International（CC BY-NC-ND 4.0）。仓库代码的 MIT/Apache 许可不能替代动作数据许可。
- G1 重定向卡片说明使用数值优化、Interaction Mesh 与 IK，并施加末端位姿、关节位置和速度约束；未包含动力学或执行器约束。
- G1 CSV 顺序为 root `XYZQXQYQZQW` 加 29 个关节，30 FPS。映射顺序固定为：left/right hip pitch/roll/yaw、knee、ankle pitch/roll、waist yaw/roll/pitch、双臂 shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw。
- `dance1_subject2` 属于 LAFAN1 的 Dance / Free dancing 类别，不是步行动捕数据，不能作为本研究步态训练集或步行结论证据。
- 研究候选改为上游列出的 `walk1_*`、`walk2_*`、`walk3_*`、`walk4_*` G1 重定向序列。候选文件需逐个记录来源修订、大小、哈希、受试者、序列名和动作类别。

在候选步行序列的来源、许可、动作类别、采样率、受试者/序列数、G1 映射、接触标注或轨迹检测及人工抽样核验都记录完成前，训练入口必须阻断；不得把 dance 文件描述成步行数据。

## 标签与数据划分

以左脚连续两次触地之间为一个完整周期：

\[ \varphi(t)=2\pi\frac{t-t_k^L}{t_{k+1}^L-t_k^L}\pmod{2\pi}. \]

只在完整周期内生成标签；站立和动作过渡另行记录。右脚触地相位从其时间戳相对于左脚周期计算并报告分布，不预设为 \(\pi\)。如果接触由轨迹推断，必须在重定向后的动作上检测并抽样人工核验。

先按受试者或完整动作序列划分 train/validation/test，再生成重叠窗口。每个窗口按时间排序、形状 `10 × 29`，标签对应最后一帧；窗口不能跨序列、跨 reset 或跨集合。训练标准化统计只从 train 帧估计。

## 编码器与验证

输出为二维单位向量：`z=(sin(phi), cos(phi))`，由原始二维头经过带 `eps` 的归一化得到；损失为 `mean(1 - z · z*)`。比较当前时间池化 1D-CNN 与保留时间位置的 1D-CNN，使用同一输入、输出头、损失和数据划分；只用 validation 选架构，冻结后才评估 test。

至少报告 held-out 圆周误差、每周期推进/完整绕行、左脚触地误差、右脚实测触地相位分布、固定时钟基线。阈值由 validation 根据标签不确定度和任务所需分辨率确定；单位范数不作为学习证据。

## 运行时与控制实验

每个环境拥有独立 10 帧历史；reset 清空并用当前姿态填满，避免 episode 串线。每个控制状态只算一次 latent，actor 和对应接触奖励复用同一缓存；奖励明确使用该状态的接触与 latent。所有公式、输出、奖励使用 `(sin, cos)`，还原统一为 `atan2(z[0], z[1])`。接触标签不进入编码器推理、actor 输入或部署逻辑；critic 保留固定时钟。

策略 rollout 在无接触标签输入下覆盖稳定步行、不同速度、速度变化和扰动，同时保留站立、跌倒和过渡。held-out 动捕通过而 on-policy 不通过时，停止控制性能主张并回到数据/窗口/架构修正。

预注册五组：固定时钟基线、Actor 消融、Reward 消融、联合方法、每 episode 固定随机二维单位向量对照。相同 PPO 预算、环境、命令分布和网络规模；主指标为平地测试集 XY 速度误差，主比较为联合方法 vs 固定时钟；每个训练 seed 是统计单位，rollout episode 不充当独立训练样本。先做小规模工程试跑，再按 seed 方差和预注册最小实际意义差异计算正式 seed 数，目标功效至少 80%。

## 当前状态（2026-10-19 实验执行结果）

数据、许可、逐文件哈希、MuJoCo 足端运动学与人工接触审计已完成，清单状态为 `READY`。已生成 50 Hz、10×29、按 subject 划分且 train-only 标准化的数据集，并完成两个监督式 1D-CNN 架构的 validation 选择。新编码器的 held-out 相位门通过（测试圆周误差 0.863 rad；固定时钟 1.514 rad），但无接触输入的 G1 rollout 只完成约 1.1% 的完整相位绕行，连续推进门失败。因此不运行正式五组 PPO，不写控制性能提升结论；旧 dance 编码器和旧 PPO 结果均不属于本实验的证据。
