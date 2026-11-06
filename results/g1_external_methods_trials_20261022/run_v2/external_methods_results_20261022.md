# G1 外部方法顺序实测

状态：完整矩阵完成；19/19 次完整训练运行。

完整实测已完成。全603周期macro均值最高为tcn100_rate/direct：49.3 ± 3.0%；其504周期macro为54.3 ± 3.6%。最低一致性MAE配置为tcn100_rate/direct。正式完整门禁通过0/21个输出评估（三种子×七种输出；FLD直接/跟踪共享检查点）。这是开发集配置比较，不能视作最终独立验证。

固定已有 train/validation 开发划分；55820 个一致性训练端点，45236 个全验证端点。三个配对种子：20261021、20261022、20261023。监督均为最终第32轮；TCTST遮蔽预训练、FLD动力学预训练另各32轮。不按验证分数挑轮次。

504周期指检测器一致性子集，99周期指事件计数不一致子集；两者均不代表独立触地真值。表中点误差先逐序列平均，再取四序列均值；门禁同样按四序列macro。±为种子间样本标准差。

| 模型 / 输出 | 帧 | 种子数 | 一致性MAE rad | 全窗口MAE rad | 504门禁 % | 99门禁 % | 603门禁 % | 正式完整门禁通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tcn10 / direct | 10 | 3 | 0.769 ± 0.036 | 0.938 ± 0.016 | 43.0 ± 7.9 | 11.6 ± 0.5 | 38.0 ± 6.4 | 0/3 |
| modern10 / direct | 10 | 3 | 0.745 ± 0.040 | 0.917 ± 0.033 | 50.1 ± 5.5 | 18.6 ± 4.6 | 45.3 ± 4.0 | 0/3 |
| tcn100_rate / direct | 100 | 3 | 0.718 ± 0.033 | 0.905 ± 0.023 | 54.3 ± 3.6 | 21.9 ± 2.7 | 49.3 ± 3.0 | 0/3 |
| tctst_scratch100 / direct | 100 | 3 | 0.767 ± 0.090 | 0.938 ± 0.068 | 40.5 ± 6.8 | 12.1 ± 9.3 | 36.2 ± 7.0 | 0/3 |
| tctst_masked100 / direct | 100 | 3 | 1.571 ± 0.000 | 1.571 ± 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0/3 |
| fld51 / direct | 51 | 3 | 0.742 ± 0.023 | 0.929 ± 0.025 | 46.9 ± 5.6 | 9.3 ± 1.8 | 41.0 ± 5.1 | 0/3 |
| fld51 / tracked | 51 | 3 | 0.797 ± 0.010 | 0.972 ± 0.020 | 52.2 ± 1.8 | 14.8 ± 3.7 | 46.5 ± 1.3 | 0/3 |

正式门禁：全603周期macro≥90%，各序列和各长度层≥80%，并满足每条件/每长度层至少10周期等覆盖要求。按半开周期排除终端重置帧，净推进2π容差π/6，最大相邻步进严格小于π/2。正式可冻结状态直接使用v4.2评分器输出。

## 三种子逐次结果

| 模型 / 输出 | seed | 一致性MAE rad | 全窗口MAE rad | 504门禁 % | 99门禁 % | 603门禁 % |
|---|---:|---:|---:|---:|---:|---:|
| tcn10/direct | 20261021 | 0.8041 | 0.9485 | 37.73 | 12.12 | 33.69 |
| tcn10/direct | 20261022 | 0.7321 | 0.9196 | 52.05 | 11.60 | 45.33 |
| tcn10/direct | 20261023 | 0.7701 | 0.9456 | 39.25 | 11.09 | 35.04 |
| modern10/direct | 20261021 | 0.7292 | 0.9129 | 51.18 | 19.46 | 46.11 |
| modern10/direct | 20261022 | 0.7901 | 0.9518 | 44.24 | 22.60 | 40.94 |
| modern10/direct | 20261023 | 0.7152 | 0.8854 | 55.01 | 13.59 | 48.81 |
| tcn100_rate/direct | 20261021 | 0.6835 | 0.8789 | 58.28 | 20.87 | 52.71 |
| tcn100_rate/direct | 20261022 | 0.7492 | 0.9227 | 53.34 | 19.91 | 48.26 |
| tcn100_rate/direct | 20261023 | 0.7208 | 0.9132 | 51.23 | 24.94 | 47.07 |
| tctst_scratch100/direct | 20261021 | 0.8134 | 0.9780 | 39.71 | 12.63 | 35.40 |
| tctst_scratch100/direct | 20261022 | 0.6639 | 0.8596 | 47.66 | 21.03 | 43.54 |
| tctst_scratch100/direct | 20261023 | 0.8240 | 0.9752 | 34.05 | 2.53 | 29.68 |
| tctst_masked100/direct | 20261021 | 1.5707 | 1.5707 | 0.00 | 0.00 | 0.00 |
| tctst_masked100/direct | 20261022 | 1.5708 | 1.5708 | 0.00 | 0.00 | 0.00 |
| tctst_masked100/direct | 20261023 | 1.5706 | 1.5712 | 0.00 | 0.00 | 0.00 |
| fld51/direct | 20261021 | 0.7149 | 0.9065 | 48.02 | 10.07 | 42.40 |
| fld51/direct | 20261022 | 0.7561 | 0.9562 | 51.81 | 10.55 | 45.28 |
| fld51/direct | 20261023 | 0.7539 | 0.9244 | 40.76 | 7.21 | 35.44 |
| fld51/tracked | 20261021 | 0.7955 | 0.9750 | 52.16 | 17.60 | 46.92 |
| fld51/tracked | 20261022 | 0.8079 | 0.9902 | 54.04 | 10.55 | 47.52 |
| fld51/tracked | 20261023 | 0.7875 | 0.9507 | 50.45 | 16.22 | 45.05 |

## 配对差

全部为候选减对照；MAE负值为改善，门禁正值为改善。仅三个种子的描述性结果，不作跨受试者显著性推断。

| 候选 vs 对照 | MAE差 rad（逐种子） | 504门禁差百分点（逐种子） | 603门禁差百分点（逐种子） |
|---|---|---|---|
| modern10/direct vs tcn10/direct | -0.075, +0.058, -0.055 | +13.449, -7.814, +15.763 | +12.417, -4.389, +13.778 |
| tctst_scratch100/direct vs tcn100_rate/direct | +0.130, -0.085, +0.103 | -18.567, -5.677, -17.178 | -17.314, -4.724, -17.389 |
| tctst_masked100/direct vs tctst_scratch100/direct | +0.757, +0.907, +0.747 | -39.709, -47.659, -34.047 | -35.398, -43.539, -29.677 |
| fld51/tracked vs fld51/direct | +0.081, +0.052, +0.034 | +4.139, +2.223, +9.689 | +4.521, +2.248, +9.613 |

## 训练收敛记录

仅用于判断是否完成有效拟合；不同目标的损失数值不可直接比较。所有验证预测均来自最后固定轮次。

| 模型 | 阶段 | 首轮损失（种子均值） | 末轮损失（种子均值） |
|---|---|---:|---:|
| tcn10 | supervised | 0.259462 | 0.018987 |
| modern10 | supervised | 0.229906 | 0.007980 |
| tcn100_rate | supervised | 0.174856 | 0.001274 |
| tctst_scratch100 | supervised | 0.509277 | 0.002809 |
| tctst_masked100 | supervised | 0.504729 | 0.501638 |
| tctst_masked100 | masked_pretrain | 0.864919 | 0.823375 |
| fld51 | supervised | 0.418899 | 0.078830 |
| fld51 | fld_pretrain | 0.558006 | 0.139890 |

## 分层覆盖与504周期表现

| 模型 / 输出 | 分组 | 周期数 | 完成率 % |
|---|---|---:|---:|
| tcn10/direct | walk1_subject1 | 149 | 63.8 ± 2.9 |
| tcn10/direct | walk2_subject1 | 128 | 42.2 ± 2.8 |
| tcn10/direct | walk3_subject1 | 128 | 36.5 ± 14.5 |
| tcn10/direct | walk4_subject1 | 99 | 29.6 ± 12.8 |
| tcn10/direct | short | 173 | 34.9 ± 8.9 |
| tcn10/direct | medium | 182 | 48.4 ± 8.1 |
| tcn10/direct | long | 149 | 51.5 ± 5.1 |
| modern10/direct | walk1_subject1 | 149 | 70.2 ± 5.1 |
| modern10/direct | walk2_subject1 | 128 | 49.7 ± 2.0 |
| modern10/direct | walk3_subject1 | 128 | 34.1 ± 15.6 |
| modern10/direct | walk4_subject1 | 99 | 46.5 ± 12.4 |
| modern10/direct | short | 173 | 40.3 ± 5.3 |
| modern10/direct | medium | 182 | 56.6 ± 5.4 |
| modern10/direct | long | 149 | 57.3 ± 6.0 |
| tcn100_rate/direct | walk1_subject1 | 149 | 74.3 ± 2.2 |
| tcn100_rate/direct | walk2_subject1 | 128 | 47.4 ± 1.2 |
| tcn100_rate/direct | walk3_subject1 | 128 | 44.3 ± 5.9 |
| tcn100_rate/direct | walk4_subject1 | 99 | 51.2 ± 10.7 |
| tcn100_rate/direct | short | 173 | 38.7 ± 6.0 |
| tcn100_rate/direct | medium | 182 | 61.5 ± 1.5 |
| tcn100_rate/direct | long | 149 | 66.9 ± 3.9 |
| tctst_scratch100/direct | walk1_subject1 | 149 | 72.7 ± 6.2 |
| tctst_scratch100/direct | walk2_subject1 | 128 | 37.0 ± 7.8 |
| tctst_scratch100/direct | walk3_subject1 | 128 | 31.0 ± 9.8 |
| tctst_scratch100/direct | walk4_subject1 | 99 | 21.2 ± 7.9 |
| tctst_scratch100/direct | short | 173 | 19.5 ± 6.9 |
| tctst_scratch100/direct | medium | 182 | 50.9 ± 7.7 |
| tctst_scratch100/direct | long | 149 | 60.4 ± 5.7 |
| tctst_masked100/direct | walk1_subject1 | 149 | 0.0 ± 0.0 |
| tctst_masked100/direct | walk2_subject1 | 128 | 0.0 ± 0.0 |
| tctst_masked100/direct | walk3_subject1 | 128 | 0.0 ± 0.0 |
| tctst_masked100/direct | walk4_subject1 | 99 | 0.0 ± 0.0 |
| tctst_masked100/direct | short | 173 | 0.0 ± 0.0 |
| tctst_masked100/direct | medium | 182 | 0.0 ± 0.0 |
| tctst_masked100/direct | long | 149 | 0.0 ± 0.0 |
| fld51/direct | walk1_subject1 | 149 | 62.9 ± 2.7 |
| fld51/direct | walk2_subject1 | 128 | 38.5 ± 4.3 |
| fld51/direct | walk3_subject1 | 128 | 37.2 ± 7.8 |
| fld51/direct | walk4_subject1 | 99 | 48.8 ± 11.7 |
| fld51/direct | short | 173 | 37.0 ± 6.4 |
| fld51/direct | medium | 182 | 54.9 ± 4.3 |
| fld51/direct | long | 149 | 50.3 ± 5.7 |
| fld51/tracked | walk1_subject1 | 149 | 69.1 ± 5.3 |
| fld51/tracked | walk2_subject1 | 128 | 39.6 ± 1.6 |
| fld51/tracked | walk3_subject1 | 128 | 38.5 ± 2.4 |
| fld51/tracked | walk4_subject1 | 99 | 61.6 ± 2.7 |
| fld51/tracked | short | 173 | 40.7 ± 0.7 |
| fld51/tracked | medium | 182 | 60.3 ± 2.5 |
| fld51/tracked | long | 149 | 56.4 ± 2.9 |

## 范围与复核

- 本次是G1的29关节输入适配实测。ModernTCN使用固定10帧新相位头；TCTST按论文实现，100帧窗口左补8点以右对齐下采样；FLD使用官方编码器并增加零谱/零幅保护和末端相位换算。
- TCTST原论文使用100Hz的200点、IMU输入及100轮训练；本次50Hz的100点、关节输入、32轮。v2恢复前20轮学习率预热及原始sin/cos平方误差，推理时归一化，速度损失仍为0.02权重；不使用验证驱动学习率、早停或选checkpoint。不能称原论文完整复现，外部2.729%周期RMSE不能直接与本次MAE或门禁对比。
- FLD先32轮五步重建推进，再冻结编码器训练读出；tracked输出在同一检查点上，以固定alpha=0.1校正预测速度积分。重置仅依赖序列/真实帧缺口，不读取参考触地。
- 新32轮TCN对照与此前8轮旧结果预算不同；二者不能当成同预算配对。
- 未读取test数组和已查看新跑步序列；未重拟合标准化。验证仍是单一留出受试者的开发集，不是新的独立确认。未修改部署输入契约，未启动PPO。
- 每次运行保留最终/部分检查点、原始预测、逐周期记录、训练日志、配置及哈希。partial checkpoint没有自动续训实现；失败重试从头训练并保留失败attempt。
- 预测重算和产物哈希已核验19次完整运行；详见 independent_prediction_audit.json。

- 训练配方修订：v1的TCTST首个种子25轮损失约1.00，在生成任何TCTST验证分数前中断。run_v2/protocol.json冻结学习率预热和原始状态回归修订；v1中断日志/第24轮检查点保留，不进入种子均值。详见相邻amendment_v2_training_recipe.md。
- 继承结果保持原协议身份：已完成ModernTCN/TCN对照引用run_v1原始manifest和产物，run_v2记录其hash；新TCTST及后续运行使用v2协议，不把继承结果改写成新协议训练。

- ModernTCN同种子完整复训：预测hash一致=True；权重内容hash一致=True。

来源：[TCTST](https://arxiv.org/html/2506.15150v1) · [ModernTCN官方仓库](https://github.com/luodhhh/ModernTCN) · [FLD](https://arxiv.org/html/2402.13820v1)。全文/源码来源快照见相邻g1_external_methods_search_20261022目录。
