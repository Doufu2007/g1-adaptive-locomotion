# 有界训练域×相位矩阵

| 实验ID | 对应假设 | 数据集 | baseline | 指标 | 已控混淆/负对照 | 完成判定 |
|---|---|---|---|---|---|---|
| EXP-01 | H1：rough经验在同phase改善粗糙行走 | train20264601..03，eval20264701..03 | 相同D501起点的FE/FC | completion/fall | RE-FE与RC-FC同额外251训练预算；完整Adam/teacher/LR/RNG；地图隔离 | 各phasecompletion difference>=.10 and fall rate difference<=-.10、各2/3seed有利才域点效应，非phase收益 |
| ABL-01 | H1：隔离额外优化与训练域 | 同3来源seed；12cell | flat额外训练FC/FE | completion/fall及交互 | 完整2×2，额外训练/模型家族/地图隔离；只变训练地形 | 固定12×251=4626432train，480eval；报告completion/fall rate两phase域差及8family |
| EXP-02 | H2：rough训练后E完整管线有优势 | 同未见map/初态steady/push | 强RC及FC | completion/fall | RE对两个clock门；不含匹配反馈移除，不作因果归因 | 两对比completion difference>=.10 and fall rate difference<=-.10、方向>=2/3，各rough保护且RE-FE flat保护全过；只候选 |
| ROB-01 | H2 secondary：rough及flat代价 | flat0与3rough | RC/FC/FE | MAE/tilt/heading/位移 | 20s全窗早倒补0，第一终止，不删零接触或边界失败 | rough MAE+.02、tilt+.02、heading+.2、位移损失.5；flat另完成损失.10/跌倒+.10，固定预算到期即停 |
