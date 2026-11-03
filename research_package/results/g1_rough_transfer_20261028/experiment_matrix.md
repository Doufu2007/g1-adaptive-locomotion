# 有限rough迁移矩阵

| 实验ID | 对应假设 | 数据集 | baseline | 指标 | 已控混淆/负对照 | 完成判定 |
|---|---|---|---|---|---|---|
| EXP-01 | H1：冻结E在粗糙迁移提高持续行走 | 3新terrain seed×5patch×3训练seed×2场景 | 同修正DC及原AC | completion/fall rate | 同地图/起点/计算预算；保留AC/AE/DC/DE全部501，无调参 | 480回合上限；completion difference>=.10 and fall rate difference<=-.10 against DC/AC，各至少2/3seed有利且保护门通过，只产生探索候选 |
| ROB-01 | H1 secondary：平地代价与推力恢复 | flat新5初态及相同rough | 同修正clock DC | 20s MAE/tilt/heading/displacement、push recovery | 同command和Y速度脉冲；早倒补零，未暴露推力单列 | 固定480回合统一停止，MAE差<=.02、tilt差<=.02、heading差<=.2、displacement差>=-.5；所有flat代价与push recovery全报，不按结果调难度 |
| EXP-02 | H2：机制关联，仅探索 | 同480raw | clock input固定1/.6Hz | phase/contact cycles | 不含移除反馈负对照，不作因果归因 | 固定480回合后只描述phase/contact cycles；complete cycles>=3时计算相关，否则不足；不作因果证明 |
