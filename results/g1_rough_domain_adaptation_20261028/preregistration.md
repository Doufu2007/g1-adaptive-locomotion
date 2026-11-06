# S前瞻本地封存

Registration status: LOCAL-ONLY. 具体选择由操作者依用户持续有限探索授权制定，不冒称公开注册或逐项用户选择。S是新2×2探索，不是R续跑或独立确认。设计/实施计划/源码/测试/CPU预检/fresh review/独立audit及六完整D501来源于任何采样前哈希封存。

Primary: rough完成率及native跌倒率。先3map×5初态×2scenario在每source seed内平均，再平均三个seed。RE-RC、RE-FE、RC-FC、交互各两终点，统一8Holm；tCI和精确符号翻转n3最小p=.25。不能把480回合当480独立训练。

Exploratory: 域收益、phase候选都仅探索；复用原3来源，不作新seed或状态反馈因果确认。RE须对RC及FC完成+.10、跌倒-.10，各2/3seed方向，加全部rough和flat保护；阈值详情见PROJECT_PLAN，任何失败停止此有限分支。

Stopping: 12×251更新/64×24=4626432train、3012PPO/60240Adam；固定checkpoint0/1/25/100/251，只最终主比较；480eval最多480000有效步。4训练smoke共12288train/8PPO160Adam，8评测smoke最多80eval另账。完整性/容量/非有限/磁盘<10GiB立即停并留故障成本；无自动循环恢复/预算扩展。

Guardrail: rough RE对两个clock MAE恶化<=.02m/s、tilt+.02rad、heading+.2rad、位移损失<=.5m；flat RE相对FE另完成损失<=.10/跌倒增<=.10，和同四保护。原R/D负结果/阈值不变。
