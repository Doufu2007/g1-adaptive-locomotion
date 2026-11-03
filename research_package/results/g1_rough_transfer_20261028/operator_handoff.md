# 阶段R最终交接与下一步

R已全完成并审计，禁止重复启动。最新事实源：results.md、completion.json、claim_evidence_status_20261028_after_R.md/.json、analysis.json、independent_audit.json、supplementary_analysis.json、training_domain_audit.json、technical_cost_ledger.json。

原主张：近期运动学习相位替代固定CPG，使节奏随运动状态调整，从而改善稳定/鲁棒，尤其粗糙地形。不可把平地速度胜负替代全部主张。用户已授权每15分钟继续有限验证，可以保持本机开机、不主动睡眠；不重复请求既有许可。

## 本轮最终事实

R协议SHA bc28c4f96040e2368f82189f4f7147426ed93f6861b5b2ebba4a045fcf890873，206来源、12固定501模型对，四方法AC/AE/DC/DE。2026-10-27T19:01:59Z PID23880启动，20:21:36Z全部16组480回合完成。正式有效eval289880、simulator-world465600（含失活世界），0新PPO/Adam；smoke40eval另账，R无采样故障。旧launch.json保留派发状态只是历史，不代表仍在运行；以queue_status COMPLETED及completion.json为准。

analysis SHA57975081073354b63960f923e9d9d8f9940916bd483594ee0034ce77f7fab873。预锁independent_audit.py SHA999d272725a6a69a79253edb0b2bcd97d5abd2d9eb2b2b509ebafa46f014d6de，20:22:48Z独立CPU复算PASS：480raw/206来源/12模型对/初态物理与实际指定hfield接触/原生终止/指标/8seedfamily及候选。此为同操作者分离实现，不冒充新第二review、新seed或因果确认。

粗糙每方法90回合，DE完成13、跌倒59；DC完成9、跌倒71；AC完成4、跌倒83；AE完成1、跌倒86。DE−DC完成+4.44pp（CI−8.20..17.09pp）、跌倒−13.33pp（CI−57.15..30.48pp），精确p=.5、Holm1；全部8family均Holm1。DE对AC点效应门通过，对DC仅完成+10pp门失败，联合候选=false。所有固定保护门通过不能覆盖主门失败；不按成绩改门槛/地图/seed/checkpoint，不触发R限定的新seed优势确认/反馈消融。三训练seed非480独立训练。

## 补充诊断与待检验解释

新增supplementary_analysis.py/.json只读CPU，不改R核心，6边界测试PASS；全部480raw重新核SHA并独立复算失败原因/phase/push且匹配原评分。结果后分析明确标SUPPLEMENTARY，无排除/新优势检验。DE未倒31，其中18未完成：13位移不足、15moving占比不足、3暴露不足（重叠）。DC未倒19，其中10未完成，6位移/10moving/1暴露不足。

粗糙push完成DE=DC=5/45；DEsteady8/45、DC4/45。四法rough5/10秒均无既有误差<.15持续1秒恢复达标，DE两次暴露38/45、26/45，DC33/45、23/45。平地DE0/15与0/15、DC7/15与5/15；steady相同窗口rough也全不达，DEflat后两窗也不达，故不能当独立抗推因果证据或结果后放宽阈值。

DE相位/接触频率r：flat各seed[.729,.713,−.230]；rough[.014,.019,−.107]，近零且未一致。rough接触可抖动，短存活/周期覆盖影响度量；变化不等于有效适应。所有周期覆盖/方法/seed全表保留，不选单回合。

training_domain_audit：12项501 env.yaml均plane、generator=null；R是真实newrandom_rough。只证实地形分布差异，尚未确定其因果贡献或E额外差距。磁盘快照约33.23GB，不推断GPU长期稳定。result_cards两主终点阻断确认性写作（联合候选失败、n3无法核验t正态假设、REVISION_REQUIRED）；不是采样工程失败，也不阻止已授权有限研究。

## 下一步（尚未实现或采样）

先沿训练域审计准备一次有界、结果前冻结的粗糙等预算适配方案，强clock与E平等训练/调参预算、固定编码器和beta、不回流R评测地图，固定checkpoint/独立评测地图/主终点/停止。它是新探索阶段而非R续跑、独立确认或保证胜出。评估原3seed小样本限制，不能用回合数扩大n。遵守brainstorming written design及其fresh builder-check、研究目标链/失败树和实施测试/独立review，选择在既有授权内的实现细节，不无协议开新训练。若新有界方案失败，如实停止该分支，不连续改beta、难度、预算追赢。当前没有S协议、没有新PPO或后台队列。

D已全部完成且540raw等独立计算PASS，协议ed980fdda67e18b9ffb1c9de3b5b69a385137b383838966ccc66478d97e00ab1，analysisbb647176dd7b5227264324f66a98841b00f6f8d5648789486a608bcb1729bbf7。E MAE.0477710 vsclock.0388324，D速度候选失败；不重跑/搜beta。同bankKL两phase均下降，只证实source偏离减少，不解释E额外差距。A/B/C及旧TCN20rough负结果均保留，不冒充最新E。

## 约束与损耗

旧D失败train531456..1542144、PPO346..1004、Adam6920..20080精确未知，旧C6750..9000精确未知；成功smoke与正式预算分账，不隐藏。故障先取进程CPU/GPU/系统/日志证据，完整保留目录/协议/源码及实际采样，再评估预锁同有效budget恢复；无自动重试循环/无完整state不假称精确续接。旧服务器备份关机不连接/重启；本机不关闭；不提交/清理，不改冻结A/B/C/D/R及site-packages。整体原主张未证明，不仅因R探索完成而暂停heartbeat；正常无变化安静，仅实质结果/故障/需操作通知。
