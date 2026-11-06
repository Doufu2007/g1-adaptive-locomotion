# 阶段D：源策略KL约束的单因素修正

## 授权与问题
用户已授权按四步持续原因排查和修正验证。C表明改回clock不能一致修复E，而同clock输入训练后的E在2/3seed明显落后；source输入E也有兼容性损失。这里测试一个训练因素：限制对源策略的累计偏离是否能防止低LR训练后的能力退化。不是证明漂移是唯一根因，也不承诺E必胜。

## 固定干预与备选裁决
原生PPO损失加 beta*mean(KL(pi_source(.|o) || pi_current(.|o)))，Gaussian KL在29动作维相加、batch均值，beta固定0.1。源teacher在source9900载入后深拷贝，同当前actor观测（包括各自phase），teacher参数/归一化均冻结，禁止借clock作为E的隐藏输入。不改变奖励/相位适配器/critic/优化器或调度。只支持当前非循环、单GPU、无RND/无symmetry的原生配置，其他明确拒绝。beta0要与原生PPO多minibatch的actor/critic/Adam/RNG逐项一致；新正则梯度只到current actor，教师无随机采样、不改变随机序列。保存教师源状态和beta，恢复不得把当前actor当新teacher。

操作者选择：beta0.1来自常规归一化surrogate与KL的量级，非成绩优化。更换reward会混入目标变化；进一步调学习率只约束局部更新；本轮选择累计source约束，更贴近C的训练路径信号。只跑此beta，不临时加试参数。

## 矩阵、预算与控制
新增clock_source_anchor/E_source_anchor两组×原3seed20261901/02/03×501更新，64world×24steps=每项769536，总4617216训练transitions。仍初始1e-5、原adaptive调度（非固定LR）；源actor/sourceSHA、最佳E SHA不变，freshcritic和freshAdam不变。复用A同seed clock_low/E_low作为无anchor对照。复用前必须逐项审计源actor、freshcritic、物理hash、pre-env commoncfg、把新algorithm class归一回原生PPO后的common_agent SHA；所有检查在首次采样前完成；还须空Adam、source0 checkpoint保存的CPU/CUDA RNG、PYTHONHASHSEED和backend逐项匹配A。新两组训练彼此全配对。

两个worker固定seed分片，worker0 seeds01/03，worker1 seed02；各seed两phase按Random(20263890+seed)固定顺序。不重复启动，原A/B/C冻结源码和site-packages均不修改，不连接旧服务器，不关闭本机。

固定checkpoint0/1/25/100/251/501。前5各3初态steady；501固定10初态20263411..20、steady/speed_change/push，均20秒；每项45回合，总270新增回合，A同等270回合复算，总分析540。3/10world保持A相同cohort，只对相同point做方法配对。终点固定501，不择优checkpoint。正式smoke在封存后两phase各2更新3072，加各10帧评测，合计6144训练transitions+20评测transitions、4次PPO更新，全部另记；失败保留原目录/成本、先修订后恢复，不能删除坏成绩。

## 预定分析与可证伪解释
主指标完整20秒body-forward速度MAE，fall后余段按零速度。先每训练seed内10初态平均，再做3seed配对效应量/95%tCI；四项family仅用501 steady、每seed十初态；speed_change/push为探索描述，不能事后挪为主指标。四项family统一Holm：E_anchor-clock_anchor、E_anchor-E_low、clock_anchor-clock_low、交互(E_anchor-E_low)-(clock_anchor-clock_low)。中间checkpoint只说明过程，不生成胜出结论。航向、倾斜、位移和新跌倒全报告。

在每phase/seed原始source首批capture的同一冻结1536观测上，比较A与anchor各固定checkpoint的KL(source||current)，该bank只作机制诊断，不参与训练/调参。锁定全部6个旧capture SHA，诊断独立CPU执行，不改变正式训练RNG或采样预算；不同策略自身轨迹上的KL不得互相比作减少漂移的证据。

训练每minibatch记录原旧新KL与sourceKL的均值/最大值、beta、原entropy、PPO losses、LR；教师起点/终点state SHA完全不变。原native record_update仍只记录20个旧新KL；sourceKL不能混入该通道。beta0等价测试是native保真门。

若anchor改善E相对无anchor且缓解两坏seed，支持加入source KL这一训练干预有效；只有同一冻结观测bank也显示sourceKL下降，才进一步支持限制累计漂移的解释；仅两phase都改善而差距不缩小，说明共同训练修正，未解释E额外差距。若没有防止退化，不把假设当成立；这轮是有限单因素尝试，不继续扩budget搜beta。候选门沿用：E_anchor-clock_anchor均差<=-.005m/s、至少2/3seed改善，所有保护门均用相同配对steady回合、每seed等权均值，新增跌倒逐episode检查。0新增跌倒、倾斜增<=.02rad、heading增<=.2rad、位移减<=.5m；还须相对未改的clock_low通过完整同速度门（均差<=-.005m/s且至少2/3seed改善）和全部保护门，防把clock改差制造优势。全部通过才事前锁新seed独立确认（样本量按配对波动与实际效应确定），探索三seed不能证明稳定优越。不通过就明确本修正不能支持主张，复核是否存在已识别技术原因，禁止临时改主指标追赢。

## 来源与工程实现
新增scripts/g1_source_anchor_ppo_20261027.py为native单GPU无扩展PPO等价更新加一项正则；scripts/g1_source_anchor_experiment_20261027.py复用原a.train/a.evaluate（进程内接线），A/B/C源文件不改。测试覆盖beta0精确等价、正则解析值和梯度、teacher冻结/RNG/保存载入、配置配对/拒绝重复缺失。封存spec/plan/code/tests、原生PPO和MLP/分布模块、A/B/C协议与analysis/源actor/最佳E、全部复用checkpoint/ONNX/manifest来源。原始raw SHA和模型SHA复算，source成本和effective成本分开。
