# PPO 起始更新与价值网络初始化诊断

用户在上一轮报告后要求“继续”。本轮仅本机执行，保留已有试验，不启动远端或关机，不追加完整501更新训练。

问题：首次更新的学习率起点及fresh critic，是否导致较大的策略漂移和第一轮行走退化？这是已知原结果后的探索性新问题；不把第一轮效应当作501更新慢走的完整因果解释。

固定模型：动作策略来自baseline model_9900.pt；E仍为seed20261026、SHA256 8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346。原训练种子20261901/02/03，原奖励/配置不变。源价值网络与当前均113维；运行前核对critic观测项顺序及除phase外的参数。复用旧critic只是一个初始化处理，不假定它在新奖励/E状态上正确。

每个种子、每种相位（clock/E）实际采集一次64环境×24步，按原actor warm-start、fresh critic、empty optimizer、初始lr0.001执行一次真实PPO更新。保存更新前的完整rollout、原始奖励、time_outs、初始/归一化更新后的critic、actor、RNG和每个小批次的实际KL/LR/梯度/参数漂移。

在同一保存的rollout上做2×2回放：初始lr0.001/0.00001 × fresh/source9900 critic。所有variant使用同一actor、空optimizer、观测、动作、原奖励、dones、old log-prob、old action distribution、更新前RNG及原20个小批次算法。warm critic按记录的下一观测逐步更新自身normalizer、重算V、timeout bootstrap及GAE；其余数据不动。fresh标准回放不装记录器，必须与真实装记录器的更新结果一致；未一致则停止解释并修回放，不按成绩换数据。

预算硬上限：6项实际首批采集（9216 transitions）及6次真实首轮更新；24项冻结数据回放（480个minibatch optimizer步骤）；合计30次PPO更新，无新增数据交互训练。每个单元均保留，不依结果筛选。

E三种子评测四个事先选定策略：before_update、default_fresh、low_lr_fresh、default_source_critic。每策略3个固定初态20263111/12/13，0.5m/s、20秒，共36正式回合。双处理low_lr_source_critic仅做更新机制回放，不据结果补选进入评测。最多两个0.2秒技术smoke，不作为科学样本。

主要比较family只有两项：low_lr_fresh减default_fresh、default_source_critic减default_fresh的完整20秒前向速度MAE，先每训练种子内平均，再按3种子计算差值/95% t区间、两项BH。before_update作描述性参照；clock回放、KL和参数/动作漂移为机制描述，无额外显著性筛选。保护指标为跌倒、倾斜、累计航向及初始方向位移。

支持分支：三个种子MAE同向降低、95%区间在0以下且q<.05；没有新增跌倒、均值倾斜增加≤0.02rad、航向绝对变化增加≤0.2rad、初始方向位移损失≤0.5m。只降低动作漂移而不改善MAE，只能称限制了首轮变化；改善MAE但保护指标失败为取舍；方向混合/区间跨0为未见一致证据。不得据结果再试学习率、增加更新或换种子。

严格核查：所有首批数据和来源SHA；sourcecritic结构/观测兼容性；初始actor/critic与原训练manifest比对；fresh回放重现V/returns/advantages、每小批次次序及真实更新结果；记录器不消耗RNG、不改变梯度；encoder冻结；ONNX与CPU模型一致；评测配置、初始状态及物理匹配。PyTorch/Warp物理不保证逐位复现，不能把闭环评测称精确物理反事实。PYTHONHASHSEED在新进程启动前固定，记录实际backend设置，不回填旧试验的设置。

失败保留：非有限值、回放不一致、观测不兼容时保留文件并停止科学解释，只做必要技术修复；不覆盖失败。结果无益或n=3不稳定不作为自动扩跑理由。交付全部原始数据、检查点、日志、中文报告及证据机检。

## 首次采样前技术修订

第一项在0个训练步、0次optimizer更新时被原critic初态SHA断言拦下；actor和启动物理一致，critic的观测项兼容性通过。原Linux torch2.8与本机Windows torch2.6的随机critic哈希不同，尚未隔离具体差异源，不能宣称重现原远端初始化。旧协议、代码和失败原样保留。改在local_reconstruction_v2子目录执行同一固定矩阵和预算；保留原actor逐位一致检查，改为核验本机clock/E同seed的fresh critic逐位相同及各回放起点相同。结论限制为本机同数据匹配诊断，不能当作原远端首轮的精确反事实。所有成绩采集仍未开始；模型、种子、处理、预算和评分不变。另修正ONNX API必填verbose参数，属静态接口修复。

第二次技术修订：v2在首次采集1536步后、首次optimizer之前因TensorDict迭代默认产生批行而不是键，导出断言索引报TypeError；不是观测数值不一致。更正为.keys()，并在断言前保存完整batch与pre-step snapshots便于以后无新增采样恢复。v2全文件和代码快照保留，使用local_reconstruction_v3继续；额外计入1536技术失败采集步（0次optimizer），科学矩阵仍9216步、30更新、36回合，总采集上限10752。不按成绩换数据或扩展处理档位。
