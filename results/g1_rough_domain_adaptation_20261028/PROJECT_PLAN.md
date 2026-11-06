# 阶段S：一次有界的训练地形对照

## 目的、授权与已知结果

用户原主张是近期运动学习相位替代固定CPG，让节奏随运动调整并改善稳定/鲁棒，尤其粗糙地形；用户已要求每15分钟持续有限验证。此方案的实现、预算和种子由操作者在该授权内决定，不冒称用户逐项选择。不得修改A/B/C/D/R冻结源、site-packages、既有主终点或负结果，不提交/清理，不连接旧服务器或关闭本机。

R的480回合独立计算审计通过：DE粗糙完成13/90、跌倒59/90，DC为9/90、71/90；完成率+4.44pp未达到预定+10pp，三seed推断不确定，未形成联合候选。受推完成两者均5/45，粗糙相位/触地频率关联未一致。D平地速度失败保留。R的训练域审计确认12项续训策略均为plane，R评测是真实random_rough；域差异明确，是否造成失败尚未检验。

问题S1：相同额外训练预算下，rough训练相对flat训练是否改善同一phase管线的粗糙行走？S2：在两者都接受rough训练后，学习相位管线是否优于强clock？这测试有限适配条件下的完整管线和训练域效应，不是原理的终局判断，也不是状态反馈因果验证。仅新增rough训练而对照停止训练无法排除额外优化，因此使用完整2×2训练域×phase对照。

## 四方法与训练预算（结果前决定）

|方法|额外训练域|phase|起点|
|---|---|---|---|
|FC|flat|clock|对应D clock seed的model_501.pt|
|FE|flat|learned|对应D learned seed的model_501.pt|
|RC|rough|clock|与同seed FC完全相同D模型|
|RE|rough|learned|与同seed FE完全相同D模型|

三个来源训练seed固定20261901/02/03，不按R成绩选择。每cell固定额外251次原生PPO更新、64world×24step/update，即385536 transitions；12cell合计4626432 train transitions、3012 PPO、60240 Adam minibatch steps。采用251以将四方法总预算限制在约一次D实验的规模，非最优训练时长宣称；若不足不得自动延长。训练相对checkpoint固定0/1/25/100/251，仅251参与主比较；不挑中间成绩。单worker、每cell fresh child，固定Random(20264590)打散12cell。不会运行12个同时占GPU的进程。

原生SourceAnchoredPPO及beta=.1保持；加载每个D501的actor、critic、Adam、normalizer、原source9900 teacher和其coefficient，均不得重置/改成新teacher。显式从source infos.factorial_state.learning_rate恢复算法learning_rate，并核所有Adam param-group lr相同；不使用构造器默认LR覆盖历史adaptive状态。原PPO/Adam/奖励/critic输入结构/归一化更新规则不改。encoder始终为冻结最新seed20261026、SHA8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346。teacher使用各管线自己的phase观测，参数及normalizer冻结，不给E隐藏clock。

模型和算法状态是warm start；仿真从新环境开始，不能叫D的精确轨迹续接。完成环境和runner构造并加载后，恢复D501保存的Python/NumPy/CPU-CUDA RNG；同phase的两个域在第一次动作采样前须完全一致。重置相对iteration和common_step_counter为0，调用一次native learn(251, init_at_random_ep_len=True)，不反复learn(1)。随机episode长度来自恢复的同一RNG，记录源RNG及首次采样前RNG。D模型中的infos含完整RNG，缺失即停止，不能静默换种子。

## 训练域唯一变化与真实地形暴露

基底始终是study.make_cfg(flat, phase, seed, play=False, num_envs=64, horizon=20)，保留训练命令范围0.3..0.8m/s、无侧向/角命令和原有训练噪声/随机化/奖励；不把原生rough整套actor的height_scan或不同奖励接进来。actor98维，critic维度与对应D source完全一致。所有四方法共用R验证过的rough求解器容量：ccd_iterations500、contact_sensor_maxmatch500、njmax1500、nconmax128。

rough仅移入native random_rough地形实体，与R同noise_range(.02,.10)m、noise_step.02、horizontal_scale.1、vertical_scale.005、border_width.25、1row×5col、patch32×16m、curriculum=False。训练地图seed按来源seed顺序为20264601/02/03，相同来源seed的RC/RE使用同一完整heightfield数组。R评测地图20264201..03不得参与训练。每次reset以world id模5指定patch，事件替换原randomize_terrain且保持在reset_base之前；flat使用同一事件，其无terrain_origins时无操作。不根据训练/评测分数改变地形难度或重采地图。

采样前核：同域C/E公共配置摘要一致；同phase跨域的actor/critic/optimizer/teacher/normalizer及RNG完全相同；跨域去除terrain实体和允许的terrain origin差异后配置相同。phase之间D501权重本来不同，不伪造相同actor声明。跨phase/域核机器人实际body_mass/body_ipos/friction等，按机器人body/geom名称规范化匹配，不能用额外hfield引起的数组长度差异误判，也不能隐藏真正机器人动力学差异。同域核hfield完整数组和geom映射相同。非配对即零step停止，先保留证据再修订；不得先采样再忽略初始化差异。

训练保存完整静态hfield/尺寸/geom数组与runtime机器人物理。每24step rollout边界读取真实foot-geom接触（R已验证的原生14 foot collision geoms、efc有效、dist<margin），累计指定patch内部接触及非指定接触；这只是每更新采样时刻的训练暴露覆盖，明确不冒充全帧覆盖。每个rough cell在固定251个边界样本中必须有实际指定hfield足接触，否则实验无效；不把根位置代替接触。全程逐step检查原生contact/constraint容量未饱和、动作/状态/奖励有限。teacher/encoder逐checkpoint和结束核冻结，逐更新记录20旧策略KL与20 sourceKL、native minibatch数、budget和学习率，不能把sourceKL下降当性能证明。

## 未见地图上的固定评测

四个新方法只评相对251最终权重：flat及3个未训练rough地图20264701/02/03，每地图15world=3来源seed×5初态20264811..15，每world steady/push两场景，20秒50Hz；共480回合，最多480000有效eval、0评测PPO。固定评测construct seed20264200，沿用R已验证的98维构造、求解容量、patch和reset顺序；改变的只为预定模型/地图/初态路由。评测Random(20264790)固定打散16组，每组fresh child、单worker。S评测地图与训练地图及R地图两两不交；保存完整数组及SHA，拒绝相同地图物理hash泄漏。

允许通过新的显式adapter复用冻结R的rollout/physical/contact实现：只在S fresh child设置S输出目录、S协议校验、S政策入口与S slot矩阵，不修改R文件或R结果，不调用R的queue/lock来重采R。actor输入路由必须逐world绑定正确source seed和domain，先做4方法/3seed/新地图的CPU ONNX路由gold test；R所有物理/初态/command/指定patch接触和原生终止保障沿用，adapter依赖须封存。

命令固定.5m/s、yaw0；push为5/10秒世界Y速度+.3/−.3m/s。每回合在第一次原生终止截断、不拼接reset；容量/非有限/模型绑定错误立即整队停留故障目录。真实指定patch内部foot接触比例不足95%、零接触或离区仍计未完成，不删失败。MAE按完整20秒，早倒余段速度补0。

## 主终点、统计family与可证伪判决

持续行走完成定义原样保留：20秒无倒、初始方向位移≥5m、body-forward>.1m/s比例≥.8、rough有效足地接触点至少95%在指定patch内部。主要终点为rough完成率与跌倒率；先每来源seed内等权平均3map×5初态×2scenario，再平均3seed。统计单位是三个匹配来源管线seed，不是480独立训练/新seed确认；有重复使用D来源和一次固定适配随机流的限制。

预定4对比：RE−RC（rough训练后的phase管线差）、RE−FE（E的训练域差）、RC−FC（clock的训练域差）、(RE−FE)−(RC−FC)（交互），各完成/跌倒，共8项family统一Holm。全部报告每seed差、均差、95%tCI与精确seed符号翻转p；n3最小双侧p=.25，不能宣布统计显著优势。不能只挑交互或正tCI；跨域因子包括在不同物理环境采集的整个学习过程，不能缩成某一相位组件的因果归因。

S探索候选须RE同时对RC和FC满足：rough完成率均差≥.10、跌倒率均差≤−.10，两项各至少2/3来源seed有利；rough MAE不恶化>.02m/s、位移不损失>.5m、tilt不增>.02rad、heading不增>.2rad。另对RE相对FE的flat完成率损失≤.10、跌倒率增加≤.10、MAE增加≤.02m/s、位移损失≤.5m、tilt增加≤.02rad、heading增加≤.2rad；原始flat全部数据及RC代价都报告。这些是新的S前瞻保护，不能回写R候选。

若RE相对FE的rough完成≥+.10且跌倒≤−.10，并各至少2/3seed有利，可描述有限预算下rough训练域有利的点效应；若clock也同样改善，解释为两管线共同受益，不能专归学习相位。即使该域门过，phase候选不通过也不能声称E胜clock。

若全部候选门通过且工程审计通过，仅形成新探索候选；随后才能另行前锁新训练seed和地图的独立确认及同权重匹配开环/反馈移除对照，不能把S当确认。若phase门失败或无结论，结束本次域适配分支并如实汇总原主张证据，不自动改beta/奖励/编码器/地图/训练预算继续追赢。n3 CI跨零照实，既不放宽门也不把点效应包装成确认。不存在“必须调到支持原主张”的成功条件。

## 工程门、故障、有限smoke

正式采样前完成目标链/失败树/完整计划包、CPU测试与语法、零步环境配置和源模型状态预检、fresh独立设计builder-check及实现review，全部来源/12cell路由/源D501状态/环境代码与版本封存。只读/零步构造不算成功smoke，不宣称性能证据。

封存后四方法各固定第一来源seed做2更新/64world真实训练smoke：共12288 train、8 PPO、160 Adam；再四smoke模型各flat/固定首个未见rough地图×1world×10帧，最多80 eval、0评测PPO。每项fresh child；粗糙smoke必须确有指定hfield足接触，完整恢复状态、teacher/encoder冻结、native20 minibatch、checkpoint和ONNX路由均过。smoke是工程覆盖，不用于挑seed/预算/方法。任何故障保留原目录/协议/源码和实际train/eval/PPO/Adam，先评估并锁修订，不自动重试循环。无完整仿真state不能声称从中间checkpoint精确续接。

正式队列只能启动一次，先查launch/queue与实际进程。进程消失不是完成；必须核实际updates/transitions/原生trace/评测raw并独立复算。全部480raw/物理/初态/模型/真实hfield接触/原生终止/指标/8family/候选，由结果前封存的独立CPU脚本复算；同操作者独立算术不冒充第二review。历史D/C不精确损耗继续独立账。磁盘低于10GB、容量饱和或非有限时停留原目录收证据，不清理旧资产或随意缩预算。

## 实现分工与限制

新增S contract、experiment、evaluation adapter、独立audit及测试，R及D原生实现只读复用；不重构旧项目。方案中的actor98维、beta、PPO预算、地图/初态seed、主终点、family与停止由操作者决定并在任何S采样前封存。builder可决定文件内部组织和正常日志格式，不能改科学设计或用中间checkpoint/新地图替代失败项。

此书面设计须经fresh builder-check澄清后形成实施计划；目标链/失败树/机读计划与零采样预检仍未完成前不启动训练。整体研究目标未完成，不能仅因设计、探索、队列启动或工程PASS就标主张已证实。

## 设计检查与零采样来源核验

fresh独立builder-check已返回“nothing important left”，并只读核对D原生完整恢复及R评测接口；没有新采样或代码修改。CPU核验六项D501均含17项Adam参数状态、每项step=10020、完整CPU/CUDA及Python/NumPy RNG和原teacher/beta=.1。clock来源20261902实际末学习率为2.25e-5，其他约1e-5，均与其optimizer param-group一致；因此必须按各来源恢复，不能统一用新构造器1e-5覆盖。跨phase比较是完整历史管线对照，不宣称当前LR数值相等；同phase跨训练域状态必须完全相同。证据在新S目录source_state_audit.json，0环境步、0新增PPO。

## Question / Estimand

S1估计同phase下rough对flat额外训练的三来源seed平均完成率/跌倒率差；S2估计rough训练后RE对RC完整管线差。计量对象是三复用来源seed在固定三未见map、五初态、两场景上的条件均值，不是480独立训练。

## Fair baseline / 对照公平

FC/FE/RC/RE全部额外251次原生PPO，训练计算预算相同。每同phase同seed跨域完整载入同一D501 actor/critic/Adam/teacher/normalizer/实际adaptive LR/RNG；clock02的2.25e-5历史保留，跨phase不宣称历史权重相等。全部最终251权重在相同未见map/物理/初态/推力上比较，原R地图不训练。

## Preregistration / 预注册

仅本地前瞻封存，无公开注册。任何S环境采样前锁定设计、预算、矩阵、独立审计源码和依赖SHA；smoke与正式预算分账。不因中间结果换seed/主指标/checkpoint/β或扩预算。新S阈值不回写R。

## Risk / 风险

完整状态或LR/RNG漏恢复、真实机器人动力学不配对、capacity溢出和train/eval数组重合会破坏可解释性：全部在采样前或逐step fail closed。三来源seed及反馈未消融限制确认性/因果主张，更多回合不能补足；候选失败即停止此分支。技术中断保留目录/源码和实际损耗，先证据与预锁恢复，无自动重试。

## 采样前实现审查澄清

原生训练配置没有play模式的randomize_terrain事件，S四臂在reset_base前统一插入同一deterministic事件；已有时替换。rough场景origin初始化会先消耗Torch随机数，因此S Env子类在scene/terrain构造后、全部manager/timer/command/startup构造前的load_managers入口按同source seed重设Python/NumPy/CPU-CUDA随机流，四臂共同采用，保留原生随机化规则；随后完整加载源并在首次采样前恢复D501 RNG。common_step_counter在mjlab load恢复副作用之后归零。CPU只证明接口/构造顺序契约，真实零步物理/初始化配对仍是smoke硬门。
