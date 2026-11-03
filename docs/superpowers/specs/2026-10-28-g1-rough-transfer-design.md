# 阶段R：最佳E与clock的冻结策略粗糙地形迁移

## 用户目标、授权和问题
用户确认的原主张为：从近期运动学习相位替代固定CPG，让节奏随运动状态调整，从而改善稳定/鲁棒，尤其粗糙地形。2026-10-28用户明确要求“继续探索剩余没有证明的主张”。本方案在已授权持续有限验证范围内，由操作者选择下列可逆实现和预算；不追溯修改A/B/C/D指标，不再重复D训练、不搜beta，不请求重复许可。

问题R1：已冻结的最新E策略，相对公平clock策略，能否在未参与其PPO训练的新随机高度地形上，提高持续行走完成率并减少跌倒？这是零PPO迁移探索，不是充分粗糙地形训练的终局比较。失败只否定这些冻结策略在此条件下的优势，不否定原理。问题R2：相位响应、实际接触节奏与扰动恢复有何关联？R2只描述机制线索；没有同权重移除状态反馈的干预不能声称适应导致收益。

## 已知证据与保留
D分析SHA bb647176dd7b5227264324f66a98841b00f6f8d5648789486a608bcb1729bbf7，经540raw等独立计算审计。D E恒速MAE .0477710、clock .0388324；速度候选失败、保护门通过，四family CI跨零。同观测bank漂移两组皆减少，未解释E额外差距。旧不同编码器粗糙38/90对26/90未胜。R不把它们改为成功，也不以R替代D。

## 设计与有限预算（操作者决定，全部前瞻）
完整保留四种方法：A的clock_low501/E_low501、D的clock_anchor501/E_anchor501；原三个训练seed20261901/02/03，最新最佳编码器seed20261026、SHA8e6b15dcf409f51948f49c18fc04b686ba502e53586476db9c78d8c35a0fc346。不选中间checkpoint，不按成绩选训练seed或方法。四方法×flat及3个rough生成seed20264201/02/03×每组15world(3训练seed×5初态20264311..15)×steady/push两场景×20秒/50Hz=480回合，最多480000正式有效评测transitions，0新增PPO/Adam。每方法每rough生成seed对5个不同patch分别评估，15个rough patch共享于所有方法/训练seed/场景；flat另5初态。不将480回合说成480独立训练。

顺序使用Random(20264290)固定打散16个method×terrain组；单worker，fresh child逐组，最大15world，避免同时占用多份rough仿真。固定0.5m/s，yaw0、无侧向/角速度命令、无观测噪声、无随机推力。push在5/10秒给世界Y方向+.3/-.3m/s，与既有协议一致。只记录首次终止前，不拼接重置。故障保留原目录/源码和真实额外transitions，同模型/世界/预算恢复必须先预锁，不自动无限重试。

## 98维输入与地形审计
当前策略actor98维不含height_scan。以最新study.make_cfg(flat,phase)为基底，只把rough地形实体移入，不添加height_scan，不换rough策略、normalizer、reward或动作接口。四方法均使用相同native rough求解容量：ccd_iterations500、contact_sensor_maxmatch500、njmax1500、nconmax128；flat对照也用同容量，说明它是新cohort而非D回合重演。

rough只使用原生HfRandomUniformTerrainCfg的random_rough，noise_range(.02,.10)m、noise_step.02、horizontal_scale.1、vertical_scale.005，border_width.25；一行五列patch，每patch32×16m，generator curriculum=False，seed按上。这个noise_range是原生生成器高度值范围，不称±10cm或所有粗糙地形。保留生成高度场的完整数组/尺寸/静态geom位姿/索引与SHA，不只保存种子。32×16取代早期32×8以减少边界暴露，本轮各法完全一致，不与旧绝对值直接比较。

reset前用确定性事件将world index模5映射到row0、col0..4，先赋terrain.env_origins，再公开env.reset(seed=eval_seed,env_ids=[i])。同slot在各方法采样前核actor观测结构、公共配置规范化SHA、实际body_mass/body_ipos/geom_friction与hfield物理数据SHA；核物理起点position/quaternion/joint_pos/velocity和命令，phase值不同为唯一方法输入差异。训练seed只是不同策略，物理世界逐slot配对。

每帧读取Warp接触数组的nacon、worldid、geom pair、pos、dist、includemargin及efc_address。以原生14个foot collision geom与地形geom配对、efc_address至少一项>=0且dist<includemargin为有效足地接触；记录每world实际foot-hfield接触点、hfield索引、是否在指定patch距离边界>.25m，及其他足地接触数。只判指定patch；接触到邻patch/边界不能当指定rough暴露。脚部接触点确有heightfield，不用根部位置替代。每帧若有效足地接触为0，单独保留，不把腾空计作粗糙接触。另存足部site位置、root、原始contact，便于检查。

## 主终点、统计单位和比较family
每回合“持续行走完成”定义为：20秒内不跌倒、初始方向位移>=5m、body-forward速度>.1m/s占比>=.8、rough有效足地接触点中至少95%在指定patch内部。flat没有rough暴露条件；不满足暴露的rough回合计未完成而不删。报告所有边界离开、早期跌倒和零接触回合。该终点描述本次有界测试区持续行走，不能叫真实导航成功。

主要效果是rough完成率差与rough跌倒率差；先每训练seed内等权平均3terrain×5world×2scenario，再三个训练seed等权；同时拆steady/push与每terrain/seed全表。四预定对比：DE-DC、AE-AC、DE-AE、(DE-AE)-(DC-AC)，各两终点，共8项family统一Holm。报告seed差、均差、95%tCI和精确seed符号翻转双侧p；n=3最小双侧p=.25，因此R永远不是统计显著优势的独立确认，不因480回合扩大n。零方差非零差也不能算p=0。训练seed独立性限于这些匹配PPO运行，terrain和初态为交叉共享评测块；结果只条件于当前15rough patch，地图泛化仍需新地图。

MAE仍完整20秒、跌倒余段补零；位移、倾斜RMS、绝对航向、存活、root及真实foot rough暴露全报告，作为次要/代价指标，不能只选支持E的姿态数。push恢复用既有速度误差<.15m/s持续1秒、第一次截至第二次推力，早跌倒未暴露分开报告；失败不当恢复、未暴露不当稳健。阶段D速度失败结论保持。

## 前瞻裁决与停止
R探索候选必须DE同时相对DC及AC满足：完成率均差>=.10，跌倒率均差<=-.10；两项分别至少2/3训练seed朝有利方向；rough完整MAE不恶化>.02m/s、初始方向位移不减少>.5m，倾斜不增加>.02rad、航向不增加>.2rad。对比AC是保护，不通过削弱clock制造优势。flat代价全部报告，不把粗糙收益自动说成平地/全面改进。

若候选通过，才另行事前锁定新训练seed和新地形的独立确认，以及同E权重的状态反馈移除/匹配开环相位干预，才能判断“适应带来收益”。R只提供候选，不直接确认。若失败或CI不确定，如实报告当前固定策略无优势/证据不足，不加地图、改阈值、挑种子或继续搜beta追赢；审计是否确有环境/分布技术原因，新的粗糙等预算训练只能作为另行有限前瞻方案，不能假称本轮已完成原主张。若所有组几乎全过或全倒，该任务在本难度下缺乏区分度，标无结论，不根据结果调难度再冒称同一确认。

固定上限16组/480回合；一组出现完整性/配对/非有限值/接触容量溢出即停止并保留，先查故障。全部完成自动独立原始SHA/起点/物理/指标/暴露/统计复算。没有完整仿真状态时中断组重跑须分账，不伪称精确续接。仅实质结果/故障通知。

## 采样前工程门与独立阅读
不改A/B/C/D冻结源或site-packages，新增独立脚本、测试和结果目录。设计由builder-check独立阅读；实现测试包含实际contact配对/世界路由、heightfield物理hash、种子×地图矩阵、早终止补零、完成与候选规则、n3精确符号翻转p/统一Holm、防重启/缺失拒绝。CPU配置预检无需环境步进；两phase各1world×10帧的flat/rough真实smoke，总最多40个额外eval transitions，0PPO，在前瞻封存之后运行，故障原样分账。正式source manifest冻结脚本/测试/设计/计划/12个501 checkpoint与ONNX/训练manifest/编码器/环境与原生地形实现/依赖版本、父协议和分析SHA。smoke只核工程和配对，不用于挑方法/难度。

## Builder-check裁决（正式采样前）
- 2026-10-28只读独立builder审查已完成，未启动仿真。必须替换原randomize_terrain事件为确定性事件，且保留在reset_base前，不依赖reset外预赋值。
- 全回合有效足地接触数为零时，rough暴露率为null/不可判，完成为false。
- R候选只按既定点效应、至少2/3种子方向和代价门裁决；CI跨零照实报告不确定，不会单独否决探索候选，也不等于确认。独立优势确认仍需新训练seed和地图。
- “几乎全过或全倒”只作区分度描述，不覆盖固定候选门；不得以其临时改难度/预算。

## 机读方案字段与风险
- question/estimand：对限定15个未见rough patch，先在每个训练seed内等权平均地图、初态、场景，再求DE相对DC/AC的完成率和跌倒率差。
- falsifier：任一clock对照的完成率差<.10、跌倒率差>-.10、方向不足2/3或保护门失败，均不能形成探索候选。
- fair baseline：四方法使用相同生成地图、真实物理、初态、命令、推力和20秒预算；全部501 checkpoint保留，0调参/0新增训练。
- preregistration：LOCAL-ONLY；本地结果前封存，不冒充公开注册。
- budget：480回合最多480000有效eval，40额外smoke，0PPO/Adam，16组单worker固定顺序。
- risk：配对错误或容量饱和立即停止并保留技术失败；n3精度不足及flat训练到rough分布偏移限制claim，只探索，不承诺胜出。

对照公平 fair_baseline：同样rough求解器、98维传感输入、完整501模型、同map/物理/初态/命令/推力/有效评测预算，四方法均无新调参。

实现审查采样前补充：raw逐有效step保存原生terminated/timed_out，跌倒率从raw首终止复核，禁止拼接自动reset。rough烟雾测试各phase必须在10帧内至少记录1个指定hfield足部有效接触，按raw contact points/offsets独立重算；此为工程覆盖条件，不按行走分数筛选，不增加40步smoke预算。
