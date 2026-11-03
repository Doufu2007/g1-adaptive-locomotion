# 源策略KL单因素修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. 用户已授权持续执行；本计划按当前工作目录原地追加新文件，不提交/不清理、不改A/B/C封存源。

**Goal:** 在原生PPO中仅增加固定源策略KL约束，公平检验clock/E是否避免累计能力退化。
**Architecture:** 新PPO子类保留native feedforward单GPU更新和随机序列；新实验入口复用A train/evaluate并在初始save前核A配对。另行封存来源、短测试和两worker正式队列，全部完成自动分析。
**Tech Stack:** Python3.13、torch2.6+cu124、rsl_rl原生PPO、mjlab、ONNX。
**Spec:** docs/superpowers/specs/2026-10-27-g1-source-anchor-design.md

## Global Constraints
- beta=.1，KL(source||current)动作维求和/样本均值，教师同输入且冻结，不隐藏clock。
- 6×501×64×24=4617216正式训练transitions，270新增评测，固定checkpoint0/1/25/100/251/501；smoke2×2更新=6144+20评测另记。
- 初始1e-5且原adaptive；A配对5字段首次采样前PASS，算法类差异需规范化成native后hash。
- A/B/C/site-packages不改、不提交清理、不连接旧服务器/不关闭本机。
- 4项seed级family统一Holm；候选必须同时过anchored clock和原clock_low保护门。

## Review Focus
- teacher复制时分布含图，不允许保存非leaf图而deepcopy失败；先从无缓存copy，载入后保持源。
- anchor不能使20次原KL仪表多计，不额外采样随机action。
- beta0须actor/critic/Adam/RNG对原生精确等价，多minibatch捕获细微漂移。
- runner构造修改cfg的pre/post摘要差异，pre-save比较缓存pre-env/pre-construction摘要。
- 初态/物理/source链、无效采样、重复/缺失回合在分析时拒绝而非悄悄平均。

### Task1: 原生等价PPO加源约束
**Files:** scripts/g1_source_anchor_ppo_20261027.py；tests/test_g1_source_anchor_20261027.py。
**Interfaces:** SourceAnchoredPPO继承PPO，anchor_coefficient=.1；source_kl(observations,current_params)返回每样本KL；save/load保留source teacher状态。
- [ ] 先写测试：不存在类应RED；beta0在同合成TensorDict storage、同seed多minibatch，对native的actor/critic/Adam/RNG全精确一致；初始sourceKL0、手工偏移与torch Normal KL解析一致且只有current梯度；教师不采样不改RNG；保存恢复teacher不是新actor；20次KL仪表兼容。
- [ ] 验证RED后实现最小非循环无扩展原生update+一项betaKL；不支持RND/symmetry/multigpu时明确拒绝。
- [ ] GREEN并记录，native源码SHA未变。

### Task2: 封存、采样前配对与评测
**Files:** scripts/g1_source_anchor_experiment_20261027.py；同tests追加。
**Interfaces:** lock/train/evaluate/cell/verify_smoke/queue/analyze；只在独立子进程内把a.OUT/a.lock、registry.load_rl_cfg和runner初始save接线。
- [ ] 先测规范化common_agent摘要不受新classname影响，phase/seed/budget或其他配置改变会拒绝；固定6任务/worker分片和四contrast缺失重复拒绝；候选必须同时过两clock对照。
- [ ] 实现lock先验证C原链和A原controls并绑定policy/checkpoint；缓存make_cfg前digest，source0 save首次采样前核五字段，teacher frozenSHA和beta入manifest。
- [ ] 复用a.train/a.evaluate固定budget/checkpoints；新sourceKL日志入losses，保持a.record_update20次。2phase有效smoke在source封存后各2更新3072+10帧评测；每项pre_sampling_pairing PASS。
- [ ] 完整相关unit/语法、独立代码review后smoke PASS才能正式queue。

### Task3: 正式执行与完整分析
- [ ] 2worker固定seed分片与seed内随机phase顺序；6jobs全完成后analyze。失败保留原目录/源码/额外实际成本，恢复必须先锁修订，不能按成绩重来。
- [ ] 对270新+270A回合rawSHA/指标重算，所有point同cohort初始化和物理配对；6×501预算和教师源SHA不变。
- [ ] 四seed级contrastCI/效应量/Holm、过程sourceKL、全部场景保护指标与双clock候选门。只形成候选才预锁独立确认；失败如实报告本修正有限结果，不无限追赢。

Self-review: Task1生产子类供Task2 cfg.class_name引用，save教师元数据由Task2 manifest核；Task2输出完整manifest/raw/updates由Task3重评分；命名/接口一致。用户已批准持续执行分支，无新权限问题。
