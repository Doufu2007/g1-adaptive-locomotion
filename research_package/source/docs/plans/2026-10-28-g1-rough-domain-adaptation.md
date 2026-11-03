# G1 Rough Domain Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本聊天已授权自动持续有限验证，操作者原地实施；仅技能明确要求的独立设计/代码审查派fresh reviewer，不开启额外实现代理。不提交。

**Goal:** 执行一次训练域×phase的2×2等预算适配，检验rough经验和学习相位完整管线的有限收益，保留全部负结果。

**Architecture:** 新增S contract负责矩阵、配置和裁决，experiment负责完整D501状态warm start及原生PPO单次连续learn、封存与串行队列；evaluation adapter在fresh child内显式复用冻结R的真实接触/评测，不写R。独立audit只读原始数组和模型/预算绑定，单独实现评分与统计。

**Tech Stack:** Windows Python3.13、现有torch/CUDA、mjlab/MuJoCo-Warp、native rsl_rl、ONNX CPU路由、numpy/scipy；不安装或修改依赖。

**Spec:** `docs/superpowers/specs/2026-10-28-g1-rough-domain-adaptation-design.md`

## Global Constraints

- 不改A/B/C/D/R冻结来源及site-packages，不提交/清理，不连旧服务器或关闭本机。
- 四方法FC/FE/RC/RE，来源seed20261901/02/03，各251×64×24；正式train4626432、3012PPO、60240Adam。
- 固定checkpoint0/1/25/100/251；主评测只最终251；actor98维，编码器冻结，原source9900 teacher/beta=.1不变。
- 完整加载D501 actor/critic/Adam/normalizer/teacher，恢复其实际LR和RNG；新仿真不称精确续接。clock seed02的源LR为2.25e-5。
- 训练地图20264601/02/03，评测rough20264701/02/03及flat0；禁止R20264201..03回流训练。eval初态20264811..15，construct seed20264200。
- 480正式eval、最大480000有效transitions；单worker fresh children。额外smoke12288train/8PPO/160Adam、最多80eval分账。
- 采样前设计/计划包、CPU配置与来源状态预检和独立实现review通过，完整协议封存；真实smoke通过才启动正式队列一次。
- 主要8项family/所有保护/停止按spec，不调阈值、beta、难度、预算追赢；三来源seed只探索。

## Review Focus

1. 原生load不恢复算法LR属性：必须显式恢复并比对Adam param groups，不能默认1e-5。
2. optimizer/critic/teacher不完整载入会换训练过程：source state和相对checkpoint0逐tensor精确核验，不能只比actor。
3. rough场景多geom影响索引和contact容量：按robot名称规范化物理，实际hfield映射与接触且容量严格未满。
4. S adapter误写R或路由错seed：只在fresh child绑S目录/协议/slot/政策，gold路由及拒绝旧seed/缺失模型。
5. 中断后误跳过、拼接或扩大预算：只接受源/完整预算/trace/audit均匹配的完成项，保留所有失败和实际损耗，无自动重试。

---

### Task 1: 配置、矩阵与裁决契约

**Files:** Create `scripts/g1_rough_domain_contract_20261028.py`; Test `tests/test_g1_rough_domain_20261028.py`.

**Interfaces:** `training_jobs()->list[dict]`、`evaluation_jobs()->list[dict]`、`slots()->list[dict]`、`source_phase(method)->str`、`training_map(seed)->int`；`make_training_cfg(study,method,seed)`；`paired_analysis(rows)->dict`。复用R纯CPUcontact/episode定义，S自己的矩阵/统计不改旧module globals。

- [ ] RED：`test_complete_fixed_matrix`断言12个唯一train cell、每251updates/385536transitions、16eval组×30回合；train/eval/R地形seed无交；固定order重复调用相同。
- [ ] RED：`test_missing_duplicate_seed_rejected`针对480键矩阵缺一行/重一行拒绝；`test_holm_eight_and_exact_seed_p`断言n3最小p=.25、统一8项校正，零方差非零差不能p0。
- [ ] RED：`test_candidate_requires_strong_clock_and_flat_guards`构造只有RE−FC过但RE−RC失败、只有fall过、flat MAE+.0201等，全部false；全门边界恰好过才true。
- [ ] 实现矩阵与统计，沿用R≥3seed精确符号翻转/tCI数值函数，只绑定S contrasts和候选；新的flat保护与rough域效应判决独立输出。
- [ ] CPU配置gold：四方法play=False/64world；98actor结构、对应Dcritic输入结构、same-domain公共配置摘要；仅rough terrain移入，solver500/500/1500/128、reset事件顺序固定；报告配置SHA，不构造或步进环境。
- [ ] Run `python -m unittest discover -s tests -p test_g1_rough_domain_20261028.py -v`（使用现有用户site依赖bootstrap），保存RED/GREEN日志；无提交。

### Task 2: 完整来源恢复与连续原生训练

**Files:** Create `scripts/g1_rough_domain_experiment_20261028.py`; extend同一CPU测试。

**Interfaces:** `source_entry(method,seed)->dict`；`restore_source(runner,source_checkpoint,torch)->dict`；`train(method,seed,smoke=False)->None`；`verify_protocol()->dict`。OUT固定`results/g1_rough_domain_adaptation_20261028`。

- [ ] RED：`test_full_native_source_restore`使用已审计六source state和最小runner/alg接口测试，拒绝missing critic/optimizer/teacher、beta变化、RNG缺失；检查native load_cfg=None、strict=True。
- [ ] RED：`test_restore_lr_from_source_not_constructor`给构造器1e-5、source2.25e-5，须算法及param groups均2.25e-5；不一致source拒绝；`test_restore_rng_after_setup`断言restore发生在env/runner构造之后、第一次动作采样之前。
- [ ] 实现source strict hashes与来源状态核验、native完整load、显式LR恢复、递归张量状态digest和CPU/CUDA RNG恢复；相对iteration0但source原生iter500/完成501均登记。
- [ ] 训练按A已验证模式保存native checkpoint和ONNX，但本地实现自己的S文件/状态，不改A globals或调用A.train；配置class使用冻结SourceAnchoredPPO，原生一次learn(251或2)。
- [ ] 每step记录精确64transitions并检查finite和容量；每update记录a.record_update的20旧KL和20sourceKL、LR、native20 minibatch，并保存进度；每rollout边界真实指定patchcontact采样。
- [ ] checkpoint0核actor/critic/Adam/teacher/normalizer等与source完全相同；同phase跨域RNG/model初始化一致；same-domain真实机器人动力学及terrain配对。允许的地形差异显式规范化，真正机器人参数不匹配立即0step失败。
- [ ] 所有checkpoint和结束核teacher/encoder冻结、native累计Adam step=10020+20×相对update。归一化按原生更新，不把动态actor normalizer冻结；teacher normalizer保持eval。

### Task 3: S评测adapter与原始数据

**Files:** Create `scripts/g1_rough_domain_evaluation_20261028.py`; extend CPU测试。

**Interfaces:** `policy_entry(method,source_seed,smoke)->dict`、`evaluate(method,terrain_seed,smoke=False)`、`analyze()->dict`；真实rollout复用R实现，fresh child内只绑定新S OUT/lock/policy_entry/slot contract及预定矩阵。

- [ ] RED：`test_fresh_adapter_has_no_R_writes`验证OUT和pairing均S目录，原R OUT/协议文件不改；拒绝未声明method/seed/map，拒绝未完成251source或错误ONNX SHA。
- [ ] RED：`test_all_policy_routes_gold`用不同常数策略替身覆盖4方法×3来源seed×5slot，逐world动作精确对应，不按slot错误共用seed01。
- [ ] 实现adapter，policy方法名尾E保持learned输入，其余clock；eval construct seed20264200、3新地图/5新初态；R的physics/contact/terminal/zero-tail/完成逻辑不改。
- [ ] CPU ONNX真实shape[1,98]和动作29验证；smoke/final政策路由分开。最终raw全部源SHA、全物理hfield、起点/command前配对、真实接触与原生终止保留。
- [ ] `analyze()`仅完整480raw时运行；结果包括所有4方法×map×scenario×seed、主要family、所有rough与flat保护、训练域门及不可确认解释；缺失/重复/非有限拒绝。

### Task 4: 来源封存、独立audit与计划机检

**Files:** S结果目录新增`PROJECT_PLAN.md`、`experiment_matrix.md`、`preregistration.md`、`reproducibility-checklist.md`、`prepare_plan_package.py`、`target-chain.json`、`failure-tree.json`、`plan_spec.json`、`plan_package.json`；新增`independent_audit.py`与`independent_audit_plan.json`。

- [ ] 机读target chain明确S1/S2、3来源seed、8family、反证条件与不扩预算；失败树分phase失败/域有利但phase失败/候选/技术无效，不以欠功效证明无差异。
- [ ] 跑研究方案gate与完整package；WARN仅可为未隔离feedback和n3探索，明确降claim，任何fairness/falsifier critical先修。
- [ ] independent audit在任何正式S采样前封存代码SHA；独立CPU从raw重算native终止、完整20s MAE、初始方向位移/tilt/heading/moving、指定hfield暴露、seed contrasts/8Holm/全部候选门，并核来源checkpoint/RNG/原生连续更新/预算/teacher/encoder。
- [ ] RED：`test_protocol_refuses_changed_source_and_duplicate_launch`任何冻结source hash变化、已有incomplete目录或active启动标记拒绝；没有完整数据不得返回analysis或audit PASS。
- [ ] 实现`lock()`，包含新S全部代码/测试/设计/计划/机检报告/审核/零采样preflight、六D501完整原生状态及manifest/env/agent、冻结D/R transitive source、独立audit源码/计划、包版本；源快照只写S。
- [ ] fresh独立实现review依requesting-code-review技能执行；回复需明确CPU/静态范围，不冒称GPU通过。按修复后新源码重新跑preflight，再封存最终协议，不带未解决Critical/Important采样。

### Task 5: 真实smoke及唯一正式队列

**Interfaces:** CLI `--stage preflight|lock|smoke|train|evaluate|queue|analyze`；train需method/source-seed，eval需method/terrain-seed；所有environment采样入口先verify_protocol。

- [ ] 锁前确认无S训练或采样进程；隐藏fresh child四训练smoke，每方法首个source seed2更新/64world，共12288train/8PPO160Adam；全部8个flat/rough评测smoke各10帧，总80eval。固定协议不按smoke成绩改预算或方法。
- [ ] 独立核checkpoint0完整恢复、native预算/20KL/sourceKL、teacher/encoder、ONNX真输出、机器人动力学和真实指定hfield接触；结果目录和失败实际成本完整保留。
- [ ] 全smoke工程PASS后唯一隐藏`queue`，12训练cell逐fresh child、后16eval组逐fresh child，单worker；首次实际更新progress确认，不能只派发就写成功。
- [ ] 任何非零child退出立即队列FAILED，不自动重试；已有完成项只在完整校验后跳过，incomplete保留需预锁恢复方案；不从中间checkpoint冒称完整仿真续接。
- [ ] 每15分钟健康无变化安静；实质结果/故障才通知。写launch/runtime_confirmation/queue_status/cost ledger，smoke/有效正式/故障分账。

### Task 6: 结果审计、有限分支结论与交接

- [ ] 全12×251/480raw完成后自动主analysis；再运行预锁独立audit，并保留source/完整预算和全部fault成本。
- [ ] 所有8family、点效应/CI/精确p/统一Holm、两个clock候选门和flat保护全报告；n3探索无确认性措辞。
- [ ] 域改善但phase失败时只支持有限域适配描述，不改写成相位机制成功。失败结束此分支，不自动延长251、换地形/seed/checkpoint/β。
- [ ] 若形成候选，另行锁新seed/map确认和同权重feedback干预；在该独立验证结论形成前不标整体原主张证明完成。
- [ ] 更新S handoff和heartbeat事实源；旧R/D的负结果、协议和未知损耗均保留。

## Self-review

完整2×2控制额外优化而非只给rough组加训练；恢复actor/critic/Adam/teacher/LR/RNG有Task2，地图隔离和实际暴露有Task1/3，记录与校验有Task4/6，有限smoke和单次队列有Task5。当前n3/反馈未隔离限制为明确探索边界，不能用机检PASS或更多回合替代新seed/因果验证。上述预算/seed/接口均为操作者预先决定；未执行的步骤保持未勾选。
