# G1 接触事件相位研究实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将接触事件定义的 G1 步态相位研究落成可核查的数据溯源、无泄漏窗口、冻结编码器、运行时接线、五组预注册配置和 fail-closed 评测门。

**Architecture:** `phasor/data.py` 负责来源清单、完整周期标签、序列级 split、窗口和 train-only 标准化；`phasor/encoder.py` 提供时间池化与保留时间位置的两种 1D-CNN；`train_encoder.py` 只消费已核实清单并按 validation 选择架构。运行时 `PhaseHistoryState` 管理逐环境历史和同一步 latent 缓存；G1 配置暴露五组实验条件；离线评测报告圆周误差、接触事件相位和控制门。

**Tech Stack:** Python 3.10+, NumPy, PyTorch, unittest/pytest, mjlab configuration.

**Spec:** `docs/superpowers/specs/2026-10-19-g1-phase-contact-study.md`

## Global Constraints

- 输入固定为 `10 × 29`，输出顺序固定为 `(sin(phi), cos(phi))`。
- 只用完整左脚接触周期；右脚相位从数据计算并报告，不预设 `pi`。
- 先按受试者或完整序列 split，再生成重叠窗口；标准化只 fit train。
- `dance1_subject2.csv` 是 dance，不能作为步行训练数据。
- 接触标签只用于离线训练/评测，不能进入运行时编码器、actor 或部署。
- 测试集不参与架构、阈值或调参；没有相位门就不启动正式 PPO。
- 不编造数据来源、许可、标签质量或控制增益。

## Review Focus

- 未核实的动作文件是否会绕过数据门；测试 `dance1_subject2` 必须失败。
- 左脚周期标签是否跨站立/过渡或跨序列；测试完整周期和边界。
- 右脚相位是否被硬编码为 `pi`；测试非半周期的触地事件。
- `(sin, cos)` 与 `atan2(z[0], z[1])` 是否全链路一致；测试四个已知向量。
- reset、缓存和 critic 时钟是否跨环境串线；测试每环境 reset、同一步重复调用和五组配置。

### Task 1: 数据来源清单与标签管线

**Files:**
- Create: `docs/data/g1_phase_data_provenance.json`
- Create: `src/tasks/velocity/phasor/data.py`
- Create: `tests/test_phase_data.py`
- Modify: `scripts/phasor_contact_phase_experiment.py` to consume the shared label/split functions.

**Interfaces:**
- `validate_provenance(manifest) -> None` raises on unresolved licence, non-walk action, missing G1 map or missing contact verification.
- `left_cycle_phase(contacts, timestamps) -> PhaseLabels` computes labels only between complete left touchdowns and returns right-touchdown phase observations.
- `split_sequences(sequence_ids, fractions, seed) -> dict[str, list[str]]` assigns whole sequences before windows.
- `make_windows(joint_pos, labels, window=10) -> WindowBatch` labels the last frame and rejects boundary crossing.
- `fit_normalizer(train_frames) -> Normalizer`; validation/test only call `transform`.

- [ ] Write failing tests for provenance blocking, non-half-period right contacts, split-before-window, and train-only normalization.
- [ ] Run the focused tests and observe expected failures.
- [ ] Implement the manifest, label builder, sequence split, window builder and normalizer.
- [ ] Run focused tests and the existing phasor tests.
- [ ] Commit Task 1.

### Task 2: Encoder architectures and supervised training

**Files:**
- Modify: `src/tasks/velocity/phasor/encoder.py`
- Modify: `src/tasks/velocity/phasor/train_encoder.py`
- Create: `tests/test_phase_encoder.py`

**Interfaces:**
- `PhaseEncoder(..., preserve_time=False)` keeps current pooled architecture when false and uses time-position-preserving convolution when true.
- `circular_phase_loss(pred, target) -> Tensor` computes `mean(1 - pred·target)`.
- `normalize_phase(raw, eps=1e-8) -> Tensor` returns finite unit vectors.
- `train_and_select(...)` fits both architectures on train, selects only by validation, and returns a frozen checkpoint plus selection record.

- [ ] Write failing tests for output order, finite normalization, loss, and validation-only selection.
- [ ] Run tests to observe failures.
- [ ] Implement both heads and train-only selection; remove reconstruction/smoothing from the phase objective.
- [ ] Run focused and existing tests.
- [ ] Commit Task 2.

### Task 3: Runtime history, angle order and reward alignment

**Files:**
- Modify: `src/tasks/velocity/phasor/runtime.py`
- Modify: `src/tasks/velocity/mdp/observations.py`
- Modify: `src/tasks/velocity/mdp/rewards.py`
- Create: `tests/test_phase_runtime_contract.py`

**Interfaces:**
- `PhaseHistoryState.update` fills a reset environment's full history with the current normalized pose, updates each environment once per control step, and caches one `(sin, cos)` latent.
- `phase_to_angle(z) -> Tensor` uses `atan2(z[...,0], z[...,1])`.
- `feet_gait_learned` uses the cached latent and current contact state without contact input to the encoder.

- [ ] Write failing tests for reset fill, per-environment update, same-step reuse, and known angle vectors.
- [ ] Run tests to observe failures.
- [ ] Implement the runtime and reward fixes.
- [ ] Run focused tests and all phasor tests.
- [ ] Commit Task 3.

### Task 4: Five pre-registered G1 configurations

**Files:**
- Modify: `src/tasks/velocity/config/g1/env_cfgs.py`
- Modify: `src/tasks/velocity/velocity_env_cfg.py`
- Create: `tests/test_phase_conditions.py`
- Modify: `paper/PHASOR-experiment-protocol-2026-10-15.md` with the new five-group table and status.

**Interfaces:**
- Add factories `unitree_g1_flat_fixed_clock_env_cfg`, `unitree_g1_flat_actor_ablation_env_cfg`, `unitree_g1_flat_reward_ablation_env_cfg`, `unitree_g1_flat_joint_phase_env_cfg`, `unitree_g1_flat_random_vector_env_cfg`.
- All five keep critic fixed-clock phase; random vector is episode-fixed and unit norm.

- [ ] Write failing tests for group factor matrix, fixed critic, random episode vector, and dimension equality.
- [ ] Run tests to observe failures.
- [ ] Implement the factories and registry wiring.
- [ ] Run focused and full phasor tests.
- [ ] Commit Task 4.

### Task 5: Phase/rollout gates and control analysis contract

**Files:**
- Create: `scripts/evaluate_phase_study.py`
- Create: `tests/test_phase_evaluation.py`
- Create: `docs/PHASE_CONTROL_PREREGISTRATION.md`

**Interfaces:**
- `evaluate_phase_predictions(...)` reports circular error, within-cycle progression, left-touchdown error, empirical right-touchdown phase distribution, and fixed-clock comparison.
- `evaluate_rollout_gate(...)` fails when held-out motion passes but on-policy phase does not.
- `control_summary(...)` aggregates by training seed, computes XY speed error and guardrails, and never relabels training termination as test fall probability.

- [ ] Write failing tests for metric definitions, non-independent episode aggregation, and fail-closed rollout gate.
- [ ] Run tests to observe failures.
- [ ] Implement evaluation helpers and preregistration document.
- [ ] Run focused, full unit tests, and static syntax checks.
- [ ] Commit Task 5.

## Execution order

Run Tasks 1–5 in order. Because the verified source is dance and no local walk/contact snapshot is currently present, the implementation must finish with a documented `DATA_NOT_READY` status and must not fabricate encoder, rollout or PPO results.
