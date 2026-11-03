# G1 Phase Gate v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable v4 phase-cycle gate, train and select encoder v2 using train/validation only, and permit test, confirmation rollout, ready lock, then PPO only after each prior gate passes.

**Architecture:** Add a standalone v4 gate module and scripts so v3 remains unchanged. Add an encoder v2 module whose shared feature transform and forward path are used by training and rollout evaluation; persist each candidate and raw validation evidence in a new v4 results directory.

**Tech Stack:** Python 3.13, NumPy, PyTorch 2.6/CUDA when available, unittest/pytest, JSON artifacts.

**Spec:** `docs/superpowers/specs/2026-10-20-g1-phase-gate-v4.md`

**Execution status:** Tasks 1–2 completed. Task 3 completed through validation-only selection and failed closed because none of 60 encoder-v2 candidates met the frozen gate; Tasks 4–5 are blocked by that result. After selection failed, a diagnostic command opened test arrays for descriptive statistics; this protocol deviation is recorded in `results/g1_phase_gate_v4/protocol_deviation_test_split_diagnostic_v1.md`. No test performance claim, new rollout, ready lock, or PPO run was made.

## Global Constraints

- Keep all v3 files and results unchanged.
- `net_tolerance_rad = pi/6` and `max_abs_step_rad = pi/2` are frozen before training.
- Training may read only train arrays; validation may select candidates; test and rollout may not affect training, candidate selection, or thresholds.
- Report every cycle by condition and period-length stratum; no post-hoc exclusions.
- The v4 macro validation completion floor is `0.90`; each condition and length stratum floor is `0.80`.
- No ready lock or PPO run exists unless frozen validation, one-time test, and independent confirmation all pass.

## Review Focus

- Reset, sequence, missing-frame, and condition boundaries must never be bridged by a cycle or a long-context window.
- Cycles with insufficient history must be counted as uncovered and must not silently disappear from coverage.
- The jump test must apply the strict per-period maximum absolute increment `< pi/2`, regardless of period length.
- Reference-clock failure or encoder validation completion below `0.90` must prevent freezing.
- Candidate selection must use cycle completion before point error, and the trainer must not read test keys.

### Task 1: v4 cycle scorer and calibration contract

**Files:**
- Create: `src/tasks/velocity/phasor/phase_gate_v4.py`
- Create: `tests/test_g1_phase_gate_v4.py`
- Create: `results/g1_phase_gate_v4/gate_contract_v4.json`

**Interfaces:**
- `score_phase_cycles_v4(...) -> dict` returns per-cycle records, coverage counts, completion rates, condition strata, period-length strata, point error, and boundary failures.
- `calibrate_phase_gate_v4(reference_phase, encoder_phase, sequence_ids, frame_ids, touchdown, ...) -> dict` freezes validation-label period tertiles and computes reference/encoder rates using identical scoring.
- `select_v4_candidate(reports) -> str` orders by macro cycle completion, condition-macro point error, then shorter context.
- `encoder_freeze_eligible(report, contract) -> bool` rejects completion below `0.90` and any failed coverage/reference/condition/length requirement.

- [x] Write tests for exact `pi/6` net tolerance and strict `< pi/2` jump boundary.
- [x] Write tests for reset/gap/condition boundaries and insufficient-history coverage accounting.
- [x] Write tests that calibration reports reference and encoder rates and rejects encoder completion below `0.90`.
- [x] Write tests that every cycle remains represented in condition and length strata and completion-first selection wins over point error.
- [x] Run the focused tests and observe the expected missing-module failures.
- [x] Implement the scorer and contract; run focused tests again.

### Task 2: frozen calibration and v3 baseline audit

**Files:**
- Create: `scripts/calibrate_g1_phase_gate_v4.py`
- Create: `scripts/evaluate_g1_phase_gate_v4.py`
- Create: `results/g1_phase_gate_v4/calibration_v4.json`
- Create: `results/g1_phase_gate_v4/current_encoder_v1_v4_audit.json`

**Interfaces:**
- Calibration reads validation labels and the existing v1 validation predictions only; it freezes `pi/6`, `pi/2`, the period tertiles, and all condition/coverage floors before v2 training.
- Baseline audit evaluates existing v1 validation and already-collected confirmation predictions without modifying thresholds or choosing v2 hyperparameters.

- [x] Calibrate thresholds using reference labels and record source hashes before training.
- [x] Re-score the existing v1 encoder on validation and the historical independent rollout with the v4 scorer.
- [x] Confirm the v1 validation encoder is marked `freeze_eligible: false` below `0.90`; retain the expected roughly `0.43` macro completion as an explicit failure.

### Task 3: encoder v2 objective and train/validation selection

**Files:**
- Create: `src/tasks/velocity/phasor/encoder_v2.py`
- Create: `scripts/train_g1_phase_encoder_v2.py`
- Create: `tests/test_g1_phase_encoder_v2.py`
- Create: `results/g1_phase_gate_v4/candidate_validation_v2.json`

**Interfaces:**
- `build_phase_v2_features(pose_history) -> Tensor` appends per-joint frame differences and scalar speed to normalized pose history.
- `phase_v2_loss(pred_now, pred_next, target_now, target_next, ...) -> dict[str, Tensor]` returns circular, rate, reverse-progress, jump, and total losses.
- `train_encoder_v2(dataset_path, contract, output_dir) -> dict` loads train arrays and validation arrays only, trains the predeclared 10/30-frame candidates, evaluates both with v4, and promotes no checkpoint unless every validation requirement passes.

- [x] Test shared feature shapes and motion-speed values.
- [x] Test that the rate loss is lower for a correctly predicted positive label increment than for a stalled prediction.
- [x] Test that reverse and over-limit jumps increase their named penalties.
- [x] Test candidate selection prioritizes v4 completion and that the trainer accesses only train/validation arrays.
- [x] Run focused tests and observe expected failures, then implement the v2 model/loss/trainer.
- [x] Train with the frozen seed and hyperparameters; write candidate reports and complete raw validation predictions.
- [x] Apply the freeze gate: no candidate passed macro completion, condition, and period-stratum requirements, so no checkpoint was frozen or exported.

### Task 4: frozen test and independent confirmation

**Files:**
- Create: `scripts/evaluate_g1_phase_gate_v4.py` test/rollout modes
- Create: `scripts/collect_g1_phase_confirmation_v4.py`
- Create: `results/g1_phase_gate_v4/frozen_candidate_v2.json`
- Create: `results/g1_phase_gate_v4/test_report_v4.json`
- Create: `results/g1_phase_gate_v4/independent_confirmation/`

- [ ] If and only if Task 3 passes, freeze checkpoint and calibration hashes before reading test arrays.
- [ ] Evaluate test exactly once with the frozen v4 scorer; do not update checkpoint, thresholds, or selection.
- [ ] Collect exactly one new confirmation rollout using the pre-registered policy seed and new environment seed `20261052`; use the same v2 feature and inference path as offline evaluation.
- [ ] Report episode boundaries, terminal contacts, history coverage, all conditions, all period strata, and failures.

### Task 5: ready lock and PPO handoff

**Files:**
- Create: `scripts/lock_g1_phase_gate_v4.py`
- Create: `results/g1_phase_gate_v4/ready.lock.json` only after all prior gates pass

- [ ] Test that the lock refuses missing, changed, or failing artifacts.
- [ ] Create and verify the ready lock only after frozen test and independent rollout pass.
- [ ] Start PPO only if the lock verifies; otherwise leave PPO blocked with the specific failed gate and evidence.

## Execution order

Complete Tasks 1–5 in order. Any failed gate terminates only its dependent downstream steps; preserve the failed report and do not open test/rollout/PPO for a candidate that did not pass validation.
