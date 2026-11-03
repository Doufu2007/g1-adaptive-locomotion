# Pre-registration: G1 phase encoder v2.1d context sweep

- Date: 2026-10-21
- Status: registered before either v2.1d training run
- Parent recipe: v2.1b; this is a context-length follow-up
- Evaluation contract: `g1-phase-gate-v4.2`; thresholds and calibration are frozen
- Data: `results/g1_phase_dataset_20261019/phase_dataset.npz`
- Dataset SHA-256: `e204c9dd0a48527ac4c82cfd0c539fba7d584f1faea6242ad0eba66970008b38`
- Test split: not loaded by this experiment

## Question

Does giving the v2.1b encoder enough temporal context to span a typical G1 gait
cycle improve cycle completion and point accuracy on the existing validation
subject? Compare exactly two context lengths: 50 and 60 frames.

## Rationale and verified data counts

The context identifiability sweep reports typical cycle lengths around 45–70
frames. The learned v2.1b model used 30 frames. This experiment changes only
the context length so each input can cover approximately one gait cycle.

The often-cited estimate of about 13,555 windows does not describe this
training recipe. Before registration, the frozen pair builder was run on the
train split: it yields 73,226 rows and 301,551 anchor-target pairs, split by
pair kind as 70,708 / 229,895 / 948. Validation has 45,236 rows; the unchanged
window-history rule marks 45,076 rows valid at context 50 and 45,036 at context
60. Pair construction does not filter on context history, matching v2.1b.

## Fixed procedure

Run `scripts/train_g1_phase_encoder_v2_1d.py` once with `--context-frames 50`
and once with `--context-frames 60`. Preserve the v2.1b settings:

- seed 20261020; 30 complete epochs; batch size 512; hidden size 64
- AdamW, learning rate 3e-4, weight decay 0.1; dropout 0.5; gradient clip 1.0
- the same v2.1b phase model, pair builder, lags, loss terms, weights, and
  validation scorer
- fit only on train; score and select only on validation
- no early stopping; complete all 30 epochs for both contexts

The only training-input change is `context_frames`. The v2.1b source files and
their existing artifacts remain unchanged. Each context writes to separate
v2.1d report, prediction, and candidate-checkpoint paths.

## Outcomes and decision rules

The primary metric is validation macro cycle completion under v4.2. The
historical v2.1b comparison is epoch 1: macro completion `0.6962456093142473`,
condition-macro point error `0.7716245022746507 rad`.

- An improvement signal is macro completion **greater than 0.6962456093**.
  This alone does not qualify an encoder for use.
- A v2.1d candidate qualifies only if the full v4.2 gate passes: macro
  completion at least 0.90, every condition at least 0.80, every period-length
  stratum at least 0.80, reference-clock and coverage checks pass, and the
  condition-macro point error is at most `π/6` (`0.5235987756 rad`).
- A best macro result at or below 0.65 is recorded as the pre-registered stop
  bar. It does not interrupt training; both runs still finish all 30 epochs.
- A result in `(0.65, 0.6962456093]` is unresolved against the v2.1b baseline.
- If both candidates qualify, rank by macro completion descending, point error
  ascending, then context length ascending. Any selected checkpoint remains a
  validation-selected research candidate, not a frozen deployment encoder.

No PPO run or deployment decision is part of this experiment. A later behavior
study requires its own registration. The existing test split has previously
been opened, so it cannot serve as a blind confirmation set; a separate
independent holdout is required for confirmatory claims.

## Provenance and reporting

The training script records source hashes, dataset hash, pair counts, history
coverage, all epoch predictions, and the complete v4.2 report for each epoch.
Report all 30 epochs for each context, including failed gate components and
point error. Do not describe a validation result as deployment performance or
as evidence of policy improvement.
