# Pre-registration: G1 phase encoder v2.1e — cycle-integral weight sweep

- Date: 2026-10-21
- Status: registered before any v2.1e training run
- Parent recipe: v2.1b; this is a loss-weighting follow-up, not a context or capacity change
- Evaluation contract: `g1-phase-gate-v4.2`; thresholds and calibration are frozen
- Data: `results/g1_phase_dataset_20261019/phase_dataset.npz`
- Dataset SHA-256: `e204c9dd0a48527ac4c82cfd0c539fba7d584f1faea6242ad0eba66970008b38`
- Test split: not loaded by this experiment

## Question

Gate v4.2 credits a cycle only when the unwrapped predicted phase advances one
turn inside it. The training objective contains exactly one row per complete
cycle that supervises that quantity. Does raising that row's weight — and
nothing else — move the gate?

This is the last untested model-side lever whose failure mode is named by the
existing diagnostics rather than inferred from a gate miss.

## Verified rationale (measured before registration)

Every number below was read from disk and reproduced, not assumed.

**What the gate grades.** On the best v2.1b epoch (epoch 1, macro 0.696246),
`net_progression_diagnosis.json` reports 603 evaluable cycles: 69.3% already
land within `pi/6` of exactly one turn, split into 113 cycles that advance too
little and 72 that advance too much. Median net error `-0.132` rad, q25
`-0.402`, q75 `+0.152` — the interquartile range already sits inside the band.
The sd is 2.80 rad, driven by a tail that includes 45 whole-turn slips.
`corr_period_net_error` is 0.640, and the per-stratum means run `-0.56` (short),
`-0.34` (medium), `+1.37` (long): the model under-advances on short cycles and
over-advances on long ones. The failure is in the cycle integral.

**What the objective supervises.** `build_multiscale_pairs` emits 301,551
anchor-target pairs for the train split: 70,708 at lag 1, 229,895 at fixed lags
4/8/16/32, and 948 cycle rows. Measured weight shares of the increment term are
0.3958 / 0.5972 / **0.0070**. The median training cycle is 59 frames and the
largest fixed lag is 32, so for a median cycle no fixed-lag row spans a cycle
boundary on both sides; the single cycle row per cycle is the only full-cycle
supervision, and it receives 0.70% of the weight.

**What is already excluded.** Four independent lines say the ceiling is not
context, capacity, epoch count, or subject shift:

| lever | evidence | result |
|---|---|---|
| context length | k-NN, no fitting, contexts 30/60/90 | 0.7996 / 0.8657 / 0.9496 rad — worse with more context |
| context length | CNN, contexts 30/50/60 | 0.7716 / 0.7447 / 0.7832 rad — flat |
| capacity and span | dilated model, 151 frames, 101,250 params | 0.714 rad, pre-registered verdict `INCONCLUSIVE_DO_NOT_DECIDE` |
| epoch count | v2.1b and v2.1d curves | peak at epoch 1, monotone decline thereafter |
| subject shift | `cross_subject_coherence.json` | cross-subject neighbour spread 0.2171 rad vs within-subject 0.2171 rad — no shift |

A within-subject re-split is therefore not informative and is **not** part of
this experiment.

**The prior is not favourable.** Eight estimators across five method families
land between 0.60 and 0.70 macro and 0.71–0.87 rad point error, none within
0.19 rad of the `pi/6` cap. This experiment is registered as a genuine test with
a real chance of a negative result, not as a step expected to succeed.

## Fixed procedure

Run `scripts/train_g1_phase_encoder_v2_1e.py` three times, once per arm:

| arm | `--cycle-weight` | cycle share of increment weight |
|---|---:|---:|
| `control` | 2.0 | 0.70% (v2.1b verbatim) |
| `w20` | 20.0 | 6.60% |
| `w60` | 60.0 | 17.49% |

Held fixed exactly as v2.1b: seed 20261020; 30 complete epochs; batch 512;
hidden 64; context 30; AdamW lr 3e-4 with weight decay 0.1; dropout 0.5;
gradient clip 1.0; lags 1/4/8/16/32; point weight 1.0; reverse 3.0; jump 1.0;
the same pair builder, model, bounded increment loss, and validation scorer.
Fit on train only, score and select on validation only. No early stopping.

`CYCLE_WEIGHT` is the only changed quantity. The point term is provably
invariant: it is weighted by row weight divided by that row's kind multiplier,
which cancels the multiplier. Only the allocation of the increment term moves.

The `control` arm is an equivalence check on the whole harness. It must
reproduce the frozen v2.1b epoch-1 figures (macro `0.6962456093142473`, point
error `0.7716245022746507`) to within `1e-6`. If it does not, the deviation is
recorded and reported; the three arms remain mutually comparable because they
share one code path and differ in one constant, but the control claim is then
`not reproduced` and no absolute comparison to published numbers is made.

## Outcomes and decision rules

Primary metric: validation macro cycle completion under v4.2, best epoch.

1. **Candidate.** An arm qualifies only on a full v4.2 pass: macro at least
   0.90, every condition at least 0.80, every period-length stratum at least
   0.80, coverage and reference-clock checks pass, and condition-macro point
   error at most `pi/6 = 0.5235987756` rad. Ranking across arms: macro
   descending, point error ascending, cycle weight ascending. A selected
   checkpoint remains a validation-selected research candidate, not a frozen
   deployment encoder.
2. **Improvement signal.** Best macro strictly greater than `0.6962456093`.
   This alone does not qualify an encoder for use.
3. **Mechanism test.** The hypothesis is specific: raising the cycle weight
   reduces the *cycle-integral* error. A gain is counted as supporting the
   mechanism only if, against `control` on the best epoch, the arm shows **both**
   (a) a lower net-turn error sd and (b) fewer whole-turn slips, alongside
   (c) higher macro completion. A macro gain with flat or worse (a) and (b) is
   recorded as a point-accuracy gain and does **not** support the mechanism.
4. **Pre-registered stop.** If neither `w20` nor `w60` beats
   `0.6962456093`, record `CYCLE_WEIGHT_NOT_THE_BINDING_CONSTRAINT` and treat
   the loss-side line of encoder development as closed: the graded integral is
   then not recoverable by reweighting supervision of the same inputs, which
   joins the eight estimators above and the 101k-parameter upper bound as
   evidence that the ceiling is information-limited at this input contract.
5. **Unresolved.** A best macro in `(0.65, 0.6962456093]` is recorded as
   unresolved against the v2.1b baseline. At or below 0.65 is the stop bar.

No PPO run or deployment decision is part of this experiment. A later behaviour
study requires its own registration. The existing test split has previously been
opened, so it cannot serve as a blind confirmation set; a separate independent
holdout is required for confirmatory claims.

## Provenance and reporting

The trainer records source hashes for itself and for every frozen module it
imports, the dataset hash, pair counts by kind, the realised increment-weight
shares, all 30 epoch predictions, the full v4.2 report per epoch, and a per-epoch
net-turn decomposition (within-tolerance share, under/over counts, whole-turn
slips, sd, quartiles, period correlation). Report all 30 epochs for all three
arms, including failed gate components. Do not describe a validation result as
deployment performance or as evidence of policy improvement.
