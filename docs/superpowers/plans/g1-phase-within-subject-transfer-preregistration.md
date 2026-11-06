# Pre-registration: is the 0.70 ceiling caused by the change of subject?

- Date: 2026-10-21
- Gate: `g1-phase-gate-v4.2`, unmodified; all thresholds frozen
- Split used: validation subject only (subject 1). **Test split is never loaded.**
- Status: registered before any run. Not edited after the fact; corrections go in
  an appended amendment section.

## The question this answers

Seven arms, all derived from the same v2.1b recipe, land at 0.628–0.703 macro
against a 0.90 bar: v2.1b, v2.1c (cycle-integral supervision), v2.1d (50 and 60
frame context), the 151-frame upper-bound CNN, and v2.1e at cycle weights
2/20/60. The levers covered loss, weighting, context and capacity.

Those seven are **not seven independent confirmations**. They share one dataset,
one feature pipeline, one normalisation, one split and one scorer. A single
defect in any of those would reproduce identically across all of them. The
agreement is therefore consistent with a property of the task *and* with a
shared pipeline fault, and this experiment separates the one shared variable
that can be separated cheaply:

> the training subjects are 3, 4 and 5; the scored subject is 1.

## Design

Subject 1 has four sequences. Two folds, each holding out two sequences:

| fold | validation | within-subject training arm |
|---|---|---|
| A | `walk3_subject1`, `walk4_subject1` | `walk1_subject1`, `walk2_subject1` |
| B | `walk1_subject1`, `walk2_subject1` | `walk3_subject1`, `walk4_subject1` |

Each fold is run twice, so the design is 2 x 2:

- **arm W (within)**: train on the fold's within-subject sequences.
- **arm C (cross, data-matched)**: train on two sequences drawn from the training
  subjects, chosen by a fixed rule stated below.

Arm C exists because arm W trains on roughly a third of v2.1b's rows (v2.1b sees
73,226 rows / 948 cycles). Without C, a *worse* W result could be caused by the
smaller training set rather than by the subject change. Selecting the C sequence
pair by touchdown count, which is the quantity the cycle term counts, controls
that.

### Fixed selection rule for arm C

For each fold, from the six training sequences
(`walk1_subject5`, `walk2_subject3`, `walk2_subject4`, `walk3_subject3`,
`walk3_subject4`, `walk3_subject5`), choose the **pair** whose total left-foot
touchdown count is closest to arm W's total; break ties by total row count.
The rule is applied by script, and the chosen pair is recorded in the artifact.

Indicative counts (touchdowns): W-A 333, W-B 274; the closest available pairs
are 354 and 247 respectively. The residual mismatch is recorded, not corrected.

### Held fixed

Everything else is the v2.1e control arm, which reproduced v2.1b bit-for-bit
across all 30 epochs (max |delta macro| = 0.000e+00):

- architecture `PhaseEncoderV21B(context_frames=30, hidden=64, dropout=0.5)`
- loss `phase_v21b_loss` with `POINT_WEIGHT`, `REVERSE_WEIGHT`, `JUMP_WEIGHT`
  and `CYCLE_WEIGHT = 2.0` unchanged
- the stored `x` representation, i.e. normalised with statistics fit on the
  training subjects. **This is deliberately not refit**: refitting it would be a
  second changed variable.
- seed 20261020, 30 epochs, batch 512, AdamW, lr 3e-4, weight decay 0.1,
  grad clip 1.0, no early stopping
- the frozen `calibration_v4_amendment1.json` period-length cuts
- point-error cap pi/6; macro bar 0.90; per-condition and per-stratum bars 0.80

## Why the normalisation is held fixed, and what that costs

The stored features are subject-1 joints normalised by training-subject
statistics — the same representation any deployed encoder would receive. Keeping
it fixed means arm W changes **only** who is in the training set.

The cost is stated plainly: if this experiment comes back negative, it does not
distinguish "the subject change is not the limit" from "the subject change is the
limit but it acts through the normalisation statistics". That follow-up
(refit normalisation on the within-subject training half) is a separate
experiment and is not run here.

## Decision rules (fixed before the run)

Let `W_mean` be the mean best-epoch macro of the two within arms and `C_mean`
the same for the two cross arms. Best epoch is by macro, as in every prior arm.

1. **`TRANSFER_IS_THE_LIMIT`** — any of the four runs passes the full v4.2 gate
   (macro >= 0.90, every condition >= 0.80, every stratum >= 0.80, point error
   <= pi/6, coverage and reference-clock checks pass). The same recipe clears the
   bar once the subject does not change.
2. **`SUBJECT_CHANGE_CONTRIBUTES`** — no full pass, but `W_mean - C_mean >= 0.05`
   macro. Direction-of-evidence only: single seed, two folds, one subject.
3. **`NOT_TRANSFER_LIMIT`** — no full pass and both `W_mean` and `C_mean` at or
   below the existing 0.65 stop bar. The ceiling survives removing the subject
   change and survives matching the training-set size.
4. **`UNRESOLVED`** — anything else, including a within-vs-cross gap below 0.05.

The 0.05 in rule 2 is a pre-declared effect size, **not** a measured noise band.
No multi-seed noise band exists for this recipe; one run per cell is what this
experiment can afford.

## What this cannot show

- Nothing about a structurally different learner. It varies the data partition,
  not the architecture. The PINN/neural-ODE family proposed separately — estimate
  phase *rate* from state and integrate, rather than regress phase position from a
  window — is untouched by this experiment.
- Nothing about subjects other than 1.
- No claim about the test split, which stays unread.

## Artifacts this will produce

- `results/g1_phase_gate_v4_2/v21f_{within,cross}_fold{A,B}_epoch_sweep.json`
- matching `..._epoch_predictions.npz`
- `results/g1_phase_gate_v4_2/v21f_transfer_summary.{md,json}`

## Amendment 1 (2026-10-21, appended before any run)

The indicative arm-C figures in "Fixed selection rule for arm C" are **wrong**.
They were computed by hand from a partial sum table and are superseded by the
rule's actual output, obtained by evaluating the rule over all fifteen pairs
before launching the runs:

| fold | within touchdowns W | rule's pair | chosen touchdowns | mismatch | chosen rows |
|---|---|---|---|---|---|
| A | 333 | `walk1_subject5` + `walk3_subject3` | 322 | 11 | 25,250 |
| B | 274 | `walk2_subject3` + `walk3_subject4` | 277 | 3 | 24,022 |

The rule itself is unchanged and is the binding part; only the pre-computed
illustration was wrong. It failed in the permissive direction — the rule matches
training-set size better than the text claimed (mismatch 11 and 3 touchdowns,
versus the 21 and 27 implied by 354 and 247), so the cross arm is a *tighter*
size control than registered.

Row counts, the 0.05 effect size, the four decision rules and everything held
fixed are untouched. Corrected before the first run; no result existed when this
was written.
