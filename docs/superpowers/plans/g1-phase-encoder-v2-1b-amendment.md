# Amendment — encoder v2.1b

- Date: 2026-10-20
- Supersedes: nothing. This is a *second* run under the v2.1 programme, not a
  re-registration of the first.
- Gate under evaluation: `g1-phase-gate-v4.2`, thresholds unchanged.
- Test split: **not read.** Fit on train (subjects 5/3/4), selection on validation
  (subject 1) only.

## Why this amendment exists

`g1-phase-encoder-v2-1-preregistration.md` pre-registered run v2.1a. v2.1a was
terminated at epoch 5 on its **training curve alone** (`v21a_terminated.json`): loss
rose 1.9489 → 2.7474 across epochs 1–4 and was still 2.5555 at epoch 5. The
mechanism is arithmetic, not empirical — dividing every lag's absolute error by π/6
makes a lag-32 row cost ~11 and a lag-1 row ~0.02, so a near-constant phase output
pays a plateau of ~1.06 that the weight-1.0 point term cannot climb out of.

v2.1a therefore carries **no information about attainability**. Its pre-declared stop
rule was explicitly *not* invoked.

## What changed, and when

The running sources:

```
scripts/train_g1_phase_encoder_v2_1b.py     927e300ebaa4e0410641ef39e7ece17542c78f9275ced2cb78175da6ef1b1bbd
src/tasks/velocity/phasor/encoder_v2_1b.py  0ec18a619fcd8d830adeeeacf892c13af8b6ff77737f6b69fe7e74cc807b10a6
```

Both hashes were taken while the run was in flight from files that had not been
edited since launch.

| item | v2.1a | v2.1b |
|---|---|---|
| increment error | `smooth_l1(Δpred − Δtrue / (π/6))` | `1 − cos(Δpred − Δtrue)` |
| optimizer | Adam, lr 1e-3 | AdamW, lr 3e-4, wd 0.1 |
| dropout | none | 0.5 |
| gradient clip | none | 1.0 |
| everything else | — | **identical**: pairs, lags 1/4/8/16/32, cycle base weight 1.0, point weight 1.0, lag-1 rate 2.0, cycle 2.0, reverse 3.0, jump 1.0, context 30, seed 20261020, hidden 64, batch 512, 30 epochs |

**These choices were made after seeing v2.1a diverge.** They are a legitimate
follow-up, but they are not part of one pre-registered study, and nothing here should
be reported as if they were. The bounded form `1 − cos` lies in [0, 2] for every lag,
so scale domination is removed by construction rather than by tuning a divisor.

## Carried over unchanged (pre-declared before any v2.1 result)

- point-error cap = π/6 = 0.5236 rad, reusing the contract's own net tolerance
- ridge floor = 0.6239 (v4.2)
- stop threshold = 0.65
- selection: `freeze_eligible` AND point error ≤ cap; rank macro desc, point asc, epoch asc

## No early stopping in this run

Unlike v2.1a, the training curve here is healthy. Stopping it because the *validation*
curve degrades would be peeking at the selection metric mid-run. All 30 epochs are
scored, and the selection rule is applied to the complete sweep.

## Pre-declared reading of the outcome

Both branches are already written in the v2.1 pre-registration:

- best epoch ≤ 0.65 → the 0.90/0.80 bar is unattainable on this data and target
  definition; move to data or target definition, not another loss variant.
- best epoch > 0.65 but no epoch eligible → **the point-error cap is the binding
  constraint and must be re-examined on its own terms, not lowered to admit a
  checkpoint.**
