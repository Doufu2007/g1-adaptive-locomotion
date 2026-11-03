# Pre-registration — encoder v2.1

- Date: 2026-10-20
- Gate under evaluation: `g1-phase-gate-v4.2` (endpoint-corrected, thresholds unchanged)
- Test split: **not read.** All fitting on train (subjects 5/3/4), all selection on
  validation (subject 1).

The constants below live in the code that was launched, not in this note:

```
scripts/train_g1_phase_encoder_v2_1.py   acc6c14c460c7be687b925b4ed663e151b9417326656f3ab20a7e7a009d71647
src/tasks/velocity/phasor/encoder_v2_1.py 12d5c4754762ac1ef8cce7a4afa41f03fdefc8aa380529ffd856847fa867a559
src/tasks/velocity/phasor/phase_gate_v4_2.py 18794a59cf8ec56aa965b40270a84b43fe752128979dcc79a7716c495c742810
```

These hashes were taken at launch. If a later run reports different hashes, the
pre-registration no longer applies to it.

## Why a new objective

Step-1 diagnostics (`results/g1_phase_gate_v4_2/diagnostic_step1_v4_2.json`) measured
three things that decide the objective:

1. **No stall.** Zero of twelve reference-phase bins have a median advance ratio below
   0.5 for any estimator (worst bin 0.68 for the ridge, 0.46 for v1). The clock never
   freezes.
2. **Phase-dependent rate distortion.** v1 advances at 0.69–0.86× the reference rate
   through the first two-thirds of the cycle and 1.02–1.09× in the last third. The net
   still reaches ≈6.2 rad because the errors cancel — which is why every estimator's
   *median* net is inside the π/6 band while only 41–62% of cycles complete.
3. **Frame-level backtracking.** Forward ratio is 0.801 (v1), 0.888 (ridge), 1.000
   (reference). One frame in five goes backwards in v1, spread evenly across phase.

Encoder v2 paired only lag-1 frames, so it could not see either failure. The v2.1
objective scores increments at lags 1/4/8/16/32 and over the full cycle.

4. **Completion tracks absolute phase error.** Median point error is 0.329 rad for
   completed vs 1.089 rad for failed ridge cycles (3.3×); 0.458 vs 1.034 for v1 (2.3×).
   Increments are differences and cannot pin phase, so a full point term is retained.

## Fixed configuration

| item | value |
|---|---|
| context | 30 frames |
| seed | 20261020 |
| epochs | 30 (every epoch scored; no early stopping) |
| batch | 512 |
| learning rate | 1e-3 (Adam) |
| hidden | 64 (same `PhaseEncoderV2` class) |
| multi-scale lags | 1, 4, 8, 16, 32 |
| weights | point 1.0, lag-1 rate 2.0, other scales 1.0, full cycle 2.0, reverse 3.0, jump 1.0 |
| cycle weighting | each complete cycle contributes base weight 1.0 regardless of length |
| cycle rule | half-open `[touchdown_k, touchdown_{k+1})`, matching gate v4.2 |

Nothing above is tuned. The reverse weight is 3.0 (v2 used 2.0) because the measured
forward ratio is 0.80, and backtracking is one of the two failure modes.

## Selection rule (validation only)

Eligible = `encoder_freeze_eligible` (v4.2) **and** `condition_macro_point_error_rad <= π/6`.

The point-error cap reuses the contract's own frozen net tolerance and is declared here
before any v2.1 result exists. It is not applied retroactively to v4.2, whose
`encoder_freeze_eligible` stays a pure correctness fix. The cap exists because net
progression and maximum step are both differences: `phase = reference + 1.0` — a constant
57° misalignment — scores macro 1.0000 in both v4.1 and v4.2.

Ranking among eligible epochs: macro completion desc, then point error asc, then epoch asc.

## Pre-declared stop rule

The bar to beat is **0.6239**, the linear ridge under v4.2 (0.6528 under v4.1), and the
best of every non-oracle estimator tried:

| estimator | macro (v4.2) |
|---|---|
| reference clock (oracle) | 0.9984 |
| ridge λ=1e4 | 0.6239 |
| regularized MLP (32, do 0.5, wd 0.1) | ≈0.64 (v4.1 scoring) |
| plain MLP 2×256 | ≈0.59 (v4.1 scoring) |
| encoder v1 | 0.4128 |
| encoder v2, best of 60 | 0.1049 |

**If the best v2.1 epoch is ≤ 0.65**, the conclusion is that the 0.90/0.80 bar is
unattainable under this data (one validation subject, three training subjects) and target
definition (cycle fraction between left touchdowns) — and the next move is data or target
definition, not another loss variant.

**If v2.1 exceeds 0.65 but no epoch is eligible**, the point-error cap is the binding
constraint and must be re-examined on its own terms, not lowered to admit a checkpoint.

## What a pass does not authorise

A validation pass does not open PPO. The test split is burned
(`protocol_deviation_test_split_diagnostic_v1.md`); blind confirmation requires a newly
collected holdout, used exactly once, followed by the independent confirmation rollout and
the ready lock.
