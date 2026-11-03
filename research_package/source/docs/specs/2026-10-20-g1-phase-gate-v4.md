# G1 phase gate v4 design

## Purpose

Replace the v3 phase-cycle checks with a pre-registered, period-level gate that detects whether an encoder advances through each complete left-foot touchdown cycle. Keep every v3 file and result unchanged. Train encoder v2 with an explicit phase-rate objective and enough temporal/speed input to address phase stalls at slow walking speeds.

## Frozen v4 gate

- A cycle is the contiguous prediction interval between consecutive left-foot touchdown frames inside one sequence or episode segment. It must include both endpoints, have at least 10 historical frames at its start, and contain every intervening frame. Cycles never cross a sequence, episode reset, missing-frame gap, or condition change.
- Net-cycle tolerance is exactly `pi/6` radians, the pre-registered task resolution. It is not derived from a single frame or cycle length.
- Every cycle must have positive unwrapped net progression within `pi/6` of `2*pi`, and its maximum absolute frame-to-frame phase increment must be strictly below `pi/2` radians. No per-frame quantile controls a period maximum.
- Calibration computes reference-clock and encoder completion rates using the same cycle scorer. Reference completion must be at least `0.90`. An encoder below `0.90` macro completion is explicitly rejected for freezing.
- Candidate validation requires at least 10 complete, evaluable cycles for each validation condition, at least 10 cycles in each frozen period-length stratum overall, reference-clock pass, encoder macro completion at least `0.90`, every condition at least `0.80`, and every period-length stratum at least `0.80`.
- Rollout confirmation keeps the existing episode, terminal-contact, history, and coverage checks: each scheduled condition must have at least 10 evaluable cycles across at least 2 episode segments. Terminated and truncated segments remain visible in coverage and reports.
- Report all cycles by condition and by frozen short/medium/long period strata. Use validation reference-period tertiles as frozen cut points; retain periods beyond either cut in the corresponding outer stratum. Do not remove slow conditions, long cycles, or failures to improve a rate.
- Pointwise circular error remains an auxiliary metric and cannot override the cycle gate.

## Encoder v2

- Keep the per-frame circular loss `mean(1 - z dot z*)`.
- Add adjacent-frame phase-rate loss against the wrapped label increment, a squared penalty for negative predicted increments, and a penalty for increments exceeding the v4 jump limit.
- Evaluate two predeclared context candidates: 10-frame and 30-frame pose history. Both use the same train-only-normalized pose history plus joint-velocity differences and a scalar joint-speed feature. Context construction pads only at sequence or continuous-segment starts and never crosses a split or gap.
- Offline validation, test evaluation, and rollout inference call the same feature-building and checkpoint-forward functions. No EMA or hidden smoothing is used.
- Candidate selection is lexicographic: highest validation macro cycle completion first; then lowest equal-weight condition point error; then shorter context. Point error is not the primary selection score.
- Training uses only the train split. Validation selects candidates and calibrates the frozen limits. Test and rollout are not read or collected until a candidate passes validation and its checkpoint and thresholds are frozen.

## Confirmation and PPO boundary

After validation passes, freeze the selected checkpoint, its hash, the v4 contract, and calibration. Evaluate the test split once without changing any decision. Then collect one new confirmation rollout with a new environment seed, score it with the same v4 implementation, and retain every episode and condition. Create a ready lock only if test and independent confirmation satisfy the frozen gate and coverage rules. PPO may start only after the ready lock exists and verifies every referenced hash. Any failed or uncovered gate leaves PPO blocked and preserves all raw artifacts.

## Data and provenance

The source is the already split, train-normalized dataset at `results/g1_phase_dataset_20261019/phase_dataset.npz`. Split IDs and normalization stay fixed. The v4 work records the source hashes, current commit, existing dirty state, code hashes, environment, and seeds. The v3 tree is read-only for this task.
