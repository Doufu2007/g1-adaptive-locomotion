# PHASOR re-wire — one-arm behavioural check (2026-10-21)

Written before any run. Nothing here may be edited after the run starts;
corrections go in an appended amendment.

## What is already wrong, and what is already fixed

The four August PHASOR arms did not deliver a phase (see
`results/g1_phase_gate_v4/progress_v4.md`, 2026-10-21 entries): the actor's
`phase` term was a near-constant unit vector, the angle was decoded as
`atan2(cos, sin)`, the shared history buffer advanced twice per control step
and was never reset, and the `foot_gait` term was replaced at the same time as
the observation, so two factors moved.

The current working tree already fixes all four: `rewards.py:456-459` decodes
through `phase_to_angle`, `observations.py` caches on `common_step_counter`,
`env_cfgs.py:293` registers `reset_phasor_phase_history`, and the reward reuses
the actor's latent. The wiring probe passes on this tree
(`results/phasor_local_pilot_20261020/probe_wiring.json`, regenerated
2026-10-21: phase terms separate, gait terms separate, actor dims 285 = 285,
both unit norm, encoder path resolving inside this repository).

## This run is a deviation, recorded as one

`docs/superpowers/specs/2026-10-16-contact-supervised-g1-phase-encoder.md`
says a PPO run starts only after the encoder passes the phase gate. The
runtime encoder scores **0.4128** against a 0.90 bar and is recorded
`encoder_freeze_eligible: false`. This run therefore **violates that rule on
purpose**. It is admitted only because it answers a question that the gate
cannot: whether the corrected pipeline produces a policy that walks at all.
It is not evidence for any claim about learned phase versus a fixed clock, and
no abstract, report table or manuscript may cite it as such.

It is also **not experiment B**. B is registered at 4096 environments, three
paired seeds, both arms, `max_iterations=10001`
(`results/phasor_rough_equalbudget_20261020/`). This run is one arm at the
local hardware ceiling.

## Configuration

| field | value | why |
|---|---|---|
| task | `Unitree-G1-Rough-PHASOR` | the corrected arm |
| environments | 1024 | measured ceiling of the RTX 4050 Laptop, 6.1 GB (`rough_scale_probe.json`) |
| iterations | 10001 | the registered budget, so the curve covers the scoring window |
| seed | 20261030 | throwaway seed, already used by the 4096-env pilot; never a B seed |
| encoder | `phase_encoder_runtime_v1.pt` (`54d6f41b`) | the artifact the audited pipeline loads; v2.1b epoch 1 scores better but expects a 30-frame window and cannot drop in |
| critic | analytic 0.6 s clock | per the 10-19 spec, unchanged |

Expected duration from the measured 5.50 s/iteration at 1024 environments:
**about 15.3 hours**. Checkpoints every 100 iterations, so a crash is
recoverable.

## Readouts, fixed now

Every 100 iterations, from TensorBoard tags that the registered analysis plan
already reads:

1. `Train/mean_episode_length` — does the policy survive?
2. `Episode_Termination/fell_over`
3. `Episode_Reward/track_linear_velocity` — does it move under command?
4. `Episode_Reward/foot_gait` — does the learned-phase gait term rise?
5. `Policy/mean_std`
6. `Perf/total_fps`

After training, one added measurement, not used to select anything: run the
final checkpoint and record the learned phase's **advance per gait cycle** in
the runtime path. The gate figure 0.4128 is offline; this is the first
in-simulator reading of the same quantity. Pre-declared rule: report it
against the one-turn-per-cycle requirement regardless of the sign of the
result.

## Decision rule, written before the numbers

- **WALKS**: episode length in the last 1000 iterations is at least 10× its
  value at iteration 100, and `Episode_Termination/fell_over` is below 5.
  Interpretation: the corrected pipeline admits locomotion, so the August
  stand-still at 1000 iterations was a wiring artefact or early-training
  transient, not a property of the learned clock.
- **STANDS**: episode length stays under 10× its iteration-100 value while the
  baseline's August curve (budget-matched window) reaches at least 100×.
  Interpretation: a signal that scores 0.4128 on the gate is not enough to
  drive a gait, and the encoder — not the wiring — is the limit.
- **UNREADABLE**: anything else, including a crash before iteration 5000, a
  degenerate run, or a curve that has not separated by the last window. No
  claim either way.

One seed, one arm. No fall-rate comparison against any other run may be drawn
from this file, in either direction.
