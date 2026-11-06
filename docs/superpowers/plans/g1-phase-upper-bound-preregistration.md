# Pre-registration — bidirectional ±1 cycle upper bound on phase identifiability

Written before the run. Launch date 2026-10-20.

## The question

Probe (c) in `results/g1_phase_gate_v4_2/REPORT_v4_2.md` swept the context length at 30, 60 and
90 frames and found no improvement in point error. That result is confounded: the estimator was a
k-NN over a **PCA-32** projection, so growing the input from 870 to 2610 dimensions while holding
the projection at 32 lets information loss substitute for lack of information. The probe therefore
supports only "under the best non-parametric estimator available here, a longer context did not
help" — not "the information is absent".

This run removes that confound. It asks one question:

> With a parametric model that sees one full cycle in **both** directions, and with no projection
> bottleneck, does absolute phase become recoverable?

This is the last explanation for the ~0.78 rad floor that has a mechanism and has not been tested.

## Design — one variable changes

Everything else is held fixed against the v2.1b run (`scripts/train_g1_phase_encoder_v2_1b.py`):

| element | v2.1b | this run |
|---|---|---|
| window | 30 frames, trailing only | **151 frames, ±75 around the anchor** |
| architecture | 2 × Conv1d(k=5), flatten, linear | **5 × Conv1d(k=5, dilations 1/2/4/8/16), centre tap, linear** |
| receptive field | 9 frames | **125 frames** |
| objective | `phase_v21b_loss` (multi-scale lags 1/4/8/16/32 + full-cycle integral, bounded increment) | **identical, imported unchanged** |
| feature path | `build_phase_v2_features` (pose + finite-difference velocity + RMS speed, 59 dims) | **identical, imported unchanged** |
| normalisation | stored `x` = `(raw - train_mean) / train_std` | **identical, verified to 0.000000 on the stored rows** |
| split | train = subjects 3/4/5 (6 sequences), validation = subject 1 (4 sequences) | **identical** |
| gate | v4.2, thresholds untouched | **identical** |
| seed / epochs / optimiser | 20261020 / 30 / AdamW 3e-4, wd 0.1, dropout 0.5, clip 1.0 | **identical** |

Period on this data is 45–70 frames, so ±75 covers at least one full cycle in each direction.

## Pre-declared decision rule

On `condition_macro_point_error_rad` at the best validation epoch:

- **≤ 0.60 rad** — the probe (c) confound was real and context does carry the missing information.
  The next step is a proper parametric sweep over context and capacity, not a contract change.
- **≥ 0.75 rad** — the input side is closed. Together with (a), (d), (e) and (f), no remaining
  candidate explains the floor, and the question becomes one about the target as posed.
- **0.60 – 0.75 rad** — inconclusive. Report it and do not use it to decide either way.

The rule is stated on point error alone because that is the quantity probe (c) was measured on and
the only one comparable across estimators. Macro completion is reported but is not part of the
rule: comparing macro across a best-of-30 selection would confuse attainability with selection.

## What this run may and may not be used for

**May**: be reported as a diagnostic upper bound, together with the epoch it was taken at.

**May not**: produce a frozen checkpoint, unlock PPO, or be counted as a passing candidate.
Selection is best-epoch-on-validation, which peeks at the metric used for selection; the number it
produces is optimistically biased and is only meaningful in the direction of an upper bound. Even
if it cleared 0.90 macro, PPO would stay blocked, because a candidate selected this way has not
been validated under the pre-registered freeze rule.

The test split is not read. `phase_gate_v4.py` is not modified. No threshold is changed.

## Known limitations, stated in advance

- Cross-subject generalisation is the binding constraint on capacity: validation is entirely
  subject 1, and v2.1b memorised its 6 training sequences in under one epoch. A larger model may
  therefore overfit harder, and the best-epoch selection is doing real work here. That biases the
  result toward *not* finding an improvement, so a ≥0.75 outcome is evidence, while a ≤0.60
  outcome is a genuine positive.
- A single seed. A split between arms at 0.60–0.75 cannot be interpreted as a mechanism.

---

## Amendment 1 — 2026-10-20, written after the run

**The original text is left as written.** Editing a pre-registration after seeing its result
destroys the only thing it is for. This section records a reasoning error found on re-reading it,
and what replaces it.

### What the run produced

Best `condition_macro_point_error_rad` = **0.714 rad at epoch 13**; best macro completion
**0.6281 at epoch 24**. The run was flat (macro 0.0000, point 0.91–1.01) for twelve epochs, escaped
at epoch 13, then trained to point error **0.006 rad** by epoch 30 while validation stayed at
0.72–0.77. Under the rule above, 0.714 falls in the middle band → **INCONCLUSIVE_DO_NOT_DECIDE**,
and that is the verdict. Artifact: `results/g1_phase_gate_v4_2/upper_bound_epoch_sweep.json`
(reconstructed from stdout — the trainer crashed in `sha256_file` after training when the E: volume
dropped, before it could write its own JSON).

### The sentence that is wrong

> "That biases the result toward *not* finding an improvement, so a ≥0.75 outcome is evidence,
> while a ≤0.60 outcome is a genuine positive."

**The first half is false.** Overfitting is not a conservative bias in the direction of the null.
A model that drives training error to 0.006 rad has demonstrated that *a* mapping exists from this
input to this target on the training sequences; what it has not demonstrated is that the mapping
transfers to a held-out subject. Those are two different failures, and a high validation error
cannot distinguish them:

- "the input does not carry the information" — the information is absent; and
- "the input carries it, but six sequences from three subjects are not enough to learn the
  transferable part, and a 101k-parameter model finds the memorising solution instead".

A ≥0.75 outcome from this arm is consistent with both. It therefore cannot be used as evidence for
the first, which is what "the input side is closed" claims.

**The second half survives.** A ≤0.60 outcome is still a genuine positive: a model cannot reach an
error lower than the information allows.

### What replaces the ≥0.75 branch

The ≥0.75 branch is **void as a decision rule**. A high validation error from an arm whose training
error is two orders of magnitude lower is uninformative about attainability. To make the branch
live again, the arm would have to separate "cannot fit" from "cannot transfer", which needs either
per-subject adaptation as a third arm or enough subjects that the held-out one is not the only
generalisation test — neither of which this run has.

### The limitation this run exposed that the pre-registration did not anticipate

The escape at epoch 13 is not a smooth descent; validation macro jumps 0.0000 → 0.5781 and point
error 0.939 → 0.714 in one epoch. **A best-of-30 selection over a curve with a discontinuous
take-off is selecting among few effectively distinct states, not 30.** The reported 0.714 is the
minimum of a set whose effective size is closer to 2 (pre- and post-transition), and the
optimistic bias of best-epoch selection on such a curve is larger than the 30-epoch count implies.
This does not change the verdict — it reinforces it.
