# Contact-Supervised G1 Phase Encoder Design

## Intent

Replace the current dance-window autoencoder checkpoint with a new encoder trained to infer G1 locomotion phase from joint-position history. The previous checkpoint has very low G1 contact agreement and barely advances around the circle. The new checkpoint must pass a frozen phase/contact gate before any PPO pilot can start.

The user-selected gate is contact agreement of at least 0.90 and continuous coverage of a complete phase cycle. This work is limited to collecting phase-training evidence, training and freezing one encoder checkpoint, and rerunning the G1 validity gate. It does not change or write the paper and does not start PPO.

## Goals

- Train on actual G1 velocity-task states and foot contacts, rather than using dance-motion reconstruction and temporal smoothness as a proxy for locomotion phase.
- Keep the 10-frame, 29-joint input and 50 Hz runtime interface so the checkpoint can replace the existing encoder without changing policy observation dimensions.
- Split data by simulator seed before fitting normalization or training, and keep validation and test rollouts independent of training.
- Preserve a decoder during pretraining so held-out reconstruction error remains measurable and reportable.
- Save the new model separately from `phasor_encoder_v2.pt`; never overwrite the frozen checkpoint or its prior results.
- Report checkpoint and data hashes, split membership, validation reconstruction error, G1 contact agreement, phase coverage, and directional phase progression.

## Data and labels

Use the existing frozen G1 fixed-clock policy checkpoint `model_9900.pt` with SHA-256 `949fae33632b9a3592e6346a6155eea059ee2fd39487207a0a0cea5d35edd8b4`. Collect 1,250 control steps for four parallel environments at each of five predeclared simulator seeds: `20261021`, `20261022`, and `20261023` for training; `20261024` for validation; and `20261025` for the locked test. Use the existing forward command range 0.5–1.0 m/s, zero lateral and yaw commands, and the 0.020 s (50 Hz) control interval. Each environment is an independent trajectory within its simulator-seed group; no window may cross a trajectory boundary.

Record episode IDs during collection. If a termination resets an environment, start a separate trajectory segment and never form windows across the reset. Every planned train and validation segment must contain at least one complete alternating touchdown cycle to be usable. Retain every invalid segment and its reason; if any planned train or validation segment is invalid, record acquisition failure and do not silently drop it. A locked-test segment failure makes the phase/contact gate fail.

Record per-step joint positions, left/right contact state from `feet_ground_contact`, command, simulator seed, environment ID, and trajectory step. Ignore stationary-command samples for phase-label generation. Derive phase targets only from valid alternating left/right touchdown events: anchor left touchdown at cycle 0, right touchdown at cycle 0.5, and linearly interpolate phase between successive valid touchdown events. Exclude intervals with missing or non-alternating touchdown events instead of filling them with clock labels. The fixed clock is not used to construct the targets. The dance motion file is not used because its original source and redistribution terms remain unverified.

## Training design

Reuse the existing 1-D CNN phase encoder and decoder, with a 10-frame input window and unit-circle output. Use a circular supervised phase loss, `mean(1 - cosine(theta_pred - theta_target))`, plus normalized-window reconstruction MSE weighted by `0.1`. Remove the adjacent-window latent-smoothing loss, which can be minimized by a nearly constant embedding and does not encode phase direction. Keep the existing Adam optimizer, learning rate `1e-3`, batch size `256`, 300-epoch ceiling, and pretraining random seed `42`; select the saved checkpoint by validation phase/contact metrics, without examining the locked test set.

Compute per-joint normalization statistics from training trajectories only. Build all windows after assigning whole simulator seeds to train, validation, and test. Do not fit preprocessing on validation or test. If contact-derived labels do not contain complete alternating cycles for a trajectory, mark that trajectory invalid and retain its failure record; do not silently replace it with clock-derived labels.

## Gate and reporting

Evaluate the frozen checkpoint on validation first. If it fails either gate component, stop before opening the locked test set and do not start PPO. If validation passes, evaluate the unchanged checkpoint on simulator seed `20261025` using the same 4-environment, 1,250-step G1 rollout and the existing contact-agreement formula (`threshold=0.56`, offsets `0.0` and `0.5`). The test gate passes only when the macro-averaged contact agreement is at least `0.90`, no planned test trajectory is invalid, and every test trajectory has a contiguous time-ordered segment with positive net phase advance of at least `2π` and preserved touchdown order. The macro average gives equal weight to each trajectory; invalid trajectories are retained and force failure rather than being excluded. Report circular coverage and net-to-total angular travel as descriptive metrics, with no additional cutoff. Also report per-trajectory agreement, winding direction, touchdown counts, and validation reconstruction MSE. If a gate component fails, no PPO pilot is run.

Select checkpoints by this fixed validation ordering: a checkpoint satisfying the complete validation gate ranks above one that fails; within the same pass/fail class, higher macro contact agreement ranks first, then lower circular phase loss. Exact ties retain the earlier checkpoint. Validation reconstruction MSE is reported as a guardrail and does not override the phase/contact ranking.

The results are conditional on the one frozen policy checkpoint used to collect trajectories. Multiple simulator seeds estimate simulation-state variation; they do not count as independent policy-training or encoder-training repetitions. Report this limitation with the checkpoint hash.

## Implementation boundaries

- Create a contact-labelled G1 trajectory collector and a phase-target builder with explicit episode and trajectory boundaries.
- Extend encoder training to consume the split manifest and save an independent checkpoint such as `phasor_encoder_g1_contact_v3.pt`.
- Extend the diagnostic path to evaluate that checkpoint on validation/test data without fitting phase alignment on held-out labels.
- Add tests for touchdown-label construction, full-cycle winding, trajectory-boundary handling, and train-only normalization before implementing those behaviors.
- Do not change the current v2 checkpoint, prior result bundles, frozen PPO matrix, paper files, or default experiment status.

## Implementation calls

- Keep collection, training, and locked-test evaluation as explicit command-line stages in one experiment script.
- Record environment/package versions and CPU/GPU choice in each dataset and training artifact.
- Treat any planned segment without a valid complete phase-labelled cycle as a failed acquisition; retain it and do not replace its seed or exclude it from the gate.
