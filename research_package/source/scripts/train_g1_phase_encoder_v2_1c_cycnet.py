"""Lever 2: add gate-aligned cycle-net supervision to the v2.1b recipe.

Why this arm exists
-------------------
The v2.1b objective never sees the quantity the gate scores. Its point term is
`1 - cos(pred - reference)`, which is blind to a constant phase offset, while the
gate's criterion is the *difference* between the unwrapped phase at the two ends
of a cycle. Its increment term is `1 - cos(pred_delta - target_delta)`, and
`pred_delta` is an angle between two unit vectors, so it is always folded into
(-pi, pi]; a full-turn target cannot be expressed in it at all.

The measured failure is exactly in that blind spot. Over 603 validation cycles,
the model is smooth and monotone (96.5% of adjacent frames advance, total
backward motion 0.257 rad), and cycles that pass accumulate 6.30 rad of forward
progression against a 2*pi target. Cycles that fail accumulate 9.04 rad, and they
are the long ones: mean period 122 frames against 64 for passing cycles.
`corr(period, net error) = +0.640` and the predicted rate runs 1.99x too fast in
the slowest true-rate quintile.

So this arm trains the gate's own criterion, as a continuous relaxation:

    L_cycle = mean over sampled training cycles of ((net_pred - net_ref) / (2*pi))**2

where `net_pred` is the sum of the model's own wrapped adjacent phase increments
along the cycle, i.e. the same unwrapping the gate performs. The error is read in
turns for two reasons: 2*pi is the target itself rather than a tuned constant, so
the term stays in the same numeric range as the point loss; and errors of a whole
turn -- the failure class this arm exists to address -- keep a useful gradient,
which a saturating surrogate would remove.

Everything else is imported unchanged from the pinned v2.1b recipe: the model,
the context windows, the seed, the epoch count, the optimizer, and the complete
existing loss (point + multi-scale increment + reverse + jump). The only
difference from v2.1b is the added term, so any change is attributable to it.

Pre-registered before the run
-----------------------------
    cycle_net_weight = 1.0, cycles_per_batch = 4, error scale = 2*pi
    seed = 20261020, epochs = 30, batch = 512, hidden = 64 (unchanged)
    no early stopping, no threshold change, no test split read
    selection: identical rule and identical gate to v2.1b

Scope, stated before the result is known
----------------------------------------
This is not expected to open the gate. Even if every one of the 45 whole-turn
slips is repaired, validation macro reaches 463/603 = 0.768, below the 0.90 bar,
and the worst condition sits at 0.52 against a 0.80 requirement. The question
this arm answers is narrower and answerable: does aligning the training objective
with the gate criterion move the gate criterion at all?
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402  (pinned, unchanged)
    BATCH_SIZE,
    CONTEXT_FRAMES,
    DATA_PATH,
    DEVICE,
    DROPOUT,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN,
    LEARNING_RATE,
    POINT_ERROR_CAP_RAD,
    RIDGE_FLOOR_MACRO,
    SEED,
    STOP_THRESHOLD_MACRO,
    V4_CALIBRATION_PATH,
    WEIGHT_DECAY,
    PairDataset,
    build_context_windows,
    evaluate,
    set_seed,
    sha256_file,
    split_arrays,
)
from src.tasks.velocity.phasor.encoder_v2 import predict_phase_v2  # noqa: E402
from src.tasks.velocity.phasor.encoder_v2_1 import (  # noqa: E402
    CYCLE_WEIGHT,
    LAG1_RATE_WEIGHT,
    MULTISCALE_WEIGHT,
    NET_TOLERANCE_RAD,
    build_multiscale_pairs,
)
from src.tasks.velocity.phasor.encoder_v2_1b import (  # noqa: E402
    PhaseEncoderV21B,
    phase_v21b_loss,
)

OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"

#: One-epoch wiring check only; writes to its own paths and is deleted after use.
SMOKE = os.environ.get("G1_CYCLESUP_SMOKE") == "1"
EPOCH_COUNT = 1 if SMOKE else EPOCHS
REPORT_PATH = OUT_DIR / ("v21c_cyclesup_smoke.json" if SMOKE else "v21c_cyclesup_epoch_sweep.json")
PREDICTIONS_PATH = OUT_DIR / ("v21c_cyclesup_smoke_predictions.npz" if SMOKE else "v21c_cyclesup_epoch_predictions.npz")

#: Pre-registered before the run; not tuned on any gate result.
CYCLE_NET_WEIGHT = 1.0
CYCLE_NET_TOLERANCE = NET_TOLERANCE_RAD
#: Error is measured in turns. 2*pi is the target itself, not a tuned scale: the
#: term reads "squared whole-cycle drift". A squared error normalised by the
#: gate tolerance instead would average ~29 at epoch 1 against a point loss of
#: ~0.28, and a saturating surrogate would give the whole-turn slips -- the
#: failures this arm exists to fix -- essentially zero gradient.
CYCLE_NET_SCALE = 2.0 * np.pi
#: Cycles sampled per optimisation step. The pair loop takes ~589 steps per
#: epoch, so a single per-epoch step on the cycle term would be outvoted ~589:1
#: and could only ever return a false negative.
CYCLES_PER_BATCH = 4


def build_cycle_indexes(split: dict[str, np.ndarray], history_valid: np.ndarray) -> list[np.ndarray]:
    """Metric frames of every cycle, matching the gate's convention exactly.

    The gate builds `[touchdown_k, touchdown_{k+1}]` and then drops the terminal
    sample, so the scored interval is `[touchdown_k, touchdown_{k+1})`. Cycles
    with a frame gap, a non-positive period, fewer than three metric samples, or
    any frame without usable history are dropped here for the same reasons the
    gate would mark them unscorable.
    """
    sequences = np.asarray(split["sequence"])
    frames = np.asarray(split["frame"]).astype(np.int64)
    touchdowns = np.asarray(split["left_touchdown"]).astype(bool)
    cycles: list[np.ndarray] = []
    for name in sorted({str(value) for value in sequences}):
        rows = np.flatnonzero(np.asarray([str(value) for value in sequences]) == name)
        rows = rows[np.argsort(frames[rows])]
        frame_to_row = {int(frames[row]): int(row) for row in rows}
        events = [int(row) for row in rows if touchdowns[row]]
        for start_row, end_row in zip(events[:-1], events[1:]):
            start_frame = int(frames[start_row])
            end_frame = int(frames[end_row])
            if end_frame <= start_frame:
                continue
            expected = range(start_frame, end_frame)  # terminal sample excluded
            metric = [frame_to_row.get(frame) for frame in expected]
            if any(row is None for row in metric):
                continue
            index = np.asarray(metric, dtype=np.int64)
            if len(index) < 3 or not bool(np.all(history_valid[index])):
                continue
            cycles.append(index)
    return cycles


def sample_cycles(cycles: list[np.ndarray], count: int, generator: torch.Generator,
                  device: torch.device, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw `count` cycles and pad them to the widest one drawn.

    Padding per draw rather than once globally keeps the extra forward cost near
    the sampled length; a single global pad would be 3581 wide because one
    training cycle between two touchdowns spans that many frames.
    """
    picked = torch.randint(0, len(cycles), (count,), generator=generator).tolist()
    width = max(len(cycles[index]) for index in picked)
    padded = np.zeros((count, width), dtype=np.int64)
    mask = np.zeros((count, width), dtype=np.float32)
    for position, index in enumerate(picked):
        cycle = cycles[index]
        padded[position, :len(cycle)] = cycle
        mask[position, :len(cycle)] = 1.0
    target = targets[picked]
    return (torch.from_numpy(padded).to(device), torch.from_numpy(mask).to(device), target.to(device))


def cycle_net_reference(cycles: list[np.ndarray], reference_rad: np.ndarray) -> torch.Tensor:
    """The label's own net progression over each cycle, under the gate's unwrap."""
    values = []
    for cycle in cycles:
        unwrapped = np.unwrap(reference_rad[cycle])
        values.append(float(unwrapped[-1] - unwrapped[0]))
    return torch.tensor(values, dtype=torch.float32)


def cycle_net_prediction(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of the model's own wrapped adjacent increments along each cycle.

    `prediction` is already gathered per cycle, shape [cycles, width, 2]. This is
    the differentiable form of what the gate computes with `np.unwrap`: the gate
    takes the difference between the unwrapped endpoints, which equals the sum of
    the wrapped adjacent differences along the path.
    """
    theta = torch.atan2(prediction[..., 0], prediction[..., 1])
    delta = theta[:, 1:] - theta[:, :-1]
    delta = delta - (2.0 * np.pi) * torch.round(delta / (2.0 * np.pi))
    return (delta * mask[:, 1:]).sum(dim=-1)


def net_statistics(report: dict[str, Any]) -> dict[str, Any]:
    """Net-progression anatomy of a validation report, independent of the pass rule.

    Reported alongside macro because macro is a thresholded view: two arms can
    share a completion rate while failing on different parts of the net-error
    distribution, and only the distribution says whether the over-advance on long
    cycles actually moved.
    """
    records = [row for row in report["cycle_records"] if row["evaluable"]]
    error = np.asarray([row["encoder_net_progression_rad"] - 2.0 * np.pi for row in records])
    periods = np.asarray([row["period_frames"] for row in records], dtype=float)
    slip = np.abs(np.abs(error) - 2.0 * np.pi) < 1.0
    statistics: dict[str, Any] = {
        "n_cycles": int(len(records)),
        "mean_net_error": float(error.mean()),
        "sd_net_error": float(error.std()),
        "share_within_tolerance": float(np.mean(np.abs(error) <= CYCLE_NET_TOLERANCE)),
        "whole_turn_slip_count": int(slip.sum()),
        "mean_period_of_slip": float(periods[slip].mean()) if slip.any() else 0.0,
    }
    for name, row in report["period_length_strata"].items():
        mask = np.zeros(len(records), dtype=bool)
        for position, record in enumerate(records):
            mask[position] = record["period_length_stratum"] == name
        if mask.any():
            statistics[f"mean_net_error_{name}"] = float(error[mask].mean())
    return statistics


def main() -> int:
    if REPORT_PATH.exists():
        raise FileExistsError(f"{REPORT_PATH.relative_to(ROOT)} exists; preserve it and use a new run")

    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_windows_np, train_history = build_context_windows(train, CONTEXT_FRAMES)
    validation_windows_np, validation_history = build_context_windows(validation, CONTEXT_FRAMES)
    train_windows = torch.from_numpy(train_windows_np).to(DEVICE)
    validation_windows = torch.from_numpy(validation_windows_np).to(DEVICE)

    train_cycles = build_cycle_indexes(train, train_history)
    train_phase = (train["y"].astype(np.float64) * (2.0 * np.pi)).astype(np.float32)
    cycle_targets = cycle_net_reference(train_cycles, train_phase.astype(np.float64))
    cycle_generator = torch.Generator().manual_seed(SEED)

    anchor, target, lag, kind, weights = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool),
    )
    multiplier = np.where(kind == 0, LAG1_RATE_WEIGHT, np.where(kind == 2, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
    base_weights = weights / multiplier
    phase_tensor = torch.from_numpy(train_phase).to(DEVICE)

    dataset = PairDataset(anchor, target, weights, base_weights, kind)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )

    reference_validation = validation["y"].astype(np.float64) * (2.0 * np.pi)
    base = {
        "sequence": validation["sequence"],
        "frame": validation["frame"],
        "touchdowns": validation["left_touchdown"].astype(bool),
        "history_valid": validation_history,
        "conditions": sorted(np.unique(validation["sequence"]).tolist()),
    }

    set_seed(SEED)
    model = PhaseEncoderV21B(context_frames=CONTEXT_FRAMES, hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"training cycles: {len(train_cycles)}  pairs: {len(anchor)}  "
          f"cycles per step: {CYCLES_PER_BATCH}  steps/epoch: {len(loader)}", flush=True)

    epochs: list[dict[str, Any]] = []
    predictions_store: dict[str, np.ndarray] = {}
    for epoch in range(1, EPOCH_COUNT + 1):
        model.train()
        keys = ("loss", "point_loss", "increment_loss", "cycle_net_loss",
                "negative_progress_penalty", "jump_penalty")
        accumulators: dict[str, list[float]] = {key: [] for key in keys}
        for index_anchor, index_target, weight, base_weight, batch_kind in loader:
            index_anchor = index_anchor.to(DEVICE, non_blocking=True)
            index_target = index_target.to(DEVICE, non_blocking=True)
            predicted_anchor = predict_phase_v2(model, train_windows[index_anchor])
            predicted_target = predict_phase_v2(model, train_windows[index_target])
            losses = phase_v21b_loss(
                predicted_anchor, predicted_target,
                phase_tensor[index_anchor], phase_tensor[index_target],
                row_weights=weight.to(DEVICE), base_weights=base_weight.to(DEVICE),
                kind=batch_kind.to(DEVICE),
            )

            # Gate-aligned term, evaluated on the same step as everything else so
            # it is not outvoted by the pair loop's ~589 steps per epoch.
            cycle_padded, cycle_mask, cycle_target = sample_cycles(
                train_cycles, CYCLES_PER_BATCH, cycle_generator, DEVICE, cycle_targets,
            )
            cycle_windows = train_windows[cycle_padded.reshape(-1)]
            cycle_prediction = predict_phase_v2(model, cycle_windows).reshape(
                cycle_padded.shape[0], cycle_padded.shape[1], 2,
            )
            net_pred = cycle_net_prediction(cycle_prediction, cycle_mask)
            cycle_loss = (((net_pred - cycle_target) / CYCLE_NET_SCALE) ** 2).mean()

            total = losses["loss"] + CYCLE_NET_WEIGHT * cycle_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            accumulators["loss"].append(float(total.detach().cpu()))
            accumulators["cycle_net_loss"].append(float(cycle_loss.detach().cpu()))
            for key in ("point_loss", "increment_loss",
                        "negative_progress_penalty", "jump_penalty"):
                accumulators[key].append(float(losses[key].detach().cpu()))

        prediction, report = evaluate(model, validation_windows, reference_validation, base, period_cuts)
        predictions_store[f"epoch_{epoch:02d}"] = prediction.astype(np.float32)
        record = {
            "epoch": epoch,
            "train": {key: float(np.mean(values)) for key, values in accumulators.items()},
            "macro_completion": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
            "net_statistics": net_statistics(report),
            "freeze_eligible": report["encoder_freeze_eligible"],
            "point_error_within_cap": bool(report["condition_macro_point_error_rad"] <= POINT_ERROR_CAP_RAD),
        }
        epochs.append(record)
        print(f"ep {epoch:2d} loss {record['train']['loss']:.4f} cyc {record['train']['cycle_net_loss']:.4f} "
              f"point {record['train']['point_loss']:.4f} | val macro {record['macro_completion']:.4f} "
              f"pt {record['condition_macro_point_error_rad']:.3f} "
              f"cond {' '.join(f'{v:.2f}' for v in record['condition_completion'].values())} "
              f"strata {' '.join(f'{v:.2f}' for v in record['period_length_strata'].values())} "
              f"| net {record['net_statistics']['mean_net_error']:+.2f} "
              f"slip {record['net_statistics']['whole_turn_slip_count']} "
              f"long {record['net_statistics'].get('mean_net_error_long', 0.0):+.2f}", flush=True)
        if len(epochs) == 1:
            np.savez_compressed(PREDICTIONS_PATH, **predictions_store)

    eligible = [r for r in epochs if r["freeze_eligible"] and r["point_error_within_cap"]]
    ranked = sorted(eligible, key=lambda r: (-r["macro_completion"], r["condition_macro_point_error_rad"], r["epoch"]))
    best_by_macro = max(epochs, key=lambda r: r["macro_completion"])
    selected = ranked[0] if ranked else None

    if selected is not None:
        status = "READY_SELECTED_CANDIDATE"
    elif best_by_macro["macro_completion"] <= STOP_THRESHOLD_MACRO:
        status = "STOP_BAR_UNATTAINABLE_ON_THIS_DATA_AND_TARGET"
    else:
        status = "BLOCKED_NO_EPOCH_PASSED_V4_2_WITH_POINT_CAP"

    payload = {
        "schema": "g1.phase_encoder_v21c_cyclesup_validation.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "arm": "lever_2_gate_aligned_cycle_net_supervision",
        "source_split": "train_only_fit_validation_only_selection",
        "test_split_read": False,
        "preregistered": {
            "basis": "v2.1b recipe with one added term",
            "cycle_net_weight": CYCLE_NET_WEIGHT,
            "cycles_per_batch": CYCLES_PER_BATCH,
            "cycle_net_scale_rad": CYCLE_NET_SCALE,
            "cycle_net_tolerance_rad": CYCLE_NET_TOLERANCE,
            "cycle_net_definition": "((sum of wrapped adjacent increments along a sampled training cycle) - (label net)) / (2*pi), squared, mean over the cycles drawn in that step; added to every optimisation step",
            "design_note": "an earlier draft applied this term once per epoch; that was stopped before any epoch output existed because ~589 pair steps per epoch would outvote it ~589:1 and could only return a false negative",
            "context_frames": CONTEXT_FRAMES, "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "hidden": HIDDEN,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "grad_clip": GRAD_CLIP,
            "point_error_cap_rad": POINT_ERROR_CAP_RAD,
            "selection": "freeze_eligible AND point_error <= cap; rank macro desc, point asc, epoch asc",
            "stop_rule": f"best macro <= {STOP_THRESHOLD_MACRO} => bar unattainable on this data/target",
            "ridge_floor_macro_v4_2": RIDGE_FLOOR_MACRO,
        },
        "dataset_sha256": sha256_file(DATA_PATH),
        "train_cycle_count": len(train_cycles),
        "pair_count": int(len(anchor)),
        "status": status,
        "selected_epoch": selected["epoch"] if selected else None,
        "best_epoch_by_macro": best_by_macro["epoch"],
        "best_macro_completion": best_by_macro["macro_completion"],
        "beats_ridge_floor": bool(best_by_macro["macro_completion"] > RIDGE_FLOOR_MACRO),
        "checkpoint_frozen": False,
        "ppo_blocked": True,
        "epochs": epochs,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(PREDICTIONS_PATH, **predictions_store)

    print(json.dumps({
        "status": status,
        "best_epoch_by_macro": payload["best_epoch_by_macro"],
        "best_macro_completion": payload["best_macro_completion"],
        "beats_ridge_floor": payload["beats_ridge_floor"],
        "outputs": [str(REPORT_PATH.relative_to(ROOT)), str(PREDICTIONS_PATH.relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
