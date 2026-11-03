"""Is the 0.70 ceiling caused by the change of subject?

Pre-registered in
``docs/superpowers/plans/g1-phase-within-subject-transfer-preregistration.md``.

Seven arms derived from the v2.1b recipe all land at 0.628-0.703 macro, but they
share one dataset, one feature pipeline, one normalisation, one split and one
scorer, so their agreement cannot distinguish "the task is hard" from "the shared
pipeline is wrong". The one shared variable that can be removed cheaply is the
subject change: they train on subjects 3/4/5 and score on subject 1.

This script runs a 2 x 2 over subject 1's four sequences:

    fold A   validation walk3+walk4   within-train walk1+walk2
    fold B   validation walk1+walk2   within-train walk3+walk4

each with two arms:

    within  train on that fold's subject-1 sequences
    cross   train on two sequences from the training subjects, chosen by a fixed
            rule (closest total touchdown count, ties by row count)

The cross arm exists only to control training-set size: the within arm sees about
a third of v2.1b's rows, so a worse within result would otherwise be explicable
by less data rather than by the subject change.

Everything else is the v2.1e control arm, which reproduced v2.1b bit-for-bit
over all 30 epochs: same model, loss, weights, seed, optimiser, epochs, stored
feature representation, frozen period cuts and frozen gate thresholds.

Validation data only for scoring; the test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    build_context_windows,
    split_arrays,
)
from scripts.train_g1_phase_encoder_v2_1e import (  # noqa: E402
    BATCH_SIZE,
    DATA_PATH,
    DEVICE,
    DROPOUT,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN,
    LEARNING_RATE,
    OUT_DIR,
    POINT_ERROR_CAP_RAD,
    SEED,
    STOP_THRESHOLD_MACRO,
    V21B_BASELINE_MACRO,
    V21B_BASELINE_POINT,
    V4_CALIBRATION_PATH,
    WEIGHT_DECAY,
    PairDataset,
    evaluate,
    net_turn_summary,
    rescale_cycle_rows,
    set_seed,
    sha256_file,
)
from src.tasks.velocity.phasor.encoder_v2 import predict_phase_v2  # noqa: E402
from src.tasks.velocity.phasor.encoder_v2_1 import (  # noqa: E402
    CYCLE_WEIGHT,
    JUMP_WEIGHT,
    KIND_CYCLE,
    KIND_LAG1,
    LAG1_RATE_WEIGHT,
    MULTISCALE_WEIGHT,
    MULTI_SCALE_LAGS,
    POINT_WEIGHT,
    REVERSE_WEIGHT,
    build_multiscale_pairs,
)
from src.tasks.velocity.phasor.encoder_v2_1b import (  # noqa: E402
    PhaseEncoderV21B,
    phase_v21b_loss,
)

SUBJECT = "subject1"
VALIDATION_SEQUENCES = (
    "walk1_subject1",
    "walk2_subject1",
    "walk3_subject1",
    "walk4_subject1",
)
TRAINING_POOL = (
    "walk1_subject5",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
)
FOLDS = {
    "A": {"validation": ("walk3_subject1", "walk4_subject1"),
          "within": ("walk1_subject1", "walk2_subject1")},
    "B": {"validation": ("walk1_subject1", "walk2_subject1"),
          "within": ("walk3_subject1", "walk4_subject1")},
}


def subset(split: dict[str, np.ndarray], sequences: tuple[str, ...]) -> dict[str, np.ndarray]:
    mask = np.isin(split["sequence"], np.asarray(sequences))
    if not mask.any():
        raise ValueError(f"no rows for sequences {sequences}")
    return {key: value[mask] for key, value in split.items()}


def touchdown_total(split: dict[str, np.ndarray]) -> int:
    return int(np.asarray(split["left_touchdown"]).astype(bool).sum())


def choose_cross_pair(train: dict[str, np.ndarray], target: int) -> tuple[tuple[str, str], dict[str, int]]:
    """Fixed rule: closest total touchdown count, ties broken by total row count."""
    per_sequence_td: dict[str, int] = {}
    per_sequence_rows: dict[str, int] = {}
    for name in TRAINING_POOL:
        mask = train["sequence"] == name
        per_sequence_td[name] = int(train["left_touchdown"][mask].astype(bool).sum())
        per_sequence_rows[name] = int(mask.sum())
    best: tuple[tuple[int, int], tuple[str, str]] | None = None
    for a, b in combinations(sorted(TRAINING_POOL), 2):
        key = (abs(per_sequence_td[a] + per_sequence_td[b] - target),
               per_sequence_rows[a] + per_sequence_rows[b])
        if best is None or key < best[0]:
            best = (key, (a, b))
    assert best is not None
    return best[1], {
        "target_touchdowns": int(target),
        "chosen_touchdowns": int(sum(per_sequence_td[s] for s in best[1])),
        "chosen_rows": int(sum(per_sequence_rows[s] for s in best[1])),
        "absolute_touchdown_mismatch": int(best[0][0]),
        "rule": "closest total left-foot touchdown count; ties by total row count",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLDS), required=True)
    parser.add_argument("--arm", choices=("within", "cross"), required=True)
    args = parser.parse_args()

    report_path = OUT_DIR / f"v21f_{args.arm}_fold{args.fold}_epoch_sweep.json"
    predictions_path = OUT_DIR / f"v21f_{args.arm}_fold{args.fold}_epoch_predictions.npz"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError(f"{report_path.name} exists; preserve it and use a new label")

    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    subject1 = subset(split_arrays(archive, "validation"), VALIDATION_SEQUENCES)
    validation = subset(subject1, FOLDS[args.fold]["validation"])
    within = subset(subject1, FOLDS[args.fold]["within"])

    cross_note = None
    if args.arm == "within":
        train = within
    else:
        training_only = split_arrays(archive, "train")
        pair, cross_note = choose_cross_pair(training_only, touchdown_total(within))
        train = subset(training_only, pair)
        cross_note["chosen_sequences"] = list(pair)

    print(f"fold {args.fold} arm {args.arm}: train rows {len(train['frame'])} "
          f"({touchdown_total(train)} touchdowns) | validation rows {len(validation['frame'])} "
          f"({touchdown_total(validation)} touchdowns)", flush=True)

    train_windows_np, _ = build_context_windows(train, CONTEXT_FRAMES)
    validation_windows_np, validation_history = build_context_windows(validation, CONTEXT_FRAMES)
    train_windows = torch.from_numpy(train_windows_np).to(DEVICE)
    validation_windows = torch.from_numpy(validation_windows_np).to(DEVICE)

    anchor, target, _lag, kind, raw_weight = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool),
    )
    multiplier = np.where(kind == KIND_LAG1, LAG1_RATE_WEIGHT,
                          np.where(kind == KIND_CYCLE, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
    base_weights = raw_weight / multiplier
    # Exactly what v2.1e's control arm does at CYCLE_WEIGHT: the function returns
    # a float32 copy there, so the weight tensors reach the loss as float32 and the
    # run stays numerically identical to the v2.1b recipe rather than promoting.
    row_weights = rescale_cycle_rows(raw_weight, kind, CYCLE_WEIGHT)

    train_phase = (train["y"].astype(np.float64) * (2.0 * np.pi)).astype(np.float32)
    phase_tensor = torch.from_numpy(train_phase).to(DEVICE)

    dataset = PairDataset(anchor, target, row_weights, base_weights, kind)
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

    epochs: list[dict[str, Any]] = []
    predictions_store: dict[str, np.ndarray] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        accumulators: dict[str, list[float]] = {key: [] for key in
                                                ("loss", "point_loss", "increment_loss",
                                                 "negative_progress_penalty", "jump_penalty")}
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
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            for key in accumulators:
                accumulators[key].append(float(losses[key].detach().cpu()))

        prediction, report = evaluate(model, validation_windows, reference_validation, base, period_cuts)
        predictions_store[f"epoch_{epoch:02d}"] = prediction.astype(np.float32)
        record = {
            "epoch": epoch,
            "train": {key: float(np.mean(values)) for key, values in accumulators.items()},
            "macro_completion": report["macro_cycle_completion_fraction"],
            "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
            "condition_completion": {k: v["completion_fraction"]
                                     for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"]
                                     for k, v in report["period_length_strata"].items()},
            "freeze_eligible": report["encoder_freeze_eligible"],
            "point_error_within_cap": bool(
                report["condition_macro_point_error_rad"] <= POINT_ERROR_CAP_RAD),
            "net_turn": net_turn_summary(report),
        }
        epochs.append(record)
        print(f"ep {epoch:2d} loss {record['train']['loss']:.4f} "
              f"point {record['train']['point_loss']:.4f} "
              f"inc {record['train']['increment_loss']:.4f} | "
              f"val macro {record['macro_completion']:.4f} "
              f"pt {record['condition_macro_point_error_rad']:.3f} "
              f"net_sd {record['net_turn'].get('net_error_sd', float('nan')):.2f} "
              f"within {record['net_turn'].get('share_within_tolerance', float('nan')):.3f} "
              f"slips {record['net_turn'].get('whole_turn_slip_count', -1)}", flush=True)

    eligible = [r for r in epochs if r["freeze_eligible"] and r["point_error_within_cap"]]
    ranked = sorted(eligible, key=lambda r: (-r["macro_completion"],
                                             r["condition_macro_point_error_rad"], r["epoch"]))
    best_by_macro = max(epochs, key=lambda r: r["macro_completion"])
    selected = ranked[0] if ranked else None

    if selected is not None:
        status = "READY_SELECTED_CANDIDATE"
    elif best_by_macro["macro_completion"] <= STOP_THRESHOLD_MACRO:
        status = "STOP_BAR_UNATTAINABLE_ON_THIS_DATA_AND_TARGET"
    else:
        status = "BLOCKED_NO_EPOCH_PASSED_V4_2_WITH_POINT_CAP"

    payload = {
        "schema": "g1.phase_encoder_v21f_within_subject.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "source_split": "validation_subject_only; test split never loaded",
        "test_split_read": False,
        "preregistration": "docs/superpowers/plans/g1-phase-within-subject-transfer-preregistration.md",
        "fold": args.fold,
        "arm": args.arm,
        "subject": SUBJECT,
        "sequences": {
            "train": sorted(str(x) for x in np.unique(train["sequence"])),
            "validation": sorted(str(x) for x in np.unique(validation["sequence"])),
        },
        "rows": {"train": int(len(train["frame"])), "validation": int(len(validation["frame"]))},
        "touchdowns": {"train": touchdown_total(train), "validation": touchdown_total(validation)},
        "cross_arm_selection": cross_note,
        "preregistered": {
            "context_frames": CONTEXT_FRAMES, "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "hidden": HIDDEN,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "grad_clip": GRAD_CLIP,
            "multi_scale_lags": list(MULTI_SCALE_LAGS),
            "weights": {"point": POINT_WEIGHT, "reverse": REVERSE_WEIGHT,
                        "jump": JUMP_WEIGHT, "cycle": CYCLE_WEIGHT},
            "only_changed_variable": "the train/validation partition of subject 1's sequences",
            "point_error_cap_rad": POINT_ERROR_CAP_RAD,
            "selection": "freeze_eligible AND point_error <= cap; rank macro desc, point asc, epoch asc",
            "stop_rule": f"best macro <= {STOP_THRESHOLD_MACRO}",
        },
        "dataset_sha256": sha256_file(DATA_PATH),
        "running_source_sha256": {
            "scripts/train_g1_phase_encoder_v2_1f_within.py": sha256_file(Path(__file__).resolve()),
            "scripts/train_g1_phase_encoder_v2_1e.py": sha256_file(
                ROOT / "scripts/train_g1_phase_encoder_v2_1e.py"),
            "scripts/train_g1_phase_encoder_v2_1b.py": sha256_file(
                ROOT / "scripts/train_g1_phase_encoder_v2_1b.py"),
            "src/tasks/velocity/phasor/encoder_v2_1.py": sha256_file(
                ROOT / "src/tasks/velocity/phasor/encoder_v2_1.py"),
            "src/tasks/velocity/phasor/encoder_v2_1b.py": sha256_file(
                ROOT / "src/tasks/velocity/phasor/encoder_v2_1b.py"),
            "src/tasks/velocity/phasor/phase_gate_v4_2.py": sha256_file(
                ROOT / "src/tasks/velocity/phasor/phase_gate_v4_2.py"),
        },
        "pair_count": int(len(anchor)),
        "pair_kind_rows": {"lag1": int((kind == KIND_LAG1).sum()),
                           "scale": int(((kind != KIND_LAG1) & (kind != KIND_CYCLE)).sum()),
                           "cycle": int((kind == KIND_CYCLE).sum())},
        "reference_for_comparison": {
            "v21b_cross_subject_macro": V21B_BASELINE_MACRO,
            "v21b_cross_subject_point_error_rad": V21B_BASELINE_POINT,
        },
        "status": status,
        "selected_epoch": selected["epoch"] if selected else None,
        "best_epoch_by_macro": best_by_macro["epoch"],
        "best_macro_completion": best_by_macro["macro_completion"],
        "best_point_error_rad": best_by_macro["condition_macro_point_error_rad"],
        "checkpoint_frozen": False,
        "ppo_blocked": True,
        "epochs": epochs,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, **predictions_store)
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "fold": args.fold, "arm": args.arm, "status": status,
        "best_epoch_by_macro": payload["best_epoch_by_macro"],
        "best_macro_completion": payload["best_macro_completion"],
        "best_point_error_rad": payload["best_point_error_rad"],
        "outputs": [str(report_path.relative_to(ROOT)), str(predictions_path.relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
