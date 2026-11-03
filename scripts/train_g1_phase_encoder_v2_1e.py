"""Train the pre-registered v2.1e sweep: how much should the cycle integral count?

Every v2.1b-derived run supervises the gait phase with 301,551 anchor-target
pairs, of which 948 -- one per complete cycle -- span a whole cycle. Those 948
rows carry 0.70% of the increment-term weight; the other 99.30% is split across
fixed lags of 1/4/8/16/32 frames, and the median training cycle is 59 frames, so
no fixed lag can see a cycle boundary on both sides.

Gate v4.2 grades exactly that integral: a cycle counts only if the unwrapped
predicted phase advances one turn within pi/6. On the best v2.1b epoch 69.3% of
cycles already do, the shortfall splits 113 too-small / 72 too-large, and the
net error correlates with the cycle period at r = 0.64. The typical cycle is
right; the integral tails are not.

This script varies one number, CYCLE_WEIGHT, and nothing else. It reproduces the
v2.1b training loop line for line -- same pairs, lags, model, loss, seed,
optimizer, 30 epochs, validation-only selection -- and rescales only the
KIND_CYCLE rows' increment-term weight. The point term is unaffected by
construction: it is weighted by row weight divided by the row's kind multiplier,
which is invariant to that multiplier.

    arm control   cycle weight   2.0  -> 0.70% of increment weight (v2.1b verbatim)
    arm w20       cycle weight  20.0  -> 6.60%
    arm w60       cycle weight  60.0  -> 17.49%

No threshold moves, no early stopping, no checkpoint is frozen unless an epoch
passes the full v4.2 gate. Train is fit only; selection reads validation only;
the test split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_g1_phase_encoder_v2_1b import (  # noqa: E402
    CONTEXT_FRAMES,
    build_context_windows,
    split_arrays,
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
    NET_TOLERANCE_RAD,
    REVERSE_WEIGHT,
    build_multiscale_pairs,
)
from src.tasks.velocity.phasor.encoder_v2_1b import (  # noqa: E402
    DROPOUT,
    GRAD_CLIP,
    LEARNING_RATE,
    POINT_WEIGHT,
    WEIGHT_DECAY,
    PhaseEncoderV21B,
    phase_v21b_loss,
)
from src.tasks.velocity.phasor.phase_gate_v4_2 import score_phase_cycles_v4_2  # noqa: E402

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_CALIBRATION_PATH = ROOT / "results" / "g1_phase_gate_v4" / "calibration_v4_amendment1.json"
OUT_DIR = ROOT / "results" / "g1_phase_gate_v4_2"

SEED = 20261020
EPOCHS = 30
BATCH_SIZE = 512
HIDDEN = 64

POINT_ERROR_CAP_RAD = NET_TOLERANCE_RAD
STOP_THRESHOLD_MACRO = 0.65

#: Frozen v2.1b epoch-1 figures; the control arm must reproduce them.
V21B_BASELINE_MACRO = 0.6962456093142473
V21B_BASELINE_POINT = 0.7716245022746507
V21B_BASELINE_PAIRS = 301551
WINDOW_TOLERANCE = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, anchor: np.ndarray, target: np.ndarray, weight: np.ndarray,
                 base_weight: np.ndarray, kind: np.ndarray) -> None:
        self.anchor = torch.from_numpy(anchor)
        self.target = torch.from_numpy(target)
        self.weight = torch.from_numpy(weight)
        self.base_weight = torch.from_numpy(base_weight)
        self.kind = torch.from_numpy(kind)

    def __len__(self) -> int:
        return len(self.anchor)

    def __getitem__(self, index: int):
        return (self.anchor[index], self.target[index], self.weight[index],
                self.base_weight[index], self.kind[index])


def rescale_cycle_rows(weight: np.ndarray, kind: np.ndarray, cycle_weight: float) -> np.ndarray:
    """Multiply only the KIND_CYCLE increment weights by cycle_weight / CYCLE_WEIGHT.

    ``build_multiscale_pairs`` bakes the frozen multipliers in, so this restores
    the intent without editing the frozen module. At cycle_weight == CYCLE_WEIGHT
    the returned array equals the input exactly.
    """
    rescaled = weight.astype(np.float32, copy=True)
    if cycle_weight == CYCLE_WEIGHT:
        return rescaled
    factor = np.float32(cycle_weight / CYCLE_WEIGHT)
    rescaled[kind == KIND_CYCLE] *= factor
    return rescaled


def weight_shares(weight: np.ndarray, kind: np.ndarray) -> dict[str, float]:
    total = float(weight.sum())
    return {
        "lag1": float(weight[kind == KIND_LAG1].sum()) / total,
        "scale": float(weight[(kind != KIND_LAG1) & (kind != KIND_CYCLE)].sum()) / total,
        "cycle": float(weight[kind == KIND_CYCLE].sum()) / total,
    }


def net_turn_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Describe the per-cycle net turn, which is the quantity the gate grades.

    ``error`` is the encoder's net progression minus one turn. Two tail
    statistics are reported and they are **not** interchangeable:

    ``whole_turn_slip_count``
        the project's published definition (``scripts/summarise_phase_arm_epochs.py``):
        ``| |error| - 2*pi | < 1``, i.e. the phase lost or gained a whole turn.
        This is the one the pre-registered mechanism criterion refers to.
    ``far_tail_count``
        ``|error| > 3``, a broader count of cycles far from one turn, including
        halfway slips. Reported for shape only; never compare it to a published
        slip count.
    """
    records = [r for r in report.get("cycle_records", []) if r.get("evaluable", True)]
    if not records:
        return {"cycle_count": 0}
    error = np.asarray([r["encoder_net_progression_rad"] - 2.0 * np.pi for r in records], dtype=np.float64)
    period = np.asarray([r["period_frames"] for r in records], dtype=np.float64)
    within = np.abs(error) <= NET_TOLERANCE_RAD
    slips = np.abs(np.abs(error) - 2.0 * np.pi) < 1.0
    correlation = float(np.corrcoef(period, error)[0, 1]) if period.std() > 0.0 else float("nan")
    return {
        "cycle_count": int(len(records)),
        "share_within_tolerance": float(within.mean()),
        "count_too_small": int((error < -NET_TOLERANCE_RAD).sum()),
        "count_too_large": int((error > NET_TOLERANCE_RAD).sum()),
        "whole_turn_slip_count": int(slips.sum()),
        "whole_turn_slip_definition": "| |net-2pi| - 2pi | < 1.0 rad (published)",
        "far_tail_count": int((np.abs(error) > 3.0).sum()),
        "far_tail_definition": "|net-2pi| > 3.0 rad (broad tail; not the published slip count)",
        "net_error_sd": float(error.std()),
        "net_error_median": float(np.median(error)),
        "net_error_q25": float(np.percentile(error, 25)),
        "net_error_q75": float(np.percentile(error, 75)),
        "corr_period_net_error": correlation,
    }


def evaluate(model: torch.nn.Module, windows: torch.Tensor, reference: np.ndarray,
             base: dict[str, Any], period_cuts: tuple[float, float]) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), 2048):
            predictions.append(predict_phase_v2(model, windows[start:start + 2048]).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    report = score_phase_cycles_v4_2(
        prediction, reference, base["sequence"], base["frame"], base["touchdowns"],
        condition_ids=base["sequence"], episode_ids=base["sequence"],
        history_valid=base["history_valid"],
        period_cuts_frames=period_cuts, required_conditions=base["conditions"],
        minimum_cycles_per_condition=10, minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )
    return prediction, report


def strip_cycle_records(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in report.items():
        if key == "cycle_records":
            out["cycle_record_count"] = len(value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-weight", type=float, required=True)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()

    report_path = OUT_DIR / f"v21e_{args.label}_epoch_sweep.json"
    predictions_path = OUT_DIR / f"v21e_{args.label}_epoch_predictions.npz"
    checkpoint_path = OUT_DIR / f"phase_encoder_v21e_{args.label}_candidate.pt"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError(f"{report_path.name} exists; preserve it and use a new label")
    if args.cycle_weight <= 0.0:
        raise ValueError("cycle weight must be positive")

    archive = np.load(DATA_PATH, allow_pickle=False)
    frozen = json.loads(V4_CALIBRATION_PATH.read_text(encoding="utf-8"))
    period_cuts = tuple(float(x) for x in frozen["period_length_cut_points_frames"])

    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_windows_np, _ = build_context_windows(train, CONTEXT_FRAMES)
    validation_windows_np, validation_history = build_context_windows(validation, CONTEXT_FRAMES)
    train_windows = torch.from_numpy(train_windows_np).to(DEVICE)
    validation_windows = torch.from_numpy(validation_windows_np).to(DEVICE)

    anchor, target, lag, kind, raw_weight = build_multiscale_pairs(
        train["sequence"], train["frame"], train["left_touchdown"].astype(bool),
    )
    if len(anchor) != V21B_BASELINE_PAIRS:
        raise ValueError(f"pair count {len(anchor)} != frozen baseline {V21B_BASELINE_PAIRS}")

    multiplier = np.where(kind == KIND_LAG1, LAG1_RATE_WEIGHT,
                          np.where(kind == KIND_CYCLE, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
    base_weights = raw_weight / multiplier  # invariant to the cycle weight
    row_weights = rescale_cycle_rows(raw_weight, kind, args.cycle_weight)
    shares = weight_shares(row_weights, kind)

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
    best_state: dict[str, torch.Tensor] | None = None
    best_record: dict[str, Any] | None = None

    print(f"cycle weight {args.cycle_weight:g}  shares "
          f"lag1 {shares['lag1']:.4f} scale {shares['scale']:.4f} cycle {shares['cycle']:.4f}", flush=True)

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
            "condition_completion": {k: v["completion_fraction"] for k, v in report["condition_completion"].items()},
            "period_length_strata": {k: v["completion_fraction"] for k, v in report["period_length_strata"].items()},
            "freeze_eligible": report["encoder_freeze_eligible"],
            "point_error_within_cap": bool(report["condition_macro_point_error_rad"] <= POINT_ERROR_CAP_RAD),
            "net_turn": net_turn_summary(report),
        }
        epochs.append(record)
        print(f"ep {epoch:2d} loss {record['train']['loss']:.4f} point {record['train']['point_loss']:.4f} "
              f"inc {record['train']['increment_loss']:.4f} | val macro {record['macro_completion']:.4f} "
              f"pt {record['condition_macro_point_error_rad']:.3f} "
              f"net_sd {record['net_turn'].get('net_error_sd', float('nan')):.2f} "
              f"within {record['net_turn'].get('share_within_tolerance', float('nan')):.3f} "
              f"slips {record['net_turn'].get('whole_turn_slip_count', -1)}", flush=True)
        if record["freeze_eligible"] and record["point_error_within_cap"]:
            if best_record is None or (
                -record["macro_completion"], record["condition_macro_point_error_rad"], record["epoch"]
            ) < (
                -best_record["macro_completion"], best_record["condition_macro_point_error_rad"], best_record["epoch"]
            ):
                best_record = record
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

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

    control_delta = {
        "macro_delta_vs_v21b": best_by_macro["macro_completion"] - V21B_BASELINE_MACRO,
        "point_delta_vs_v21b": best_by_macro["condition_macro_point_error_rad"] - V21B_BASELINE_POINT,
    }

    payload = {
        "schema": "g1.phase_encoder_v21e_validation.v1",
        "gate_version": "g1-phase-gate-v4.2",
        "source_split": "train_only_fit_validation_only_selection",
        "test_split_read": False,
        "label": args.label,
        "preregistered": {
            "cycle_weight": args.cycle_weight,
            "context_frames": CONTEXT_FRAMES, "seed": SEED, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "hidden": HIDDEN,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "grad_clip": GRAD_CLIP,
            "multi_scale_lags": list(MULTI_SCALE_LAGS),
            "weights": {"point": POINT_WEIGHT, "reverse": REVERSE_WEIGHT, "jump": JUMP_WEIGHT},
            "only_changed_parameter": "CYCLE_WEIGHT (increment-term weight on the one-row-per-cycle pairs)",
            "point_error_cap_rad": POINT_ERROR_CAP_RAD,
            "selection": "freeze_eligible AND point_error <= cap; rank macro desc, point asc, epoch asc",
            "stop_rule": f"best macro <= {STOP_THRESHOLD_MACRO} => bar unattainable on this data/target",
        },
        "dataset_sha256": sha256_file(DATA_PATH),
        "running_source_sha256": {
            "scripts/train_g1_phase_encoder_v2_1e.py": sha256_file(Path(__file__).resolve()),
            "scripts/train_g1_phase_encoder_v2_1b.py": sha256_file(ROOT / "scripts/train_g1_phase_encoder_v2_1b.py"),
            "src/tasks/velocity/phasor/encoder_v2_1.py": sha256_file(ROOT / "src/tasks/velocity/phasor/encoder_v2_1.py"),
            "src/tasks/velocity/phasor/encoder_v2_1b.py": sha256_file(ROOT / "src/tasks/velocity/phasor/encoder_v2_1b.py"),
            "src/tasks/velocity/phasor/phase_gate_v4_2.py": sha256_file(ROOT / "src/tasks/velocity/phasor/phase_gate_v4_2.py"),
        },
        "pair_count": int(len(anchor)),
        "pair_kind_rows": {"lag1": int((kind == KIND_LAG1).sum()),
                           "scale": int(((kind != KIND_LAG1) & (kind != KIND_CYCLE)).sum()),
                           "cycle": int((kind == KIND_CYCLE).sum())},
        "increment_weight_shares": shares,
        "control_comparison": {
            "v21b_baseline_macro": V21B_BASELINE_MACRO,
            "v21b_baseline_point_error_rad": V21B_BASELINE_POINT,
            "best_macro_this_arm": best_by_macro["macro_completion"],
            "point_error_this_arm": best_by_macro["condition_macro_point_error_rad"],
            **control_delta,
            "reproduced_within_tolerance": bool(
                abs(control_delta["macro_delta_vs_v21b"]) <= WINDOW_TOLERANCE
                and abs(control_delta["point_delta_vs_v21b"]) <= WINDOW_TOLERANCE
            ) if args.cycle_weight == CYCLE_WEIGHT else None,
        },
        "status": status,
        "selected_epoch": selected["epoch"] if selected else None,
        "best_epoch_by_macro": best_by_macro["epoch"],
        "best_macro_completion": best_by_macro["macro_completion"],
        "best_point_error_rad": best_by_macro["condition_macro_point_error_rad"],
        "beats_v21b": bool(best_by_macro["macro_completion"] > V21B_BASELINE_MACRO),
        "checkpoint_frozen": False,
        "ppo_blocked": True,
        "epochs": epochs,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, **predictions_store)
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if selected is not None and best_state is not None:
        torch.save({
            "state_dict": best_state,
            "context_frames": CONTEXT_FRAMES,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "epoch": selected["epoch"],
            "cycle_weight": args.cycle_weight,
            "macro_completion": selected["macro_completion"],
            "condition_macro_point_error_rad": selected["condition_macro_point_error_rad"],
        }, checkpoint_path)
        payload["checkpoint_frozen"] = True
        payload["checkpoint_sha256"] = sha256_file(checkpoint_path)
        report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "label": args.label,
        "cycle_weight": args.cycle_weight,
        "status": status,
        "best_epoch_by_macro": payload["best_epoch_by_macro"],
        "best_macro_completion": payload["best_macro_completion"],
        "best_point_error_rad": payload["best_point_error_rad"],
        "beats_v21b": payload["beats_v21b"],
        "outputs": [str(report_path.relative_to(ROOT)), str(predictions_path.relative_to(ROOT))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
