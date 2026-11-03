"""Train v2 candidates from train data and select only with the frozen v4 gate."""

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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.encoder_v2 import PhaseEncoderV2, phase_v2_loss, predict_phase_v2
from src.tasks.velocity.phasor.phase_gate_v4 import PHASE_GATE_VERSION, score_phase_cycles_v4, select_v4_candidate


DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
V4_DIR = ROOT / "results" / "g1_phase_gate_v4"
CALIBRATION_PATH = V4_DIR / "calibration_v4_amendment1.json"
CONTRACT_PATH = V4_DIR / "gate_contract_v4.json"
CANDIDATE_DIR = V4_DIR / "candidates_v2"
REPORT_PATH = V4_DIR / "candidate_validation_v2.json"
PREDICTIONS_PATH = V4_DIR / "candidate_validation_v2_predictions.npz"
EPOCH_SELECTION_REPORT_PATH = V4_DIR / "candidate_validation_v2_epoch_selection.json"
EPOCH_SELECTION_PREDICTIONS_PATH = V4_DIR / "candidate_validation_v2_epoch_selection_predictions.npz"
FROZEN_PATH = V4_DIR / "phase_encoder_v2_frozen.pt"
SEED = 20261020
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
HIDDEN = 64
CONTEXTS = (10, 30)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_validation_split(archive: Any, split: str) -> dict[str, np.ndarray]:
    """Read one permitted split; test keys are deliberately inaccessible here."""
    if split not in {"train", "validation"}:
        raise ValueError("the test split is forbidden for encoder-v2 fitting or selection")
    names = {
        "x": f"x_{split}",
        "phase_cycles": f"y_{split}_cycles",
        "sequence": f"sequence_{split}",
        "frame": f"frame_{split}",
        "left_touchdown": f"left_touchdown_{split}",
    }
    arrays = {alias: np.asarray(archive[key]) for alias, key in names.items()}
    count = len(arrays["phase_cycles"])
    if arrays["x"].shape != (count, 10, 29):
        raise ValueError(f"{split} pose windows must have shape [N, 10, 29]")
    if any(arrays[key].shape != (count,) for key in ("sequence", "frame", "left_touchdown")):
        raise ValueError(f"{split} labels and metadata must align")
    return arrays


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def checkpoint_epochs(total_epochs: int) -> tuple[int, ...]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    return tuple(range(1, total_epochs + 1))


def build_context_windows(data: dict[str, np.ndarray], context_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Build continuous pose context; 30-frame history starts from stored 10-frame windows."""
    if context_frames == 10:
        return np.asarray(data["x"], dtype=np.float32).copy(), np.ones(len(data["frame"]), dtype=bool)
    if context_frames != 30:
        raise ValueError("the frozen v2 recipe has only 10-frame and 30-frame candidates")
    count = len(data["frame"])
    result = np.empty((count, context_frames, 29), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    groups: dict[str, list[int]] = {}
    for index, sequence in enumerate(data["sequence"]):
        groups.setdefault(str(sequence), []).append(index)
    for indexes in groups.values():
        order = sorted(indexes, key=lambda index: int(data["frame"][index]))
        frames = np.asarray([data["frame"][index] for index in order], dtype=np.int64)
        if len(np.unique(frames)) != len(frames):
            raise ValueError("duplicate frames found while building phase-v2 contexts")
        segment_starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
        segment_ends = np.r_[segment_starts[1:], len(order)]
        for start, end in zip(segment_starts, segment_ends):
            segment_order = order[int(start):int(end)]
            for position, row_index in enumerate(segment_order):
                if position == 0:
                    stored = np.asarray(data["x"][row_index], dtype=np.float32)
                    pad_count = context_frames - stored.shape[0]
                    first_pose = stored[0]
                    window = np.concatenate((np.repeat(first_pose[None, :], pad_count, axis=0), stored), axis=0)
                else:
                    previous = result[segment_order[position - 1]]
                    latest_pose = np.asarray(data["x"][row_index, -1], dtype=np.float32)
                    window = np.concatenate((previous[1:], latest_pose[None, :]), axis=0)
                result[row_index] = window
                valid[row_index] = position + 10 >= context_frames
    return result, valid


def build_cycle_balanced_pairs(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair contiguous train frames and give each complete cycle equal total weight."""
    groups: dict[str, list[int]] = {}
    for index, sequence in enumerate(data["sequence"]):
        groups.setdefault(str(sequence), []).append(index)
    pair_rows: list[tuple[int, int]] = []
    pair_weights: list[float] = []
    for indexes in groups.values():
        order = sorted(indexes, key=lambda index: int(data["frame"][index]))
        frames = np.asarray([data["frame"][index] for index in order], dtype=np.int64)
        by_frame = {int(data["frame"][index]): index for index in order}
        touchdown_frames = sorted(int(data["frame"][index]) for index in order if data["left_touchdown"][index])
        for start, end in zip(touchdown_frames[:-1], touchdown_frames[1:]):
            period = end - start
            if period <= 0:
                continue
            interval = np.arange(start, end + 1, dtype=np.int64)
            if not all(int(frame) in by_frame for frame in interval):
                continue
            weight = 1.0 / float(period)
            for frame in range(start, end):
                pair_rows.append((by_frame[frame], by_frame[frame + 1]))
                pair_weights.append(weight)
    if not pair_rows:
        raise ValueError("training split has no contiguous complete touchdown-cycle pairs")
    pairs = np.asarray(pair_rows, dtype=np.int64)
    return pairs[:, 0], pairs[:, 1], np.asarray(pair_weights, dtype=np.float32)


def _predict(model: PhaseEncoderV2, windows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            pose = torch.from_numpy(windows[start:start + batch_size]).to(DEVICE)
            predictions.append(predict_phase_v2(model, pose).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def fit_candidate(
    train_windows: np.ndarray,
    train_phase: np.ndarray,
    train_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    context_frames: int,
    epoch_callback: Any | None = None,
) -> tuple[PhaseEncoderV2, list[dict[str, float]]]:
    set_reproducible_seed(SEED)
    model = PhaseEncoderV2(context_frames=context_frames, hidden=HIDDEN).to(DEVICE)
    first, second, pair_weights = train_pairs
    dataset = TensorDataset(
        torch.from_numpy(train_windows[first]),
        torch.from_numpy(train_windows[second]),
        torch.from_numpy(train_phase[first]),
        torch.from_numpy(train_phase[second]),
        torch.from_numpy(pair_weights),
    )
    generator = torch.Generator().manual_seed(SEED + context_frames)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_losses: list[dict[str, float]] = []
    for epoch_index, epoch_number in enumerate(checkpoint_epochs(EPOCHS)):
        model.train()
        accumulators: dict[str, list[float]] = {key: [] for key in ("loss", "point_loss", "rate_loss", "negative_progress_penalty", "jump_penalty")}
        for pose_now, pose_next, phase_now, phase_next, weights in loader:
            pose_now = pose_now.to(DEVICE)
            pose_next = pose_next.to(DEVICE)
            phase_now = phase_now.to(DEVICE)
            phase_next = phase_next.to(DEVICE)
            weights = weights.to(DEVICE)
            predicted_now = predict_phase_v2(model, pose_now)
            predicted_next = predict_phase_v2(model, pose_next)
            losses = phase_v2_loss(
                predicted_now,
                predicted_next,
                phase_now,
                phase_next,
                pair_weights=weights,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            for key in accumulators:
                accumulators[key].append(float(losses[key].detach().cpu()))
        epoch_record = {"epoch": epoch_number, **{key: float(np.mean(values)) for key, values in accumulators.items()}}
        epoch_losses.append(epoch_record)
        if epoch_callback is not None:
            epoch_callback(model, epoch_number, epoch_record)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval(), epoch_losses


def _candidate_checkpoint(
    model: PhaseEncoderV2,
    context_frames: int,
    epoch: int,
    dataset_sha256: str,
    calibration_sha256: str,
    train_sequences: list[str],
    validation_sequences: list[str],
) -> dict[str, Any]:
    return {
        "schema": "g1.phase_encoder_checkpoint.v2",
        "gate_version": PHASE_GATE_VERSION,
        "state_dict": model.state_dict(),
        "joint_dim": 29,
        "context_frames": int(context_frames),
        "feature_dim": 59,
        "features": ["train_normalized_joint_position", "adjacent_joint_position_difference", "rms_joint_speed"],
        "feature_builder": "src.tasks.velocity.phasor.encoder_v2.build_phase_v2_features",
        "inference": "src.tasks.velocity.phasor.encoder_v2.predict_phase_v2",
        "hidden": HIDDEN,
        "seed": SEED,
        "epochs": EPOCHS,
        "checkpoint_epoch": int(epoch),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "dataset_sha256": dataset_sha256,
        "calibration_sha256": calibration_sha256,
        "phase_unit": "radians",
        "output_convention": "(sin(phi), cos(phi))",
        "ema_or_filter": None,
    }


def main() -> int:
    for path in (DATA_PATH, CALIBRATION_PATH, CONTRACT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if FROZEN_PATH.exists():
        raise FileExistsError(f"refusing to overwrite a previously frozen checkpoint: {FROZEN_PATH}")
    if EPOCH_SELECTION_REPORT_PATH.exists() or EPOCH_SELECTION_PREDICTIONS_PATH.exists():
        raise FileExistsError("epoch-selection outputs already exist; preserve them and choose a new run directory")
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if calibration.get("gate_version") != PHASE_GATE_VERSION or contract.get("version") != PHASE_GATE_VERSION:
        raise ValueError("v4 calibration and contract versions must match the trainer")
    if calibration.get("thresholds_valid") is not True:
        raise ValueError("v4 reference calibration or cycle coverage failed; refusing to train candidates")

    dataset_archive = np.load(DATA_PATH, allow_pickle=False)
    train = load_train_validation_split(dataset_archive, "train")
    validation = load_train_validation_split(dataset_archive, "validation")
    dataset_sha256 = sha256_file(DATA_PATH)
    calibration_sha256 = sha256_file(CALIBRATION_PATH)
    trainer_sha256 = sha256_file(Path(__file__).resolve())
    gate_module_sha256 = sha256_file(ROOT / "src" / "tasks" / "velocity" / "phasor" / "phase_gate_v4.py")
    encoder_module_sha256 = sha256_file(ROOT / "src" / "tasks" / "velocity" / "phasor" / "encoder_v2.py")
    period_cuts = tuple(float(x) for x in calibration["period_length_cut_points_frames"])
    validation_conditions = sorted(np.unique(validation["sequence"]).tolist())
    train_sequences = sorted(np.unique(train["sequence"]).tolist())
    validation_sequences = validation_conditions
    train_phase = (train["phase_cycles"].astype(np.float32) * (2.0 * np.pi)).astype(np.float32)
    validation_phase = (validation["phase_cycles"].astype(np.float32) * (2.0 * np.pi)).astype(np.float32)
    train_pairs = build_cycle_balanced_pairs(train)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    detailed_reports: dict[str, dict[str, Any]] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    training_history: dict[str, list[dict[str, float]]] = {}
    for context_frames in CONTEXTS:
        context_name = f"context_{context_frames}"
        train_windows, _ = build_context_windows(train, context_frames)
        validation_windows, validation_history = build_context_windows(validation, context_frames)

        def evaluate_epoch(
            model: PhaseEncoderV2,
            epoch: int,
            epoch_loss: dict[str, float],
            *,
            context: int = context_frames,
            windows: np.ndarray = validation_windows,
            valid_history: np.ndarray = validation_history,
        ) -> None:
            candidate_name = f"context_{context}_epoch_{epoch:02d}"
            prediction = _predict(model, windows)
            report = score_phase_cycles_v4(
                prediction,
                validation_phase,
                validation["sequence"],
                validation["frame"],
                validation["left_touchdown"].astype(bool),
                condition_ids=validation["sequence"],
                episode_ids=validation["sequence"],
                history_valid=valid_history,
                period_cuts_frames=period_cuts,
                required_conditions=validation_conditions,
                minimum_cycles_per_condition=10,
                minimum_cycles_per_length_stratum=10,
                minimum_episode_segments_per_condition=1,
            )
            checkpoint = _candidate_checkpoint(
                model,
                context,
                epoch,
                dataset_sha256,
                calibration_sha256,
                train_sequences,
                validation_sequences,
            )
            checkpoint_path = CANDIDATE_DIR / f"phase_encoder_v2_{candidate_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            summaries[candidate_name] = {
                "context_frames": context,
                "epoch": epoch,
                "macro_cycle_completion_fraction": report["macro_cycle_completion_fraction"],
                "condition_macro_point_error_rad": report["condition_macro_point_error_rad"],
                "candidate_gate_pass": report["encoder_freeze_eligible"],
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_epoch_loss": epoch_loss,
                "validation_report": {
                    key: value for key, value in report.items() if key != "cycle_records"
                },
            }
            detailed_reports[candidate_name] = report
            raw_predictions[candidate_name] = prediction
            print(json.dumps({
                "candidate": candidate_name,
                "macro_completion": report["macro_cycle_completion_fraction"],
                "condition_rates": {
                    key: value["completion_fraction"]
                    for key, value in report["condition_completion"].items()
                },
                "length_rates": {
                    key: value["completion_fraction"]
                    for key, value in report["period_length_strata"].items()
                },
                "freeze_eligible": report["encoder_freeze_eligible"],
            }, separators=(",", ":")))

        _, epoch_losses = fit_candidate(
            train_windows,
            train_phase,
            train_pairs,
            context_frames=context_frames,
            epoch_callback=evaluate_epoch,
        )
        training_history[context_name] = epoch_losses

    eligible = {name: report for name, report in summaries.items() if report["candidate_gate_pass"]}
    selection_pool = eligible if eligible else summaries
    selected_name = select_v4_candidate(selection_pool)
    validation_pass = bool(eligible)
    if validation_pass:
        selected_checkpoint_path = ROOT / summaries[selected_name]["checkpoint"]
        selected_checkpoint = torch.load(selected_checkpoint_path, map_location="cpu", weights_only=True)
        selected_checkpoint["frozen_after_validation"] = True
        selected_checkpoint["selected_candidate"] = selected_name
        torch.save(selected_checkpoint, FROZEN_PATH)
        frozen_checkpoint_sha256 = sha256_file(FROZEN_PATH)
    else:
        selected_checkpoint_path = None
        frozen_checkpoint_sha256 = None

    np.savez_compressed(EPOCH_SELECTION_PREDICTIONS_PATH, **raw_predictions, reference_phase=validation_phase)
    result = {
        "schema": "g1.phase_encoder_v2_validation.v2",
        "gate_version": PHASE_GATE_VERSION,
        "status": "VALIDATION_PASS_FROZEN" if validation_pass else "BLOCKED_NO_CANDIDATE_PASSED_V4",
        "source_split": "train_only_fit_validation_only_selection",
        "plan_amendment": "results/g1_phase_gate_v4/PLAN_AMENDMENT_v2_epoch_selection.md",
        "test_split_read_for_fit_or_selection": False,
        "new_rollout_used_for_fit_or_selection": False,
        "dataset_sha256": dataset_sha256,
        "calibration_sha256": calibration_sha256,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "code_sha256": {
            "training_script": trainer_sha256,
            "phase_gate_v4_module": gate_module_sha256,
            "encoder_v2_module": encoder_module_sha256,
        },
        "training_config": {
            "seed": SEED,
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden": HIDDEN,
            "contexts": list(CONTEXTS),
            "checkpoint_epochs_evaluated": list(checkpoint_epochs(EPOCHS)),
            "cycle_balanced_pair_weights": True,
            "circular_loss_weight": 1.0,
            "rate_loss_weight": 2.0,
            "negative_progress_weight": 2.0,
            "jump_penalty_weight": 1.0,
            "rate_scale_rad": float(np.pi / 6.0),
            "jump_limit_rad": float(np.pi / 2.0),
            "ema_or_filter": None,
        },
        "candidate_selection": {
            "primary": "highest v4 macro cycle completion among candidates passing all frozen requirements",
            "tie_break_1": "lowest equal-weight validation condition point error",
            "tie_break_2": "shorter context",
            "tie_break_3": "earlier epoch",
            "candidate_count": len(summaries),
            "selected_candidate": selected_name,
            "passing_candidate_names": sorted(eligible),
        },
        "training_history_by_context": training_history,
        "candidates": summaries,
        "selected_candidate_full_cycle_report": detailed_reports[selected_name],
        "frozen_checkpoint": str(FROZEN_PATH.relative_to(ROOT)) if validation_pass else None,
        "frozen_checkpoint_sha256": frozen_checkpoint_sha256,
        "predictions_path": str(EPOCH_SELECTION_PREDICTIONS_PATH.relative_to(ROOT)),
    }
    EPOCH_SELECTION_REPORT_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "selected_candidate": selected_name,
        "passing_candidates": sorted(eligible),
        "candidate_count": len(summaries),
        "frozen_checkpoint_sha256": frozen_checkpoint_sha256,
        "report": str(EPOCH_SELECTION_REPORT_PATH.relative_to(ROOT)),
    }, indent=2))
    return 0 if validation_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
