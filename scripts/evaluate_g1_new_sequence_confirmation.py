"""One-shot evaluation of the frozen TCN seeds on the pinned new walk sequence.

The candidate is not used for training, normalization, model selection, or
threshold tuning. The source record and raw-file hash are checked before CSV
values are parsed. See the confirmation amendment and source-record template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import run_g1_contact_followup_comparison as base
from scripts.build_g1_walk_phase_dataset import (
    G1_XML,
    SOURCE_FPS,
    TARGET_FPS,
    _model_contract,
    forward_kinematics,
    load_csv,
    phase_from_touchdown_times,
)
from src.tasks.velocity.phasor.data import (
    ContactDetectionConfig,
    detect_foot_contacts,
    make_windows,
    resample_joint_positions,
)


REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "07ff8845a94d59d4195a8ac04fe918d4e6338239"
CANDIDATE_PATH = "g1/walk1_subject3.csv"
LICENSE = "CC BY-NC-ND-4.0"
SEQUENCE_ID = "walk1_subject3"
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
MODEL_DIR = ROOT / "results" / "g1_contact_followup_20261021" / "run_v1_replay_optimized"
PROTOCOL = ROOT / "results" / "g1_phase_dataset_20261019" / "new_sequence_confirmation_amendment_20261022.md"
SOURCE_RECORD_TEMPLATE = ROOT / "results" / "g1_phase_dataset_20261019" / "candidate_source_record.template.json"
DEFAULT_OUT = ROOT / "results" / "g1_contact_followup_20261022" / "new_sequence_confirmation_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROZEN_CHECKPOINTS = {
    20261021: "b7250066a445d5eab8313463b52db300b746dc7e4e700741c14bb40f00f28a4a",
    20261022: "4a626147ccfdff82da7dfbf5e358a4e50a93e48364e8015657ce880585e9f5d7",
    20261023: "8f864b13c60bd6d0a5a994dbb457753021bf1663e7d5c45805bd90a9e41066e9",
}
PROTOCOL_FROZEN_AT_UTC = datetime.fromisoformat("2026-10-21T16:56:04+00:00")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_record(record: dict[str, Any], csv_path: Path) -> dict[str, Any]:
    expected = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "resolved_commit": REVISION,
        "path": CANDIDATE_PATH,
        "sequence_id": SEQUENCE_ID,
        "license": LICENSE,
    }
    mismatches = {key: (record.get(key), value) for key, value in expected.items() if record.get(key) != value}
    if mismatches:
        raise ValueError(f"source record does not match frozen candidate: {mismatches}")
    url = str(record.get("source_url", ""))
    expected_url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{CANDIDATE_PATH}"
    if url.rstrip("?") != expected_url:
        raise ValueError("source_url must identify the official pinned-revision file endpoint")
    retrieved = record.get("retrieved_at_utc")
    if not isinstance(retrieved, str):
        raise ValueError("source record requires retrieved_at_utc")
    timestamp = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("retrieved_at_utc must include a timezone")
    if timestamp.astimezone(timezone.utc) < PROTOCOL_FROZEN_AT_UTC:
        raise ValueError("candidate retrieval time predates the frozen confirmation protocol")
    if record.get("first_access_after_protocol_frozen") is not True:
        raise ValueError("source record must attest that candidate content was first accessed after freeze")

    file_hash = sha256(csv_path)
    file_size = csv_path.stat().st_size
    if record.get("sha256") != file_hash:
        raise ValueError("candidate CSV SHA-256 does not match its source record")
    if record.get("bytes") != file_size:
        raise ValueError("candidate CSV byte size does not match its source record")
    return {"sha256": file_hash, "bytes": file_size}


def consistency_rows(frame_indices: np.ndarray, labels: Any) -> tuple[np.ndarray, int]:
    keep = np.zeros(len(frame_indices), dtype=bool)
    retained_cycles = 0
    target_indices = frame_indices.astype(np.int64)
    left = labels.left_touchdown_indices.astype(np.int64)
    right = labels.right_touchdown_indices.astype(np.int64)
    for start, end in zip(left[:-1], left[1:]):
        count = int(np.count_nonzero((right >= start) & (right < end)))
        if count == 1:
            keep |= (target_indices >= start) & (target_indices < end)
            retained_cycles += 1
    return keep, retained_cycles


def evaluate_seed(
    checkpoint_path: Path,
    *,
    seed: int,
    x: np.ndarray,
    y_cycles: np.ndarray,
    consistency_mask: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    if sha256(checkpoint_path) != FROZEN_CHECKPOINTS[seed]:
        raise ValueError(f"frozen checkpoint hash mismatch for seed {seed}")
    model = base.TCNEncoder().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    prediction = base.predict(model, x, batch_size=2048)
    errors = base.circular_abs_error(prediction, y_cycles)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        values = errors[mask]
        if not len(values):
            raise ValueError("frozen confirmation subset contains no evaluable frames")
        return {
            "n": int(len(values)),
            "mae_rad": float(values.mean()),
            "median_rad": float(np.median(values)),
            "p90_rad": float(np.quantile(values, 0.9)),
        }

    prediction_path = output_dir / f"tcn_seed_{seed}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        prediction=prediction.astype(np.float32),
        error_rad=errors.astype(np.float32),
        target_phase_cycles=y_cycles.astype(np.float32),
        consistency_mask=consistency_mask.astype(np.uint8),
    )
    return {
        "seed": seed,
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction_sha256": sha256(prediction_path),
        "primary_consistency_subset": summarize(consistency_mask),
        "full_sequence_valid_windows": summarize(np.ones(len(errors), dtype=bool)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"preserving existing confirmation output: {args.out_dir}")
    if args.csv.name != f"{SEQUENCE_ID}.csv" or not args.csv.is_file():
        raise ValueError(f"expected the frozen candidate filename {SEQUENCE_ID}.csv")
    if not args.source_record.is_file() or not PROTOCOL.is_file() or not SOURCE_RECORD_TEMPLATE.is_file():
        raise FileNotFoundError("source record or frozen protocol assets are missing")

    source_record = json.loads(args.source_record.read_text(encoding="utf-8"))
    source_identity = validate_source_record(source_record, args.csv)
    raw = load_csv(args.csv)
    if int(source_record.get("frame_count", -1)) != len(raw):
        raise ValueError("candidate CSV frame count does not match source record")
    if len(raw) < 2:
        raise ValueError("candidate sequence must contain at least two frames")

    original_archive = DATA_DIR / "phase_dataset.npz"
    with np.load(original_archive, allow_pickle=False) as archive:
        mean = archive["train_mean"].copy()
        std = archive["train_std"].copy()
    model, data, qpos, feet = _model_contract(G1_XML)
    source_times = np.arange(len(raw), dtype=np.float64) / SOURCE_FPS
    foot_positions = forward_kinematics(raw, model, data, qpos, feet)
    contacts = detect_foot_contacts(foot_positions, fps=SOURCE_FPS, config=ContactDetectionConfig())
    left_times = source_times[np.flatnonzero(contacts[:, 0] & ~np.concatenate(([False], contacts[:-1, 0])))]
    right_times = source_times[np.flatnonzero(contacts[:, 1] & ~np.concatenate(([False], contacts[:-1, 1])))]
    target_times, joint_positions = resample_joint_positions(
        source_times, raw[:, 7:], target_hz=TARGET_FPS
    )
    labels = phase_from_touchdown_times(target_times, left_times, right_times)
    normalized = ((joint_positions - mean) / std).astype(np.float32)
    windows = make_windows(normalized, labels, sequence_ids=[SEQUENCE_ID] * len(normalized), window=10)
    if len(windows.labels) == 0:
        raise ValueError("candidate has no complete phase-labeled 10-frame windows")
    consistency_mask, retained_cycles = consistency_rows(
        windows.label_frame_indices, labels
    )
    if not consistency_mask.any():
        raise ValueError("candidate has no complete cycle with exactly one detected right touchdown")

    args.out_dir.mkdir(parents=True)
    checkpoint_records: list[dict[str, Any]] = []
    for seed in FROZEN_CHECKPOINTS:
        checkpoint = MODEL_DIR / f"filtered_train_tcn_seed_{seed}.pt"
        checkpoint_records.append(evaluate_seed(
            checkpoint,
            seed=seed,
            x=windows.windows,
            y_cycles=windows.labels,
            consistency_mask=consistency_mask,
            output_dir=args.out_dir,
        ))
    primary_values = [row["primary_consistency_subset"]["mae_rad"] for row in checkpoint_records]
    full_values = [row["full_sequence_valid_windows"]["mae_rad"] for row in checkpoint_records]
    result = {
        "schema": "g1.new_sequence_confirmation.v1",
        "status": "COMPLETED_ONCE",
        "source_record": source_record,
        "candidate_identity": source_identity,
        "candidate_frame_count": int(len(raw)),
        "candidate_resampled_frame_count": int(len(joint_positions)),
        "candidate_subject_seen_in_training": True,
        "claim_scope": "new sequence only; not new-subject generalization",
        "test_split_loaded": False,
        "normalization_source": "frozen original training split train_mean/train_std; no refit",
        "window_frames": 10,
        "valid_window_count": int(len(windows.labels)),
        "consistency_valid_window_count": int(consistency_mask.sum()),
        "consistency_retained_cycle_count": retained_cycles,
        "label_source": "frozen height-speed kinematic detector; not independent contact ground truth",
        "primary_metric": {
            "definition": "mean circular absolute phase error on complete left cycles with exactly one detected right touchdown",
            "seed_mae_rad": primary_values,
            "mean_across_frozen_seeds_rad": float(np.mean(primary_values)),
            "sd_across_frozen_seeds_rad": float(np.std(primary_values, ddof=1)),
        },
        "full_sequence_supplement": {
            "seed_mae_rad": full_values,
            "mean_across_frozen_seeds_rad": float(np.mean(full_values)),
            "sd_across_frozen_seeds_rad": float(np.std(full_values, ddof=1)),
        },
        "frozen_seed_results": checkpoint_records,
        "protocol_sha256": sha256(PROTOCOL),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "base_model_code_sha256": sha256(Path(base.__file__).resolve()),
        "source_archive_sha256": sha256(original_archive),
        "g1_model_sha256": sha256(G1_XML),
        "evaluation_limits": [
            "The confirmation sequence belongs to subject 3, which appears in training; it does not establish cross-subject generalization.",
            "All targets use the same kinematic contact detector; phase error is not an estimate of contact-label accuracy.",
            "The three seeds quantify training variability and are not independent data samples.",
        ],
    }
    (args.out_dir / "confirmation_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.out_dir),
        "candidate_frames": int(len(raw)),
        "valid_windows": int(len(windows.labels)),
        "consistency_windows": int(consistency_mask.sum()),
        "primary_mean_mae_rad": float(np.mean(primary_values)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
