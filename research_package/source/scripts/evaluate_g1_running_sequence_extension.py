"""One-shot, preregistered TCN evaluation on a previously unused G1 run clip."""
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

from scripts import evaluate_g1_new_sequence_confirmation as frozen
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
CANDIDATE_PATH = "g1/run1_subject5.csv"
SEQUENCE_ID = "run1_subject5"
LICENSE = "CC BY-NC-ND-4.0"
UPSTREAM_GIT_BLOB_OID = "666cd6f5fcbfba80bc35b16ffcb7fecaa9187349"
UPSTREAM_BYTES = 2_439_981
PROTOCOL = ROOT / "results" / "g1_phase_dataset_20261019" / "new_running_sequence_extension_preregistration_20261022.md"
DATA_DIR = ROOT / "results" / "g1_phase_dataset_20261019"
MODEL_DIR = ROOT / "results" / "g1_contact_followup_20261021" / "run_v1_replay_optimized"
DEFAULT_OUT = ROOT / "results" / "g1_contact_followup_20261022" / "new_running_sequence_extension_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTOCOL_FROZEN_AT_UTC = datetime.fromisoformat("2026-10-21T17:46:27+00:00")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
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
        "action_type": "run",
        "subject": "5",
        "license": LICENSE,
        "upstream_git_blob_oid": UPSTREAM_GIT_BLOB_OID,
        "upstream_bytes": UPSTREAM_BYTES,
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source record does not match the frozen candidate: {mismatches}")
    expected_url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{CANDIDATE_PATH}"
    if str(record.get("source_url", "")).rstrip("?") != expected_url:
        raise ValueError("source_url must identify the pinned upstream file endpoint")
    retrieved = record.get("retrieved_at_utc")
    if not isinstance(retrieved, str):
        raise ValueError("source record requires retrieved_at_utc")
    timestamp = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.astimezone(timezone.utc) < PROTOCOL_FROZEN_AT_UTC:
        raise ValueError("candidate retrieval must be timestamped after the extension freeze")
    if record.get("first_access_after_protocol_frozen") is not True:
        raise ValueError("source record must attest that candidate contents were first accessed after freeze")

    file_size = csv_path.stat().st_size
    if file_size != UPSTREAM_BYTES or record.get("bytes") != file_size:
        raise ValueError("candidate CSV byte size does not match the frozen upstream metadata")
    blob_oid = git_blob_oid(csv_path)
    if blob_oid != UPSTREAM_GIT_BLOB_OID:
        raise ValueError("candidate CSV Git blob object ID does not match the frozen upstream metadata")
    file_hash = file_sha256(csv_path)
    if record.get("sha256") != file_hash:
        raise ValueError("candidate CSV SHA-256 does not match its source record")
    return {"sha256": file_hash, "git_blob_oid": blob_oid, "bytes": file_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"preserving existing extension output: {args.out_dir}")
    if args.csv.name != f"{SEQUENCE_ID}.csv" or not args.csv.is_file():
        raise ValueError(f"expected frozen candidate filename {SEQUENCE_ID}.csv")
    if not args.source_record.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError("source record or frozen extension protocol is missing")

    source_record = json.loads(args.source_record.read_text(encoding="utf-8"))
    source_identity = validate_source_record(source_record, args.csv)
    raw = load_csv(args.csv)
    if int(source_record.get("frame_count", -1)) != len(raw):
        raise ValueError("candidate CSV frame count does not match its source record")
    if len(raw) < 2:
        raise ValueError("candidate sequence must contain at least two frames")

    original_archive = DATA_DIR / "phase_dataset.npz"
    with np.load(original_archive, allow_pickle=False) as archive:
        mean = archive["train_mean"].copy()
        std = archive["train_std"].copy()
    model, data, qpos, feet = _model_contract(G1_XML)
    source_times = np.arange(len(raw), dtype=np.float64) / SOURCE_FPS
    foot_positions = forward_kinematics(raw, model, data, qpos, feet)
    contacts = detect_foot_contacts(
        foot_positions, fps=SOURCE_FPS, config=ContactDetectionConfig()
    )
    left_edges = contacts[:, 0] & ~np.concatenate(([False], contacts[:-1, 0]))
    right_edges = contacts[:, 1] & ~np.concatenate(([False], contacts[:-1, 1]))
    left_times = source_times[np.flatnonzero(left_edges)]
    right_times = source_times[np.flatnonzero(right_edges)]
    target_times, joint_positions = resample_joint_positions(
        source_times, raw[:, 7:], target_hz=TARGET_FPS
    )
    labels = phase_from_touchdown_times(target_times, left_times, right_times)
    normalized = ((joint_positions - mean) / std).astype(np.float32)
    windows = make_windows(
        normalized, labels, sequence_ids=[SEQUENCE_ID] * len(normalized), window=10
    )
    if len(windows.labels) == 0:
        raise ValueError("candidate has no complete phase-labeled 10-frame windows")
    consistency_mask, retained_cycles = frozen.consistency_rows(
        windows.label_frame_indices, labels
    )
    if not consistency_mask.any():
        raise ValueError("candidate has no complete cycle with exactly one detected right touchdown")

    args.out_dir.mkdir(parents=True)
    checkpoint_records: list[dict[str, Any]] = []
    for seed in frozen.FROZEN_CHECKPOINTS:
        checkpoint = MODEL_DIR / f"filtered_train_tcn_seed_{seed}.pt"
        checkpoint_records.append(
            frozen.evaluate_seed(
                checkpoint,
                seed=seed,
                x=windows.windows,
                y_cycles=windows.labels,
                consistency_mask=consistency_mask,
                output_dir=args.out_dir,
            )
        )

    primary_values = [row["primary_consistency_subset"]["mae_rad"] for row in checkpoint_records]
    full_values = [row["full_sequence_valid_windows"]["mae_rad"] for row in checkpoint_records]
    result = {
        "schema": "g1.new_running_sequence_extension.v1",
        "status": "COMPLETED_ONCE",
        "source_record": source_record,
        "candidate_identity": source_identity,
        "candidate_frame_count": int(len(raw)),
        "candidate_resampled_frame_count": int(len(joint_positions)),
        "candidate_action": "running",
        "candidate_subject": "5",
        "candidate_subject_seen_in_training": True,
        "claim_scope": "one new running sequence; same-subject cross-action transfer only",
        "not_claimed": [
            "new walking sequence confirmation",
            "new-subject generalization",
            "independent foot-contact label accuracy",
        ],
        "test_split_loaded": False,
        "normalization_source": "frozen original training split train_mean/train_std; no refit",
        "training_action_domain": "walking sequences",
        "window_frames": 10,
        "valid_window_count": int(len(windows.labels)),
        "consistency_valid_window_count": int(consistency_mask.sum()),
        "consistency_retained_cycle_count": retained_cycles,
        "detected_left_touchdown_count": int(left_edges.sum()),
        "detected_right_touchdown_count": int(right_edges.sum()),
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
        "protocol_sha256": file_sha256(PROTOCOL),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "original_confirmation_evaluator_sha256": frozen.sha256(Path(frozen.__file__).resolve()),
        "base_model_code_sha256": frozen.sha256(Path(base.__file__).resolve()),
        "source_archive_sha256": file_sha256(original_archive),
        "g1_model_sha256": file_sha256(G1_XML),
        "evaluation_limits": [
            "The candidate is a running sequence while frozen training data contain walking sequences, so the result includes an action-distribution shift.",
            "The candidate belongs to subject 5, which appears in training; this does not establish new-subject generalization.",
            "All phase targets use the same kinematic contact detector; phase error is not contact-label accuracy.",
            "The three seeds quantify training variability and are not independent data samples.",
        ],
    }
    (args.out_dir / "running_sequence_extension_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.out_dir),
                "candidate_frames": int(len(raw)),
                "valid_windows": int(len(windows.labels)),
                "consistency_windows": int(consistency_mask.sum()),
                "primary_mean_mae_rad": float(np.mean(primary_values)),
                "claim_scope": result["claim_scope"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
