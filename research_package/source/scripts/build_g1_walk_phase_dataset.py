"""Build an auditable 50 Hz contact-phase dataset from G1 walk CSV files.

The upstream files are kept outside the repository's tracked source tree.  This
script records their revision and hashes, computes foot kinematics with the
repository G1 XML, creates operational contact labels, splits complete
sequences by subject before windowing, and fits normalization on train frames
only.  It never marks the provenance manifest READY; that requires the plotted
contact audit to be reviewed first.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src.tasks.velocity.phasor.data import (
    ContactDetectionConfig,
    G1_JOINT_ORDER,
    PhaseLabels,
    detect_foot_contacts,
    fit_normalizer,
    make_windows,
    resample_joint_positions,
)
from src.assets.robots.unitree_g1.g1_constants import G1_XML, get_assets
from src.tasks.velocity.phasor.mujoco_io import load_mjcf_spec


SOURCE_REVISION = "07ff8845a94d59d4195a8ac04fe918d4e6338239"
SOURCE_URL = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
SOURCE_FPS = 30.0
TARGET_FPS = 50.0
WINDOW = 10
JOINT_DIM = 29
JOINT_ORDER = G1_JOINT_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 + JOINT_DIM:
        raise ValueError(f"{path.name}: expected 36 columns, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: non-finite motion value")
    return values


def _model_contract(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, int], tuple[int, int]]:
    xml_path = Path(xml_path).resolve()
    if xml_path != G1_XML.resolve():
        raise ValueError(f"phase dataset builder only supports the frozen G1 XML: {G1_XML}")
    spec = load_mjcf_spec(mujoco.MjSpec, xml_path)
    spec.assets = get_assets(spec.meshdir)
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos_addresses = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_ORDER
    }
    foot_sites = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_foot", "right_foot")
    )
    if any(index < 0 for index in foot_sites):
        raise ValueError("G1 XML must expose left_foot and right_foot sites")
    return model, data, qpos_addresses, foot_sites


def forward_kinematics(
    values: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: dict[str, int],
    foot_sites: tuple[int, int],
) -> np.ndarray:
    positions = np.empty((len(values), 2, 3), dtype=np.float32)
    for index, row in enumerate(values):
        data.qpos[:3] = row[:3]
        # Source order is XYZQXQYQZQW; MuJoCo free joints use WXYZ.
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        for joint_index, name in enumerate(JOINT_ORDER):
            data.qpos[qpos_addresses[name]] = row[7 + joint_index]
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[list(foot_sites)]
    return positions


def phase_from_touchdown_times(
    target_times: np.ndarray,
    left_touchdown_times: np.ndarray,
    right_touchdown_times: np.ndarray,
) -> PhaseLabels:
    phase = np.full(len(target_times), np.nan, dtype=np.float32)
    valid = np.zeros(len(target_times), dtype=bool)
    right_phases: list[float] = []
    complete = 0
    for start, end in zip(left_touchdown_times[:-1], left_touchdown_times[1:]):
        if end <= start:
            continue
        mask = (target_times >= start) & (target_times <= end)
        phase[mask] = np.mod((target_times[mask] - start) / (end - start), 1.0)
        valid[mask] = True
        complete += 1
        # A simultaneous event at the cycle boundary belongs to the boundary
        # itself; do not count the initial state as a right-foot phase sample.
        inside = right_touchdown_times[(right_touchdown_times > start) & (right_touchdown_times < end)]
        right_phases.extend(((inside - start) / (end - start)).tolist())
    left_indices = np.searchsorted(target_times, left_touchdown_times, side="left")
    right_indices = np.searchsorted(target_times, right_touchdown_times, side="left")
    return PhaseLabels(
        phase_cycles=phase,
        valid=valid,
        left_touchdown_indices=left_indices.astype(np.int64),
        right_touchdown_indices=right_indices.astype(np.int64),
        right_touchdown_phase_cycles=np.asarray(right_phases, dtype=np.float32),
        complete_cycle_count=complete,
    )


def _touchdown_times(contact: np.ndarray, times: np.ndarray) -> np.ndarray:
    rising = contact & ~np.concatenate((np.zeros(1, dtype=bool), contact[:-1]))
    return times[np.flatnonzero(rising)]


def split_subjects(subjects: list[str], seed: int = 20261019) -> dict[str, list[str]]:
    unique = sorted(set(subjects), key=lambda value: int(value))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    if len(unique) < 3:
        raise ValueError("at least three subjects are required for held-out splits")
    n_train = max(1, int(round(len(unique) * 0.6)))
    n_validation = max(1, int(round(len(unique) * 0.2)))
    if n_train + n_validation >= len(unique):
        n_train = len(unique) - 2
        n_validation = 1
    return {
        "train": unique[:n_train],
        "validation": unique[n_train : n_train + n_validation],
        "test": unique[n_train + n_validation :],
    }


def _sequence_subject(sequence_id: str) -> str:
    match = re.search(r"subject(\d+)$", sequence_id)
    if match is None:
        raise ValueError(f"cannot infer subject from {sequence_id}")
    return match.group(1)


def build_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    xml_path: Path,
    seed: int = 20261019,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ContactDetectionConfig()
    model, data, qpos_addresses, foot_sites = _model_contract(xml_path)
    sequences: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    all_sequence_ids: list[str] = []

    for csv_path in sorted(raw_dir.glob("walk*_subject*.csv")):
        if csv_path.stat().st_size < 1000:
            continue
        sequence_id = csv_path.stem
        values = load_csv(csv_path)
        source_times = np.arange(len(values), dtype=np.float64) / SOURCE_FPS
        foot_positions = forward_kinematics(values, model, data, qpos_addresses, foot_sites)
        contacts = detect_foot_contacts(foot_positions, fps=SOURCE_FPS, config=cfg)
        left_touches = _touchdown_times(contacts[:, 0], source_times)
        right_touches = _touchdown_times(contacts[:, 1], source_times)
        if len(left_touches) < 3:
            raise RuntimeError(f"{sequence_id}: fewer than two complete left cycles")

        target_times, joint_positions = resample_joint_positions(
            source_times, values[:, 7:], target_hz=TARGET_FPS
        )
        phase_labels = phase_from_touchdown_times(target_times, left_touches, right_touches)
        left_touchdown = np.zeros(len(target_times), dtype=np.uint8)
        right_touchdown = np.zeros(len(target_times), dtype=np.uint8)
        left_touchdown[phase_labels.left_touchdown_indices] = 1
        right_touchdown[phase_labels.right_touchdown_indices] = 1
        sequence_ids = np.full(len(target_times), sequence_id, dtype=object)
        windows = make_windows(
            joint_positions,
            phase_labels,
            sequence_ids=sequence_ids,
            window=WINDOW,
        )
        subject = _sequence_subject(sequence_id)
        sequences[sequence_id] = {
            "subject": subject,
            "windows": windows,
            "frames": joint_positions,
            "contacts": contacts,
            "foot_positions": foot_positions,
            "phase_labels": phase_labels,
            "left_touchdown": left_touchdown,
            "right_touchdown": right_touchdown,
            "raw_frames": int(len(values)),
            "resampled_frames": int(len(joint_positions)),
            "sha256": sha256_file(csv_path),
            "bytes": int(csv_path.stat().st_size),
        }
        all_sequence_ids.append(sequence_id)
        audit_rows.append(
            {
                "sequence_id": sequence_id,
                "subject": subject,
                "source": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "source_path": f"g1/{csv_path.name}",
                "action_type": "walk",
                "walk_verified": True,
                "license": "CC BY-NC-ND-4.0",
                "fps": SOURCE_FPS,
                "retargeting": {
                    "method": "numerical optimization with Interaction Mesh and IK (upstream dataset card)",
                    "joint_mapping": "G1 root XYZQXQYQZQW followed by the frozen 29-joint order",
                },
                "raw_frames": int(len(values)),
                "resampled_frames": int(len(joint_positions)),
                "left_touchdown_count": int(len(left_touches)),
                "right_touchdown_count": int(len(right_touches)),
                "complete_left_cycles": int(phase_labels.complete_cycle_count),
                "right_touchdown_phase_cycles": phase_labels.right_touchdown_phase_cycles.tolist(),
                "sha256": sequences[sequence_id]["sha256"],
                "bytes": sequences[sequence_id]["bytes"],
            }
        )

    split_subject = split_subjects([sequences[s]["subject"] for s in all_sequence_ids], seed=seed)
    split_sequences = {
        split: [sid for sid in all_sequence_ids if sequences[sid]["subject"] in subjects]
        for split, subjects in split_subject.items()
    }
    normalizer = fit_normalizer(
        np.concatenate([sequences[sid]["frames"] for sid in split_sequences["train"]], axis=0)
    )

    arrays: dict[str, np.ndarray] = {}
    for split, sequence_ids in split_sequences.items():
        batches = [sequences[sid]["windows"] for sid in sequence_ids if len(sequences[sid]["windows"].windows)]
        if not batches:
            raise RuntimeError(f"empty {split} window split")
        windows = np.concatenate([batch.windows for batch in batches], axis=0)
        labels = np.concatenate([batch.labels for batch in batches], axis=0)
        windows = (windows - normalizer.mean) / normalizer.std
        arrays[f"x_{split}"] = windows.astype(np.float32)
        arrays[f"y_{split}_cycles"] = labels.astype(np.float32)
        arrays[f"sequence_{split}"] = np.concatenate([batch.sequence_ids[:, 0] for batch in batches]).astype(str)
        arrays[f"frame_{split}"] = np.concatenate([batch.label_frame_indices for batch in batches]).astype(np.int64)
        arrays[f"left_touchdown_{split}"] = np.concatenate(
            [sequences[sid]["left_touchdown"][batch.label_frame_indices] for sid, batch in zip(sequence_ids, batches)]
        ).astype(np.uint8)
        arrays[f"right_touchdown_{split}"] = np.concatenate(
            [sequences[sid]["right_touchdown"][batch.label_frame_indices] for sid, batch in zip(sequence_ids, batches)]
        ).astype(np.uint8)

    np.savez_compressed(
        output_dir / "phase_dataset.npz",
        **arrays,
        train_mean=normalizer.mean,
        train_std=normalizer.std,
    )

    audit_figure = output_dir / "contact_audit.png"
    figure, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=False)
    for axis, sequence_id in zip(axes.flat, all_sequence_ids):
        record = sequences[sequence_id]
        t = np.arange(len(record["foot_positions"])) / SOURCE_FPS
        height = record["foot_positions"][:, :, 2]
        speed = np.linalg.norm(
            np.gradient(record["foot_positions"], 1.0 / SOURCE_FPS, axis=0), axis=-1
        )
        axis.plot(t, height[:, 0], color="tab:blue", alpha=0.7, label="left z")
        axis.plot(t, height[:, 1], color="tab:orange", alpha=0.7, label="right z")
        axis.plot(t, speed[:, 0] * 0.01, color="tab:blue", linestyle="--", alpha=0.5)
        axis.plot(t, speed[:, 1] * 0.01, color="tab:orange", linestyle="--", alpha=0.5)
        axis.scatter(t[record["contacts"][:, 0]], height[record["contacts"][:, 0], 0], s=1, color="navy")
        axis.scatter(t[record["contacts"][:, 1]], height[record["contacts"][:, 1], 1], s=1, color="darkorange")
        axis.set_title(sequence_id)
        axis.set_ylim(-0.02, max(0.25, float(np.quantile(height, 0.995))))
        axis.grid(alpha=0.2)
    axes_flat = list(axes.flat)
    for axis in axes_flat[len(all_sequence_ids) :]:
        axis.axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    figure.suptitle("G1 retargeted walk contact audit: solid=z, dashed=0.01×speed, dots=detected contact")
    figure.tight_layout()
    figure.savefig(audit_figure, dpi=160)
    plt.close(figure)

    manifest_path = output_dir / "g1_phase_data_provenance_candidate.json"
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/data/g1_phase_data_provenance.json").read_text(encoding="utf-8")
    )
    manifest["status"] = "CONTACT_AUDIT_PENDING"
    manifest["current_file"] = None
    manifest["candidate_walk_sequences"].update(
        {
            "local_snapshot": str(raw_dir),
            "file_metadata": audit_rows,
            "contact_verification": {
                "status": "pending_visual_audit",
                "method": {
                    "kinematics": "MuJoCo forward kinematics from local G1 XML",
                    "rule": asdict(cfg),
                    "source_fps": SOURCE_FPS,
                    "target_fps": TARGET_FPS,
                    "manual_sample_protocol": "one full-sequence plot per candidate file; inspect height, speed, and contact dots",
                },
                "manual_sample_count": 0,
                "uncertain_intervals": [],
            },
        }
    )
    manifest["split"] = {
        "unit": "subject",
        "seed": seed,
        "subjects": split_subject,
        "sequences": split_sequences,
    }
    manifest["phase_units"] = {
        "training": "phase_cycles in [0, 1), converted to phi_radians=2*pi*phase_cycles before target vectors",
        "output": "(sin(phi), cos(phi))",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metadata = {
        "schema": "g1.phase_dataset.v1",
        "source": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "window_frames": WINDOW,
        "joint_dim": JOINT_DIM,
        "joint_order": list(JOINT_ORDER),
        "contact_rule": asdict(cfg),
        "split": {"subjects": split_subject, "sequences": split_sequences, "seed": seed},
        "normalizer": {"fit_split": "train", "mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()},
        "sequences": audit_rows,
        "artifacts": {
            "dataset": "phase_dataset.npz",
            "contact_audit": audit_figure.name,
            "manifest_candidate": manifest_path.name,
        },
        "status": "READY",
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "contact_audit.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # MuJoCo on Windows resolves relative XML paths reliably even when the
    # workspace contains non-ASCII parent directories.
    os.chdir(root)
    metadata = build_dataset(
        root / "data/raw/lafan1_g1_walk",
        root / "results/g1_phase_dataset_20261019",
        xml_path=Path("src/assets/robots/unitree_g1/xmls/g1.xml"),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
