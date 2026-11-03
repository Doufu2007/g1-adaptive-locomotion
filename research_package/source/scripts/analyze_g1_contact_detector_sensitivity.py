"""Describe sensitivity of G1 kinematic touchdown events to detector rules.

This analysis is restricted to the four validation sequences and the frozen
144-cycle review map. It does not change labels or claim independent contact
ground truth. See the dated preregistration in results/g1_contact_followup_20261022/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_g1_walk_phase_dataset import (
    SOURCE_FPS,
    G1_XML,
    _model_contract,
    forward_kinematics,
    load_csv,
)
from src.tasks.velocity.phasor.data import (
    ContactDetectionConfig,
    _resolve_contact_candidates,
    detect_foot_contacts,
)


RAW_DIR = ROOT / "data" / "raw" / "lafan1_g1_walk"
MANUAL_PATH = (
    ROOT / "results" / "g1_phase_dataset_20261019" / "contact_consistency_filter_v1"
    / "touchdown_review_v1" / "manual_touchdown_audit_v1"
    / "manual_review_queue_B001-B144_v1.csv"
)
PROTOCOL = ROOT / "results" / "g1_contact_followup_20261022" / "followup_extension_preregistration_20261022.md"
OUT_DIR = ROOT / "results" / "g1_contact_followup_20261022" / "detector_sensitivity_v1"
SEQUENCES = tuple(f"walk{i}_subject1" for i in range(1, 5))
MATCH_TOLERANCE_S = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def detector_specs(base: ContactDetectionConfig) -> list[tuple[str, str, float, float, float, float]]:
    """Return detector id, type, and on/off thresholds in preregistered order."""
    specs: list[tuple[str, str, float, float, float, float]] = [
        ("baseline", "combined", base.height_on, base.height_off, base.speed_on, base.speed_off),
        ("height_only", "height", base.height_on, base.height_off, base.speed_on, base.speed_off),
        ("speed_only", "speed", base.height_on, base.height_off, base.speed_on, base.speed_off),
    ]
    for height_on in (0.02, 0.03, 0.04):
        for speed_on in (0.375, 0.5, 0.625):
            if np.isclose(height_on, base.height_on) and np.isclose(speed_on, base.speed_on):
                continue
            specs.append((
                f"combined_h{height_on:.3f}_s{speed_on:.3f}",
                "combined",
                height_on,
                height_on + 0.02,
                speed_on,
                speed_on + 0.3,
            ))
    return specs


def detect_variant(
    positions: np.ndarray,
    *,
    kind: str,
    height_on: float,
    height_off: float,
    speed_on: float,
    speed_off: float,
    min_contact_frames: int = 2,
    min_release_frames: int = 2,
) -> np.ndarray:
    velocity = np.gradient(positions.astype(np.float64), 1.0 / SOURCE_FPS, axis=0, edge_order=1)
    speed = np.linalg.norm(velocity, axis=-1)
    height = positions[..., 2]
    result = np.zeros((len(positions), 2), dtype=bool)
    for foot in range(2):
        if kind == "height":
            on = height[:, foot] <= height_on
            off = height[:, foot] >= height_off
        elif kind == "speed":
            on = speed[:, foot] <= speed_on
            off = speed[:, foot] >= speed_off
        elif kind == "combined":
            on = (height[:, foot] <= height_on) & (speed[:, foot] <= speed_on)
            off = (height[:, foot] >= height_off) | (speed[:, foot] >= speed_off)
        else:
            raise ValueError(f"unknown detector kind: {kind}")
        result[:, foot] = _resolve_contact_candidates(
            on,
            off,
            min_contact_frames=min_contact_frames,
            min_release_frames=min_release_frames,
        )
    return result


def touchdown_times(contact: np.ndarray) -> np.ndarray:
    rising = contact & ~np.concatenate((np.zeros(1, dtype=bool), contact[:-1]))
    return np.flatnonzero(rising).astype(np.float64) / SOURCE_FPS


def match_events(reference: np.ndarray, candidate: np.ndarray, tolerance_s: float) -> dict[str, Any]:
    i = j = 0
    shifts: list[float] = []
    while i < len(reference) and j < len(candidate):
        delta = float(candidate[j] - reference[i])
        if abs(delta) <= tolerance_s:
            shifts.append(delta)
            i += 1
            j += 1
        elif delta < -tolerance_s:
            j += 1
        else:
            i += 1
    matched = len(shifts)
    return {
        "reference_event_count": int(len(reference)),
        "candidate_event_count": int(len(candidate)),
        "matched_event_count": matched,
        "reference_match_fraction": float(matched / len(reference)) if len(reference) else None,
        "candidate_match_fraction": float(matched / len(candidate)) if len(candidate) else None,
        "median_absolute_shift_s": float(np.median(np.abs(shifts))) if shifts else None,
        "p90_absolute_shift_s": float(np.quantile(np.abs(shifts), 0.9)) if shifts else None,
        "signed_shifts_s": shifts,
    }


def floats_from_cell(value: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value or "")]


def load_review_rows() -> list[dict[str, str]]:
    with MANUAL_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 144:
        raise ValueError(f"expected 144 frozen review rows; found {len(rows)}")
    return rows


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"preserving existing analysis directory: {OUT_DIR}")
    if not PROTOCOL.is_file() or not MANUAL_PATH.is_file():
        raise FileNotFoundError("frozen extension protocol or manual cycle map is missing")

    base = ContactDetectionConfig()
    specs = detector_specs(base)
    manual_rows = load_review_rows()
    if set(row["sequence_id"] for row in manual_rows) != set(SEQUENCES):
        raise ValueError("manual review mapping does not match the four frozen validation sequences")

    model, data, qpos, feet = _model_contract(G1_XML)
    event_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    sequence_records: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}

    for sequence_id in SEQUENCES:
        csv_path = RAW_DIR / f"{sequence_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        source_hashes[csv_path.name] = sha256(csv_path)
        raw = load_csv(csv_path)
        foot_positions = forward_kinematics(raw, model, data, qpos, feet)
        baseline = detect_foot_contacts(foot_positions, fps=SOURCE_FPS, config=base)
        times_by_detector: dict[str, list[np.ndarray]] = {}
        sequence_records[sequence_id] = {"raw_frames": int(len(raw)), "detectors": {}}

        for detector_id, kind, h_on, h_off, s_on, s_off in specs:
            contact = baseline if detector_id == "baseline" else detect_variant(
                foot_positions,
                kind=kind,
                height_on=h_on,
                height_off=h_off,
                speed_on=s_on,
                speed_off=s_off,
                min_contact_frames=base.min_contact_frames,
                min_release_frames=base.min_release_frames,
            )
            foot_times = [touchdown_times(contact[:, foot]) for foot in range(2)]
            times_by_detector[detector_id] = foot_times
            sequence_records[sequence_id]["detectors"][detector_id] = {
                "kind": kind,
                "height_on_m": h_on,
                "height_off_m": h_off,
                "speed_on_m_s": s_on,
                "speed_off_m_s": s_off,
                "left_touchdown_count": int(len(foot_times[0])),
                "right_touchdown_count": int(len(foot_times[1])),
            }
            for foot, foot_name in enumerate(("left", "right")):
                if detector_id == "baseline":
                    continue
                comparison = match_events(
                    times_by_detector["baseline"][foot], foot_times[foot], MATCH_TOLERANCE_S
                )
                event_rows.append({
                    "sequence_id": sequence_id,
                    "foot": foot_name,
                    "detector_id": detector_id,
                    "kind": kind,
                    "height_on_m": h_on,
                    "height_off_m": h_off,
                    "speed_on_m_s": s_on,
                    "speed_off_m_s": s_off,
                    **comparison,
                })

        for row in (item for item in manual_rows if item["sequence_id"] == sequence_id):
            start_values = floats_from_cell(row["left_touchdown_start_raw_times_s"])
            end_values = floats_from_cell(row["left_touchdown_end_raw_times_s"])
            if len(start_values) != 1 or len(end_values) != 1:
                raise ValueError(f"{row['blind_id']}: expected one frozen start/end timestamp")
            start, end = start_values[0], end_values[0]
            baseline_count = int(row["right_touchdown_count"])
            for detector_id, _, *_ in specs:
                right_times = times_by_detector[detector_id][1]
                count = int(np.count_nonzero((right_times >= start) & (right_times < end)))
                if detector_id == "baseline" and count != baseline_count:
                    raise ValueError(
                        f"{row['blind_id']}: reconstructed baseline event count {count} "
                        f"does not match frozen map {baseline_count}"
                    )
                cycle_rows.append({
                    "blind_id": row["blind_id"],
                    "sequence_id": sequence_id,
                    "cycle_id": row["cycle_id"],
                    "category": row["category"],
                    "baseline_retained": row["retained"].lower() == "true",
                    "detector_id": detector_id,
                    "right_touchdown_count": count,
                    "retained_by_same_rule": count == 1,
                    "classification_changed": (count == 1) != (row["retained"].lower() == "true"),
                })

    OUT_DIR.mkdir(parents=True)
    with (OUT_DIR / "event_matching.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    with (OUT_DIR / "review_cycle_detector_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cycle_rows[0]))
        writer.writeheader()
        writer.writerows(cycle_rows)

    aggregate: dict[str, Any] = {}
    for detector_id, *_ in specs:
        rows = [row for row in cycle_rows if row["detector_id"] == detector_id]
        changed = sum(bool(row["classification_changed"]) for row in rows)
        counts = {str(k): sum(row["right_touchdown_count"] == k for row in rows) for k in (0, 1, 2)}
        counts[">2"] = sum(row["right_touchdown_count"] > 2 for row in rows)
        aggregate[detector_id] = {
            "review_cycle_count": len(rows),
            "changed_retained_excluded_classification_count": int(changed),
            "right_touchdown_count_distribution": counts,
        }

    manifest = {
        "schema": "g1.contact_detector_sensitivity.v1",
        "status": "DESCRIPTIVE_SENSITIVITY_ONLY",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "manual_mapping_sha256": sha256(MANUAL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "g1_model_path": str(G1_XML.relative_to(ROOT)),
        "g1_model_sha256": sha256(G1_XML),
        "source_sequence_sha256": source_hashes,
        "split": "validation only: subject 1, four walking sequences",
        "baseline_config": {
            "height_on_m": base.height_on,
            "height_off_m": base.height_off,
            "speed_on_m_s": base.speed_on,
            "speed_off_m_s": base.speed_off,
            "min_contact_frames": base.min_contact_frames,
            "min_release_frames": base.min_release_frames,
        },
        "event_match_tolerance_s": MATCH_TOLERANCE_S,
        "aggregate_review_cycle_results": aggregate,
        "by_sequence": sequence_records,
        "interpretation_limits": [
            "All detectors use the same retargeted G1 kinematic trajectories; this is not independent contact ground truth.",
            "No labels, training rows, or frozen cycle-selection rules were changed.",
            "Review-cycle uncertainty intervals remain descriptive visual observations.",
        ],
    }
    (OUT_DIR / "sensitivity_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "detector_count": len(specs), "cycle_rows": len(cycle_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
