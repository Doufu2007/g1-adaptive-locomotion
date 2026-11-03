"""Freeze the learned-phase contact reward template from train subjects only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.build_g1_walk_phase_dataset import (
  SOURCE_FPS,
  _model_contract,
  _touchdown_times,
  forward_kinematics,
  load_csv,
  phase_from_touchdown_times,
)
from src.tasks.velocity.phasor.data import ContactDetectionConfig, G1_JOINT_ORDER, detect_foot_contacts


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results/g1_phase_dataset_20261019"
RAW_DIR = ROOT / "data/raw/lafan1_g1_walk"
METADATA_PATH = DATA_DIR / "dataset_metadata.json"
MANIFEST_PATH = ROOT / "docs/data/g1_phase_data_provenance.json"
OUTPUT_PATH = DATA_DIR / "g1_reward_template_v1.json"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def stance_duty_fractions(contact: np.ndarray) -> np.ndarray:
  """Return complete touchdown-to-touchdown stance fractions for one foot."""
  values = np.asarray(contact, dtype=bool)
  if values.ndim != 1:
    raise ValueError("contact must be one-dimensional")
  touchdown = np.flatnonzero(values & ~np.concatenate(([False], values[:-1])))
  liftoff = np.flatnonzero(~values & np.concatenate(([False], values[:-1])))
  fractions: list[float] = []
  for start, end in zip(touchdown[:-1], touchdown[1:], strict=True):
    if start == 0 or end <= start:
      continue
    candidates = liftoff[(liftoff > start) & (liftoff < end)]
    if candidates.size:
      fractions.append(float((candidates[0] - start) / (end - start)))
  return np.asarray(fractions, dtype=np.float64)


def circular_median_cycles(values: list[float] | np.ndarray) -> float:
  """Choose the observed phase minimizing total absolute circular distance."""
  phases = np.mod(np.asarray(values, dtype=np.float64), 1.0)
  if phases.ndim != 1 or phases.size == 0 or not np.isfinite(phases).all():
    raise ValueError("right touchdown phases must be a non-empty finite sequence")
  distances = np.abs(np.mod(phases[:, None] - phases[None, :] + 0.5, 1.0) - 0.5)
  return float(phases[int(np.argmin(distances.sum(axis=1)))])


def _sequence_contacts(
  sequence_id: str,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_addresses: np.ndarray,
  foot_sites: tuple[int, int],
) -> dict[str, Any]:
  csv_path = RAW_DIR / f"{sequence_id}.csv"
  values = load_csv(csv_path)
  times = np.arange(len(values), dtype=np.float64) / SOURCE_FPS
  foot_positions = forward_kinematics(values, model, data, qpos_addresses, foot_sites)
  contacts = detect_foot_contacts(foot_positions, fps=SOURCE_FPS, config=ContactDetectionConfig())
  touchdowns = [_touchdown_times(contacts[:, foot], times) for foot in range(2)]
  if len(touchdowns[0]) < 3:
    raise ValueError(f"{sequence_id} has too few left-foot cycles")
  labels = phase_from_touchdown_times(times, touchdowns[0], touchdowns[1])
  duties = [stance_duty_fractions(contacts[:, foot]) for foot in range(2)]
  return {
    "contacts": contacts,
    "labels": labels,
    "duties": duties,
    "raw_sha256": sha256_file(csv_path),
  }


def _contact_agreement(
  sequence: dict[str, Any], *, offset_cycles: tuple[float, float], threshold: float
) -> dict[str, float | int]:
  contacts = sequence["contacts"]
  labels = sequence["labels"]
  valid = labels.valid
  phase = labels.phase_cycles[valid]
  if phase.size == 0:
    raise ValueError("validation sequence has no complete left cycles")
  predicted = np.stack(
    [np.mod(phase + offset_cycles[foot], 1.0) < threshold for foot in range(2)], axis=1
  )
  observed = contacts[valid]
  per_foot = (predicted == observed).mean(axis=0)
  return {
    "valid_frames": int(valid.sum()),
    "left_contact_agreement": float(per_foot[0]),
    "right_contact_agreement": float(per_foot[1]),
    "combined_contact_agreement": float((predicted == observed).mean()),
  }


def calibrate_reward_template() -> dict[str, Any]:
  metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
  manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
  split = metadata["split"]["sequences"]
  train_ids = list(split["train"])
  validation_ids = list(split["validation"])
  if set(train_ids) & set(validation_ids) or set(train_ids) & set(split["test"]):
    raise ValueError("sequence splits overlap")
  if tuple(metadata["joint_order"]) != G1_JOINT_ORDER:
    raise ValueError("dataset joint order differs from the runtime G1 contract")
  if metadata["contact_rule"] != asdict(ContactDetectionConfig()):
    raise ValueError("contact detection rule differs from the dataset labels")

  model, data, qpos_addresses, foot_sites = _model_contract(
    ROOT / "src/assets/robots/unitree_g1/xmls/g1.xml"
  )
  train_records = {sequence_id: _sequence_contacts(sequence_id, model, data, qpos_addresses, foot_sites) for sequence_id in train_ids}
  validation_records = {
    sequence_id: _sequence_contacts(sequence_id, model, data, qpos_addresses, foot_sites)
    for sequence_id in validation_ids
  }

  right_touchdown_phases = [
    float(phase)
    for sequence_id in train_ids
    for phase in train_records[sequence_id]["labels"].right_touchdown_phase_cycles.tolist()
  ]
  right_touchdown_median = circular_median_cycles(right_touchdown_phases)
  right_offset = float(np.mod(-right_touchdown_median, 1.0))
  duty_by_foot = [
    np.concatenate(
      [train_records[sequence_id]["duties"][foot] for sequence_id in train_ids]
    )
    for foot in range(2)
  ]
  if any(values.size == 0 for values in duty_by_foot):
    raise ValueError("train split has no complete stance cycles for one or both feet")
  all_duties = np.concatenate(duty_by_foot)
  threshold = float(np.median(all_duties))
  if not 0.0 < threshold < 1.0:
    raise ValueError("calibrated stance threshold is outside (0, 1)")

  offsets = (0.0, right_offset)
  validation = {
    sequence_id: _contact_agreement(
      validation_records[sequence_id], offset_cycles=offsets, threshold=threshold
    )
    for sequence_id in validation_ids
  }
  contact_review = manifest.get("candidate_walk_sequences", {}).get("contact_verification", {})
  uncertain_ids = [
    item.get("sequence_id") for item in contact_review.get("uncertain_intervals", [])
  ]
  return {
    "schema": "g1.learned_phase_reward_template.v1",
    "status": "FROZEN",
    "fit_split": "train",
    "dataset_sha256": sha256_file(DATA_DIR / "phase_dataset.npz"),
    "dataset_metadata_sha256": sha256_file(METADATA_PATH),
    "provenance_manifest_sha256": sha256_file(MANIFEST_PATH),
    "calibration_script_sha256": sha256_file(Path(__file__).resolve()),
    "joint_order": list(G1_JOINT_ORDER),
    "contact_rule": metadata["contact_rule"],
    "train_sequences": train_ids,
    "train_raw_sha256": {
      sequence_id: train_records[sequence_id]["raw_sha256"] for sequence_id in train_ids
    },
    "training_uncertain_sequence_ids_retained": uncertain_ids,
    "right_touchdown_phase_cycles": {
      "count": len(right_touchdown_phases),
      "circular_median": right_touchdown_median,
      "offset_cycles": right_offset,
      "offset_rule": "(-right_touchdown_circular_median) mod 1",
    },
    "stance_duty_fraction": {
      "fit_rule": "median of complete left and right touchdown-to-touchdown stance fractions",
      "left_count": int(duty_by_foot[0].size),
      "right_count": int(duty_by_foot[1].size),
      "left_median": float(np.median(duty_by_foot[0])),
      "right_median": float(np.median(duty_by_foot[1])),
      "pooled_q10": float(np.quantile(all_duties, 0.10)),
      "pooled_median": threshold,
      "pooled_q90": float(np.quantile(all_duties, 0.90)),
    },
    "offset_cycles": list(offsets),
    "stance_threshold": threshold,
    "validation_split": "validation",
    "validation_sequences": validation_ids,
    "validation_contact_agreement": validation,
    "test_split_used_for_fit_or_validation": False,
  }


def main() -> None:
  report = calibrate_reward_template()
  OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
