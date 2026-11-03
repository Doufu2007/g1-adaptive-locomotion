"""Data contracts for contact-event phase supervision.

This module deliberately keeps provenance validation and window construction
separate from the simulator.  Contact labels are consumed offline only; the
runtime encoder receives joint-position history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


G1_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class ProvenanceError(ValueError):
    """Raised when a data manifest is not safe for phase training."""


@dataclass(frozen=True)
class ContactDetectionConfig:
    """Deterministic foot contact rule applied to retargeted kinematics."""

    height_on: float = 0.03
    height_off: float = 0.05
    speed_on: float = 0.5
    speed_off: float = 0.8
    min_contact_frames: int = 2
    min_release_frames: int = 2

    def __post_init__(self) -> None:
        if not (0.0 < self.height_on < self.height_off):
            raise ValueError("height_on must be positive and below height_off")
        if not (0.0 < self.speed_on < self.speed_off):
            raise ValueError("speed_on must be positive and below speed_off")
        if self.min_contact_frames < 1 or self.min_release_frames < 1:
            raise ValueError("debounce frame counts must be positive")


@dataclass(frozen=True)
class PhaseLabels:
    """Operational phase labels and contact-event audit information."""

    phase_cycles: np.ndarray
    valid: np.ndarray
    left_touchdown_indices: np.ndarray
    right_touchdown_indices: np.ndarray
    right_touchdown_phase_cycles: np.ndarray
    complete_cycle_count: int


@dataclass(frozen=True)
class WindowBatch:
    """Overlapping windows whose labels are attached to the final frame."""

    windows: np.ndarray
    labels: np.ndarray
    sequence_ids: np.ndarray
    frame_indices: np.ndarray
    label_frame_indices: np.ndarray


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, frames: np.ndarray) -> np.ndarray:
        values = np.asarray(frames, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("normalizer input must be finite")
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("normalizer input must have shape [frames, joints]")
        return (values - self.mean) / self.std


def validate_provenance(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless a walk/contact-verified manifest is complete."""

    current = manifest.get("current_file", {})
    candidate = manifest.get("candidate_walk_sequences", {})
    file_metadata = candidate.get("file_metadata", [])
    if current:
        file_metadata = [current]
    if not file_metadata:
        raise ProvenanceError("no candidate walking files are resolved")
    for record in file_metadata:
        action = str(record.get("action_type", "")).lower()
        if "walk" not in action or record.get("walk_verified") is not True:
            raise ProvenanceError("training source is not verified walking data")
        if record.get("license") != "CC BY-NC-ND-4.0":
            raise ProvenanceError("source licence is unresolved")
        mapping = record.get("retargeting", {}).get("joint_mapping")
        if not mapping:
            raise ProvenanceError("G1 joint mapping is unresolved for a candidate file")
    if len(manifest.get("g1_joint_order", [])) != 29:
        raise ProvenanceError("G1 29-joint mapping is unresolved")
    contact = candidate.get("contact_verification", {})
    if contact.get("status") != "verified" or int(contact.get("manual_sample_count", 0)) <= 0:
        raise ProvenanceError("foot-contact labels require verification before training")
    if manifest.get("status") != "READY":
        raise ProvenanceError("provenance manifest is not READY")


def _touchdown_indices(contact: np.ndarray) -> np.ndarray:
    previous = np.concatenate((np.zeros(1, dtype=bool), contact[:-1]))
    return np.flatnonzero(contact & ~previous)


def _resolve_contact_candidates(
    on_candidate: np.ndarray,
    off_candidate: np.ndarray,
    *,
    min_contact_frames: int,
    min_release_frames: int,
) -> np.ndarray:
    """Apply the shared consecutive-frame contact transition rule."""
    if on_candidate.ndim != 1 or off_candidate.shape != on_candidate.shape:
        raise ValueError("contact candidates must be matching one-dimensional arrays")
    if min_contact_frames < 1 or min_release_frames < 1:
        raise ValueError("debounce frame counts must be positive")
    output = np.zeros(on_candidate.shape, dtype=bool)
    active = False
    on_count = 0
    off_count = 0
    for index, (is_on, is_off) in enumerate(zip(on_candidate, off_candidate)):
        if active:
            if is_off:
                off_count += 1
                if off_count >= min_release_frames:
                    active = False
                    off_count = 0
            else:
                off_count = 0
        elif is_on:
            on_count += 1
            if on_count >= min_contact_frames:
                active = True
                on_count = 0
        else:
            on_count = 0
        output[index] = active
    return output


def debounce_contact_sequence(
    contact: np.ndarray,
    *,
    min_contact_frames: int = 2,
    min_release_frames: int = 2,
) -> np.ndarray:
    """Debounce a sampled binary contact signal with the training rule."""
    values = np.asarray(contact, dtype=bool)
    if values.ndim != 1:
        raise ValueError("contact must be a one-dimensional sequence")
    return _resolve_contact_candidates(
        values,
        ~values,
        min_contact_frames=min_contact_frames,
        min_release_frames=min_release_frames,
    )


def detect_foot_contacts(
    foot_positions: np.ndarray,
    *,
    fps: float,
    config: ContactDetectionConfig | None = None,
) -> np.ndarray:
    """Detect left/right contact from retargeted foot height and speed.

    The detector uses separate on/off thresholds and consecutive-frame
    debounce.  It is intentionally a kinematic operational label, not a
    claim of force-ground truth.
    """

    positions = np.asarray(foot_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError("foot_positions must have shape [time, 2, 3]")
    if len(positions) < 2 or not np.isfinite(positions).all() or fps <= 0.0:
        raise ValueError("foot_positions must be finite and contain two frames")
    cfg = config or ContactDetectionConfig()
    velocity = np.gradient(positions, 1.0 / fps, axis=0, edge_order=1)
    speed = np.linalg.norm(velocity, axis=-1)
    height = positions[..., 2]
    contacts = np.zeros((len(positions), 2), dtype=bool)
    for foot in range(2):
        on_candidate = (height[:, foot] <= cfg.height_on) & (
            speed[:, foot] <= cfg.speed_on
        )
        off_candidate = (height[:, foot] >= cfg.height_off) | (
            speed[:, foot] >= cfg.speed_off
        )
        contacts[:, foot] = _resolve_contact_candidates(
            on_candidate,
            off_candidate,
            min_contact_frames=cfg.min_contact_frames,
            min_release_frames=cfg.min_release_frames,
        )
    return contacts


def resample_joint_positions(
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    *,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a time-major joint sequence at a fixed rate."""

    times = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(joint_pos, dtype=np.float32)
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values):
        raise ValueError("timestamps and joint_pos must have matching time dimensions")
    if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("timestamps and joint_pos must be finite and non-empty")
    if np.any(np.diff(times) <= 0.0) or target_hz <= 0.0:
        raise ValueError("timestamps must increase and target_hz must be positive")
    count = int(np.floor((times[-1] - times[0]) * target_hz + 1e-9)) + 1
    new_times = times[0] + np.arange(count, dtype=np.float64) / target_hz
    new_values = np.stack(
        [np.interp(new_times, times, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)
    return new_times, new_values


def left_cycle_phase(
    contacts: np.ndarray,
    timestamps: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> PhaseLabels:
    """Build phase from consecutive left touchdowns.

    Values outside complete left-foot cycles are NaN.  Right-foot touchdown
    phases are measured from the same left-cycle timestamps and returned for
    reporting; no half-cycle offset is assumed.
    """

    contact = np.asarray(contacts, dtype=bool)
    times = np.asarray(timestamps, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError("contacts must have shape [time, 2] as [left, right]")
    if times.ndim != 1 or times.shape[0] != contact.shape[0]:
        raise ValueError("timestamps must have shape [time]")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if active_mask is None:
        active = np.ones(len(times), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (len(times),):
            raise ValueError("active_mask must have shape [time]")

    left = _touchdown_indices(contact[:, 0])
    right = _touchdown_indices(contact[:, 1])
    phase = np.full(len(times), np.nan, dtype=np.float32)
    valid = np.zeros(len(times), dtype=bool)
    right_phases: list[float] = []
    complete = 0
    for start, end in zip(left[:-1], left[1:]):
        if end <= start or not active[start : end + 1].all():
            continue
        duration = times[end] - times[start]
        values = np.mod((times[start : end + 1] - times[start]) / duration, 1.0)
        phase[start : end + 1] = values.astype(np.float32)
        valid[start : end + 1] = True
        complete += 1
        inside_right = right[(right >= start) & (right < end)]
        right_phases.extend(
            ((times[inside_right] - times[start]) / duration).astype(float).tolist()
        )

    return PhaseLabels(
        phase_cycles=phase,
        valid=valid,
        left_touchdown_indices=left,
        right_touchdown_indices=right,
        right_touchdown_phase_cycles=np.asarray(right_phases, dtype=np.float32),
        complete_cycle_count=complete,
    )


def split_sequences(
    sequence_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign complete sequences to train/validation/test before windows."""

    ids = list(dict.fromkeys(str(value) for value in sequence_ids))
    if len(ids) < 3:
        raise ValueError("at least three complete sequences are required")
    if len(fractions) != 3 or any(value <= 0.0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    counts = [max(1, int(round(len(ids) * fraction))) for fraction in fractions]
    while sum(counts) > len(ids):
        largest = int(np.argmax(counts))
        if counts[largest] > 1:
            counts[largest] -= 1
        else:
            break
    while sum(counts) < len(ids):
        counts[0] += 1
    first, second = counts[0], counts[0] + counts[1]
    return {"train": ids[:first], "validation": ids[first:second], "test": ids[second:]}


def make_windows(
    joint_pos: np.ndarray,
    labels: PhaseLabels | np.ndarray,
    *,
    sequence_ids: Sequence[str] | None = None,
    window: int = 10,
) -> WindowBatch:
    """Generate windows after sequence assignment; labels use final frames."""

    values = np.asarray(joint_pos, dtype=np.float32)
    if values.ndim != 2 or len(values) < window:
        raise ValueError("joint_pos must have shape [frames, joints] and fit the window")
    phase = labels.phase_cycles if isinstance(labels, PhaseLabels) else np.asarray(labels)
    valid = labels.valid if isinstance(labels, PhaseLabels) else np.isfinite(phase)
    if phase.shape != (len(values),) or valid.shape != (len(values),):
        raise ValueError("labels must match the frame count")
    ids = np.asarray(sequence_ids if sequence_ids is not None else ["sequence"] * len(values))
    if ids.shape != (len(values),):
        raise ValueError("sequence_ids must have shape [frames]")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    frame_rows: list[np.ndarray] = []
    sequence_rows: list[np.ndarray] = []
    label_indices: list[int] = []
    for end in range(window - 1, len(values)):
        row_ids = ids[end - window + 1 : end + 1]
        if len(set(row_ids.tolist())) != 1 or not bool(valid[end]) or not np.isfinite(phase[end]):
            continue
        windows.append(values[end - window + 1 : end + 1])
        targets.append(float(phase[end]))
        frame_rows.append(np.arange(end - window + 1, end + 1, dtype=np.int64))
        sequence_rows.append(row_ids.copy())
        label_indices.append(end)
    if not windows:
        empty = np.empty((0, window, values.shape[1]), dtype=np.float32)
        return WindowBatch(empty, np.empty(0, dtype=np.float32), np.empty((0, window), dtype=ids.dtype), np.empty((0, window), dtype=np.int64), np.empty(0, dtype=np.int64))
    return WindowBatch(
        np.stack(windows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.stack(sequence_rows),
        np.stack(frame_rows),
        np.asarray(label_indices, dtype=np.int64),
    )


def fit_normalizer(train_frames: np.ndarray, eps: float = 1e-8) -> Normalizer:
    values = np.asarray(train_frames, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("train_frames must be a non-empty finite [frames, joints] array")
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), eps).astype(np.float32)
    return Normalizer(mean=mean, std=std)
