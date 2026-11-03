from __future__ import annotations

import numpy as np


def phase_to_angle(phase_2d: np.ndarray) -> np.ndarray:
    """Decode canonical ``(sin(phi), cos(phi))`` vectors to radians."""
    values = np.asarray(phase_2d, dtype=np.float64)
    if values.shape[-1] != 2:
        raise ValueError("phase_2d must have final dimension 2")
    return np.arctan2(values[..., 0], values[..., 1])


def _phase_cycles(phase_2d: np.ndarray, *, phase_order: str = "sin_cos") -> np.ndarray:
    values = np.asarray(phase_2d, dtype=np.float64)
    if phase_order == "sin_cos":
        angles = phase_to_angle(values)
    elif phase_order == "cos_sin":
        angles = np.arctan2(values[..., 1], values[..., 0])
    else:
        raise ValueError("phase_order must be 'sin_cos' or 'cos_sin'")
    return np.mod(angles / (2.0 * np.pi), 1.0)


def _infer_phase_order(
    phase_2d: np.ndarray,
    contacts: np.ndarray,
    offsets: tuple[float, ...],
    threshold: float,
) -> str:
    """Read old saved ``(cos, sin)`` artifacts without changing new outputs."""
    contact = np.asarray(contacts, dtype=bool)
    scores: dict[str, float] = {}
    for order in ("sin_cos", "cos_sin"):
        cycles = _phase_cycles(phase_2d, phase_order=order)
        offset_values = np.asarray(offsets, dtype=np.float64)
        predicted = np.mod(cycles[..., None] + offset_values, 1.0) < threshold
        scores[order] = float(np.mean(predicted == contact))
    return max(scores, key=lambda order: (scores[order], order == "sin_cos"))


def circular_coverage_rad(angles: np.ndarray) -> float:
    """Length of the shortest circular arc containing the given angles."""
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one angle is required")
    wrapped = np.sort(np.mod(values, 2.0 * np.pi))
    if wrapped.size == 1:
        return 0.0
    gaps = np.diff(np.concatenate((wrapped, wrapped[:1] + 2.0 * np.pi)))
    return float(np.clip(2.0 * np.pi - gaps.max(), 0.0, 2.0 * np.pi))


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Unwrap one continuous sequence; call separately for each episode."""
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one angle is required")
    return np.unwrap(values)


def phase_contact_summary(
    phase_2d: np.ndarray,
    contacts: np.ndarray,
    offsets: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.56,
    phase_order: str = "auto",
) -> dict[str, object]:
    """Summarize phase/contact agreement for ``[time, env, foot]`` sequences."""
    phase = np.asarray(phase_2d, dtype=np.float64)
    contact = np.asarray(contacts, dtype=bool)
    offset_values = np.asarray(offsets, dtype=np.float64)
    if phase.ndim != 3 or phase.shape[-1] != 2:
        raise ValueError("phase_2d must have shape (time, env, 2)")
    if contact.shape != (*phase.shape[:2], len(offset_values)):
        raise ValueError("contacts must have shape (time, env, number_of_offsets)")
    if phase.shape[0] == 0 or phase.shape[1] == 0 or len(offset_values) == 0:
        raise ValueError("phase and contact sequences must be non-empty")
    if not np.isfinite(phase).all() or not np.isfinite(offset_values).all():
        raise ValueError("phase and offsets must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    if phase_order == "auto":
        phase_order = _infer_phase_order(phase, contact, tuple(offset_values), threshold)
    cycles = _phase_cycles(phase, phase_order=phase_order)
    leg_cycles = np.mod(cycles[..., None] + offset_values, 1.0)
    predicted_contact = leg_cycles < threshold
    matches = predicted_contact == contact

    previous_contact = np.concatenate(
        (np.zeros_like(contact[:1]), contact[:-1]), axis=0
    )
    touchdowns = contact & ~previous_contact
    touchdown_counts = touchdowns.sum(axis=(0, 1))
    touchdown_resultants: list[float | None] = []
    for foot in range(contact.shape[-1]):
        event_cycles = cycles[touchdowns[..., foot]]
        if event_cycles.size == 0:
            touchdown_resultants.append(None)
        else:
            vectors = np.exp(2j * np.pi * event_cycles)
            touchdown_resultants.append(float(np.abs(vectors.mean())))

    unwrapped = np.unwrap(
        _phase_cycles(phase, phase_order=phase_order) * (2.0 * np.pi), axis=0
    )
    angular_steps = np.diff(unwrapped, axis=0)
    total_travel = np.abs(angular_steps).sum(axis=0)
    net_travel = np.abs(angular_steps.sum(axis=0))
    travel_ratio = np.divide(
        net_travel,
        total_travel,
        out=np.zeros_like(net_travel),
        where=total_travel > 0.0,
    )

    return {
        "contact_agreement": float(matches.mean()),
        "contact_agreement_by_foot": matches.mean(axis=(0, 1)).tolist(),
        "touchdown_counts": touchdown_counts.astype(int).tolist(),
        "touchdown_phase_resultants": touchdown_resultants,
        "mean_net_to_total_angular_travel": float(travel_ratio.mean()),
    }


def calibrate_phase_alignment(
    phase_2d: np.ndarray,
    contacts: np.ndarray,
    offsets: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.56,
    grid_size: int = 1000,
) -> dict[str, float | int]:
    """Fit one global phase rotation/reflection to calibration contact labels."""
    phase = np.asarray(phase_2d, dtype=np.float64)
    contact = np.asarray(contacts, dtype=bool)
    offset_values = np.asarray(offsets, dtype=np.float64)
    if phase.ndim != 3 or phase.shape[-1] != 2:
        raise ValueError("phase_2d must have shape (time, env, 2)")
    if contact.shape != (*phase.shape[:2], len(offset_values)):
        raise ValueError("contacts must have shape (time, env, number_of_offsets)")
    if phase.shape[0] == 0 or phase.shape[1] == 0 or len(offset_values) == 0:
        raise ValueError("phase and contact sequences must be non-empty")
    if not np.isfinite(phase).all() or not np.isfinite(offset_values).all():
        raise ValueError("phase and offsets must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")

    phase_order = _infer_phase_order(phase, contact, tuple(offset_values), threshold)
    cycles = _phase_cycles(phase, phase_order=phase_order)
    best_agreement = -1.0
    best_direction = 1
    best_shift = 0.0
    for direction in (1, -1):
        directed_cycles = direction * cycles
        for shift in np.arange(grid_size, dtype=np.float64) / grid_size:
            leg_cycles = np.mod(directed_cycles[..., None] + shift + offset_values, 1.0)
            agreement = float(np.mean((leg_cycles < threshold) == contact))
            if agreement > best_agreement:
                best_agreement = agreement
                best_direction = direction
                best_shift = float(shift)

    return {
        "phase_direction": best_direction,
        "phase_shift_cycles": best_shift,
        "calibration_contact_agreement": best_agreement,
        "phase_order": phase_order,
    }


def apply_phase_alignment(
    phase_2d: np.ndarray,
    direction: int,
    shift_cycles: float,
    *,
    phase_order: str = "sin_cos",
) -> np.ndarray:
    """Apply a fitted global reflection and rotation to a phase sequence."""
    phase = np.asarray(phase_2d, dtype=np.float64)
    if phase.ndim < 2 or phase.shape[-1] != 2:
        raise ValueError("phase_2d must have shape (..., 2)")
    if not np.isfinite(phase).all():
        raise ValueError("phase_2d must be finite")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not np.isfinite(shift_cycles):
        raise ValueError("shift_cycles must be finite")

    cycles = _phase_cycles(phase, phase_order=phase_order)
    aligned_cycles = np.mod(direction * cycles + shift_cycles, 1.0)
    aligned_angles = aligned_cycles * (2.0 * np.pi)
    return np.stack((np.sin(aligned_angles), np.cos(aligned_angles)), axis=-1)


def evaluate_phase_alignment(
    calibration_phase: np.ndarray,
    calibration_contacts: np.ndarray,
    heldout_phase: np.ndarray,
    heldout_contacts: np.ndarray,
    offsets: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.56,
    grid_size: int = 1000,
) -> dict[str, float | int | list[float]]:
    """Fit alignment on one seed and evaluate it unchanged on held-out traces."""
    alignment = calibrate_phase_alignment(
        calibration_phase,
        calibration_contacts,
        offsets=offsets,
        threshold=threshold,
        grid_size=grid_size,
    )
    direction = int(alignment["phase_direction"])
    shift_cycles = float(alignment["phase_shift_cycles"])
    phase_order = str(alignment.get("phase_order", "sin_cos"))
    calibration_aligned = apply_phase_alignment(
        calibration_phase, direction, shift_cycles, phase_order=phase_order
    )
    heldout_aligned = apply_phase_alignment(
        heldout_phase, direction, shift_cycles, phase_order=phase_order
    )

    calibration_summary = phase_contact_summary(
        calibration_aligned,
        calibration_contacts,
        offsets=offsets,
        threshold=threshold,
    )
    heldout_summary = phase_contact_summary(
        heldout_aligned,
        heldout_contacts,
        offsets=offsets,
        threshold=threshold,
    )
    heldout_contacts_array = np.asarray(heldout_contacts, dtype=bool)
    heldout_by_env = [
        phase_contact_summary(
            heldout_aligned[:, env_id : env_id + 1],
            heldout_contacts_array[:, env_id : env_id + 1],
            offsets=offsets,
            threshold=threshold,
        )["contact_agreement"]
        for env_id in range(heldout_aligned.shape[1])
    ]

    return {
        "phase_direction": direction,
        "phase_shift_cycles": shift_cycles,
        "calibration_contact_agreement": float(
            calibration_summary["contact_agreement"]
        ),
        "heldout_contact_agreement": float(heldout_summary["contact_agreement"]),
        "heldout_contact_agreement_by_env": [float(value) for value in heldout_by_env],
    }
