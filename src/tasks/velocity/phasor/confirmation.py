"""Frozen scenario helpers for independent G1 phase confirmation rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np


CONDITION_NAMES = (
    "slow_constant",
    "medium_constant",
    "fast_constant",
    "speed_stand_transition",
)
CONSTANT_FORWARD_SPEEDS = (0.4, 0.7, 0.9)
DISTURBANCE_STEPS = (250, 500)


def quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-first root quaternions to roll, pitch, yaw in radians."""
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1:] != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must be finite with a final dimension of four")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = np.moveaxis(values / norms, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def choose_feasible_policy_candidate(candidates: list[dict[str, Any]]) -> int | None:
    """Choose only covered policies, then rank by terminations and cycle count."""
    feasible: list[tuple[int, int, int]] = []
    for candidate in candidates:
        coverage = candidate.get("coverage_status")
        termination_count = candidate.get("termination_count")
        cycle_count = candidate.get("complete_cycle_count")
        policy_seed = candidate.get("policy_seed")
        if coverage != "adequate":
            continue
        values = (termination_count, cycle_count, policy_seed)
        if any(not isinstance(value, (int, np.integer)) or value < 0 for value in values):
            raise ValueError("feasible candidate ranking requires non-negative integer counts and seed")
        feasible.append((int(termination_count), -int(cycle_count), int(policy_seed)))
    if not feasible:
        return None
    return min(feasible)[2]


def advance_episode_segments(
    *,
    block_id: int,
    reset_counts: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    block_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance per-environment episode IDs after an auto-reset step."""
    counts = np.asarray(reset_counts, dtype=np.int64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if counts.ndim != 1 or terminated.shape != counts.shape or truncated.shape != counts.shape:
        raise ValueError("reset counts and done masks must be matching one-dimensional arrays")
    if block_id < 0 or block_span <= 0:
        raise ValueError("block_id must be non-negative and block_span must be positive")

    updated_counts = counts + np.logical_or(terminated, truncated).astype(np.int64)
    if np.any(updated_counts >= block_span):
        raise ValueError("episode segment count exceeds block_span")
    episode_ids = block_id * block_span + updated_counts
    return updated_counts, episode_ids


def command_for_step(condition_id: int, step_index: int) -> float:
    """Forward command in m/s for a 50 Hz, 15 s confirmation episode."""
    if condition_id in (0, 1, 2):
        return CONSTANT_FORWARD_SPEEDS[condition_id]
    if condition_id != 3:
        raise ValueError(f"unknown confirmation condition: {condition_id}")
    if step_index < 100:
        return 0.0
    if step_index < 250:
        return 0.6
    if step_index < 350:
        return 0.0
    return 0.6


def onnx_batch_one_actions(
    session: Any,
    actor_observations: np.ndarray,
    *,
    input_name: str,
    output_name: str,
    action_dim: int,
) -> np.ndarray:
    """Run an exported fixed-batch-one actor independently for each environment."""
    observations = np.asarray(actor_observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("actor_observations must have non-empty shape [env, feature]")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    actions: list[np.ndarray] = []
    for row in observations:
        result = session.run([output_name], {input_name: row[None, :]})
        if not result:
            raise ValueError("ONNX policy returned no outputs")
        action = np.asarray(result[0], dtype=np.float32)
        if action.shape != (1, action_dim):
            raise ValueError(f"ONNX action output must have shape (1, {action_dim}), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("ONNX policy returned NaN or infinity")
        actions.append(action)
    return np.concatenate(actions, axis=0)
