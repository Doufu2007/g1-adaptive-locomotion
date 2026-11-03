"""Encoder-v2 phase source shared by training and offline rollout inference."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.velocity.phasor.encoder import normalize_phase


JOINT_DIM = 29
FEATURE_DIM = 2 * JOINT_DIM + 1
RATE_SCALE_RAD = math.pi / 6.0
JUMP_LIMIT_RAD = math.pi / 2.0


def build_phase_v2_features(pose_history: torch.Tensor) -> torch.Tensor:
    """Append per-joint frame differences and RMS joint speed to pose history."""
    pose = torch.as_tensor(pose_history)
    if pose.ndim != 3 or pose.shape[-1] != JOINT_DIM or pose.shape[1] < 2:
        raise ValueError("pose_history must have shape [batch, time>=2, 29]")
    if not torch.is_floating_point(pose) or not torch.isfinite(pose).all():
        raise ValueError("pose_history must be finite floating point values")
    velocity = torch.zeros_like(pose)
    velocity[:, 1:] = pose[:, 1:] - pose[:, :-1]
    speed = torch.sqrt(torch.mean(velocity.square(), dim=-1, keepdim=True).clamp_min(0.0))
    return torch.cat((pose, velocity, speed), dim=-1)


def target_vectors_v2(phase_rad: torch.Tensor) -> torch.Tensor:
    phase = torch.as_tensor(phase_rad)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def _weighted_mean(values: torch.Tensor, pair_weights: torch.Tensor | None) -> torch.Tensor:
    if pair_weights is None:
        return values.mean()
    weights = torch.as_tensor(pair_weights, dtype=values.dtype, device=values.device)
    if weights.shape != values.shape or not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("pair_weights must be finite, nonnegative, and aligned with pairs")
    total = weights.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError("pair_weights must have a positive sum")
    return (values * weights).sum() / total


def phase_v2_loss(
    predicted_now: torch.Tensor,
    predicted_next: torch.Tensor,
    target_now_rad: torch.Tensor,
    target_next_rad: torch.Tensor,
    *,
    pair_weights: torch.Tensor | None = None,
    rate_weight: float = 2.0,
    negative_progress_weight: float = 2.0,
    jump_weight: float = 1.0,
    rate_scale_rad: float = RATE_SCALE_RAD,
    jump_limit_rad: float = JUMP_LIMIT_RAD,
) -> dict[str, torch.Tensor]:
    """Circular point and adjacent-frame rate objective aligned with gate v4."""
    if predicted_now.shape != predicted_next.shape or predicted_now.ndim != 2 or predicted_now.shape[-1] != 2:
        raise ValueError("predicted phase vectors must both have shape [pairs, 2]")
    y_now = torch.as_tensor(target_now_rad, dtype=predicted_now.dtype, device=predicted_now.device).reshape(-1)
    y_next = torch.as_tensor(target_next_rad, dtype=predicted_now.dtype, device=predicted_now.device).reshape(-1)
    if len(y_now) != len(predicted_now) or len(y_next) != len(predicted_now):
        raise ValueError("phase labels must align with prediction pairs")
    constants = (rate_weight, negative_progress_weight, jump_weight, rate_scale_rad, jump_limit_rad)
    if any(not math.isfinite(value) or value < 0.0 for value in constants) or rate_scale_rad == 0.0 or jump_limit_rad == 0.0:
        raise ValueError("loss weights and phase scales must be finite and nonnegative; scales must be positive")
    if not torch.isfinite(predicted_now).all() or not torch.isfinite(predicted_next).all():
        raise ValueError("predicted phase vectors must be finite")

    point_now = 1.0 - (predicted_now * target_vectors_v2(y_now)).sum(dim=-1)
    point_next = 1.0 - (predicted_next * target_vectors_v2(y_next)).sum(dim=-1)
    point_loss = _weighted_mean(0.5 * (point_now + point_next), pair_weights)

    p_now = normalize_phase(predicted_now)
    p_next = normalize_phase(predicted_next)
    cross = p_next[:, 0] * p_now[:, 1] - p_next[:, 1] * p_now[:, 0]
    dot = (p_next * p_now).sum(dim=-1)
    predicted_delta = torch.atan2(cross, dot)
    target_delta = torch.atan2(torch.sin(y_next - y_now), torch.cos(y_next - y_now))
    scaled_rate_error = (predicted_delta - target_delta) / rate_scale_rad
    rate_loss = _weighted_mean(F.smooth_l1_loss(scaled_rate_error, torch.zeros_like(scaled_rate_error), reduction="none"), pair_weights)
    reverse_penalty = _weighted_mean(torch.relu(-predicted_delta / rate_scale_rad).square(), pair_weights)
    jump_penalty = _weighted_mean(torch.relu((predicted_delta.abs() - jump_limit_rad) / rate_scale_rad).square(), pair_weights)
    total_loss = (
        point_loss
        + rate_weight * rate_loss
        + negative_progress_weight * reverse_penalty
        + jump_weight * jump_penalty
    )
    return {
        "loss": total_loss,
        "point_loss": point_loss,
        "rate_loss": rate_loss,
        "negative_progress_penalty": reverse_penalty,
        "jump_penalty": jump_penalty,
        "predicted_delta_rad": predicted_delta,
        "target_delta_rad": target_delta,
    }


class PhaseEncoderV2(nn.Module):
    """Time-position-preserving CNN over pose, velocity, and speed history."""

    def __init__(self, context_frames: int, hidden: int = 64):
        super().__init__()
        if context_frames < 2 or hidden <= 0:
            raise ValueError("context_frames must be at least 2 and hidden must be positive")
        self.context_frames = int(context_frames)
        self.hidden = int(hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden * context_frames, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.context_frames, FEATURE_DIM):
            raise ValueError(
                f"expected [batch, {self.context_frames}, {FEATURE_DIM}] feature input, got {tuple(features.shape)}"
            )
        encoded = self.conv(features.permute(0, 2, 1)).flatten(start_dim=1)
        return normalize_phase(self.head(encoded))


def predict_phase_v2(model: PhaseEncoderV2, pose_history: torch.Tensor) -> torch.Tensor:
    """Run the exact shared feature path used by training and evaluation."""
    features = build_phase_v2_features(pose_history)
    return model(features)
