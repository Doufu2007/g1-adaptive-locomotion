"""Encoder-v2.1b: the same multi-scale objective with a bounded increment error.

Why a second run exists
-----------------------
v2.1a (encoder_v2_1.py) rose from 1.949 to 2.747 in training loss across epochs
1-4 and never left that basin. The cause is visible in the training curve alone,
before any gate result: the v2.1a increment term divides every lag's error by
pi/6, so a lag-32 row costs up to ~11 while a lag-1 row costs ~0.02. A model that
outputs a near-constant phase has zero predicted increments and pays a plateau of
~1.06 in that term, while the only term that could pull it out -- the point term,
weight 1.0 -- is two orders of magnitude smaller. The optimizer settled on the
plateau and then wandered.

The fix here keeps the same pairs, lags, cycle weighting, and penalties, and
changes only the increment error to a bounded angular form:

    1 - cos(predicted_delta - target_delta)

That lies in [0, 2] for every lag, so no scale can dominate by construction, and
it still penalizes both a rate distortion (any mismatch between the predicted and
true multi-step increment) and backtracking (negative predicted deltas).

Regularization is reused from the one probe that trained stably and transferred
best (`diagnostic_v2_failure_and_gate_audit_20261020.md`, 0.6389 under v4.1):
dropout 0.5 and weight decay 0.1, at AdamW lr 3e-4, with gradient clipping. The
v2 recipe had none of these, which is why both v2 and v2.1a diverged.

Gate thresholds still do not move.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.velocity.phasor.encoder import normalize_phase
from src.tasks.velocity.phasor.encoder_v2 import FEATURE_DIM
from src.tasks.velocity.phasor.encoder_v2_1 import KIND_LAG1, _delta


NET_TOLERANCE_RAD = math.pi / 6.0
JUMP_LIMIT_RAD = math.pi / 2.0

#: Pre-registered before the first v2.1b run; not tuned on gate results.
POINT_WEIGHT = 1.0
REVERSE_WEIGHT = 3.0
JUMP_WEIGHT = 1.0
DROPOUT = 0.5
WEIGHT_DECAY = 0.1
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError("row weights must have a positive sum")
    return (values * weights).sum() / total


def phase_v21b_loss(
    predicted_anchor: torch.Tensor,
    predicted_target: torch.Tensor,
    reference_anchor_rad: torch.Tensor,
    reference_target_rad: torch.Tensor,
    *,
    row_weights: torch.Tensor,
    base_weights: torch.Tensor,
    kind: torch.Tensor,
    rate_scale_rad: float = NET_TOLERANCE_RAD,
    jump_limit_rad: float = JUMP_LIMIT_RAD,
) -> dict[str, torch.Tensor]:
    """Bounded multi-scale increment loss with a point term and v2 penalties."""
    if predicted_anchor.shape != predicted_target.shape or predicted_anchor.ndim != 2 or predicted_anchor.shape[-1] != 2:
        raise ValueError("predicted phase vectors must share shape [rows, 2]")
    reference_anchor = torch.as_tensor(reference_anchor_rad, dtype=predicted_anchor.dtype, device=predicted_anchor.device).reshape(-1)
    reference_target = torch.as_tensor(reference_target_rad, dtype=predicted_anchor.dtype, device=predicted_anchor.device).reshape(-1)
    if len(reference_anchor) != len(predicted_anchor) or len(reference_target) != len(predicted_anchor):
        raise ValueError("reference angles must align with prediction rows")
    if not torch.isfinite(predicted_anchor).all() or not torch.isfinite(predicted_target).all():
        raise ValueError("predicted phase vectors must be finite")

    point_target = torch.stack((torch.sin(reference_target), torch.cos(reference_target)), dim=-1)
    point_anchor = torch.stack((torch.sin(reference_anchor), torch.cos(reference_anchor)), dim=-1)
    point = 0.5 * (
        (1.0 - (predicted_target * point_target).sum(dim=-1))
        + (1.0 - (predicted_anchor * point_anchor).sum(dim=-1))
    )
    point_loss = _weighted(point, base_weights)

    predicted_delta = _delta(predicted_anchor, predicted_target)
    target_delta = torch.atan2(torch.sin(reference_target - reference_anchor), torch.cos(reference_target - reference_anchor))
    increment_loss = _weighted(1.0 - torch.cos(predicted_delta - target_delta), row_weights)

    adjacent = kind == KIND_LAG1
    if bool(adjacent.any()):
        delta1 = predicted_delta[adjacent]
        weight1 = base_weights[adjacent]
        reverse = _weighted(torch.relu(-delta1 / rate_scale_rad).square(), weight1)
        jump = _weighted(torch.relu((delta1.abs() - jump_limit_rad) / rate_scale_rad).square(), weight1)
    else:
        reverse = torch.zeros((), dtype=predicted_anchor.dtype, device=predicted_anchor.device)
        jump = torch.zeros((), dtype=predicted_anchor.dtype, device=predicted_anchor.device)

    total = POINT_WEIGHT * point_loss + increment_loss + REVERSE_WEIGHT * reverse + JUMP_WEIGHT * jump
    return {
        "loss": total,
        "point_loss": point_loss,
        "increment_loss": increment_loss,
        "negative_progress_penalty": reverse,
        "jump_penalty": jump,
        "predicted_delta_rad": predicted_delta,
        "target_delta_rad": target_delta,
    }


class PhaseEncoderV21B(nn.Module):
    """`PhaseEncoderV2` with dropout applied before the linear head."""

    def __init__(self, context_frames: int, hidden: int = 64, dropout: float = DROPOUT):
        super().__init__()
        if context_frames < 2 or hidden <= 0:
            raise ValueError("context_frames must be at least 2 and hidden must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.context_frames = int(context_frames)
        self.hidden = int(hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * context_frames, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.context_frames, FEATURE_DIM):
            raise ValueError(
                f"expected [batch, {self.context_frames}, {FEATURE_DIM}] feature input, got {tuple(features.shape)}"
            )
        encoded = self.conv(features.permute(0, 2, 1)).flatten(start_dim=1)
        return normalize_phase(self.head(self.dropout(encoded)))
