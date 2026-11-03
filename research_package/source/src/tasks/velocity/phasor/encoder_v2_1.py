"""Encoder-v2.1 objective: multi-scale increments instead of adjacent frames only.

Why this exists
---------------
Encoder v2 constrained only lag-1 frame pairs. Step-1 diagnostics
(`results/g1_phase_gate_v4_2/diagnostic_step1_v4_2.json`) measured two failure
modes on validation subject 1 that a lag-1 objective cannot see:

  * a phase-dependent rate distortion -- the encoder advances at 0.69-0.86x the
    reference rate through the first two-thirds of the cycle and 1.02-1.09x in
    the last third, so the cycle integral lands near the right total by
    compensation while individual cycle tails miss the band;
  * frame-level backtracking -- v1 advances on only 80.1% of frame pairs and the
    ridge on 88.8%, spread evenly across phase, against 100% for the reference.

The objective below therefore scores increments at lags 1/4/8/16/32 *and* the
full-cycle integral, while keeping a real point term (cycle completion tracks
absolute phase error 2.3-3.3x between completed and failed cycles, so an
increment-only loss leaves the offset free).

Weighting: every complete cycle contributes total weight 1.0 regardless of its
length or of how many lag rows it generates, so slow and long cycles are not
outvoted by short ones.

Nothing here changes the gate. Bounds `pi/6` and `pi/2` are the contract's own.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.tasks.velocity.phasor.encoder import normalize_phase


MULTI_SCALE_LAGS = (1, 4, 8, 16, 32)
NET_TOLERANCE_RAD = math.pi / 6.0
JUMP_LIMIT_RAD = math.pi / 2.0

#: Pre-registered before the first v2.1 training run; not tuned on results.
POINT_WEIGHT = 1.0
LAG1_RATE_WEIGHT = 2.0
MULTISCALE_WEIGHT = 1.0
CYCLE_WEIGHT = 2.0
REVERSE_WEIGHT = 3.0
JUMP_WEIGHT = 1.0

KIND_LAG1 = 0
KIND_SCALE = 1
KIND_CYCLE = 2

_KIND_MULTIPLIER = {
    KIND_LAG1: LAG1_RATE_WEIGHT,
    KIND_SCALE: MULTISCALE_WEIGHT,
    KIND_CYCLE: CYCLE_WEIGHT,
}


def build_multiscale_pairs(
    sequence: np.ndarray, frame: np.ndarray, left_touchdown: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair frames inside each complete cycle at several lags.

    Returns `(anchor, target, lag, kind, weight)`. A cycle owns the half-open
    interval `[touchdown_k, touchdown_{k+1})`, matching gate v4.2. Each cycle's
    rows sum to weight 1.0 times that row's kind multiplier.
    """
    sequences = np.asarray(sequence)
    frames = np.asarray(frame, dtype=np.int64)
    touchdowns = np.asarray(left_touchdown, dtype=bool)
    if not (len(sequences) == len(frames) == len(touchdowns)):
        raise ValueError("sequence, frame, and touchdown arrays must align")

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(sequences):
        groups.setdefault(str(name), []).append(index)

    anchors: list[int] = []
    targets: list[int] = []
    lags: list[int] = []
    kinds: list[int] = []
    weights: list[float] = []
    cycle_count = 0

    for indexes in groups.values():
        order = sorted(indexes, key=lambda index: int(frames[index]))
        ordered_frames = np.asarray([frames[index] for index in order], dtype=np.int64)
        by_frame = {int(frames[index]): index for index in order}
        if len(by_frame) != len(order):
            raise ValueError("duplicate frame IDs within a sequence")
        events = [int(frames[index]) for index in order if touchdowns[index]]
        for start, end in zip(events[:-1], events[1:]):
            period = end - start
            if period < 2:
                continue
            span = np.arange(start, end, dtype=np.int64)  # half-open, cycle v4.2
            if any(int(value) not in by_frame for value in span):
                continue
            rows: list[tuple[int, int, int, int]] = []
            for lag in MULTI_SCALE_LAGS:
                if lag >= len(span):
                    continue
                kind = KIND_LAG1 if lag == 1 else KIND_SCALE
                for offset in range(len(span) - lag):
                    rows.append((int(span[offset]), int(span[offset + lag]), lag, kind))
            rows.append((int(span[0]), int(span[-1]), int(len(span) - 1), KIND_CYCLE))
            if not rows:
                continue
            share = 1.0 / float(len(rows))
            cycle_count += 1
            for anchor_frame, target_frame, lag, kind in rows:
                anchors.append(by_frame[anchor_frame])
                targets.append(by_frame[target_frame])
                lags.append(lag)
                kinds.append(kind)
                weights.append(share * _KIND_MULTIPLIER[kind])

    if cycle_count == 0:
        raise ValueError("no complete touchdown cycles were found")
    return (
        np.asarray(anchors, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(lags, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError("row weights must have a positive sum")
    return (values * weights).sum() / total


def _delta(predicted_a: torch.Tensor, predicted_b: torch.Tensor) -> torch.Tensor:
    a = normalize_phase(predicted_a)
    b = normalize_phase(predicted_b)
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    dot = (b * a).sum(dim=-1)
    return torch.atan2(cross, dot)


def phase_v21_loss(
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
    """Multi-scale increment loss with a point term and retained v2 penalties."""
    if predicted_anchor.shape != predicted_target.shape or predicted_anchor.ndim != 2 or predicted_anchor.shape[-1] != 2:
        raise ValueError("predicted phase vectors must share shape [rows, 2]")
    reference_anchor = torch.as_tensor(reference_anchor_rad, dtype=predicted_anchor.dtype, device=predicted_anchor.device).reshape(-1)
    reference_target = torch.as_tensor(reference_target_rad, dtype=predicted_anchor.dtype, device=predicted_anchor.device).reshape(-1)
    rows = len(predicted_anchor)
    if len(reference_anchor) != rows or len(reference_target) != rows:
        raise ValueError("reference angles must align with prediction rows")
    if rate_scale_rad <= 0.0 or jump_limit_rad <= 0.0:
        raise ValueError("loss scales must be positive")
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
    increment_error = (predicted_delta - target_delta) / rate_scale_rad
    increment_loss = _weighted(F.smooth_l1_loss(increment_error, torch.zeros_like(increment_error), reduction="none"), row_weights)

    adjacent = kind == KIND_LAG1
    if bool(adjacent.any()):
        delta1 = predicted_delta[adjacent]
        weight1 = base_weights[adjacent]
        reverse = _weighted(torch.relu(-delta1 / rate_scale_rad).square(), weight1)
        jump = _weighted(torch.relu((delta1.abs() - jump_limit_rad) / rate_scale_rad).square(), weight1)
    else:
        reverse = torch.zeros((), dtype=predicted_anchor.dtype, device=predicted_anchor.device)
        jump = torch.zeros((), dtype=predicted_anchor.dtype, device=predicted_anchor.device)

    total = (
        POINT_WEIGHT * point_loss
        + increment_loss
        + REVERSE_WEIGHT * reverse
        + JUMP_WEIGHT * jump
    )
    return {
        "loss": total,
        "point_loss": point_loss,
        "increment_loss": increment_loss,
        "negative_progress_penalty": reverse,
        "jump_penalty": jump,
        "predicted_delta_rad": predicted_delta,
        "target_delta_rad": target_delta,
    }
