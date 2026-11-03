"""Tests for the v2.1 multi-scale objective.

The load-bearing properties are: every complete cycle contributes total weight 1
regardless of length, increments are scored at every declared lag, and a perfect
prediction has zero loss.
"""

import unittest

import numpy as np
import torch

from src.tasks.velocity.phasor.encoder_v2_1 import (
    CYCLE_WEIGHT,
    KIND_CYCLE,
    KIND_LAG1,
    KIND_SCALE,
    LAG1_RATE_WEIGHT,
    MULTI_SCALE_LAGS,
    MULTISCALE_WEIGHT,
    build_multiscale_pairs,
    phase_v21_loss,
)


def toy(periods=(20, 45), touchdown_offset=0):
    frames = np.arange(sum(periods) + 1, dtype=np.int64)
    sequence = np.asarray(["seq"] * len(frames))
    touchdown = np.zeros(len(frames), dtype=bool)
    touchdown[0] = True
    cursor = 0
    for period in periods:
        cursor += period
        touchdown[cursor] = True
    return sequence, frames, touchdown


def phase_of(sequence, frames, touchdown):
    """Cycle-fraction phase in radians, reset at each touchdown."""
    values = np.zeros(len(frames), dtype=np.float32)
    events = np.flatnonzero(touchdown)
    for start, end in zip(events[:-1], events[1:]):
        span = np.arange(int(frames[start]), int(frames[end]))
        values[span] = (span - frames[start]) / float(frames[end] - frames[start]) * 2.0 * np.pi
    return values


class MultiscalePairTests(unittest.TestCase):
    def test_every_complete_cycle_contributes_base_weight_one(self):
        sequence, frames, touchdown = toy()
        anchor, target, lag, kind, weight = build_multiscale_pairs(sequence, frames, touchdown)
        multiplier = np.where(kind == KIND_LAG1, LAG1_RATE_WEIGHT,
                              np.where(kind == KIND_CYCLE, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
        base = weight / multiplier
        for start, end in ((0, 20), (20, 65)):
            mask = (anchor >= start) & (anchor < end)
            self.assertAlmostEqual(float(base[mask].sum()), 1.0, places=5)

    def test_all_declared_lags_appear(self):
        sequence, frames, touchdown = toy()
        _, _, lag, kind, _ = build_multiscale_pairs(sequence, frames, touchdown)
        present = set(np.unique(lag[kind != KIND_CYCLE]).tolist())
        self.assertEqual(present, set(MULTI_SCALE_LAGS))

    def test_one_cycle_row_per_cycle_and_it_spans_the_half_open_interval(self):
        sequence, frames, touchdown = toy()
        anchor, target, lag, kind, _ = build_multiscale_pairs(sequence, frames, touchdown)
        cycle_rows = kind == KIND_CYCLE
        self.assertEqual(int(cycle_rows.sum()), 2)
        spans = sorted((int(anchor[i]), int(target[i])) for i in np.flatnonzero(cycle_rows))
        self.assertEqual(spans, [(0, 19), (20, 64)])
        self.assertEqual(sorted(lag[cycle_rows].tolist()), [19, 44])

    def test_pairs_never_cross_a_touchdown(self):
        sequence, frames, touchdown = toy()
        anchor, target, _, kind, _ = build_multiscale_pairs(sequence, frames, touchdown)
        events = set(np.flatnonzero(touchdown).tolist())
        for a, b in zip(anchor, target):
            crossing = [event for event in events if a < event <= b]
            self.assertEqual(crossing, [], f"pair {a}->{b} crosses a touchdown")

    def test_short_cycles_are_dropped_rather_than_scored_on_missing_lags(self):
        sequence, frames, touchdown = toy(periods=(3,))
        _, _, lag, kind, _ = build_multiscale_pairs(sequence, frames, touchdown)
        self.assertTrue((lag[kind != KIND_CYCLE] == 1).all())


class MultiscaleLossTests(unittest.TestCase):
    def setUp(self):
        self.sequence, self.frames, self.touchdown = toy()
        self.anchor, self.target, self.lag, self.kind, self.weight = build_multiscale_pairs(
            self.sequence, self.frames, self.touchdown
        )
        self.phase = phase_of(self.sequence, self.frames, self.touchdown)
        multiplier = np.where(self.kind == KIND_LAG1, LAG1_RATE_WEIGHT,
                              np.where(self.kind == KIND_CYCLE, CYCLE_WEIGHT, MULTISCALE_WEIGHT))
        self.base = self.weight / multiplier

    def _vectors(self, phase):
        return torch.tensor(np.stack([np.sin(phase), np.cos(phase)], axis=-1), dtype=torch.float32)

    def _loss(self, predicted, reference=None):
        reference = self.phase if reference is None else reference
        return phase_v21_loss(
            self._vectors(predicted[self.anchor]),
            self._vectors(predicted[self.target]),
            torch.tensor(reference[self.anchor]),
            torch.tensor(reference[self.target]),
            row_weights=torch.tensor(self.weight),
            base_weights=torch.tensor(self.base),
            kind=torch.tensor(self.kind),
        )

    def test_perfect_prediction_has_zero_loss(self):
        losses = self._loss(self.phase)
        for key in ("loss", "point_loss", "increment_loss", "negative_progress_penalty", "jump_penalty"):
            self.assertAlmostEqual(float(losses[key]), 0.0, places=5)

    def test_constant_offset_is_penalized_by_the_point_term_only(self):
        losses = self._loss((self.phase - 1.0) % (2.0 * np.pi))
        self.assertGreater(float(losses["point_loss"]), 0.1)
        self.assertAlmostEqual(float(losses["increment_loss"]), 0.0, places=4)

    def test_backtracking_is_penalized(self):
        losses = self._loss(-self.phase * 0.5)
        self.assertGreater(float(losses["negative_progress_penalty"]), 0.0)
        self.assertGreater(float(losses["loss"]), float(self._loss(self.phase)["loss"]))


if __name__ == "__main__":
    unittest.main()
