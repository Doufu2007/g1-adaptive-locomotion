"""Regression tests for the v4.2 endpoint correction.

Every toy cycle here is built the way the real reference clock behaves: the
phase rises through the interval and is reset to zero *at* the terminal
touchdown, so the last sample of `[touchdown_k, touchdown_{k+1}]` is post-reset.
"""

import unittest

import numpy as np

from src.tasks.velocity.phasor.phase_gate_v4 import score_phase_cycles_v4
from src.tasks.velocity.phasor.phase_gate_v4_2 import (
    MIN_CYCLE_SAMPLES,
    PHASE_GATE_VERSION,
    TERMINAL_SAMPLE_POLICY,
    score_phase_cycles_v4_2,
    select_v4_2_candidate,
)


LAST_INTERNAL_FRACTION = 0.99


def vectors(angles):
    values = np.asarray(angles, dtype=np.float64)
    return np.stack((np.sin(values), np.cos(values)), axis=-1)


def toy_cycles(periods=(32,) * 12, alpha=1.0):
    """Build contiguous cycles; each terminal touchdown resets the reference to zero."""
    sequence_ids, frame_ids, touchdowns, reference, angles = [], [], [], [], []
    cursor = 0
    for period in periods:
        for step in range(period):
            share = LAST_INTERNAL_FRACTION * step / (period - 1)
            sequence_ids.append("seq")
            frame_ids.append(cursor + step)
            touchdowns.append(step == 0)
            reference.append(share)
            angles.append(alpha * share)
        cursor += period
    sequence_ids.append("seq")
    frame_ids.append(cursor)
    touchdowns.append(True)
    reference.append(0.0)
    angles.append(alpha * 0.0)
    return {
        "encoder": vectors(np.asarray(angles) * 2.0 * np.pi),
        "reference": np.asarray(reference) * 2.0 * np.pi,
        "sequence_ids": np.asarray(sequence_ids),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "left_touchdown": np.asarray(touchdowns, dtype=bool),
        "condition_ids": np.asarray(sequence_ids),
        "episode_ids": np.zeros(len(sequence_ids), dtype=np.int64),
        "history_valid": np.ones(len(sequence_ids), dtype=bool),
    }


def score(data, scorer=score_phase_cycles_v4_2):
    return scorer(
        data["encoder"],
        data["reference"],
        data["sequence_ids"],
        data["frame_ids"],
        data["left_touchdown"],
        condition_ids=data["condition_ids"],
        episode_ids=data["episode_ids"],
        history_valid=data["history_valid"],
        required_conditions=["seq"],
        minimum_cycles_per_condition=10,
        minimum_cycles_per_length_stratum=10,
        minimum_episode_segments_per_condition=1,
    )


class PhaseGateV4Dot2Tests(unittest.TestCase):
    def test_version_and_policy_are_recorded(self):
        report = score(toy_cycles())
        self.assertEqual(PHASE_GATE_VERSION, "g1-phase-gate-v4.2")
        self.assertEqual(report["gate_version"], PHASE_GATE_VERSION)
        self.assertEqual(report["terminal_sample_policy"], TERMINAL_SAMPLE_POLICY)
        self.assertEqual(report["cycle_rule"]["terminal_sample_policy"], TERMINAL_SAMPLE_POLICY)

    def test_terminal_sample_is_excluded_from_the_cycle_metrics(self):
        report = score(toy_cycles(periods=(32,), ))
        record = report["cycle_records"][0]
        self.assertEqual(record["cycle_sample_count"], 33)
        self.assertEqual(record["metric_sample_count"], 32)
        self.assertEqual(record["metric_sample_count"], record["cycle_sample_count"] - 1)
        self.assertEqual(report["excluded_terminal_sample_count"], 1)

    def test_trimmed_net_matches_the_analytic_progress(self):
        report = score(toy_cycles(periods=(32,)))
        record = report["cycle_records"][0]
        expected = LAST_INTERNAL_FRACTION * 2.0 * np.pi
        self.assertAlmostEqual(record["encoder_net_progression_rad"], expected, places=9)
        self.assertAlmostEqual(record["reference_net_progression_rad"], expected, places=9)

    def test_reference_clock_still_passes_under_the_trimmed_rule(self):
        report = score(toy_cycles())
        self.assertGreater(report["reference_clock_completion_fraction"], 0.99)
        self.assertTrue(report["reference_clock_pass"])
        self.assertTrue(report["coverage_pass"])
        self.assertTrue(report["encoder_freeze_eligible"])

    def test_alpha_090_ramp_used_to_pass_and_now_fails(self):
        data = toy_cycles(alpha=0.90)
        v41 = score(data, scorer=score_phase_cycles_v4)
        v42 = score(data, scorer=score_phase_cycles_v4_2)

        pinned = v41["cycle_records"][0]["encoder_net_progression_rad"]
        self.assertAlmostEqual(pinned, 2.0 * np.pi, delta=0.02)
        self.assertTrue(v41["encoder_freeze_eligible"])

        trimmed = v42["cycle_records"][0]["encoder_net_progression_rad"]
        self.assertAlmostEqual(trimmed, 0.90 * LAST_INTERNAL_FRACTION * 2.0 * np.pi, places=9)
        self.assertLess(trimmed, 2.0 * np.pi - np.pi / 6.0)
        self.assertFalse(v42["cycle_records"][0]["encoder_completed_one_turn"])
        self.assertEqual(v42["macro_cycle_completion_fraction"], 0.0)
        self.assertFalse(v42["encoder_freeze_eligible"])
        self.assertTrue(v42["reference_clock_pass"])

    def test_trimming_is_strictly_stricter_than_v41(self):
        for alpha in (0.98, 0.95, 0.92, 0.90, 0.85):
            with self.subTest(alpha=alpha):
                data = toy_cycles(alpha=alpha)
                v41 = score(data, scorer=score_phase_cycles_v4)
                v42 = score(data, scorer=score_phase_cycles_v4_2)
                self.assertAlmostEqual(
                    v41["cycle_records"][0]["encoder_net_progression_rad"], 2.0 * np.pi, delta=0.02
                )
                self.assertLessEqual(
                    v42["macro_cycle_completion_fraction"], v41["macro_cycle_completion_fraction"]
                )

    def test_cycle_without_enough_internal_samples_is_unscorable_not_completed(self):
        # A period-p interval holds p+1 frames; dropping the terminal one leaves p.
        report = score(toy_cycles(periods=(MIN_CYCLE_SAMPLES - 1,)))
        record = report["cycle_records"][0]
        self.assertEqual(record["metric_sample_count"], MIN_CYCLE_SAMPLES - 1)
        self.assertFalse(record["evaluable"])
        self.assertIn("cycle_interval_too_short", record["failure_reasons"])
        self.assertEqual(report["evaluable_cycle_count"], 0)
        self.assertEqual(report["short_cycle_count"], 1)
        self.assertEqual(report["condition_completion"]["seq"]["completion_fraction"], 0.0)

    def test_candidate_selection_still_prioritizes_completion(self):
        candidates = {
            "a_low_error": {
                "macro_cycle_completion_fraction": 0.70,
                "condition_macro_point_error_rad": 0.10,
                "context_frames": 30,
            },
            "b_high_completion": {
                "macro_cycle_completion_fraction": 0.92,
                "condition_macro_point_error_rad": 0.90,
                "context_frames": 30,
            },
        }
        self.assertEqual(select_v4_2_candidate(candidates), "b_high_completion")


if __name__ == "__main__":
    unittest.main()
