import unittest

import numpy as np

from src.tasks.velocity.phasor.phase_gate_v4 import (
    calibrate_phase_gate_v4,
    score_phase_cycles_v4,
    select_v4_candidate,
)


def vectors(angles):
    values = np.asarray(angles, dtype=np.float64)
    return np.stack((np.sin(values), np.cos(values)), axis=-1)


def one_cycle(angles, *, period=None, history_valid=None, episode_ids=None):
    values = np.asarray(angles, dtype=np.float64)
    count = len(values)
    touchdown = np.zeros(count, dtype=bool)
    touchdown[[0, count - 1]] = True
    kwargs = {}
    if history_valid is not None:
        kwargs["history_valid"] = np.asarray(history_valid, dtype=bool)
    if episode_ids is not None:
        kwargs["episode_ids"] = np.asarray(episode_ids)
    return score_phase_cycles_v4(
        vectors(values),
        values,
        sequence_ids=np.array(["condition_a"] * count),
        frame_ids=np.arange(count) if period is None else np.linspace(0, period, count, dtype=int),
        left_touchdown=touchdown,
        condition_ids=np.array(["condition_a"] * count),
        **kwargs,
    )


class PhaseGateV4Tests(unittest.TestCase):
    def test_period_net_tolerance_is_pi_over_six(self):
        base = np.linspace(0.0, 2.0 * np.pi, 9)
        within = base + np.linspace(0.0, np.pi / 6.0, 9)
        outside = base + np.linspace(0.0, np.pi / 6.0 + 1e-4, 9)
        self.assertEqual(one_cycle(within)["encoder_completion_fraction"], 1.0)
        self.assertEqual(one_cycle(outside)["encoder_completion_fraction"], 0.0)

    def test_cycle_jump_limit_is_strict_maximum_absolute_increment(self):
        exact_limit = np.linspace(0.0, 2.0 * np.pi, 5)
        report = one_cycle(exact_limit)
        self.assertEqual(report["cycle_records"][0]["encoder_max_abs_step_rad"], np.pi / 2.0)
        self.assertEqual(report["encoder_completion_fraction"], 0.0)

    def test_insufficient_history_and_episode_reset_remain_visible(self):
        angles = np.tile(np.linspace(0.0, 2.0 * np.pi, 9), 2)
        count = len(angles)
        frames = np.r_[np.arange(9), np.arange(9)]
        episodes = np.r_[np.zeros(9, dtype=int), np.ones(9, dtype=int)]
        touchdown = np.zeros(count, dtype=bool)
        touchdown[[0, 8, 9, 17]] = True
        history = np.ones(count, dtype=bool)
        history[0] = False
        report = score_phase_cycles_v4(
            vectors(angles),
            angles,
            sequence_ids=np.array(["walk"] * count),
            frame_ids=frames,
            left_touchdown=touchdown,
            condition_ids=np.array(["slow"] * count),
            episode_ids=episodes,
            history_valid=history,
        )
        self.assertEqual(report["observed_cycle_count"], 2)
        self.assertEqual(report["insufficient_history_cycle_count"], 1)
        self.assertEqual(report["evaluable_cycle_count"], 1)
        self.assertEqual(report["cycle_records"][0]["episode_id"], 0)
        self.assertEqual(report["cycle_records"][1]["episode_id"], 1)

    def test_gap_and_condition_change_do_not_form_a_cycle(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 9)
        frames = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        touchdown = np.zeros(9, dtype=bool)
        touchdown[[0, 8]] = True
        report = score_phase_cycles_v4(
            vectors(angles),
            angles,
            sequence_ids=np.array(["walk"] * 9),
            frame_ids=frames,
            left_touchdown=touchdown,
            condition_ids=np.array(["slow"] * 4 + ["fast"] * 5),
        )
        self.assertGreaterEqual(report["unscorable_cycle_count"], 1)
        self.assertEqual(report["evaluable_cycle_count"], 0)

    def test_invalid_reference_phase_is_counted_as_unscorable_not_dropped(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 9)
        touchdown = np.zeros(9, dtype=bool)
        touchdown[[0, 8]] = True
        reference = angles.copy()
        reference[4] = np.nan
        reference_valid = np.ones(9, dtype=bool)
        reference_valid[4] = False
        report = score_phase_cycles_v4(
            vectors(angles),
            reference,
            sequence_ids=np.array(["walk"] * 9),
            frame_ids=np.arange(9),
            left_touchdown=touchdown,
            condition_ids=np.array(["slow"] * 9),
            reference_valid=reference_valid,
        )
        self.assertEqual(report["observed_cycle_count"], 1)
        self.assertEqual(report["evaluable_cycle_count"], 0)
        self.assertIn("reference_phase_invalid", report["cycle_records"][0]["failure_reasons"])

    def test_calibration_reports_reference_and_encoder_and_rejects_low_encoder_rate(self):
        periods = [8, 9, 10]
        frames = np.concatenate([np.arange(period + 1) + 20 * i for i, period in enumerate(periods)])
        sequence = np.concatenate([np.array([f"c{i}"] * (period + 1)) for i, period in enumerate(periods)])
        touchdown = np.zeros(len(frames), dtype=bool)
        offset = 0
        reference = []
        for period in periods:
            touchdown[[offset, offset + period]] = True
            reference.extend(np.linspace(0.0, 2.0 * np.pi, period + 1).tolist())
            offset += period + 1
        reference = np.asarray(reference)
        calibration = calibrate_phase_gate_v4(
            vectors(np.zeros_like(reference)),
            reference,
            sequence,
            frames,
            touchdown,
            condition_ids=sequence,
            history_valid=np.ones(len(frames), dtype=bool),
        )
        self.assertEqual(calibration["reference_clock_completion_fraction"], 1.0)
        self.assertEqual(calibration["encoder_completion_fraction"], 0.0)
        self.assertFalse(calibration["encoder_freeze_eligible"])
        strata = calibration["reference_clock"]["period_length_strata"]
        self.assertEqual(strata["short"]["evaluable_cycle_count"], 1)
        self.assertEqual(strata["medium"]["evaluable_cycle_count"], 1)
        self.assertEqual(strata["long"]["evaluable_cycle_count"], 1)

    def test_candidate_selection_prioritizes_v4_completion_over_point_error(self):
        selected = select_v4_candidate(
            {
                "short_context": {
                    "macro_cycle_completion_fraction": 0.84,
                    "condition_macro_point_error_rad": 0.2,
                    "context_frames": 10,
                },
                "long_context": {
                    "macro_cycle_completion_fraction": 0.86,
                    "condition_macro_point_error_rad": 0.9,
                    "context_frames": 30,
                },
            }
        )
        self.assertEqual(selected, "long_context")

    def test_checkpoint_tie_break_uses_numeric_epoch_not_candidate_name(self):
        selected = select_v4_candidate(
            {
                "context_10_epoch_10": {
                    "macro_cycle_completion_fraction": 0.91,
                    "condition_macro_point_error_rad": 0.8,
                    "context_frames": 10,
                    "epoch": 10,
                },
                "context_10_epoch_2": {
                    "macro_cycle_completion_fraction": 0.91,
                    "condition_macro_point_error_rad": 0.8,
                    "context_frames": 10,
                    "epoch": 2,
                },
            }
        )
        self.assertEqual(selected, "context_10_epoch_2")


if __name__ == "__main__":
    unittest.main()
