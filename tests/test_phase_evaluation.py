import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase_study import (
    RolloutGateError,
    control_summary,
    calibrate_cycle_progression,
    evaluate_phase_predictions,
    evaluate_rollout_gate,
    estimate_seed_count,
    select_macro_phase_threshold,
    select_phase_threshold,
)


def vectors(cycles):
    angles = 2.0 * np.pi * np.asarray(cycles)
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1)


class PhaseEvaluationTests(unittest.TestCase):
    def test_phase_metrics_use_canonical_order_and_fixed_clock_baseline(self):
        target = vectors(np.linspace(0.0, 1.0, 9, endpoint=False))
        report = evaluate_phase_predictions(target, target, fixed_clock_phase=vectors(np.zeros(9)))
        self.assertAlmostEqual(report["circular_phase_error_rad"], 0.0)
        self.assertGreater(report["fixed_clock_circular_phase_error_rad"], 0.0)

    def test_rollout_gate_fails_closed_on_distribution_shift(self):
        heldout = {"circular_phase_error_rad": 0.1}
        onpolicy = {"circular_phase_error_rad": 1.0}
        with self.assertRaises(RolloutGateError):
            evaluate_rollout_gate(heldout, onpolicy, threshold_rad=0.2)

    def test_phase_threshold_is_frozen_from_validation_errors(self):
        target = vectors(np.linspace(0.0, 1.0, 8, endpoint=False))
        threshold = select_phase_threshold(target, target, task_resolution_rad=0.2)
        self.assertAlmostEqual(threshold, 0.2)

    def test_macro_phase_threshold_weights_validation_sequences_equally(self):
        target = np.zeros(101, dtype=np.float64)
        predicted = np.zeros((101, 2), dtype=np.float64)
        predicted[:, 1] = 1.0
        predicted[1:, 0] = np.sin(0.4)
        predicted[1:, 1] = np.cos(0.4)
        sequences = np.array(["short"] + ["long"] * 100)

        calibration = select_macro_phase_threshold(
            predicted,
            target,
            sequences,
            quantile=0.95,
            task_resolution_rad=0.1,
        )

        self.assertEqual(calibration["validation_sequence_count"], 2)
        self.assertAlmostEqual(calibration["validation_sequence_macro_errors_rad"]["short"], 0.0)
        self.assertAlmostEqual(calibration["validation_sequence_macro_errors_rad"]["long"], 0.4)
        self.assertAlmostEqual(calibration["phase_gate_threshold_rad"], 0.38)
        self.assertIn("equal-weight sequence-macro", calibration["threshold_source"])

    def test_cycle_progression_includes_next_touchdown_endpoint(self):
        # The cycle label is half-open: the next touchdown belongs to the
        # following cycle.  A perfect prediction must still receive the
        # endpoint when measuring the preceding touchdown-to-touchdown turn.
        angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 0.0])
        predicted = vectors(angles / (2.0 * np.pi))
        report = evaluate_phase_predictions(
            predicted,
            angles,
            left_touchdown=np.array([True, False, False, False, True]),
            cycle_ids=np.array([0, 0, 0, 0, 1]),
        )
        self.assertEqual(report["complete_cycle_count"], 1)
        self.assertAlmostEqual(report["cycle_net_progression_rad"][0], 2.0 * np.pi)
        self.assertEqual(report["cycle_completed_one_turn_fraction"], 1.0)

    def test_cycle_progression_does_not_count_short_or_unclosed_arcs(self):
        angles = np.array([0.0, 0.5 * np.pi, 0.0])
        predicted = vectors(angles / (2.0 * np.pi))
        report = evaluate_phase_predictions(
            predicted,
            angles,
            left_touchdown=np.array([True, False, False]),
            cycle_ids=np.array([0, 0, 0]),
        )
        self.assertEqual(report["complete_cycle_count"], 0)
        self.assertIsNone(report["cycle_completed_one_turn_fraction"])

    def test_cycle_progression_accepts_one_turn_with_validation_resolution_tolerance(self):
        angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 0.0])
        predicted_angles = angles.copy()
        predicted_angles[-1] = -0.05  # small endpoint error, far below pi/6
        predicted = vectors(predicted_angles / (2.0 * np.pi))
        report = evaluate_phase_predictions(
            predicted,
            angles,
            left_touchdown=np.array([True, False, False, False, True]),
            cycle_ids=np.array([0, 0, 0, 0, 1]),
            cycle_progression_tolerance_rad=np.pi / 6.0,
            maximum_forward_step_rad=np.pi / 2.0,
            maximum_reverse_step_rad=0.1,
        )
        self.assertAlmostEqual(report["cycle_completed_one_turn_fraction"], 1.0)

    def test_progression_gate_rejects_stationary_half_reverse_double_and_jump(self):
        cases = {
            "stationary": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "half": np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 0.0]),
            "reverse": np.array([0.0, 2.0, 1.0, 4.0, 2.0 * np.pi]),
            "double": np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
            "jump": np.array([0.0, 0.2, 0.4, 3.0, 2.0 * np.pi]),
        }
        for name, angles in cases.items():
            with self.subTest(name=name):
                maximum_forward = 4.0 if name in {"reverse", "double"} else 1.0
                report = evaluate_phase_predictions(
                    vectors(angles / (2.0 * np.pi)),
                    np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 0.0]),
                    left_touchdown=np.array([True, False, False, False, True]),
                    cycle_ids=np.array([0, 0, 0, 0, 1]),
                    cycle_progression_tolerance_rad=np.pi / 6.0,
                    maximum_forward_step_rad=maximum_forward,
                    maximum_reverse_step_rad=0.1,
                )
                self.assertEqual(report["cycle_completed_one_turn_fraction"], 0.0)

    def test_cycle_calibration_uses_validation_frame_resolution(self):
        frames = np.arange(97)
        phase = vectors(frames / 32.0)
        touchdown = np.zeros(97, dtype=bool)
        touchdown[[0, 32, 64, 96]] = True
        calibration = calibrate_cycle_progression(
            phase,
            np.array(["walk"] * 97),
            frames,
            touchdown,
            task_resolution_rad=np.pi / 6,
        )
        self.assertAlmostEqual(calibration["tolerance_rad"], 2.0 * np.pi / 32.0)
        self.assertAlmostEqual(calibration["validation_cycle_completion_fraction_diagnostic"], 1.0)
        self.assertTrue(calibration["resolution_calibration_pass"])

    def test_cycle_calibration_blocks_when_sampling_cannot_support_resolution(self):
        frames = np.arange(31)
        phase = vectors(frames / 10.0)
        touchdown = np.zeros(31, dtype=bool)
        touchdown[[0, 10, 20, 30]] = True
        with self.assertRaisesRegex(ValueError, "task resolution"):
            calibrate_cycle_progression(
                phase,
                np.array(["walk"] * 31),
                frames,
                touchdown,
                task_resolution_rad=np.pi / 6,
            )

    def test_control_summary_uses_seed_as_statistical_unit(self):
        report = control_summary(
            [
                {"group": "fixed_clock", "seed": 1, "xy_speed_error": 0.5, "training_termination": 0.8},
                {"group": "fixed_clock", "seed": 2, "xy_speed_error": 0.4, "training_termination": 0.9},
                {"group": "joint", "seed": 1, "xy_speed_error": 0.2},
                {"group": "joint", "seed": 2, "xy_speed_error": 0.3},
            ],
            minimum_mean_improvement=0.1,
        )
        self.assertEqual(report["statistical_unit"], "training_seed")
        self.assertEqual(report["groups"]["joint"]["seed_count"], 2)
        self.assertTrue(report["primary_comparison"]["meets_minimum_practical_improvement"])
        self.assertNotIn("test_fall_probability_by_seed", report["groups"]["fixed_clock"])

    def test_seed_planning_uses_engineering_seed_variance(self):
        count = estimate_seed_count([0.2, 0.4, 0.3, 0.5], minimum_effect=0.1)
        self.assertGreaterEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
