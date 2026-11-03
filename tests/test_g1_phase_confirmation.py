from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from scripts.evaluate_g1_phase_confirmation import evaluate_rollout_arrays
from scripts.evaluate_g1_onpolicy_phase_gate import reference_phase
from src.tasks.velocity.phasor.confirmation import (
    advance_episode_segments,
    choose_feasible_policy_candidate,
    command_for_step,
    onnx_batch_one_actions,
    quaternion_wxyz_to_rpy,
)


class PhaseFromLastJoint(nn.Module):
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        angle = windows[:, -1, 0]
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


class BatchOnePolicy:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run(self, outputs, inputs):
        observations = inputs["obs"]
        self.batch_sizes.append(observations.shape[0])
        return [np.ones((1, 29), dtype=np.float32)]


class G1PhaseConfirmationTests(unittest.TestCase):
    @staticmethod
    def _single_cycle_report(
        *,
        start: int = 28,
        end: int = 58,
        episode_at: np.ndarray | None = None,
        angle: np.ndarray | None = None,
    ):
        total = 80
        joint = np.zeros((total, 1, 29), dtype=np.float32)
        contacts = np.zeros((total, 1, 2), dtype=bool)
        contact_start = max(0, start - 1)
        contacts[contact_start : min(contact_start + 3, total), 0, 0] = True
        if end < total:
            contact_end = max(0, end - 1)
            contacts[contact_end : min(contact_end + 3, total), 0, 0] = True
        if angle is not None:
            joint[:, 0, 0] = angle
        episode_ids = (
            np.full((total, 1), 752, dtype=np.int64)
            if episode_at is None
            else np.asarray(episode_at, dtype=np.int64).reshape(total, 1)
        )
        report, predictions = evaluate_rollout_arrays(
            joint,
            contacts,
            episode_ids,
            PhaseFromLastJoint(),
            mean=np.zeros(29, dtype=np.float32),
            std=np.ones(29, dtype=np.float32),
            calibration={
                "tolerance_rad": 0.21,
                "maximum_forward_step_rad": 0.61,
                "maximum_reverse_step_rad": 0.01,
                "macro_completion_fraction_required": 0.9,
                "minimum_cycles_per_condition": 1,
                "minimum_episodes_per_condition": 1,
            },
            point_threshold_rad=0.5,
        )
        return report, predictions

    def test_termination_advances_only_the_affected_environment_segment(self) -> None:
        reset_counts, episode_ids = advance_episode_segments(
            block_id=1,
            reset_counts=np.array([0, 1, 0]),
            terminated=np.array([False, True, False]),
            truncated=np.array([False, False, True]),
            block_span=751,
        )

        np.testing.assert_array_equal(reset_counts, [0, 2, 1])
        np.testing.assert_array_equal(episode_ids, [751, 753, 752])

    def test_confirmation_conditions_cover_speed_and_standing_transitions(self) -> None:
        self.assertEqual(command_for_step(0, 0), 0.4)
        self.assertEqual(command_for_step(1, 0), 0.7)
        self.assertEqual(command_for_step(2, 0), 0.9)
        self.assertEqual(command_for_step(3, 0), 0.0)
        self.assertEqual(command_for_step(3, 100), 0.6)
        self.assertEqual(command_for_step(3, 250), 0.0)
        self.assertEqual(command_for_step(3, 350), 0.6)

    def test_controller_selection_requires_coverage_then_minimizes_terminations(self) -> None:
        candidates = [
            {"policy_seed": 31, "coverage_status": "insufficient_coverage", "termination_count": 0, "complete_cycle_count": 50},
            {"policy_seed": 32, "coverage_status": "adequate", "termination_count": 3, "complete_cycle_count": 30},
            {"policy_seed": 33, "coverage_status": "adequate", "termination_count": 1, "complete_cycle_count": 25},
        ]

        selected = choose_feasible_policy_candidate(candidates)

        self.assertEqual(selected, 33)
        self.assertIsNone(choose_feasible_policy_candidate(candidates[:1]))

    def test_fixed_batch_one_policy_is_called_once_per_environment(self) -> None:
        policy = BatchOnePolicy()
        actions = onnx_batch_one_actions(
            policy,
            np.zeros((4, 98), dtype=np.float32),
            input_name="obs",
            output_name="actions",
            action_dim=29,
        )
        self.assertEqual(policy.batch_sizes, [1, 1, 1, 1])
        self.assertEqual(actions.shape, (4, 29))
        self.assertTrue(np.isfinite(actions).all())

    def test_quaternion_conversion_reports_root_body_attitude_in_radians(self) -> None:
        quaternions = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(np.deg2rad(35.0)), np.sin(np.deg2rad(35.0)), 0.0, 0.0],
                [np.cos(np.deg2rad(30.0)), 0.0, 0.0, np.sin(np.deg2rad(30.0))],
            ],
            dtype=np.float64,
        )

        rpy = quaternion_wxyz_to_rpy(quaternions)

        np.testing.assert_allclose(rpy[0], [0.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(rpy[1, 0], np.deg2rad(70.0), places=12)
        self.assertAlmostEqual(rpy[2, 2], np.deg2rad(60.0), places=12)

    def test_evaluator_keeps_cycles_separate_at_episode_resets(self) -> None:
        total = 200
        joint = np.zeros((total, 1, 29), dtype=np.float32)
        contacts = np.zeros((total, 1, 2), dtype=bool)
        episode_ids = np.repeat(np.array([0, 1], dtype=np.int64), 100)[:, None]
        for episode in range(2):
            start = episode * 100
            local = np.arange(100)
            contacts[start : start + 100, 0, 0] = local % 20 < 3
            contacts[start : start + 100, 0, 1] = (local - 10) % 20 < 3
            reference, _, _ = reference_phase(contacts[start : start + 100, 0, 0])
            joint[start : start + 100, 0, 0] = np.nan_to_num(reference, nan=0.0)

        report, predictions = evaluate_rollout_arrays(
            joint,
            contacts,
            episode_ids,
            PhaseFromLastJoint(),
            mean=np.zeros(29, dtype=np.float32),
            std=np.ones(29, dtype=np.float32),
            calibration={
                "tolerance_rad": 0.01,
                "maximum_forward_step_rad": 0.4,
                "maximum_reverse_step_rad": 0.01,
                "macro_completion_fraction_required": 0.9,
            },
            point_threshold_rad=0.01,
            terminated=(np.arange(total) == 100)[:, None],
            truncated=np.zeros((total, 1), dtype=bool),
        )

        self.assertTrue(report["phase_rollout_gate_pass"], report)
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["termination_count"], 1)
        self.assertEqual(report["truncation_count"], 0)
        self.assertEqual(len(report["per_environment"]), 1)
        self.assertEqual(report["per_environment"][0]["termination_count"], 1)
        self.assertEqual(report["per_environment"][0]["complete_cycle_count"], 6)
        self.assertEqual(report["per_environment"][0]["unclosed_cycle_count"], 2)
        self.assertEqual(report["per_environment"][0]["episode_count"], 2)
        self.assertEqual(
            predictions["predicted_phase"].shape,
            (report["per_environment"][0]["prediction_count"], 2),
        )

    def test_episode_752_cycle_28_to_58_uses_unlabelled_terminal_touchdown(self) -> None:
        angles = np.zeros(80, dtype=np.float32)
        angles[28:59] = np.linspace(0.0, 2.0 * np.pi, 31, dtype=np.float32)
        angles[59:] = 2.0 * np.pi

        report, predictions = self._single_cycle_report(angle=angles)
        row = report["per_environment"][0]

        self.assertEqual(row["complete_cycle_count"], 1)
        self.assertEqual(row["cycle_start_frame_indices"], [28])
        self.assertEqual(row["cycle_end_frame_indices"], [58])
        self.assertEqual(row["cycle_completed_one_turn"], [True])
        endpoint = np.flatnonzero(predictions["frame_indices"] == 58)
        self.assertEqual(endpoint.tolist(), [49])
        self.assertFalse(bool(predictions["target_valid_mask"][endpoint[0]]))
        self.assertTrue(bool(predictions["left_touchdown"][endpoint[0]]))

    def test_first_touchdown_without_ten_frame_history_is_not_a_scored_cycle(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 80, dtype=np.float32)
        report, _ = self._single_cycle_report(start=2, end=32, angle=angles)
        row = report["per_environment"][0]

        self.assertEqual(row["insufficient_history_cycle_count"], 1)
        self.assertEqual(row["complete_cycle_count"], 0)
        self.assertEqual(row["cycle_coverage_status"], "insufficient_coverage")
        self.assertIsNone(row["cycle_completed_one_turn_fraction"])

    def test_touchdowns_on_opposite_sides_of_reset_do_not_form_a_cycle(self) -> None:
        episodes = np.full(80, 752, dtype=np.int64)
        episodes[40:] = 753
        report, _ = self._single_cycle_report(episode_at=episodes)
        row = report["per_environment"][0]

        self.assertEqual(row["complete_cycle_count"], 0)
        self.assertEqual(row["cycle_coverage_status"], "insufficient_coverage")
        self.assertIsNone(report["cycle_progression_pass"])
        self.assertIsNone(report["phase_rollout_gate_pass"])

    def test_constant_predicted_phase_is_a_completed_but_failed_cycle(self) -> None:
        report, _ = self._single_cycle_report()
        row = report["per_environment"][0]

        self.assertEqual(row["complete_cycle_count"], 1)
        self.assertAlmostEqual(row["cycle_net_progression_rad"][0], 0.0, places=6)
        self.assertEqual(row["cycle_completed_one_turn"], [False])
        self.assertEqual(row["cycle_completed_one_turn_fraction"], 0.0)

    def test_reverse_phase_progression_fails_a_closed_contact_cycle(self) -> None:
        angles = np.zeros(80, dtype=np.float32)
        angles[28:59] = np.linspace(0.0, -2.0 * np.pi, 31, dtype=np.float32)
        angles[59:] = -2.0 * np.pi
        report, _ = self._single_cycle_report(angle=angles)
        row = report["per_environment"][0]

        self.assertEqual(row["complete_cycle_count"], 1)
        self.assertLess(row["cycle_net_progression_rad"][0], 0.0)
        self.assertEqual(row["cycle_completed_one_turn"], [False])

    def test_large_phase_jump_fails_even_when_endpoint_advances_one_turn(self) -> None:
        angles = np.zeros(80, dtype=np.float32)
        angles[28:40] = np.linspace(0.0, 0.5, 12, dtype=np.float32)
        angles[40:59] = np.linspace(2.5, 2.0 * np.pi, 19, dtype=np.float32)
        angles[59:] = 2.0 * np.pi
        report, _ = self._single_cycle_report(angle=angles)
        row = report["per_environment"][0]

        self.assertEqual(row["complete_cycle_count"], 1)
        self.assertAlmostEqual(row["cycle_net_progression_rad"][0], 2.0 * np.pi, places=5)
        self.assertGreater(row["cycle_max_forward_step_rad"][0], 1.9)
        self.assertEqual(row["cycle_completed_one_turn"], [False])


if __name__ == "__main__":
    unittest.main()
