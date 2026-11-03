import unittest

import numpy as np
import torch

from src.tasks.velocity.phasor.encoder_v2 import (
    PhaseEncoderV2,
    build_phase_v2_features,
    phase_v2_loss,
    predict_phase_v2,
    target_vectors_v2,
)
from scripts.train_g1_phase_encoder_v2 import (
    build_context_windows,
    build_cycle_balanced_pairs,
    checkpoint_epochs,
    load_train_validation_split,
)


class PhaseEncoderV2Tests(unittest.TestCase):
    def test_features_append_joint_velocity_and_scalar_speed(self):
        pose = torch.zeros(1, 2, 29)
        pose[0, 1, 0] = 3.0
        pose[0, 1, 1] = 4.0
        features = build_phase_v2_features(pose)
        self.assertEqual(tuple(features.shape), (1, 2, 59))
        torch.testing.assert_close(features[0, 0, 29:58], torch.zeros(29))
        torch.testing.assert_close(features[0, 1, 29], torch.tensor(3.0))
        torch.testing.assert_close(features[0, 1, 30], torch.tensor(4.0))
        torch.testing.assert_close(features[0, 1, 58], torch.tensor((25.0 / 29.0) ** 0.5))

    def test_phase_rate_loss_rewards_label_matched_positive_progress(self):
        current = target_vectors_v2(torch.tensor([0.0]))
        next_label = target_vectors_v2(torch.tensor([0.2]))
        matched = phase_v2_loss(current, next_label, torch.tensor([0.0]), torch.tensor([0.2]))
        stalled = phase_v2_loss(current, current, torch.tensor([0.0]), torch.tensor([0.2]))
        self.assertLess(float(matched["rate_loss"]), float(stalled["rate_loss"]))
        self.assertAlmostEqual(float(matched["rate_loss"]), 0.0, places=6)

    def test_reverse_progress_and_large_jump_have_explicit_penalties(self):
        current = target_vectors_v2(torch.tensor([0.0]))
        reverse = target_vectors_v2(torch.tensor([-0.2]))
        jump = target_vectors_v2(torch.tensor([2.0]))
        reverse_loss = phase_v2_loss(current, reverse, torch.tensor([0.0]), torch.tensor([0.2]))
        jump_loss = phase_v2_loss(current, jump, torch.tensor([0.0]), torch.tensor([2.0]))
        self.assertGreater(float(reverse_loss["negative_progress_penalty"]), 0.0)
        self.assertGreater(float(jump_loss["jump_penalty"]), 0.0)

    def test_reverse_prediction_gradient_moves_toward_positive_label_rate(self):
        predicted_angle = torch.tensor([-0.2], requires_grad=True)
        current = target_vectors_v2(torch.tensor([0.0]))
        next_prediction = target_vectors_v2(predicted_angle)
        losses = phase_v2_loss(
            current,
            next_prediction,
            torch.tensor([0.0]),
            torch.tensor([0.2]),
        )
        losses["loss"].backward()
        self.assertLess(float(predicted_angle.grad), 0.0)

    def test_offline_prediction_uses_shared_feature_builder_and_model_contract(self):
        model = PhaseEncoderV2(context_frames=30, hidden=8)
        pose = torch.randn(3, 30, 29)
        direct = model(build_phase_v2_features(pose))
        shared = predict_phase_v2(model, pose)
        self.assertEqual(tuple(shared.shape), (3, 2))
        torch.testing.assert_close(shared, direct)
        self.assertTrue(torch.isfinite(shared).all())

    def test_training_split_loader_refuses_test_split(self):
        archive = {
            "x_train": np.zeros((2, 10, 29), dtype=np.float32),
            "y_train_cycles": np.zeros(2, dtype=np.float32),
            "sequence_train": np.array(["a", "a"]),
            "frame_train": np.array([10, 11]),
            "left_touchdown_train": np.array([1, 0], dtype=np.uint8),
            "x_validation": np.zeros((2, 10, 29), dtype=np.float32),
            "y_validation_cycles": np.zeros(2, dtype=np.float32),
            "sequence_validation": np.array(["b", "b"]),
            "frame_validation": np.array([20, 21]),
            "left_touchdown_validation": np.array([1, 0], dtype=np.uint8),
            "x_test": np.ones((2, 10, 29), dtype=np.float32),
        }
        train = load_train_validation_split(archive, "train")
        self.assertEqual(set(train), {"x", "phase_cycles", "sequence", "frame", "left_touchdown"})
        with self.assertRaisesRegex(ValueError, "test split is forbidden"):
            load_train_validation_split(archive, "test")

    def test_long_context_does_not_bridge_a_frame_gap(self):
        x = np.zeros((4, 10, 29), dtype=np.float32)
        for row in range(4):
            x[row, :, 0] = row + 1
        data = {
            "x": x,
            "sequence": np.array(["walk"] * 4),
            "frame": np.array([0, 1, 3, 4]),
        }
        windows, history_valid = build_context_windows(data, 30)
        self.assertEqual(windows.shape, (4, 30, 29))
        self.assertTrue(np.all(windows[2, :, 0] == 3.0))
        self.assertEqual(history_valid.tolist(), [False, False, False, False])

    def test_rate_training_weights_each_complete_cycle_equally(self):
        frames = np.r_[np.arange(5), np.arange(10, 19)]
        touchdown = np.zeros(len(frames), dtype=np.uint8)
        touchdown[[0, 4, 5, 13]] = 1
        data = {
            "sequence": np.array(["walk"] * len(frames)),
            "frame": frames,
            "left_touchdown": touchdown,
        }
        first, second, weights = build_cycle_balanced_pairs(data)
        pair_frames = [(int(frames[a]), int(frames[b])) for a, b in zip(first, second)]
        self.assertNotIn((4, 10), pair_frames)
        short_cycle_weight = sum(float(w) for (a, _), w in zip(pair_frames, weights) if 0 <= a < 4)
        long_cycle_weight = sum(float(w) for (a, _), w in zip(pair_frames, weights) if 10 <= a < 18)
        self.assertAlmostEqual(short_cycle_weight, 1.0)
        self.assertAlmostEqual(long_cycle_weight, 1.0)

    def test_checkpoint_selection_candidates_cover_each_fixed_training_epoch(self):
        self.assertEqual(checkpoint_epochs(4), (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
