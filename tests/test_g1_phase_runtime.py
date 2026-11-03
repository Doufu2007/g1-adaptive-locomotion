import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.velocity.mdp import observations


class G1PhaseRuntimeTests(unittest.TestCase):
  def test_normalization_accepts_numpy_and_tensor_values(self) -> None:
    for mean, std in (
      (np.arange(3, dtype=np.float64), np.ones(3, dtype=np.float64)),
      (torch.arange(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)),
      (np.arange(3, dtype=np.float32)[None, :], np.ones((1, 3), dtype=np.float32)),
    ):
      actual_mean, actual_std = observations._validated_encoder_stats(
        mean, std, joint_dim=3, device=torch.device("cpu")
      )
      self.assertEqual(actual_mean.dtype, torch.float32)
      self.assertEqual(actual_std.dtype, torch.float32)
      self.assertEqual(tuple(actual_mean.shape), (3,))
      self.assertEqual(tuple(actual_std.shape), (3,))

  def test_normalization_rejects_wrong_shape(self) -> None:
    with self.assertRaisesRegex(ValueError, "shape"):
      observations._validated_encoder_stats(
        np.zeros(4), np.ones(3), joint_dim=3, device=torch.device("cpu")
      )
    with self.assertRaisesRegex(ValueError, "shape"):
      observations._validated_encoder_stats(
        np.zeros((2, 3)), np.ones((2, 3)), joint_dim=3, device=torch.device("cpu")
      )

  def test_normalization_rejects_nonfinite_and_nonpositive_std(self) -> None:
    with self.assertRaisesRegex(ValueError, "finite"):
      observations._validated_encoder_stats(
        np.array([0.0, np.nan, 0.0]), np.ones(3), joint_dim=3, device=torch.device("cpu")
      )
    with self.assertRaisesRegex(ValueError, "finite"):
      observations._validated_encoder_stats(
        np.zeros(3), np.array([1.0, np.inf, 1.0]), joint_dim=3, device=torch.device("cpu")
      )
    with self.assertRaisesRegex(ValueError, "positive"):
      observations._validated_encoder_stats(
        np.zeros(3), np.array([1.0, 0.0, -0.5]), joint_dim=3, device=torch.device("cpu")
      )

  def test_joint_positions_are_mapped_by_name(self) -> None:
    asset = SimpleNamespace(joint_names=("knee_joint", "hip_joint"))
    positions = torch.tensor([[2.0, 1.0]])
    ordered = observations._joint_positions_in_order(
      asset, positions, ("hip_joint", "knee_joint")
    )
    torch.testing.assert_close(ordered, torch.tensor([[1.0, 2.0]]))

  def test_joint_mapping_fails_on_missing_or_duplicate_names(self) -> None:
    with self.assertRaisesRegex(ValueError, "missing"):
      observations._joint_positions_in_order(
        SimpleNamespace(joint_names=("hip_joint",)),
        torch.zeros(1, 1),
        ("hip_joint", "knee_joint"),
      )
    with self.assertRaisesRegex(ValueError, "unique"):
      observations._joint_positions_in_order(
        SimpleNamespace(joint_names=("hip_joint", "hip_joint")),
        torch.zeros(1, 2),
        ("hip_joint", "knee_joint"),
      )


if __name__ == "__main__":
  unittest.main()
