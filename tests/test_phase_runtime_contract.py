import unittest

import torch
from torch import nn

from src.tasks.velocity.phasor.runtime import PhaseHistoryState, phase_to_angle


class LastFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, history):
        self.calls += 1
        value = history[:, -1, :2]
        return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-8)


class PhaseRuntimeContractTests(unittest.TestCase):
    def test_reset_can_fill_each_environment_with_current_pose(self):
        state = PhaseHistoryState(2, 3, 2, torch.device("cpu"))
        pose = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        state.reset(fill_value=pose)
        torch.testing.assert_close(state.buffer[0], pose[0].expand(3, 2))
        torch.testing.assert_close(state.buffer[1], pose[1].expand(3, 2))

    def test_each_environment_updates_once_and_reuses_same_latent(self):
        state = PhaseHistoryState(2, 3, 2, torch.device("cpu"))
        encoder = LastFrameEncoder()
        pose = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        first = state.update(pose, 4, encoder, torch.zeros(2), torch.ones(2)).clone()
        second = state.update(torch.flip(pose, (1,)), 4, encoder, torch.zeros(2), torch.ones(2))
        self.assertEqual(encoder.calls, 1)
        torch.testing.assert_close(first, second)
        state.update(torch.flip(pose, (1,)), 5, encoder, torch.zeros(2), torch.ones(2))
        self.assertEqual(encoder.calls, 2)

    def test_angle_decoding_uses_sin_then_cos(self):
        vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
        angles = torch.remainder(phase_to_angle(vectors), 2 * torch.pi)
        torch.testing.assert_close(
            angles,
            torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
