import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.encoder import (
    PhaseEncoder,
    circular_phase_loss,
    normalize_phase,
    select_validation_candidate,
    target_vectors,
)


class PhaseEncoderTests(unittest.TestCase):
    def test_target_vectors_use_sin_then_cos(self):
        values = target_vectors(torch.tensor([0.0, torch.pi / 2, torch.pi]))
        torch.testing.assert_close(
            values,
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_normalize_phase_is_finite_and_unit_for_nonzero_rows(self):
        raw = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        result = normalize_phase(raw)
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result[0])), 1.0)
        torch.testing.assert_close(result[1], torch.zeros(2))

    def test_circular_phase_loss_is_zero_for_matching_direction(self):
        target = target_vectors(torch.tensor([0.0, torch.pi / 2]))
        self.assertAlmostEqual(float(circular_phase_loss(target, target)), 0.0)

    def test_both_architectures_keep_same_io_contract(self):
        batch = torch.randn(4, 10, 29)
        pooled = PhaseEncoder(preserve_time=False)
        positional = PhaseEncoder(preserve_time=True)
        self.assertEqual(tuple(pooled(batch).shape), (4, 2))
        self.assertEqual(tuple(positional(batch).shape), (4, 2))
        self.assertTrue(torch.isfinite(positional(batch)).all())

    def test_validation_selection_ignores_test_values(self):
        selected = select_validation_candidate(
            {"pooled": 0.20, "preserve_time": 0.25}
        )
        self.assertEqual(selected, "pooled")
        self.assertEqual(
            select_validation_candidate([("preserve_time", 0.10), ("pooled", 0.10)]),
            "pooled",
        )


if __name__ == "__main__":
    unittest.main()
