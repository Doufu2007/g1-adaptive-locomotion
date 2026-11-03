import unittest

import torch

from scripts.g1_tctst_faithful_v2 import ChannelIdentifiedTCTST


class TestTCTSTReadout(unittest.TestCase):
    def make_oversmoothed_model(self):
        torch.manual_seed(7)
        model = ChannelIdentifiedTCTST(window=100, dim=48, heads=4, layers=2)
        with torch.no_grad():
            for parameter in model.transformer.parameters():
                parameter.zero_()
        return model

    def test_phase_readout_retains_sample_signal_when_transformer_oversmooths(self):
        model = self.make_oversmoothed_model()
        inputs = torch.randn(8, 100, 29)
        predictions = model(inputs)[:, :2]

        across_samples = predictions.std(dim=0, unbiased=False).mean().item()
        self.assertGreater(across_samples, 1e-4)

    def test_phase_loss_backpropagates_to_temporal_embedding_after_oversmoothing(self):
        model = self.make_oversmoothed_model().train()
        inputs = torch.randn(8, 100, 29)
        loss = model(inputs)[:, 0].mean()

        loss.backward()

        gradient_norm = model.embedding.weight.grad.norm().item()
        self.assertGreater(gradient_norm, 1e-8)

    def test_phase_loss_reaches_early_transformer_when_last_layer_collapses(self):
        torch.manual_seed(11)
        model = ChannelIdentifiedTCTST(window=100, dim=48, heads=4, layers=2).train()
        with torch.no_grad():
            for parameter in model.transformer.layers[-1].parameters():
                parameter.zero_()

        inputs = torch.randn(8, 100, 29)
        model(inputs)[:, 0].mean().backward()

        gradient_norm = model.transformer.layers[0].linear1.weight.grad.norm().item()
        self.assertGreater(gradient_norm, 1e-8)


if __name__ == "__main__":
    unittest.main()
