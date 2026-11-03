import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.g1_tctst_faithful_v2 import ChannelIdentifiedTCTST
from scripts.run_g1_external_methods import LongRateTCN
from src.tasks.velocity.mdp import observations
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER
from src.tasks.velocity.phasor.encoder_tcn_long_rate import LongRateTCNEncoder
from src.tasks.velocity.phasor.encoder_tctst_runtime import RepairedTCTSTEncoder


class LongWindowRuntimeTests(unittest.TestCase):
    def roundtrip(self, training_model, runtime_model, architecture, window):
        runtime_model.load_state_dict(training_model.state_dict(), strict=True)
        training_model.eval()
        runtime_model.eval()
        schema = ('g1.phase_runtime_checkpoint.experimental.v4'
                  if architecture == 'tctst_repaired'
                  else 'g1.phase_runtime_checkpoint.experimental.v3')
        contract = {
            'schema': schema,
            'joint_order': list(G1_JOINT_ORDER), 'joint_dim': 29, 'window': window,
            'control_dt_seconds': .02,
            'phase_output_convention': '(sin(phi), cos(phi))',
        }
        for name in ('source_checkpoint_sha256', 'dataset_sha256', 'metadata_sha256',
                     'manifest_sha256', 'reward_template_sha256'):
            contract[name] = 'a' * 64
        checkpoint = {
            'state_dict': training_model.state_dict(), 'architecture': architecture,
            'joint_dim': 29, 'window': window, 'joint_names': list(G1_JOINT_ORDER),
            'runtime_contract': contract, 'output_convention': '(sin(phi), cos(phi))',
            'dataset_sha256': 'a' * 64, 'data_manifest_sha256': 'a' * 64,
            'mean': torch.zeros(29), 'std': torch.ones(29),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.pt'
            torch.save(checkpoint, path)
            bundle = observations._load_phasor_encoder(
                SimpleNamespace(device='cpu', step_dt=.02), str(path))
            x = torch.randn(2, window, 29)
            with torch.inference_mode():
                expected = training_model(x).float()[:, :2]
                actual = bundle.encoder(x).float()
            self.assertEqual(bundle.window, window)
            self.assertEqual(tuple(actual.shape), (2, 2))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            with self.assertRaisesRegex(ValueError, '50 Hz|control_dt'):
                observations._load_phasor_encoder(
                    SimpleNamespace(device='cpu', step_dt=.01), str(path))

    def test_tcn100_rate_checkpoint_runtime_parity(self):
        torch.manual_seed(101)
        self.roundtrip(LongRateTCN().eval(), LongRateTCNEncoder().eval(),
                       'tcn_long_rate', 100)

    def test_repaired_tctst_checkpoint_runtime_parity(self):
        torch.manual_seed(102)
        self.roundtrip(ChannelIdentifiedTCTST().eval(), RepairedTCTSTEncoder().eval(),
                       'tctst_repaired', 100)


if __name__ == '__main__':
    unittest.main()
