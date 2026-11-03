import importlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.run_g1_long_history_comparison import LongContextTCNEncoder
from src.tasks.velocity.mdp import observations
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER

try:
    metrics = importlib.import_module('scripts.g1_closed_loop_metrics')
except ModuleNotFoundError:
    metrics = SimpleNamespace()


class ClosedLoopPilotTests(unittest.TestCase):
    def test_explicit_twenty_frame_runtime_and_sampling_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'encoder.pt'
            contract = dict(schema='g1.phase_runtime_checkpoint.experimental.v2',
                            joint_order=list(G1_JOINT_ORDER), joint_dim=29, window=20,
                            control_dt_seconds=0.02,
                            phase_output_convention='(sin(phi), cos(phi))')
            for name in ('source_checkpoint_sha256', 'dataset_sha256',
                         'metadata_sha256', 'manifest_sha256', 'reward_template_sha256'):
                contract[name] = 'a' * 64
            torch.save(dict(state_dict=LongContextTCNEncoder().state_dict(),
                            architecture='tcn_long_context', joint_dim=29, window=20,
                            joint_names=list(G1_JOINT_ORDER), runtime_contract=contract,
                            output_convention='(sin(phi), cos(phi))',
                            dataset_sha256='a'*64, data_manifest_sha256='a'*64,
                            mean=torch.zeros(29), std=torch.ones(29)), path)
            rng_before_load = torch.random.get_rng_state()
            bundle = observations._load_phasor_encoder(
                SimpleNamespace(device='cpu', step_dt=0.02), str(path))
            self.assertTrue(torch.equal(rng_before_load, torch.random.get_rng_state()))
            self.assertEqual(bundle.window, 20)
            original = LongContextTCNEncoder()
            original.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
            original.eval()
            x = torch.randn(3, 20, 29)
            torch.testing.assert_close(bundle.encoder(x), original(x), rtol=1e-6, atol=1e-7)
            with self.assertRaisesRegex(ValueError, 'sampling|50 Hz|control_dt'):
                observations._load_phasor_encoder(
                    SimpleNamespace(device='cpu', step_dt=0.01), str(path))

    def test_variable_cycle_reference_and_constant_phase(self):
        fn = getattr(metrics, 'phase_metrics', None)
        self.assertTrue(callable(fn), 'phase_metrics is not implemented')
        times = np.arange(301) * .02
        contacts = np.zeros((301, 2), dtype=bool)
        angle = np.zeros(301)
        bounds = [0, 50, 120, 210, 300]
        for start, end in zip(bounds[:-1], bounds[1:]):
            contacts[start:start+3, 0] = True
            angle[start:end] = np.linspace(0, 2*math.pi, end-start, endpoint=False)
        contacts[300, 0] = True
        good = fn(times, angle, contacts)
        bad = fn(times, np.zeros_like(angle), contacts)
        self.assertEqual(good['complete_cycles'], 3)
        self.assertAlmostEqual(good['one_turn_fraction'], 1.)
        self.assertAlmostEqual(bad['one_turn_fraction'], 0.)
        self.assertEqual(fn(times, angle, np.zeros_like(contacts))['status'], 'insufficient_cycles')

    def test_fall_does_not_shorten_velocity_error_horizon(self):
        fn = getattr(metrics, 'fixed_horizon_mae', None)
        self.assertTrue(callable(fn), 'fixed_horizon_mae is not implemented')
        self.assertAlmostEqual(fn(np.array([.5, .5]), np.full(4, .5)), .25)

    def test_clock_has_constant_frequency_despite_variable_contact_periods(self):
        times = np.arange(301)*.02
        contacts = np.zeros((301, 2), dtype=bool)
        for start in (0, 50, 120, 210, 300):
            contacts[start:start+3, 0] = True
        result = metrics.phase_metrics(times, 2*np.pi*times/.6, contacts)
        self.assertIsNone(result['frequency_correlation'])
        np.testing.assert_allclose([c['predicted_frequency_hz'] for c in result['cycles']], 1/.6)
        # Float32 actuator observations retain roundoff, not real clock variation.
        quantized = metrics.phase_metrics(times, (2*np.pi*times/.6).astype(np.float32), contacts)
        self.assertIsNone(quantized['frequency_correlation'])

    def test_first_push_cannot_recover_after_second_push(self):
        n, dt = 1000, .02
        velocity = np.zeros((n, 3))
        velocity[:, 0] = .5
        velocity[250:525, 1] = .3
        raw = dict(velocity=velocity, position=np.zeros((n, 3)), time=np.arange(n)*dt,
                   angle=np.zeros(n), contact=np.zeros((n, 2), bool), scenario='push')
        result = metrics.summarize_episode(raw, np.full(n, .5), dt, False)
        self.assertIsNone(result['push_recovery'][0]['recovered_seconds'])
        self.assertAlmostEqual(result['push_recovery'][1]['recovered_seconds'], .5)


if __name__ == '__main__':
    unittest.main()
