import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.tasks.velocity.mdp import observations
from src.tasks.velocity.config.g1 import env_cfgs as g1_env_cfgs
from src.tasks.velocity.phasor import train_encoder
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER

try:
    diagnostics = importlib.import_module("src.tasks.velocity.phasor.diagnostics")
except ModuleNotFoundError as error:
    if error.name != "src.tasks.velocity.phasor.diagnostics":
        raise
    diagnostics = SimpleNamespace()


class CountingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, history):
        self.calls += 1
        phase = history[:, -1, :2]
        return phase / phase.norm(dim=1, keepdim=True).clamp_min(1e-8)


class FakeCommandManager:
    def __init__(self, num_envs):
        self.command = torch.ones(num_envs, 3)

    def get_command(self, name):
        return self.command


class PhaseStateTests(unittest.TestCase):
    def make_env(self, num_envs=4):
        joint_pos = torch.zeros(num_envs, 29)
        joint_pos[:, 0] = 1.0
        return SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=num_envs,
            common_step_counter=7,
            scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos), joint_names=G1_JOINT_ORDER)},
            command_manager=FakeCommandManager(num_envs),
        )

    def make_bundle(self, encoder):
        return SimpleNamespace(
            encoder=encoder,
            mean=torch.zeros(29),
            std=torch.ones(29),
            window=3,
            joint_dim=29,
            joint_names=G1_JOINT_ORDER,
            identity=("test-checkpoint", 1, 1, "cpu"),
            sha256="test-sha256",
        )

    def test_phase_history_advances_once_per_control_step(self):
        env = self.make_env()
        encoder = CountingEncoder()
        with patch.object(observations, "_load_phasor_encoder", return_value=self.make_bundle(encoder)):
            first = observations.learned_phase(env, "unused-checkpoint.pt")
            env.scene["robot"].data.joint_pos[:, 0] = 0.0
            env.scene["robot"].data.joint_pos[:, 1] = 1.0
            second = observations.learned_phase(env, "unused-checkpoint.pt")

        self.assertEqual(encoder.calls, 1)
        torch.testing.assert_close(second, first)

    def test_reset_clears_only_terminated_environment_histories(self):
        env = self.make_env()
        encoder = CountingEncoder()
        with patch.object(observations, "_load_phasor_encoder", return_value=self.make_bundle(encoder)):
            before = observations.learned_phase(env, "unused-checkpoint.pt")
            reset_history = getattr(observations, "reset_learned_phase_history", None)
            self.assertTrue(callable(reset_history))
            reset_history(env, torch.tensor([1, 3]), "unused-checkpoint.pt")

            env.scene["robot"].data.joint_pos[:, 0] = 0.0
            env.scene["robot"].data.joint_pos[:, 1] = 1.0
            after = observations.learned_phase(env, "unused-checkpoint.pt")

        self.assertEqual(encoder.calls, 2)
        torch.testing.assert_close(after[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(after[[1, 3]], torch.tensor([[0.0, 1.0]] * 2))

    def test_temporal_pairs_are_adjacent_in_the_original_sequence(self):
        make_pairs = getattr(train_encoder, "contiguous_window_pairs", None)
        self.assertTrue(callable(make_pairs))
        pairs = make_pairs(5, device=torch.device("cpu"))
        torch.testing.assert_close(
            pairs,
            torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
        )

    def test_motion_resampling_matches_the_policy_sampling_rate(self):
        resample = getattr(train_encoder, "resample_joint_positions", None)
        self.assertTrue(callable(resample))
        source = torch.tensor([[0.0], [1.0], [2.0]])
        sampled = resample(source, source_fps=2.0, target_fps=4.0)
        torch.testing.assert_close(
            sampled[:, 0], torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        )

    def test_sliding_windows_include_the_final_valid_window(self):
        make_windows = getattr(train_encoder, "sliding_windows", None)
        self.assertTrue(callable(make_windows))
        sequence = torch.arange(5, dtype=torch.float32).reshape(-1, 1).numpy()
        windows = make_windows(sequence, window=3)
        self.assertEqual(windows.shape, (3, 3, 1))
        torch.testing.assert_close(
            torch.from_numpy(windows[-1, :, 0]), torch.tensor([2.0, 3.0, 4.0])
        )

    def test_circular_coverage_handles_full_arcs_and_constant_latents(self):
        coverage = getattr(diagnostics, "circular_coverage_rad", None)
        self.assertTrue(callable(coverage))
        full_circle = torch.linspace(0.0, 2 * torch.pi, 1024)[:-1].numpy()
        partial_arc = torch.linspace(2.8, 3.4, 101).numpy()
        constant = torch.full((100,), -0.4).numpy()
        self.assertGreater(coverage(full_circle), 6.0)
        self.assertAlmostEqual(coverage(partial_arc), 0.6, places=2)
        self.assertAlmostEqual(coverage(constant), 0.0, places=5)

    def test_angle_unwrap_is_local_to_a_sequence(self):
        unwrap = getattr(diagnostics, "unwrap_angles", None)
        self.assertTrue(callable(unwrap))
        boundary_crossing = torch.deg2rad(torch.tensor([179.0, -179.0])).numpy()
        unwrapped = unwrap(boundary_crossing)
        self.assertAlmostEqual(float(unwrapped[-1] - unwrapped[0]), 0.0349066, places=5)

    def test_phase_contact_summary_recovers_known_two_foot_gait(self):
        summarize = getattr(diagnostics, "phase_contact_summary", None)
        self.assertTrue(callable(summarize))

        cycles = torch.arange(32, dtype=torch.float64) / 16.0 + 1.0 / 32.0
        angles = cycles * (2.0 * torch.pi)
        phase = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        phase = phase[:, None, :].numpy()
        left = [True] * 8 + [False] * 8
        right = [False] * 8 + [True] * 8
        left *= 2
        right *= 2
        contacts = torch.tensor(list(zip(left, right)), dtype=torch.bool).numpy()
        contacts = contacts[:, None, :]

        result = summarize(phase, contacts, offsets=(0.0, 0.5), threshold=0.5)

        self.assertAlmostEqual(result["contact_agreement"], 1.0)
        self.assertEqual(result["touchdown_counts"], [2, 2])
        for concentration in result["touchdown_phase_resultants"]:
            self.assertAlmostEqual(concentration, 1.0)
        self.assertAlmostEqual(result["mean_net_to_total_angular_travel"], 1.0)

    def test_phase_alignment_calibration_recovers_a_rotated_coordinate(self):
        calibrate = getattr(diagnostics, "calibrate_phase_alignment", None)
        self.assertTrue(callable(calibrate))

        target_cycles = torch.arange(32, dtype=torch.float64) / 16.0 + 1.0 / 32.0
        latent_cycles = torch.remainder(target_cycles - 0.25, 1.0)
        angles = latent_cycles * (2.0 * torch.pi)
        phase = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        phase = phase[:, None, :].numpy()
        left = [True] * 8 + [False] * 8
        right = [False] * 8 + [True] * 8
        contacts = torch.tensor(list(zip(left * 2, right * 2)), dtype=torch.bool).numpy()
        contacts = contacts[:, None, :]

        result = calibrate(
            phase, contacts, offsets=(0.0, 0.5), threshold=0.5, grid_size=64
        )

        self.assertAlmostEqual(result["calibration_contact_agreement"], 1.0)
        self.assertIn(result["phase_direction"], (-1, 1))
        self.assertGreaterEqual(result["phase_shift_cycles"], 0.0)
        self.assertLess(result["phase_shift_cycles"], 1.0)

    def test_zero_phase_preserves_batch_shape_and_removes_signal(self):
        env = self.make_env(num_envs=3)
        output = observations.zero_phase(env, command_name="twist")
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertEqual(int(torch.count_nonzero(output)), 0)

    def test_five_condition_matrix_changes_only_declared_phase_factors(self):
        configs = [
            g1_env_cfgs.unitree_g1_flat_no_phase_env_cfg(),
            g1_env_cfgs.unitree_g1_flat_env_cfg(),
            g1_env_cfgs.unitree_g1_flat_phasor_obs_env_cfg(),
            g1_env_cfgs.unitree_g1_flat_phasor_reward_env_cfg(),
            g1_env_cfgs.unitree_g1_flat_phasor_env_cfg(),
        ]
        expected_observation = [
            observations.zero_phase,
            observations.phase,
            observations.learned_phase,
            observations.phase,
            observations.learned_phase,
        ]
        expected_reward = [
            "feet_gait",
            "feet_gait",
            "feet_gait",
            "feet_gait_learned",
            "feet_gait_learned",
        ]
        baseline = configs[1]
        baseline_actor_terms = set(baseline.observations["actor"].terms)
        baseline_critic_terms = set(baseline.observations["critic"].terms)
        baseline_rewards = set(baseline.rewards)

        for config, obs_func, reward_func_name in zip(
            configs, expected_observation, expected_reward, strict=True
        ):
            self.assertEqual(
                config.observations["actor"].terms["phase"].func, obs_func
            )
            self.assertEqual(
                config.rewards["foot_gait"].func.__name__, reward_func_name
            )
            self.assertEqual(set(config.observations["actor"].terms), baseline_actor_terms)
            self.assertEqual(set(config.observations["critic"].terms), baseline_critic_terms)
            self.assertEqual(set(config.rewards), baseline_rewards)
            self.assertEqual(config.scene.num_envs, baseline.scene.num_envs)
            self.assertEqual(config.sim.mujoco.timestep, baseline.sim.mujoco.timestep)
            self.assertEqual(config.decimation, baseline.decimation)
            self.assertEqual(config.commands, baseline.commands)
            for reward_name in baseline_rewards - {"foot_gait"}:
                self.assertEqual(config.rewards[reward_name], baseline.rewards[reward_name])


if __name__ == "__main__":
    unittest.main()
