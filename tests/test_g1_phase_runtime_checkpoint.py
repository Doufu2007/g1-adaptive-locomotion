from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_g1_phase_runtime_checkpoint import build_runtime_checkpoint
from src.tasks.velocity.mdp import observations, rewards
from src.tasks.velocity.phasor.data import G1_JOINT_ORDER


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeCheckpointPreparationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
        dataset = root / "phase_dataset.npz"
        dataset.write_bytes(b"synthetic dataset bytes")
        metadata = root / "dataset_metadata.json"
        metadata.write_text(
            json.dumps({"train_sequences": ["walk_train_a"]}), encoding="utf-8"
        )
        manifest = root / "provenance.json"
        manifest.write_text(json.dumps({"status": "READY"}), encoding="utf-8")
        source = root / "source.pt"
        torch.save(
            {
                "state_dict": {"weight": torch.arange(3, dtype=torch.float32)},
                "joint_dim": 29,
                "window": 10,
                "preserve_time": True,
                "output_convention": "(sin(phi), cos(phi))",
                "mean": np.zeros(29, dtype=np.float32),
                "std": np.ones(29, dtype=np.float32),
                "dataset_sha256": sha256_file(dataset),
                "data_manifest_sha256": sha256_file(manifest),
                "train_sequences": ["walk_train_a"],
            },
            source,
        )
        template = root / "reward_template.json"
        template.write_text(
            json.dumps(
                {
                    "schema": "g1.learned_phase_reward_template.v1",
                    "status": "FROZEN",
                    "fit_split": "train",
                    "dataset_sha256": sha256_file(dataset),
                    "dataset_metadata_sha256": sha256_file(metadata),
                    "provenance_manifest_sha256": sha256_file(manifest),
                    "joint_order": list(G1_JOINT_ORDER),
                    "train_sequences": ["walk_train_a"],
                    "offset_cycles": [0.0, 0.5],
                    "stance_threshold": 0.58,
                }
            ),
            encoding="utf-8",
        )
        output = root / "runtime.pt"
        return source, dataset, metadata, manifest, template, output

    def test_runtime_checkpoint_binds_weights_to_train_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            source, dataset, metadata, manifest, template, output = paths
            source_hash_before = sha256_file(source)

            built = build_runtime_checkpoint(
                source, dataset, metadata, manifest, template, output
            )
            runtime = torch.load(output, map_location="cpu", weights_only=False)
            original = torch.load(source, map_location="cpu", weights_only=False)

            self.assertEqual(built["sha256"], sha256_file(output))
            self.assertEqual(runtime["joint_names"], list(G1_JOINT_ORDER))
            self.assertEqual(set(runtime["state_dict"]), set(original["state_dict"]))
            for name, tensor in original["state_dict"].items():
                torch.testing.assert_close(runtime["state_dict"][name], tensor)
            self.assertEqual(runtime["runtime_contract"]["dataset_sha256"], sha256_file(dataset))
            self.assertEqual(runtime["runtime_contract"]["metadata_sha256"], sha256_file(metadata))
            self.assertEqual(runtime["runtime_contract"]["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(runtime["runtime_contract"]["reward_template_sha256"], sha256_file(template))
            self.assertEqual(sha256_file(source), source_hash_before)

    def test_unverified_candidate_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            source, dataset, metadata, manifest, template, output = paths
            manifest.write_text(json.dumps({"status": "CONTACT_AUDIT_PENDING"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest hash"):
                build_runtime_checkpoint(source, dataset, metadata, manifest, template, output)

    def test_real_training_checkpoint_loads_with_frozen_runtime_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifact_dir = root / "results" / "g1_phase_dataset_20261019"
        source = artifact_dir / "phase_encoder_checkpoint.pt"
        dataset = artifact_dir / "phase_dataset.npz"
        metadata = artifact_dir / "dataset_metadata.json"
        manifest = root / "docs" / "data" / "g1_phase_data_provenance.json"
        template = artifact_dir / "g1_reward_template_v1.json"

        with tempfile.TemporaryDirectory() as temporary:
            runtime_path = Path(temporary) / "runtime.pt"
            build_runtime_checkpoint(source, dataset, metadata, manifest, template, runtime_path)
            invalid_path = Path(temporary) / "runtime_bad_contract.pt"
            invalid = torch.load(runtime_path, map_location="cpu", weights_only=False)
            invalid["runtime_contract"]["joint_order"] = list(reversed(G1_JOINT_ORDER))
            torch.save(invalid, invalid_path)
            with self.assertRaisesRegex(ValueError, "runtime contract"):
                observations._load_phasor_encoder(
                    type("CpuEnv", (), {"device": torch.device("cpu")})(), str(invalid_path)
                )
            bundle = observations._load_phasor_encoder(
                type("CpuEnv", (), {"device": torch.device("cpu")})(), str(runtime_path)
            )
            phase = bundle.encoder(torch.zeros(2, 10, 29))

        self.assertEqual(bundle.joint_names, G1_JOINT_ORDER)
        self.assertEqual(bundle.joint_dim, 29)
        self.assertEqual(bundle.window, 10)
        self.assertTrue(torch.isfinite(phase).all())
        self.assertEqual(tuple(phase.shape), (2, 2))

    def test_actor_and_reward_share_one_encoder_inference_per_control_step(self) -> None:
        from types import SimpleNamespace

        root = Path(__file__).resolve().parents[1]
        artifact_dir = root / "results" / "g1_phase_dataset_20261019"
        with tempfile.TemporaryDirectory() as temporary:
            runtime_path = Path(temporary) / "runtime.pt"
            build_runtime_checkpoint(
                artifact_dir / "phase_encoder_checkpoint.pt",
                artifact_dir / "phase_dataset.npz",
                artifact_dir / "dataset_metadata.json",
                root / "docs" / "data" / "g1_phase_data_provenance.json",
                artifact_dir / "g1_reward_template_v1.json",
                runtime_path,
            )
            asset = SimpleNamespace(
                joint_names=G1_JOINT_ORDER,
                data=SimpleNamespace(joint_pos=torch.zeros(2, 29)),
            )
            sensor = SimpleNamespace(
                data=SimpleNamespace(current_contact_time=torch.ones(2, 2))
            )
            command_manager = SimpleNamespace(
                get_command=lambda name: torch.ones(2, 3)
            )
            env = SimpleNamespace(
                device=torch.device("cpu"),
                num_envs=2,
                common_step_counter=11,
                scene={"robot": asset, "feet_ground_contact": sensor},
                command_manager=command_manager,
            )
            observations.reset_learned_phase_history(env, None, str(runtime_path))
            bundle = observations._load_phasor_encoder(env, str(runtime_path))
            calls: list[torch.Tensor] = []
            hook = bundle.encoder.register_forward_hook(lambda _module, _inputs, output: calls.append(output))
            try:
                phase = observations.learned_phase(env, str(runtime_path))
                reward = rewards.feet_gait_learned(
                    env,
                    encoder_path=str(runtime_path),
                    offset=[0.0, 0.5],
                    threshold=0.5869565217391305,
                    command_threshold=0.1,
                    command_name="twist",
                    sensor_name="feet_ground_contact",
                )
            finally:
                hook.remove()

        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(phase.shape), (2, 2))
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertTrue(torch.isfinite(phase).all())
        self.assertTrue(torch.isfinite(reward).all())


if __name__ == "__main__":
    unittest.main()
