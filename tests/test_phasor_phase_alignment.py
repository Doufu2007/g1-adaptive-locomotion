import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.analyze_phasor_phase_alignment import build_report
from src.tasks.velocity.phasor import diagnostics


def phase_and_contacts(cycles, direction=-1, shift=0.237):
    gait_cycles = np.asarray(cycles, dtype=np.float64)
    latent_cycles = np.mod(direction * (gait_cycles - shift), 1.0)
    angles = latent_cycles * (2.0 * np.pi)
    phase = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    contact_cycle = np.mod(gait_cycles, 1.0)
    contacts = np.stack(
        (
            contact_cycle < 0.56,
            np.mod(contact_cycle + 0.5, 1.0) < 0.56,
        ),
        axis=-1,
    )
    return phase[:, None, :], contacts[:, None, :]


class PhaseAlignmentTests(unittest.TestCase):
    def test_global_alignment_fitted_on_one_seed_transfers_to_heldout_seed(self):
        calibration_cycles = np.linspace(0.0, 2.0, 129, endpoint=False)
        heldout_cycles = np.linspace(0.11, 3.11, 193, endpoint=False)
        calibration_phase, calibration_contacts = phase_and_contacts(
            calibration_cycles
        )
        heldout_phase, heldout_contacts = phase_and_contacts(heldout_cycles)

        fitted = diagnostics.calibrate_phase_alignment(
            calibration_phase,
            calibration_contacts,
            threshold=0.56,
            grid_size=1000,
        )
        result = diagnostics.evaluate_phase_alignment(
            calibration_phase,
            calibration_contacts,
            heldout_phase,
            heldout_contacts,
            threshold=0.56,
            grid_size=1000,
        )

        self.assertAlmostEqual(fitted["calibration_contact_agreement"], 1.0)
        self.assertAlmostEqual(result["calibration_contact_agreement"], 1.0)
        self.assertAlmostEqual(result["heldout_contact_agreement"], 1.0)
        self.assertEqual(result["heldout_contact_agreement_by_env"], [1.0])

    def test_report_binds_recomputed_result_to_both_trajectory_bundles(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            calibration_dir = root / "calibration"
            heldout_dir = root / "heldout"
            calibration_dir.mkdir()
            heldout_dir.mkdir()

            def write_bundle(directory, seed, cycles):
                phase, contacts = phase_and_contacts(cycles)
                trajectory_path = directory / "trajectories.npz"
                np.savez_compressed(
                    trajectory_path, phase=phase, contacts=contacts
                )
                digest = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
                summary = {
                    "seed": seed,
                    "checkpoint_sha256": "a" * 64,
                    "encoder_sha256": "b" * 64,
                    "trajectory_file": trajectory_path.name,
                    "trajectory_sha256": digest,
                }
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )

            write_bundle(
                calibration_dir, 20261015, np.linspace(0.0, 2.0, 129, endpoint=False)
            )
            write_bundle(
                heldout_dir, 20261016, np.linspace(0.11, 3.11, 193, endpoint=False)
            )

            report = build_report(calibration_dir, heldout_dir)

        self.assertEqual(report["calibration_seed"], 20261015)
        self.assertEqual(report["heldout_seed"], 20261016)
        self.assertAlmostEqual(report["calibration_contact_agreement"], 1.0)
        self.assertAlmostEqual(report["heldout_contact_agreement"], 1.0)
        self.assertEqual(len(report["input_artifacts"]), 4)
        self.assertTrue(
            all(item["sha256"].startswith("sha256:") for item in report["input_artifacts"])
        )

    def test_report_rejects_trajectory_bundle_with_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            calibration_dir = root / "calibration"
            heldout_dir = root / "heldout"
            calibration_dir.mkdir()
            heldout_dir.mkdir()

            for directory, seed in (
                (calibration_dir, 20261015),
                (heldout_dir, 20261016),
            ):
                phase, contacts = phase_and_contacts(np.linspace(0.0, 1.0, 65))
                np.savez_compressed(
                    directory / "trajectories.npz", phase=phase, contacts=contacts
                )
                summary = {
                    "seed": seed,
                    "checkpoint_sha256": "a" * 64,
                    "encoder_sha256": "b" * 64,
                    "trajectory_file": "trajectories.npz",
                    "trajectory_sha256": "0" * 64,
                }
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "trajectory hash mismatch"):
                build_report(calibration_dir, heldout_dir)

    def test_cli_runs_when_called_outside_the_repository_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            calibration_dir = root / "calibration"
            heldout_dir = root / "heldout"
            calibration_dir.mkdir()
            heldout_dir.mkdir()

            for directory, seed, cycles in (
                (calibration_dir, 20261015, np.linspace(0.0, 2.0, 129, endpoint=False)),
                (heldout_dir, 20261016, np.linspace(0.11, 3.11, 193, endpoint=False)),
            ):
                phase, contacts = phase_and_contacts(cycles)
                trajectory_path = directory / "trajectories.npz"
                np.savez_compressed(
                    trajectory_path, phase=phase, contacts=contacts
                )
                digest = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
                (directory / "summary.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "checkpoint_sha256": "a" * 64,
                            "encoder_sha256": "b" * 64,
                            "trajectory_file": trajectory_path.name,
                            "trajectory_sha256": digest,
                        }
                    ),
                    encoding="utf-8",
                )

            output = root / "alignment.json"
            script = Path(__file__).resolve().parents[1] / "scripts" / (
                "analyze_phasor_phase_alignment.py"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--calibration-dir",
                    str(calibration_dir),
                    "--heldout-dir",
                    str(heldout_dir),
                    "--output",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["heldout_seed"], 20261016)


if __name__ == "__main__":
    unittest.main()
