import json
import math
import sys
from pathlib import Path

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks.velocity.phasor.data import (  # noqa: E402
    ContactDetectionConfig,
    ProvenanceError,
    debounce_contact_sequence,
    detect_foot_contacts,
    fit_normalizer,
    left_cycle_phase,
    make_windows,
    resample_joint_positions,
    split_sequences,
    validate_provenance,
)


def _manifest():
    return json.loads((ROOT / "docs/data/g1_phase_data_provenance.json").read_text())


class PhaseDataTests(unittest.TestCase):
  def test_binary_contact_debounce_matches_two_frame_hysteresis(self):
    raw = np.array([False, True, True, True, False, True, True, True, False, False])
    filtered = debounce_contact_sequence(raw, min_contact_frames=2, min_release_frames=2)
    np.testing.assert_array_equal(
        filtered,
        [False, False, True, True, True, True, True, True, True, False],
    )

  def test_contact_detector_uses_hysteresis_and_debounce(self):
    foot = np.zeros((12, 2, 3), dtype=np.float32)
    foot[:, :, 2] = 0.08
    foot[2:6, :, 2] = 0.02
    foot[6, :, 2] = 0.035
    foot[7:10, :, 2] = 0.02
    contacts = detect_foot_contacts(
        foot,
        fps=50.0,
        config=ContactDetectionConfig(
            height_on=0.03,
            height_off=0.05,
            speed_on=0.5,
            speed_off=0.8,
            min_contact_frames=2,
            min_release_frames=2,
        ),
    )
    self.assertFalse(contacts[3, 0])
    self.assertTrue(contacts[4, 0])
    self.assertTrue(contacts[5, 0])
    self.assertTrue(contacts[6, 0])
    self.assertFalse(contacts[10, 0])

  def test_resample_joint_positions_preserves_endpoints(self):
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    frames = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    new_times, new_frames = resample_joint_positions(times, frames, target_hz=10.0)
    np.testing.assert_allclose(new_times, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(new_frames[:, 0], [0.0, 1.0, 2.0])

  def test_unverified_dance_manifest_is_blocked(self):
    manifest = _manifest()
    # Build the historical dance-only shape explicitly; the repository
    # manifest now contains verified walking candidates by design.
    manifest["candidate_walk_sequences"]["file_metadata"] = [
        {"action_type": "dance", "walk_verified": False, "license": "CC BY-NC-ND-4.0",
         "retargeting": {"joint_mapping": "G1 29-joint mapping"}}
    ]
    with self.assertRaisesRegex(ProvenanceError, "walking|contact"):
        validate_provenance(manifest)


  def test_right_touchdown_phase_is_measured_not_hardcoded_half_cycle(self):
    n = 31
    contacts = np.zeros((n, 2), dtype=bool)
    contacts[[0, 10, 20, 30], 0] = True
    contacts[[4, 15, 25], 1] = True
    # Keep contacts active for one frame so the event detector sees rising edges.
    contacts[1, 0] = contacts[11, 0] = contacts[21, 0] = True
    contacts[5, 1] = contacts[16, 1] = contacts[26, 1] = True
    timestamps = np.arange(n, dtype=float) * 0.01
    labels = left_cycle_phase(contacts, timestamps)
    self.assertTrue(np.isclose(labels.phase_cycles[10], 0.0))
    np.testing.assert_allclose(labels.right_touchdown_phase_cycles, [0.4, 0.5, 0.5])
    self.assertNotAlmostEqual(labels.right_touchdown_phase_cycles[0], 0.5)


  def test_split_sequences_happens_before_window_generation(self):
    split = split_sequences(["s1", "s2", "s3", "s4"], seed=7)
    self.assertEqual(set(split), {"train", "validation", "test"})
    self.assertTrue(set(split["train"]).isdisjoint(split["validation"]))
    self.assertTrue(set(split["train"]).isdisjoint(split["test"]))
    self.assertTrue(set(split["validation"]).isdisjoint(split["test"]))

    frames = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    labels = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    sequence_ids = np.array(["a"] * 6 + ["b"] * 6)
    batch = make_windows(frames, labels, sequence_ids=sequence_ids, window=4)
    self.assertTrue(all(len(set(ids)) == 1 for ids in batch.sequence_ids))
    self.assertTrue(np.all(batch.frame_indices[:, -1] == batch.label_frame_indices))


  def test_normalizer_fits_only_train_frames(self):
    normalizer = fit_normalizer(np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32))
    transformed = normalizer.transform(np.array([[1.0, 3.0]], dtype=np.float32))
    self.assertAlmostEqual(float(transformed.mean()), -1.0)
    with self.assertRaises(ValueError):
        normalizer.transform(np.array([[math.nan, 0.0]], dtype=np.float32))


if __name__ == "__main__":
  unittest.main()
