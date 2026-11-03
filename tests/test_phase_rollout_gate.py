import unittest

import numpy as np

from scripts.evaluate_g1_onpolicy_phase_gate import reference_phase


class RolloutReferenceTests(unittest.TestCase):
    def test_two_frame_debounce_removes_single_contact_gap(self):
        contact = np.zeros(80, dtype=bool)
        contact[0:11] = True
        contact[5] = False
        contact[30:41] = True
        contact[60:71] = True
        phase, touchdown, cycle = reference_phase(contact)
        self.assertEqual(np.flatnonzero(touchdown).tolist(), [1, 31, 61])
        self.assertAlmostEqual(float(phase[16]), np.pi, places=6)
        self.assertEqual(int(cycle[16]), 0)

    def test_cycles_never_cross_episode_reset(self):
        contact = np.array(
            [
                True, True, False, False, True, True, True,
                True, True, False, False, True, True, True,
            ],
            dtype=bool,
        )
        episode_ids = np.array([0] * 7 + [1] * 7)
        phase, touchdown, cycle = reference_phase(contact, episode_ids=episode_ids)
        first_episode = np.flatnonzero(np.isfinite(phase[:7]))
        second_episode = np.flatnonzero(np.isfinite(phase[7:])) + 7
        self.assertEqual(first_episode.tolist(), [1, 2, 3, 4])
        self.assertEqual(second_episode.tolist(), [8, 9, 10, 11])
        self.assertNotEqual(int(cycle[1]), int(cycle[8]))
        self.assertTrue(touchdown[1] and touchdown[5] and touchdown[8] and touchdown[12])


if __name__ == "__main__":
    unittest.main()
