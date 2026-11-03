import unittest

from src.runtime_compat import choose_ascii_temp_path


class RuntimeCompatTests(unittest.TestCase):
  def test_non_ascii_temp_uses_first_ascii_candidate(self) -> None:
    chosen = choose_ascii_temp_path(
      r"C:\Users\研究者\AppData\Local\Temp",
      [r"C:\Windows\Temp\g1_warp", r"E:\g1_warp_temp"],
    )
    self.assertEqual(chosen, r"C:\Windows\Temp\g1_warp")

  def test_ascii_temp_is_preserved(self) -> None:
    chosen = choose_ascii_temp_path(
      r"C:\Temp",
      [r"C:\Windows\Temp\g1_warp"],
    )
    self.assertEqual(chosen, r"C:\Temp")

  def test_fails_when_no_ascii_temp_path_exists(self) -> None:
    with self.assertRaises(ValueError):
      choose_ascii_temp_path("C:\\Users\\研究者\\Temp", ["E:\\临时"])


if __name__ == "__main__":
  unittest.main()
