"""Small process-level compatibility helpers for simulator runtimes."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import ModuleType


def choose_ascii_temp_path(current: str, candidates: list[str]) -> str:
  """Keep a usable ASCII temp path, or select the first ASCII fallback."""
  if current and current.isascii():
    return current
  for candidate in candidates:
    if candidate and candidate.isascii():
      return candidate
  raise ValueError("no ASCII temporary directory path is available")


def configure_windows_ascii_temp_dir() -> str | None:
  """Point Windows temp variables at an ASCII path when Unicode would break NVRTC."""
  if os.name != "nt":
    return None

  cached_tempfile = sys.modules.get("tempfile")
  cached_temp = getattr(cached_tempfile, "tempdir", None)
  current = os.environ.get("TEMP", "") or os.environ.get("TMP", "")
  cache_is_unicode = isinstance(cached_temp, str) and not cached_temp.isascii()
  if current and current.isascii() and not cache_is_unicode:
    return current

  system_root = os.environ.get("SystemRoot", r"C:\Windows")
  system_drive = os.environ.get("SystemDrive", "C:")
  candidates = [
    str(Path(system_root) / "Temp" / "g1_warp"),
    str(Path(system_drive) / "g1_warp_temp"),
  ]
  selected = choose_ascii_temp_path("" if cache_is_unicode else current, candidates)

  last_error: OSError | None = None
  for candidate in [selected, *candidates]:
    if not candidate.isascii():
      continue
    directory = Path(candidate)
    try:
      directory.mkdir(parents=True, exist_ok=True)
      if not os.access(directory, os.W_OK):
        continue
    except OSError as error:
      last_error = error
      continue
    os.environ["TEMP"] = str(directory)
    os.environ["TMP"] = str(directory)
    if cached_tempfile is not None:
      cached_tempfile.tempdir = str(directory)
    return str(directory)

  message = "could not create a writable ASCII temporary directory for Warp/NVRTC"
  if last_error is not None:
    raise RuntimeError(message) from last_error
  raise RuntimeError(message)


def configure_warp_ascii_paths(warp_module: ModuleType) -> dict[str, str]:
  """Keep Warp's NVRTC include and cache paths ASCII on Windows."""
  if os.name != "nt":
    return {}

  temp_dir = configure_windows_ascii_temp_dir()
  if temp_dir is None:
    raise RuntimeError("failed to configure an ASCII temporary directory")

  warp_root = Path(warp_module.__file__).resolve().parent
  warp_build = __import__("warp._src.build", fromlist=["warp_home"])
  paths = {"temp_dir": temp_dir}

  if not str(warp_root).isascii():
    source_native = warp_root / "native"
    ascii_warp_root = Path(temp_dir) / "warp"
    ascii_native = ascii_warp_root / "native"
    if not source_native.is_dir() or not (source_native / "builtin.h").is_file():
      raise FileNotFoundError(f"Warp native include files are missing: {source_native}")
    shutil.copytree(source_native, ascii_native, dirs_exist_ok=True)
    if not (ascii_native / "builtin.h").is_file():
      raise FileNotFoundError(f"ASCII Warp native include copy is incomplete: {ascii_native}")
    warp_build.warp_home = str(ascii_warp_root)
    paths["native_include_dir"] = str(ascii_native)
  else:
    paths["native_include_dir"] = str(warp_root / "native")

  cache_dir = Path(temp_dir) / "warp_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  warp_module.config.kernel_cache_dir = str(cache_dir)
  paths["kernel_cache_dir"] = str(cache_dir)
  return paths
