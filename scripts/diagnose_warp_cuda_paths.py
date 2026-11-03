"""Probe whether an ASCII copy of Warp's native headers fixes NVRTC."""

from pathlib import Path
import shutil

from src.runtime_compat import configure_windows_ascii_temp_dir

temp_dir = configure_windows_ascii_temp_dir()

import warp as wp
import warp._src.build as warp_build

warp_root = Path(wp.__file__).resolve().parent
ascii_native = Path(temp_dir) / "warp" / "native"
shutil.copytree(warp_root / "native", ascii_native, dirs_exist_ok=True)
print(f"warp_root_ascii={str(warp_root).isascii()}")
print(f"native_copy={ascii_native}")
print(f"native_copy_header_exists={(ascii_native / 'builtin.h').is_file()}")
print(f"build_module_identity={warp_build is wp._src.build}")
print(f"build_home_before={warp_build.warp_home}")
wp._src.build.warp_home = str(ascii_native.parent)
ascii_cache = Path(temp_dir) / "warp_cache"
ascii_cache.mkdir(parents=True, exist_ok=True)
wp.config.kernel_cache_dir = str(ascii_cache)
print(f"build_home_after={warp_build.warp_home}")
print(f"kernel_cache_dir={wp.config.kernel_cache_dir}")


@wp.kernel
def ascii_include_probe(output: wp.array(dtype=wp.int32)):
  output[wp.tid()] = 7


output = wp.zeros(1, dtype=wp.int32, device="cuda:0")
wp.launch(ascii_include_probe, dim=1, inputs=[output], device="cuda:0")
wp.synchronize()
print(f"ascii_include_probe_result={output.numpy().tolist()}")
