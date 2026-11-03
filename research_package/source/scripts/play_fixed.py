"""Play wrapper — works around MuJoCo non-ASCII path issues."""
import sys
import os

# Patch mujoco.MjSpec.from_file to handle Chinese paths
# (MuJoCo's C library can't open non-ASCII paths)
import mujoco

_orig_from_file = mujoco.MjSpec.from_file

def _safe_from_file(path):
    spath = str(path) if not isinstance(path, str) else path
    if not os.path.exists(spath):
        raise FileNotFoundError(f"XML file not found: {spath}")
    with open(spath, encoding="utf-8") as f:
        return mujoco.MjSpec.from_string(f.read())

mujoco.MjSpec.from_file = _safe_from_file

# ── Now run play.py ──────────────────────────────────────────────────
play_path = os.path.join(os.path.dirname(__file__), "play.py")
sys.argv[0] = play_path
with open(play_path) as f:
    code = compile(f.read(), play_path, "exec")
    exec(code, {"__name__": "__main__"})
