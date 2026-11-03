"""Wrapper for play.py that patches mjlab version incompatibilities."""
import os
import sys

# ── Patch: mjlab 1.6.0 removed update_assets — provide it ────────────
def _update_assets(assets_dict, assets_dir, meshdir):
    """Read asset files from assets_dir into assets_dict."""
    from pathlib import Path
    assets_path = Path(assets_dir)
    if assets_path.exists():
        for f in assets_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(assets_path.parent))
                assets_dict[rel] = f.read_bytes()

import mjlab.utils.os
mjlab.utils.os.update_assets = _update_assets

# ── Now run the real play script ──────────────────────────────────────
sys.argv[0] = os.path.join(os.path.dirname(__file__), "play.py")
with open(sys.argv[0]) as f:
    code = compile(f.read(), sys.argv[0], "exec")
    exec(code, {"__name__": "__main__"})
