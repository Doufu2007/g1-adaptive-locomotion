"""Narrow Windows Unicode-path compatibility for MJCF loading."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def load_mjcf_spec(spec_api: Any, path: str | Path) -> Any:
    """Use MuJoCo's native loader for ASCII paths and UTF-8 text otherwise."""
    source_path = Path(os.fsdecode(path))
    if source_path.as_posix().isascii():
        return spec_api.from_file(str(source_path))
    return spec_api.from_string(source_path.read_text(encoding="utf-8"))


def install_scene_xml_unicode_compat(spec_api: Any, scene_path: str | Path) -> Any:
    """Intercept only mjlab's known scene XML path when it contains Unicode."""
    target = Path(scene_path).resolve()
    original = spec_api.from_file
    if getattr(original, "_g1_scene_xml_unicode_compat", False):
        return original

    def from_file(path: str | bytes | Path, *args: Any, **kwargs: Any) -> Any:
        source_path = Path(os.fsdecode(path))
        if source_path.resolve() == target and not source_path.as_posix().isascii():
            if args or kwargs:
                raise TypeError("mjlab scene XML loading does not accept extra parser options")
            return load_mjcf_spec(spec_api, source_path)
        return original(path, *args, **kwargs)

    from_file._g1_scene_xml_unicode_compat = True  # type: ignore[attr-defined]
    spec_api.from_file = staticmethod(from_file)
    return original


def install_mjlab_scene_xml_unicode_compat(
    mujoco_module: ModuleType | None = None,
) -> Any:
    """Install a loader wrapper scoped to the installed mjlab scene.xml."""
    if mujoco_module is None:
        import mujoco as mujoco_module

    mjlab_spec = importlib.util.find_spec("mjlab")
    if mjlab_spec is None or mjlab_spec.origin is None:
        raise ImportError("cannot locate the installed mjlab package")
    scene_path = Path(mjlab_spec.origin).parent / "scene" / "scene.xml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"installed mjlab scene XML is missing: {scene_path}")
    return install_scene_xml_unicode_compat(mujoco_module.MjSpec, scene_path)
