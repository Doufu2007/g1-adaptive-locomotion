import tempfile
import unittest
import importlib.util
from pathlib import Path

import mujoco

from src.tasks.velocity.phasor.mujoco_io import (
    install_scene_xml_unicode_compat,
    load_mjcf_spec,
)


class FakeSpec:
    calls = []

    @classmethod
    def from_file(cls, path):
        cls.calls.append(("file", str(path)))
        return "file-loaded"

    @classmethod
    def from_string(cls, xml):
        cls.calls.append(("string", xml))
        return "string-loaded"


class MujocoIoTests(unittest.TestCase):
    def setUp(self):
        FakeSpec.calls.clear()

    def test_ascii_path_keeps_native_file_loader(self):
        with tempfile.TemporaryDirectory(
            prefix="mjcf_ascii_", dir=r"C:\Windows\Temp"
        ) as directory:
            path = Path(directory) / "scene.xml"
            path.write_text("<mujoco/>", encoding="utf-8")
            self.assertEqual(load_mjcf_spec(FakeSpec, path), "file-loaded")
            self.assertEqual(FakeSpec.calls[0][0], "file")

    def test_unicode_path_uses_utf8_xml_contents(self):
        with tempfile.TemporaryDirectory(prefix="mjcf_") as directory:
            path = Path(directory) / "场景" / "scene.xml"
            path.parent.mkdir()
            path.write_text("<mujoco model='测试'/>", encoding="utf-8")
            self.assertEqual(load_mjcf_spec(FakeSpec, path), "string-loaded")
            self.assertEqual(FakeSpec.calls, [("string", "<mujoco model='测试'/>")])

    def test_only_the_configured_scene_file_is_intercepted(self):
        with tempfile.TemporaryDirectory(prefix="mjcf_") as directory:
            root = Path(directory) / "场景"
            root.mkdir()
            scene = root / "scene.xml"
            other = root / "robot.xml"
            scene.write_text("<mujoco/>", encoding="utf-8")
            other.write_text("<mujoco/>", encoding="utf-8")
            original = install_scene_xml_unicode_compat(FakeSpec, scene)
            self.assertEqual(FakeSpec.from_file(str(scene)), "string-loaded")
            self.assertEqual(FakeSpec.from_file(str(other)), "file-loaded")
            self.assertIsNotNone(original)

    def test_actual_g1_xml_and_mesh_assets_compile(self):
        from src.assets.robots.unitree_g1.g1_constants import get_spec

        model = get_spec().compile()
        self.assertGreater(model.nbody, 1)
        self.assertGreater(model.nmesh, 0)

    def test_installed_mjlab_scene_compiles_through_unicode_wrapper(self):
        from src.tasks.velocity.phasor.mujoco_io import (
            install_mjlab_scene_xml_unicode_compat,
        )

        install_mjlab_scene_xml_unicode_compat(mujoco)
        package = importlib.util.find_spec("mjlab")
        scene = Path(package.origin).parent / "scene" / "scene.xml"
        spec = mujoco.MjSpec.from_file(str(scene))
        self.assertGreater(spec.compile().nbody, 0)


if __name__ == "__main__":
    unittest.main()
