import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from mikirin.layout import ProjectLayout, discover_root
from mikirin.prompts import PromptBuilder


class RootDiscoveryTests(unittest.TestCase):
    def test_no_checkout_defaults_to_cwd_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("mikirin.layout.Path.cwd", return_value=root):
                self.assertEqual(discover_root(), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_explicit_start_without_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(discover_root(root / "."), root)

    def test_unrelated_or_invalid_ancestor_metadata_is_ignored(self) -> None:
        cases = (
            '[project]\nname = "unrelated"\n',
            '[tool.example]\nname = "mikirin"\n',
            'project = "mikirin"\n',
            '[project\n',
        )
        for metadata in cases:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "src" / "mikirin"
                source.mkdir(parents=True)
                (source / "__init__.py").touch()
                (root / "pyproject.toml").write_text(metadata, encoding="utf-8")
                child = root / "work"
                child.mkdir()
                self.assertEqual(discover_root(child), child)

    def test_name_without_source_checkout_is_not_a_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "pyproject.toml").write_text(
                '[project]\nname = "mikirin"\n', encoding="utf-8"
            )
            child = root / "work"
            child.mkdir()
            self.assertEqual(discover_root(child), child)

    def test_checkout_is_found_through_unrelated_nested_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "src" / "mikirin"
            source.mkdir(parents=True)
            (source / "__init__.py").touch()
            (root / "pyproject.toml").write_text(
                '[project]\nname = "mikirin"\n', encoding="utf-8"
            )
            child = root / "examples" / "other"
            child.mkdir(parents=True)
            (child / "pyproject.toml").write_text(
                '[project]\nname = "other"\n', encoding="utf-8"
            )
            self.assertEqual(discover_root(child), root)
            self.assertEqual(discover_root(root), root)


class PackagedPromptTests(unittest.TestCase):
    @staticmethod
    def default(name: str) -> str:
        return files("mikirin").joinpath("resources", "agent", name).read_text(
            encoding="utf-8"
        ).strip()

    def test_defaults_work_without_initialization_or_agent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            stable = PromptBuilder(layout).build().stable
            self.assertEqual(
                stable, self.default("SOUL.md") + "\n\n" + self.default("TOOLS.md")
            )
            self.assertIn("Mikirin", stable)
            self.assertIn("search_anime", stable)
            self.assertFalse(layout.agent_assets.exists())

    def test_each_override_takes_priority_and_user_data_is_preserved(self) -> None:
        for name in ("SOUL.md", "TOOLS.md"):
            for content in ("自定义内容", ""):
                with (
                    self.subTest(name=name, content=content),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    layout = ProjectLayout(Path(directory))
                    layout.initialize()
                    override = layout.agent_assets / name
                    override.write_text(content, encoding="utf-8")
                    user = layout.user_data / "USER.md"
                    memory = layout.user_data / "MEMORY.md"
                    user.write_bytes("用户画像\r\n".encode())
                    memory.write_bytes("用户记忆\r\n".encode())
                    original = {path: path.read_bytes() for path in (override, user, memory)}
                    expected = {
                        asset: content if asset == name else self.default(asset)
                        for asset in ("SOUL.md", "TOOLS.md")
                    }
                    builder = PromptBuilder(layout)
                    for _ in range(2):
                        layout.initialize()
                        self.assertEqual(
                            builder.build().stable,
                            expected["SOUL.md"] + "\n\n" + expected["TOOLS.md"],
                        )
                    self.assertEqual(original, {path: path.read_bytes() for path in original})
                    other = "TOOLS.md" if name == "SOUL.md" else "SOUL.md"
                    self.assertFalse((layout.agent_assets / other).exists())

    def test_unreadable_override_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = PromptBuilder(ProjectLayout(Path(directory)))
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    builder._read_agent_asset(builder.soul_path)


if __name__ == "__main__":
    unittest.main()
