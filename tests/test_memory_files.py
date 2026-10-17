import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from mikirin.agent.tools import MemoryWriteTool
from mikirin.memory_files import BudgetedTextFile, MemoryDriftError, TextOperation


class MemoryFileTests(unittest.TestCase):
    def test_user_1375_boundary_and_merge_advice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "USER.md"
            editor = BudgetedTextFile(path, limit=1375, initial_text="字" * 1375)
            with self.assertRaisesRegex(ValueError, "merge old entries first"):
                editor.add("新")
            self.assertEqual(path.read_text(encoding="utf-8"), "字" * 1375)
            missing = Path(directory) / "oversized.md"
            with self.assertRaisesRegex(ValueError, "merge old entries first"):
                BudgetedTextFile(missing, limit=1375, initial_text="字" * 1376)
            self.assertFalse(missing.exists())

    def test_batch_budget_is_all_or_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MEMORY.md"
            editor = BudgetedTextFile(path, limit=20, initial_text="# M\n")
            before = path.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                editor.apply_batch([
                    TextOperation("add", text="short"),
                    TextOperation("add", text="x" * 30),
                ])
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_replace_remove_and_external_drift_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "USER.md"
            editor = BudgetedTextFile(path, limit=100, initial_text="alpha\nbeta\n")
            report = editor.apply_batch([
                TextOperation("replace", old="alpha", new="A"),
                TextOperation("remove", text="beta\n"),
            ])
            self.assertEqual(path.read_text(encoding="utf-8"), "A\n")
            self.assertEqual(report.operations, 2)
            path.write_text("external", encoding="utf-8")
            with self.assertRaises(MemoryDriftError):
                editor.add("new")
            backups = list(Path(directory).glob("USER.md.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "external")

    def test_memory_tool_writes_with_sequential_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = MemoryWriteTool(root / "USER.md", root / "MEMORY.md")
            updates = []

            async def update(text: str) -> None:
                updates.append(text)

            result = asyncio.run(tool.execute(
                "call-1",
                {"target": "memory", "operations": [{"op": "add", "text": "喜欢作画"}]},
                update,
            ))
            self.assertEqual(tool.execution_mode, "sequential")
            self.assertIn("喜欢作画", (root / "MEMORY.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(result.content)["operations"], 1)
            self.assertTrue(updates)


if __name__ == "__main__":
    unittest.main()
