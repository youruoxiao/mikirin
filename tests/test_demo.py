from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mikirin.cli import build_parser, command_tui
from mikirin.demo import create_demo
from mikirin.layout import ProjectLayout
from mikirin.model_config import ModelConfigStore
from mikirin.publish import Publisher
from mikirin.query import KnowledgeQueries


class DemoTests(unittest.TestCase):
    def test_demo_is_searchable_and_repeated_setup_preserves_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            database = create_demo(layout)
            first_bytes = database.read_bytes()
            memory = layout.user_data / "MEMORY.md"
            memory.write_text("keep me", encoding="utf-8")
            self.assertEqual(create_demo(layout), database)
            self.assertEqual(database.read_bytes(), first_bytes)
            self.assertEqual(memory.read_text(encoding="utf-8"), "keep me")
            hit = KnowledgeQueries.search_entities(database, "放映部")[0]
            self.assertEqual(hit.canonical_name, "屋顶放映部")
            context = KnowledgeQueries.search_anime_context(database, "屋顶放映部")
            self.assertIn("虚构演示", context["entities"][0]["summary"])
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM episodes").fetchone()[0], 9)
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_demo_refuses_existing_real_database_and_unfinished_build(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            publisher = Publisher(layout)
            building = publisher.create_empty_build(build_id="real", code_version="test")
            original = building.read_bytes()
            with self.assertRaises(FileExistsError):
                create_demo(layout)
            self.assertEqual(building.read_bytes(), original)
            database = publisher.publish(version="real")
            pointer = layout.current_pointer.read_bytes()
            with self.assertRaises(FileExistsError):
                create_demo(layout)
            self.assertEqual(publisher.current(), database)
            self.assertEqual(layout.current_pointer.read_bytes(), pointer)

    def test_tui_model_flags_are_applied_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            create_demo(layout)
            args = build_parser().parse_args([
                "--root", directory, "tui", "--browse-only", "--model", "test-model",
                "--base-url", "https://example.invalid/v1", "--api-style", "chat-completions",
            ])
            with patch("mikirin.cli.run_tui", return_value=0) as run:
                self.assertEqual(command_tui(args), 0)
                self.assertTrue(run.call_args.kwargs["browse_only"])
            profile = ModelConfigStore(layout.manifests / "runtime_model_config.json").load()["chat"]
            self.assertEqual(profile.model, "test-model")
            self.assertEqual(profile.key_env, "MIKIRIN_API_KEY")
