import tempfile
import unittest
from datetime import date
from pathlib import Path

from mikirin.assets import RawAssetStore
from mikirin.layout import ProjectLayout


class RawAssetStoreTests(unittest.TestCase):
    def test_ingest_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "source.jsonl"
            source_file.write_text('{"id": 1}\n', encoding="utf-8")
            store = RawAssetStore(ProjectLayout(root / "project"))
            first = store.ingest(
                source_file,
                source="bgm_archive",
                source_version="2026-07-21",
                acquired_on=date(2026, 7, 26),
            )
            second = store.ingest(
                source_file,
                source="bgm_archive",
                source_version="2026-07-21",
                acquired_on=date(2026, 7, 26),
            )
            self.assertEqual(first, second)
            archived = root / "project" / first.relative_path
            self.assertTrue(archived.exists())
            self.assertIn("data/raw/bgm_archive/2026-07-26/", first.relative_path)


if __name__ == "__main__":
    unittest.main()

