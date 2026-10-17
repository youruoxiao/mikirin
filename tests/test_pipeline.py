import tempfile
import unittest
from pathlib import Path

from mikirin.pipeline import PipelineStore, TaskState


class PipelineStoreTests(unittest.TestCase):
    def test_discovery_is_idempotent_and_transition_is_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PipelineStore(Path(directory) / "pipeline.sqlite")
            store.initialize()
            created = store.discover(
                task_key="bgm:2026-07-21",
                source="bgm_archive",
                input_version="2026-07-21",
                processor_version="0.1.0",
            )
            repeated = store.discover(
                task_key="bgm:2026-07-21",
                source="bgm_archive",
                input_version="2026-07-21",
                processor_version="0.1.0",
            )
            self.assertTrue(created)
            self.assertFalse(repeated)
            store.transition("bgm:2026-07-21", TaskState.FETCHED)
            self.assertEqual(store.list_tasks()[0].state, TaskState.FETCHED)

    def test_cannot_skip_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PipelineStore(Path(directory) / "pipeline.sqlite")
            store.initialize()
            store.discover(
                task_key="one",
                source="test",
                input_version="1",
                processor_version="1",
            )
            with self.assertRaises(ValueError):
                store.transition("one", TaskState.PARSED)


if __name__ == "__main__":
    unittest.main()

