import tempfile
import unittest
from pathlib import Path

from mikirin.vector_bundle import VectorBundlePublisher
from mikirin.vector_index import VectorIndexWriter


class VectorBundleTests(unittest.TestCase):
    def _build(self, publisher: VectorBundlePublisher, value: float = 1.0):
        staging = publisher.create_staging()
        index_dir = staging / "episode_cards"
        with VectorIndexWriter(
            index_dir,
            dimensions=8,
            bits=16,
            embedding_fingerprint="embed-v1",
            schema_versions={"episode_card": "card-v1"},
        ) as writer:
            writer.add(
                "episode_card",
                "ep1",
                [value] * 8,
                subject_entity_id="s1",
                episode_entity_id="ep1",
            )
        return publisher.write_manifest(
            staging,
            embedding_fingerprint="embed-v1",
            index_directories={"episode_cards": index_dir},
            source_database={"build_id": "build-1"},
        )

    def test_side_build_publish_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = VectorBundlePublisher(Path(tmp) / "vectors")
            staged = self._build(publisher)
            self.assertFalse(publisher.pointer.exists())
            published = publisher.publish(staged.directory)
            self.assertTrue(publisher.pointer.exists())
            self.assertFalse(staged.directory.exists())
            current = publisher.current(expected_embedding_fingerprint="embed-v1")
            self.assertIsNotNone(current)
            self.assertEqual(current.bundle_fingerprint, published.bundle_fingerprint)

    def test_runtime_fingerprint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = VectorBundlePublisher(Path(tmp) / "vectors")
            staged = self._build(publisher)
            publisher.publish(staged.directory)
            with self.assertRaises(RuntimeError):
                publisher.current(expected_embedding_fingerprint="embed-v2")

    def test_second_build_does_not_modify_current_until_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = VectorBundlePublisher(Path(tmp) / "vectors")
            first = self._build(publisher)
            published = publisher.publish(first.directory)
            second = self._build(publisher, value=0.5)
            current = publisher.current()
            self.assertEqual(current.bundle_fingerprint, published.bundle_fingerprint)
            self.assertTrue(second.directory.exists())


if __name__ == "__main__":
    unittest.main()
