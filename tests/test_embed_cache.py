import tempfile
import unittest
from pathlib import Path

from mikirin.embed_cache import EmbeddingCache


class EmbeddingCacheTests(unittest.TestCase):
    def test_batch_roundtrip_and_dimension_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EmbeddingCache(Path(tmp)) as cache:
                cache.put_many(
                    "fp",
                    [("first", [1.0, 0.0, 0.0]), ("second", [0.0, 1.0, 0.0])],
                )
                first = cache.get("fp", "first", dimensions=3)
                self.assertIsNotNone(first)
                self.assertGreater(first[0], 0.99)
                self.assertIsNone(cache.get("fp", "first", dimensions=4))
                self.assertEqual(len(cache.get_int8("fp", "first", dimensions=3)), 3)

    def test_legacy_file_is_readable_and_promotable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with EmbeddingCache(directory) as cache:
                key = cache._key("legacy", "text")
                path = directory / key[:2] / f"{key}.int8"
                path.parent.mkdir(parents=True)
                path.write_bytes(bytes([255, 128, 128]))
                self.assertTrue(
                    cache.promote(
                        source_fingerprint="legacy",
                        source_text="text",
                        target_fingerprint="current",
                        target_text="text",
                        dimensions=3,
                    )
                )
                self.assertIsNotNone(cache.get("current", "text", dimensions=3))

    def test_invalid_vectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EmbeddingCache(Path(tmp)) as cache:
                with self.assertRaises(ValueError):
                    cache.put("fp", "empty", [])
                with self.assertRaises(ValueError):
                    cache.put("fp", "nan", [float("nan")])


if __name__ == "__main__":
    unittest.main()
