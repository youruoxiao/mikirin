import tempfile
import unittest
import urllib.error
import sqlite3
from pathlib import Path
from unittest.mock import patch

from mikirin.vector_index import (
    EmbeddingClient,
    EmbeddingConfig,
    PermanentEmbeddingError,
    VectorIndex,
    VectorIndexWriter,
    canonical_base_url,
    lsh_probe_keys,
    quantize_int8,
    simhash_sign,
    build_projection,
)


def make_vector(seed: int, dims: int = 64) -> list[float]:
    import random

    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dims)]


def perturb(vector: list[float], amount: float = 0.05) -> list[float]:
    return [x + amount * (1 if i % 2 else -1) for i, x in enumerate(vector)]


class VectorIndexTests(unittest.TestCase):
    def test_permanent_embedding_http_errors_are_not_retried(self):
        config = EmbeddingConfig(
            provider="test",
            base_url="https://example.invalid/v1",
            model="embedding-test",
            dimensions=2,
        )
        error = urllib.error.HTTPError(
            config.base_url + "/embeddings", 402, "Payment Required", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaisesRegex(PermanentEmbeddingError, "HTTP 402"):
                EmbeddingClient(config, "secret", max_retries=9).embed_documents(["x"])
        mocked.assert_called_once()

    def test_lsh_multiprobe_index_returns_exact_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "lsh"
            with VectorIndexWriter(
                directory,
                dimensions=64,
                bits=128,
                seed=4,
                embedding_fingerprint="embed-v1",
                schema_versions={"community_chunk": "chunk-v1"},
                lsh_bands=8,
                lsh_band_bits=8,
                lsh_probe_radius=2,
            ) as writer:
                for seed in range(200):
                    writer.add("community_chunk", f"c{seed}", make_vector(seed))
            index = VectorIndex(directory)
            try:
                results = index.search(make_vector(42), coarse=50, top=5)
                self.assertEqual(results[0][1], "c42")
                self.assertEqual(index.lsh_bands, 8)
            finally:
                index.close()
    def test_lsh_probe_radius_two_is_deterministic(self):
        probes = list(lsh_probe_keys(0, 8, 2))
        self.assertEqual(len(probes), 1 + 8 + 28)
        self.assertEqual(len(set(probes)), len(probes))

    def test_embedding_fingerprint_is_canonical_and_complete(self):
        first = EmbeddingConfig(
            provider="example",
            base_url="https://EMBED.example/v1/",
            model="m1",
            dimensions=64,
            query_prefix="q:",
        )
        same = EmbeddingConfig(
            provider="example",
            base_url="https://embed.example/v1",
            model="m1",
            dimensions=64,
            query_prefix="q:",
        )
        changed = EmbeddingConfig(
            provider="example",
            base_url="https://embed.example/v1",
            model="m1",
            dimensions=64,
            query_prefix="query:",
        )
        self.assertEqual(canonical_base_url(first.base_url), "https://embed.example/v1")
        self.assertEqual(first.fingerprint(), same.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())

    def test_writer_requires_side_build_and_tracks_scope_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "index"
            with VectorIndexWriter(
                directory,
                dimensions=8,
                bits=16,
                embedding_fingerprint="embed-v1",
                schema_versions={"episode_card": "card-v1"},
            ) as writer:
                writer.add(
                    "episode_card",
                    "ep1",
                    [1.0] * 8,
                    subject_entity_id="s1",
                    episode_entity_id="ep1",
                )
                writer.add(
                    "community_chunk",
                    "chunk1",
                    [-1.0] * 8,
                    subject_entity_id="s2",
                    episode_entity_id="ep2",
                )
            with self.assertRaises(FileExistsError):
                VectorIndexWriter(directory, dimensions=8, bits=16)
            index = VectorIndex(directory)
            try:
                results = index.search(
                    [1.0] * 8,
                    object_kinds={"episode_card"},
                    subject_entity_id="s1",
                )
                self.assertEqual([item[1] for item in results], ["ep1"])
                self.assertTrue(index.index_fingerprint)
            finally:
                index.close()
            db = sqlite3.connect(directory / "meta.db")
            try:
                indexes = {row[1] for row in db.execute("PRAGMA index_list(vector_meta)")}
            finally:
                db.close()
            self.assertIn("vector_meta_subject_row", indexes)
            self.assertIn("vector_meta_episode_row", indexes)
            self.assertIn("vector_meta_kind_row", indexes)

    def test_similar_vectors_rank_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            writer = VectorIndexWriter(directory, dimensions=64, bits=128, seed=1)
            base = make_vector(42, 64)
            near = perturb(base)
            far = make_vector(7, 64)
            writer.add("episode", "ep_base", base)
            writer.add("episode", "ep_far", far)
            writer.add("episode", "ep_near", near)
            writer.close()

            index = VectorIndex(directory)
            results = index.search(base, coarse=3, top=3)
            index.close()
            self.assertEqual(results[0][1], "ep_base")
            self.assertEqual(results[1][1], "ep_near")
            self.assertEqual(results[2][1], "ep_far")

    def test_quantize_is_symmetric(self):
        vector = make_vector(1, 32)
        quantized = quantize_int8(vector)
        self.assertEqual(len(quantized), 32)
        self.assertTrue(all(0 <= b <= 255 for b in quantized))

    def test_projection_is_deterministic(self):
        self.assertEqual(build_projection(64, 32, 9), build_projection(64, 32, 9))
        self.assertNotEqual(build_projection(64, 32, 9), build_projection(64, 32, 10))

    def test_signature_similarity(self):
        projection = build_projection(64, 128, 1)
        base = quantize_int8(make_vector(5, 64))
        near = quantize_int8(perturb(make_vector(5, 64)))
        far = quantize_int8(make_vector(99, 64))
        sig_base = simhash_sign(base, projection)
        near_dist = (sig_base ^ simhash_sign(near, projection)).bit_count()
        far_dist = (sig_base ^ simhash_sign(far, projection)).bit_count()
        self.assertLess(near_dist, far_dist)


if __name__ == "__main__":
    unittest.main()
