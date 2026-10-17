import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mikirin.episode_card import CARD_SCHEMA_VERSION
from mikirin.semantic_search import SemanticEpisodeSearch
from mikirin.vector_bundle import VectorBundlePublisher
from mikirin.vector_index import EmbeddingConfig, VectorIndexWriter


class SemanticSearchContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        manifests = self.root / "data" / "manifests"
        manifests.mkdir(parents=True)
        self.config = {
            "provider": "example",
            "base_url": "https://embed.example/v1",
            "model": "embed-v1",
            "key_env": "TEST_EMBED_KEY",
            "dimensions": 8,
            "distance": "cosine",
            "query_prefix": "q:",
            "document_prefix": "d:",
            "normalize": True,
            "input_normalization_version": "unicode-nfc-lf-v1",
        }
        self.config_path = manifests / "runtime_model_config.json"
        self._write_config(self.config)
        self.database = self.root / "content.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE build(id TEXT PRIMARY KEY)")
            db.execute("INSERT INTO build VALUES ('build-1')")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_config(self, config):
        self.config_path.write_text(
            json.dumps({"roles": {"embedding": config}}), encoding="utf-8"
        )

    def _publish_index(self, *, build_id="build-1"):
        config = EmbeddingConfig(
            provider=self.config["provider"],
            base_url=self.config["base_url"],
            model=self.config["model"],
            dimensions=self.config["dimensions"],
            distance=self.config["distance"],
            query_prefix=self.config["query_prefix"],
            document_prefix=self.config["document_prefix"],
            normalize=self.config["normalize"],
            input_normalization_version=self.config["input_normalization_version"],
        )
        publisher = VectorBundlePublisher(self.root / "data" / "vector_index")
        staging = publisher.create_staging()
        index_dir = staging / "episode_cards"
        with VectorIndexWriter(
            index_dir,
            dimensions=8,
            bits=16,
            embedding_fingerprint=config.fingerprint(),
            schema_versions={"episode_card": CARD_SCHEMA_VERSION},
        ) as writer:
            writer.add("episode_card", "ep1", [1.0] * 8)
        publisher.write_manifest(
            staging,
            embedding_fingerprint=config.fingerprint(),
            index_directories={"episode_cards": index_dir},
            source_database={"build_id": build_id},
        )
        publisher.publish(staging)

    def test_missing_bundle_explicitly_falls_back_to_fts(self):
        semantic = SemanticEpisodeSearch(self.root, self.database)
        self.assertFalse(semantic.available)
        self.assertEqual(semantic.status()["fallback"], "fts")
        self.assertEqual(semantic.fallback_reason, "no-published-vector-bundle")

    def test_matching_bundle_requires_api_key(self):
        self._publish_index()
        with patch.dict(os.environ, {}, clear=True):
            semantic = SemanticEpisodeSearch(self.root, self.database)
        self.assertFalse(semantic.available)
        self.assertEqual(semantic.fallback_reason, "embedding-api-key-missing")

    def test_matching_contract_and_database_enable_vector_search(self):
        self._publish_index()
        with patch.dict(os.environ, {"TEST_EMBED_KEY": "secret"}, clear=False):
            semantic = SemanticEpisodeSearch(self.root, self.database)
        try:
            self.assertTrue(semantic.available)
            self.assertIsNone(semantic.status()["fallback"])
        finally:
            semantic.close()

    def test_runtime_config_mismatch_disables_vector_search(self):
        self._publish_index()
        changed = dict(self.config)
        changed["query_prefix"] = "changed:"
        self._write_config(changed)
        with patch.dict(os.environ, {"TEST_EMBED_KEY": "secret"}, clear=False):
            semantic = SemanticEpisodeSearch(self.root, self.database)
        self.assertFalse(semantic.available)
        self.assertTrue(semantic.fallback_reason.startswith("index-contract-mismatch:"))

    def test_content_database_mismatch_disables_vector_search(self):
        self._publish_index(build_id="different-build")
        with patch.dict(os.environ, {"TEST_EMBED_KEY": "secret"}, clear=False):
            semantic = SemanticEpisodeSearch(self.root, self.database)
        self.assertFalse(semantic.available)
        self.assertEqual(semantic.fallback_reason, "content-database-mismatch")


if __name__ == "__main__":
    unittest.main()
