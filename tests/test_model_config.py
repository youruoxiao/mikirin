import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mikirin.model_config import EmbeddingProfile, ModelConfigStore, ModelProfile


class ModelConfigTests(unittest.TestCase):
    def test_roles_are_independent_and_secrets_are_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            store = ModelConfigStore(path)
            store.set_provider("chat", "openai", model="example-chat")
            profiles = store.load()
            self.assertEqual(profiles["organizer"].provider, "deepseek")
            self.assertEqual(profiles["chat"].model, "example-chat")
            self.assertNotIn("api_key", path.read_text(encoding="utf-8"))

    def test_key_is_read_only_from_named_environment_variable(self) -> None:
        profile = ModelConfigStore(Path("unused")).load()["organizer"]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"}, clear=False):
            self.assertEqual(profile.key(), "secret")
        self.assertNotIn("secret", repr(profile))

    def test_embedding_is_an_independent_formal_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            store = ModelConfigStore(path)
            embedding = EmbeddingProfile(
                provider="openai-compatible",
                base_url="https://embeddings.example/v1",
                model="embed-v1",
                api_style="openai-embeddings",
                key_env="EMBEDDING_API_KEY",
                dimensions=1024,
                query_prefix="query: ",
                document_prefix="document: ",
                normalize=False,
            )
            saved = store.set_custom("embedding", embedding)
            profiles = store.load()
            self.assertEqual(saved, embedding)
            self.assertIsInstance(profiles["embedding"], EmbeddingProfile)
            self.assertEqual(profiles["embedding"].dimensions, 1024)
            self.assertEqual(profiles["chat"].provider, "deepseek")
            self.assertEqual(profiles["organizer"].provider, "deepseek")

    def test_existing_openai_embeddings_manifest_loads_and_gets_stable_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(
                '{"schema_version": 1, "roles": {"embedding": {'
                '"api_style": "openai-embeddings", "provider": "openai", '
                '"base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small", '
                '"key_env": "OPENAI_API_KEY", "dimensions": 1536, "distance": "cosine", '
                '"query_prefix": "", "document_prefix": "", "normalize": true}}}',
                encoding="utf-8",
            )
            profile = ModelConfigStore(path).load()["embedding"]
            self.assertEqual(profile.input_normalization_version, "unicode-nfc-lf-v1")

    def test_save_without_embedding_does_not_drop_existing_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            store = ModelConfigStore(path)
            store.set_custom(
                "embedding",
                EmbeddingProfile(
                    provider="test",
                    base_url="https://example.test/v1",
                    model="embed",
                    api_style="openai-embeddings",
                    key_env="EMBED_KEY",
                    dimensions=3,
                ),
            )
            store.save({"chat": store.load()["chat"], "organizer": store.load()["organizer"]})
            self.assertEqual(store.load()["embedding"].model, "embed")

    def test_embedding_key_reads_only_environment(self) -> None:
        profile = EmbeddingProfile(
            provider="test",
            base_url="https://example.test",
            model="embed",
            api_style="openai-embeddings",
            key_env="EMBED_KEY",
            dimensions=1,
        )
        with patch.dict(os.environ, {"EMBED_KEY": "secret"}, clear=False):
            self.assertEqual(profile.key(), "secret")
        self.assertNotIn("secret", repr(profile))

    def test_embedding_validation_rejects_invalid_values(self) -> None:
        base = dict(
            provider="test",
            base_url="https://example.test",
            model="embed",
            api_style="openai-embeddings",
            key_env="EMBED_KEY",
            dimensions=1,
        )
        invalid = [
            {"base_url": "http://example.test"},
            {"dimensions": 0},
            {"distance": "euclidean"},
            {"query_prefix": 1},
            {"document_prefix": None},
            {"normalize": "true"},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    ModelConfigStore(Path("unused")).set_custom(
                        "embedding", EmbeddingProfile(**{**base, **changes})
                    )

    def test_chat_and_organizer_model_profile_api_remains_compatible(self) -> None:
        profile = ModelProfile(
            provider="custom",
            base_url="https://example.test/v1",
            model="chat",
            api_style="responses",
            key_env="CHAT_KEY",
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                ModelConfigStore(Path(directory) / "models.json").set_custom("chat", profile),
                profile,
            )


if __name__ == "__main__":
    unittest.main()
