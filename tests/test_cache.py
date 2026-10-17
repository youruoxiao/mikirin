import tempfile
import unittest
from pathlib import Path

from mikirin.cache import ModelCache, ModelInvocation


class ModelCacheTests(unittest.TestCase):
    def test_cache_is_replayable_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ModelCache(Path(directory))
            invocation = ModelInvocation(
                kind="llm",
                input_hash="sha256:input",
                prompt_hash="sha256:prompt",
                model_name="example",
                model_revision="r1",
                parameters={"temperature": 0},
                code_version="a0",
                parser_version="v1",
            )
            cache.put(invocation, raw_response="raw", validated_response={"ok": True})
            self.assertEqual(cache.get(invocation)["validated_response"], {"ok": True})
            with self.assertRaises(RuntimeError):
                cache.put(invocation, raw_response="changed", validated_response={"ok": True})


if __name__ == "__main__":
    unittest.main()

