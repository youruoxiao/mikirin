import hashlib
import tempfile
import unittest
from pathlib import Path

from mikirin.sources.moegirl import git_blob_sha1_bytes, git_blob_sha1_file


class MoegirlDatasetTests(unittest.TestCase):
    def test_git_blob_hash_matches_git_object_contract(self) -> None:
        content = b"hello\n"
        expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
        self.assertEqual(git_blob_sha1_bytes(content), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_bytes(content)
            self.assertEqual(git_blob_sha1_file(path), expected)


if __name__ == "__main__":
    unittest.main()

