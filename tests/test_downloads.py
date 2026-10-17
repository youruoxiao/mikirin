import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mikirin.downloads import AssemblyPlan


class DownloadAssemblyTests(unittest.TestCase):
    def test_contiguous_fragments_are_verified_and_assembled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"abc")
            (root / "b.bin").write_bytes(b"defg")
            expected = hashlib.sha256(b"abcdefg").hexdigest()
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "expected_size": 7,
                        "expected_sha256": expected,
                        "fragments": [
                            {"start": 0, "end": 2, "path": "a.bin"},
                            {"start": 3, "end": 6, "path": "b.bin"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output.bin"
            AssemblyPlan.from_json(plan_path).assemble(output)
            self.assertEqual(output.read_bytes(), b"abcdefg")

    def test_gap_is_rejected_before_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"abc")
            (root / "b.bin").write_bytes(b"ef")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "expected_size": 6,
                        "expected_sha256": "0" * 64,
                        "fragments": [
                            {"start": 0, "end": 2, "path": "a.bin"},
                            {"start": 4, "end": 5, "path": "b.bin"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "缺口"):
                AssemblyPlan.from_json(plan_path).validate_fragments()


if __name__ == "__main__":
    unittest.main()

