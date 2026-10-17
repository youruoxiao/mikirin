import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mikirin.range_fetch import HttpRangeFragmentFetcher, RangeSpec


class RangeSpecTests(unittest.TestCase):
    def test_resume_boundary_is_exact_and_inclusive(self) -> None:
        spec = RangeSpec(100, 199)
        self.assertEqual(spec.length, 100)
        self.assertEqual(spec.remaining(0), (100, 199))
        self.assertEqual(spec.remaining(37), (137, 199))
        self.assertIsNone(spec.remaining(100))

    def test_oversized_existing_fragment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RangeSpec(100, 199).remaining(101)

    def test_response_ignoring_requested_end_is_truncated_at_fragment_boundary(self) -> None:
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 100-999/1000"}

            def __enter__(self):
                self._done = False
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                if self._done:
                    return b""
                self._done = True
                return b"x" * 200

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fragment.bin"
            with patch("urllib.request.urlopen", return_value=Response()):
                HttpRangeFragmentFetcher(max_attempts=1).fetch(
                    url="https://example.invalid/file",
                    spec=RangeSpec(100, 149),
                    destination=destination,
                )
            self.assertEqual(destination.stat().st_size, 50)


if __name__ == "__main__":
    unittest.main()
