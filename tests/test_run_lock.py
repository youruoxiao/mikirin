import tempfile
import unittest
from pathlib import Path

from mikirin.run_lock import exclusive_process_lock


class ProcessLockTests(unittest.TestCase):
    def test_second_lock_is_rejected_and_lock_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "organizer.lock"
            with exclusive_process_lock(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_process_lock(path):
                        pass
            with exclusive_process_lock(path):
                pass


if __name__ == "__main__":
    unittest.main()
