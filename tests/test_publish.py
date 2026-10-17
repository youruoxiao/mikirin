import tempfile
import unittest
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher


class PublisherTests(unittest.TestCase):
    def test_publish_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            publisher = Publisher(layout)
            publisher.create_empty_build(build_id="build-1", code_version="test")
            first = publisher.publish(version="1")
            self.assertEqual(publisher.current(), first)

            publisher.create_empty_build(build_id="build-2", code_version="test")
            second = publisher.publish(version="2")
            self.assertEqual(publisher.current(), second)
            self.assertEqual(publisher.rollback(version="1"), first)
            self.assertEqual(publisher.current(), first)


if __name__ == "__main__":
    unittest.main()

