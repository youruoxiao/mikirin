import tempfile
import unittest
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.quality import QualityInspector


class QualityInspectorTests(unittest.TestCase):
    def test_empty_build_is_not_publishable_as_real_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Publisher(ProjectLayout(Path(directory))).create_empty_build(
                build_id="build-quality", code_version="test"
            )
            report = QualityInspector.inspect(database)
            self.assertIn("核心表为空: subjects", report.fatal_issues)
            with self.assertRaises(RuntimeError):
                QualityInspector.assert_publishable(report)


if __name__ == "__main__":
    unittest.main()
