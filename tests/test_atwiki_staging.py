import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.sources.atwiki_staging import AtwikiStager


class AtwikiStagingTests(unittest.TestCase):
    def test_raw_page_becomes_normalized_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            layout.initialize()
            html = b'<div id="wikibody"><p>\xe7\x9b\xa3\xe7\x9d\xa3 \xe5\xb1\xb1\xe7\x94\xb0</p><p>\xe5\x8e\x9f\xe7\x94\xbb<br>\xe4\xbd\x90\xe8\x97\xa4</p><div id="atwiki-page-tags"></div></div>'
            digest = hashlib.sha256(html).hexdigest()
            raw_path = layout.raw / "atwiki" / "2026-07-26" / f"{digest}.html"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(html)
            state_path = layout.manifests / "atwiki" / "crawl-state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "pages": {
                            "106": {
                                "status": "ok",
                                "record": {
                                    "title": "作品",
                                    "url": "https://w.atwiki.jp/sakuga/pages/106.html",
                                    "fetched_at": "2026-07-26T00:00:00+00:00",
                                    "content_hash": f"sha256:{digest}",
                                    "relative_path": raw_path.relative_to(layout.root).as_posix(),
                                },
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report = AtwikiStager(layout).stage(source_version="fixture")
            self.assertEqual(report.pages, 1)
            output = json.loads(
                (layout.staging / "atwiki" / "fixture" / "pages.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(output["source_page_id"], 106)
            self.assertEqual(output["role_sections"][1]["role"], "原画")


if __name__ == "__main__":
    unittest.main()

