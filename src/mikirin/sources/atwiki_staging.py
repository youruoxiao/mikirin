from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.layout import ProjectLayout
from mikirin.sources.atwiki_parse import extract_wikibody_text, split_role_sections


@dataclass(frozen=True, slots=True)
class AtwikiStagingReport:
    source_version: str
    pages: int
    empty_bodies: int
    output_directory: str


class AtwikiStager:
    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def stage(self, *, source_version: str) -> AtwikiStagingReport:
        state_path = self.layout.manifests / "atwiki" / "crawl-state.json"
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle).get("pages", {})
        failed = [page_id for page_id, value in state.items() if value.get("status") != "ok"]
        if failed:
            raise ValueError(f"仍有 {len(failed)} 个未成功页面，拒绝发布完整 staging")

        target = self.layout.staging / "atwiki" / source_version
        if target.exists():
            raise FileExistsError(f"atwiki staging 已存在，拒绝覆盖: {target}")
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        output_path = temporary / "pages.jsonl"
        page_count = 0
        empty_bodies = 0
        try:
            with output_path.open("w", encoding="utf-8", newline="\n") as output:
                for page_id in sorted(state, key=int):
                    record = state[page_id]["record"]
                    raw_path = self.layout.root / record["relative_path"]
                    html = raw_path.read_text(encoding="utf-8", errors="replace")
                    text = extract_wikibody_text(html)
                    if not text:
                        empty_bodies += 1
                    role_sections = [asdict(section) for section in split_role_sections(text)]
                    normalized = {
                        "source": "atwiki_sakuga",
                        "source_page_id": int(page_id),
                        "title": record["title"],
                        "url": record["url"],
                        "fetched_at": record["fetched_at"],
                        "content_hash": record["content_hash"],
                        "raw_relative_path": record["relative_path"],
                        "plain_text": text,
                        "role_sections": role_sections,
                        "page_kind": "unclassified",
                    }
                    output.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                    page_count += 1
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        report = AtwikiStagingReport(
            source_version=source_version,
            pages=page_count,
            empty_bodies=empty_bodies,
            output_directory=target.relative_to(self.layout.root).as_posix(),
        )
        atomic_write_json(target / "manifest.json", asdict(report))
        return report

