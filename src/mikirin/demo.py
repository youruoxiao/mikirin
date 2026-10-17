"""Small, wholly fictional dataset for trying the application without an API."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.run_lock import exclusive_process_lock
from mikirin.search_index import SearchIndexer


BUILD_ID = "demo-v1"
WORKS = (
    ("rooftop", "屋顶放映部", "放映部", "几个高中生修好旧放映机，在屋顶办了一场只有五个人的放映会。"),
    ("last-train", "末班电车与猫", "末班猫", "夜班司机每天都会遇到同一只猫，却从来没见过它下车。"),
    ("radio", "夏日调频", "调频", "海边小城的广播社决定用一个暑假，录下那些快要消失的声音。"),
)


def create_demo(layout: ProjectLayout) -> Path:
    """Publish a tiny demo only in an unused workspace; never replace a real DB."""
    layout.initialize()
    publisher = Publisher(layout)
    with exclusive_process_lock(layout.manifests / "demo.lock"):
        current = publisher.current()
        if current is not None:
            with closing(sqlite3.connect(current.as_uri() + "?mode=ro", uri=True)) as db:
                marker = db.execute(
                    "SELECT value FROM schema_metadata WHERE key='demo_dataset'"
                ).fetchone()
            if marker and marker[0] == BUILD_ID:
                publisher.validate(current)
                return current
            raise FileExistsError("这个目录已有知识库。请用 --root 指定一个新的演示目录。")
        if any(layout.db.glob("content.v*.db")):
            raise FileExistsError("这个目录已有数据库版本，请换一个空的演示目录。")
        building = publisher.create_empty_build(build_id=BUILD_ID, code_version=BUILD_ID)
        try:
            with closing(sqlite3.connect(building)) as db, db:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("INSERT INTO schema_metadata VALUES('demo_dataset',?)", (BUILD_ID,))
                db.execute(
                    "INSERT INTO ingestion_run(id,source,started_at,source_version,parser_version,status) "
                    "VALUES('demo','demo','2026-01-01','1','1','done')"
                )
                db.execute(
                    "INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                    "VALUES('demo','demo','builtin:demo','demo-v1','2026-01-01')"
                )
                for slug, title, alias, summary in WORKS:
                    subject = f"demo:subject:{slug}"
                    db.execute("INSERT INTO entity VALUES(?,'subject',?,'active')", (subject, title))
                    db.execute(
                        "INSERT INTO subjects(entity_id,type,name,name_cn,summary,platform,source_asset_id) "
                        "VALUES(?,2,?,?,?,'演示作品','demo')",
                        (subject, title, title, "【虚构演示】" + summary),
                    )
                    db.execute(
                        "INSERT INTO entity_alias(id,entity_id,alias_text,normalized_alias,lang,"
                        "source_kind,status,confidence,source_asset_id,build_id,license_id) "
                        "VALUES(?,?,?,?,'zh','demo','active',1,'demo',?,'demo-original')",
                        (f"demo:alias:{slug}", subject, alias, alias, BUILD_ID),
                    )
                    for number, name in enumerate(("第一次见面", "放学以后", "再试一次"), 1):
                        episode = f"demo:episode:{slug}:{number}"
                        db.execute("INSERT INTO entity VALUES(?,'episode',?,'active')",
                                   (episode, f"{title} 第{number}集 {name}"))
                        db.execute(
                            "INSERT INTO episodes(entity_id,subject_entity_id,sort,type,name,name_cn,"
                            "description,source_asset_id) VALUES(?,?,?,0,?,?,?,'demo')",
                            (episode, subject, number, name, name, "虚构演示章节。"),
                        )
                    document = f"demo:document:{slug}"
                    content = f"【虚构演示笔记，并非社区评论】{title}：{summary} 这里可以试用作品名和别名检索。"
                    db.execute(
                        "INSERT INTO source_document(id,source,source_record_id,title,content,"
                        "content_hash,source_asset_id) VALUES(?,'demo',?,?,?,?, 'demo')",
                        (document, slug, title + "（演示笔记）", content,
                         hashlib.sha256(content.encode()).hexdigest()),
                    )
                    db.execute(
                        "INSERT INTO community_document(document_id,source_kind,subject_entity_id,"
                        "license_id,redistributable,attribution_required,commercial_use_allowed) "
                        "VALUES(?,'demo',?,'demo-original',1,0,1)", (document, subject),
                    )
            SearchIndexer.rebuild(building)
            return publisher.publish(version=BUILD_ID)
        except BaseException:
            # This file was created by this invocation, never a pre-existing build.
            building.unlink(missing_ok=True)
            raise
