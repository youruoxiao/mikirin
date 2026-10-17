from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import tempfile
import unicodedata
import uuid
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            yield value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_alias(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


@dataclass(frozen=True, slots=True)
class SupplementaryLoadReport:
    source: str
    source_version: str
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PreparedBgmWebDocument:
    document_id: str
    source_record_id: str
    title: str
    url: Any
    raw_content: str
    structured_json: str
    content_hash: str
    fetched_at: Any
    source_kind: str
    subject_id: str
    episode_id: Any
    author_hash: str
    rating: Any
    posted_at_raw: Any
    license_id: str
    redistributable: int
    attribution_required: int
    commercial_use_allowed: int
    chunks: tuple[tuple[str, int, int, int, str, str], ...]


def _prepare_bgm_web_document(task: tuple[str, int, str]) -> _PreparedBgmWebDocument:
    """Parse and prepare one JSONL row; safe to run in a spawned worker."""
    path, line_number, line = task
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")

    document_id = str(record["document_id"])
    source_kind = str(record["source_kind"])
    subject_id = str(record["subject_entity_id"])
    episode_id = record.get("episode_entity_id")
    raw_content = str(record["raw_content"])
    structured = {
        "source_kind": source_kind,
        "subject_entity_id": subject_id,
        "episode_entity_id": episode_id,
        "metadata_text": record.get("metadata_text", []),
        "raw_relative_path": record.get("raw_relative_path"),
        "source_asset_hash": record.get("source_asset_hash"),
    }
    chunks = tuple(
        (
            f"{document_id}:chunk:{chunk_index}",
            chunk_index,
            char_start,
            char_end,
            chunk_content,
            "sha256:" + hashlib.sha256(chunk_content.encode("utf-8")).hexdigest(),
        )
        for chunk_index, char_start, char_end, chunk_content in _text_chunks(raw_content)
    )
    return _PreparedBgmWebDocument(
        document_id=document_id,
        source_record_id=str(record.get("source_record_id") or document_id),
        title=str(record.get("title") or ""),
        url=record.get("url"),
        raw_content=raw_content,
        structured_json=_json(structured),
        content_hash=str(record["content_hash"]),
        fetched_at=record.get("fetched_at"),
        source_kind=source_kind,
        subject_id=subject_id,
        episode_id=episode_id,
        author_hash=str(record.get("author_hash") or ""),
        rating=record.get("rating"),
        posted_at_raw=record.get("posted_at_raw"),
        license_id=str(record.get("license_id") or "unknown"),
        redistributable=int(bool(record.get("redistributable", False))),
        attribution_required=int(bool(record.get("attribution_required", False))),
        commercial_use_allowed=int(bool(record.get("commercial_use_allowed", False))),
        chunks=chunks,
    )


def _bgm_web_tasks(path: Path) -> Iterator[tuple[str, int, str]]:
    path_text = str(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield path_text, line_number, line


def _prepare_bgm_web_batch(
    tasks: tuple[tuple[str, int, str], ...],
) -> tuple[_PreparedBgmWebDocument, ...]:
    """Amortize Windows process-pool IPC over a small ordered row batch."""
    return tuple(_prepare_bgm_web_document(task) for task in tasks)


def _bgm_web_task_batches(
    tasks: Iterator[tuple[str, int, str]],
    batch_size: int = 64,
) -> Iterator[tuple[tuple[str, int, str], ...]]:
    batch: list[tuple[str, int, str]] = []
    for task in tasks:
        batch.append(task)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _prepared_bgm_web_documents(
    path: Path,
    workers: int,
) -> Iterator[_PreparedBgmWebDocument]:
    tasks = _bgm_web_tasks(path)
    if workers == 1:
        for task in tasks:
            yield _prepare_bgm_web_document(task)
        return

    # Keep only a small, fixed multiple of row batches in flight. Futures are
    # consumed in submission order so database output is deterministic, while
    # batching avoids paying Windows process IPC overhead for every document.
    batches = _bgm_web_task_batches(tasks)
    executor = ProcessPoolExecutor(max_workers=workers)
    pending: deque[Future[tuple[_PreparedBgmWebDocument, ...]]] = deque()
    try:
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(_prepare_bgm_web_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            yield from pending.popleft().result()
            try:
                pending.append(executor.submit(_prepare_bgm_web_batch, next(batches)))
            except StopIteration:
                pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class SupplementaryStagingLoader:
    @staticmethod
    def _register_asset(
        db: sqlite3.Connection,
        *,
        source: str,
        source_version: str,
        staging_directory: Path,
        parser_version: str,
    ) -> tuple[str, str]:
        manifest = staging_directory / "manifest.json"
        content_hash = _sha256(manifest)
        ingestion_id = f"ingest:{source}:{source_version}:{content_hash}"
        asset_id = f"asset:{source}:{source_version}:{content_hash}"
        now = _utc_now()
        db.execute(
            """
            INSERT INTO ingestion_run(
                id, source, started_at, completed_at, source_version,
                parser_version, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'loading')
            """,
            (ingestion_id, source, now, now, source_version, parser_version),
        )
        db.execute(
            """
            INSERT INTO source_asset(
                id, ingestion_run_id, local_path, content_hash, fetched_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (asset_id, ingestion_id, str(manifest), content_hash, now),
        )
        return ingestion_id, asset_id

    def load_moegirl(
        self,
        *,
        staging_directory: Path,
        database_path: Path,
        source_version: str,
        build_id: str,
    ) -> SupplementaryLoadReport:
        counts = {"characters": 0, "aliases": 0, "facts": 0, "mapping_candidates": 0}
        with closing(sqlite3.connect(database_path)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            ingestion_id, asset_id = self._register_asset(
                db,
                source="moegirl_dataset",
                source_version=source_version,
                staging_directory=staging_directory,
                parser_version="moegirl-loader-v1",
            )
            for record in _rows(staging_directory / "characters.jsonl"):
                entity_id = record["entity_id"]
                name = str(record["raw_name"])
                db.execute(
                    "INSERT INTO entity VALUES (?, 'character', ?, 'provisional')",
                    (entity_id, name),
                )
                db.execute(
                    """
                    INSERT INTO source_entity VALUES (
                        ?, 'moegirl_dataset', 'character', ?, ?, ?, NULL,
                        'unresolved', 0.0, ?
                    )
                    """,
                    (entity_id, str(record["source_entity_id"]), entity_id, name, asset_id),
                )
                db.execute(
                    """
                    INSERT INTO characters(
                        entity_id, name, role, infobox_raw, summary,
                        comments, collects, source_asset_id
                    ) VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?)
                    """,
                    (entity_id, name, asset_id),
                )
                counts["characters"] += 1
                for alias in record.get("aliases", []):
                    alias = str(alias).strip()
                    if not alias:
                        continue
                    alias_id = "alias:" + uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{entity_id}\0{_normalize_alias(alias)}\0moegirl_dataset"
                    ).hex
                    db.execute(
                        """
                        INSERT OR IGNORE INTO entity_alias(
                            id, entity_id, alias_text, normalized_alias, lang,
                            source_kind, status, confidence, source_asset_id,
                            build_id, license_id
                        ) VALUES (?, ?, ?, ?, NULL, 'moegirl_dataset', 'candidate',
                                  0.8, ?, ?, 'upstream')
                        """,
                        (alias_id, entity_id, alias, _normalize_alias(alias), asset_id, build_id),
                    )
                    counts["aliases"] += db.execute("SELECT changes()").fetchone()[0]
                for fact_kind in ("attributes", "voice_actors", "gender", "subjects", "fields"):
                    value = record.get(fact_kind)
                    if value in (None, [], {}):
                        continue
                    fact_id = "fact:" + uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{entity_id}\0{fact_kind}\0{asset_id}"
                    ).hex
                    db.execute(
                        "INSERT INTO entity_fact VALUES (?, ?, ?, ?, ?, 0.8)",
                        (fact_id, entity_id, fact_kind, _json(value), asset_id),
                    )
                    counts["facts"] += 1

            for record in _rows(staging_directory / "mapping_candidates.jsonl"):
                candidate_id = "candidate:" + uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{record['left_entity_id']}\0{record['right_entity_id']}\0same_character",
                ).hex
                db.execute(
                    """
                    INSERT OR IGNORE INTO resolution_candidate(
                        id, left_entity_id, right_entity_id, relation_kind,
                        confidence, evidence_json, decision_source, status,
                        source_asset_id
                    ) VALUES (?, ?, ?, 'same_character', ?, ?, ?, 'candidate', ?)
                    """,
                    (
                        candidate_id,
                        record["left_entity_id"],
                        record["right_entity_id"],
                        float(record["confidence"]),
                        _json(record.get("evidence", {})),
                        record["decision_source"],
                        asset_id,
                    ),
                )
                counts["mapping_candidates"] += db.execute("SELECT changes()").fetchone()[0]
            total = sum(counts.values())
            db.execute(
                "UPDATE ingestion_run SET status='loaded', row_count=? WHERE id=?",
                (total, ingestion_id),
            )
        return SupplementaryLoadReport("moegirl_dataset", source_version, counts)

    def load_atwiki(
        self,
        *,
        staging_directory: Path,
        database_path: Path,
        source_version: str,
    ) -> SupplementaryLoadReport:
        counts = {"documents": 0, "role_sections": 0}
        with closing(sqlite3.connect(database_path)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            ingestion_id, asset_id = self._register_asset(
                db,
                source="atwiki_sakuga",
                source_version=source_version,
                staging_directory=staging_directory,
                parser_version="atwiki-loader-v1",
            )
            for record in _rows(staging_directory / "pages.jsonl"):
                document_id = f"atwiki:page:{record['source_page_id']}:{record['content_hash']}"
                structured = {
                    "role_sections": record.get("role_sections", []),
                    "page_kind": record.get("page_kind", "unclassified"),
                    "raw_relative_path": record.get("raw_relative_path"),
                }
                db.execute(
                    """
                    INSERT INTO source_document(
                        id, source, source_record_id, title, url, content,
                        structured_json, content_hash, fetched_at, source_asset_id
                    ) VALUES (?, 'atwiki_sakuga', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        str(record["source_page_id"]),
                        record["title"],
                        record.get("url"),
                        record.get("plain_text", ""),
                        _json(structured),
                        record["content_hash"],
                        record.get("fetched_at"),
                        asset_id,
                    ),
                )
                counts["documents"] += 1
                for section_order, section in enumerate(record.get("role_sections", [])):
                    section_id = "atwiki:role-section:" + uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{document_id}\0{section_order}"
                    ).hex
                    db.execute(
                        """
                        INSERT INTO document_role_section(
                            id, document_id, section_order, role, content, source_asset_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            section_id,
                            document_id,
                            section_order,
                            str(section.get("role", "unknown")),
                            str(section.get("content", "")),
                            asset_id,
                        ),
                    )
                    counts["role_sections"] += 1
            db.execute(
                "UPDATE ingestion_run SET status='loaded', row_count=? WHERE id=?",
                (sum(counts.values()), ingestion_id),
            )
        return SupplementaryLoadReport("atwiki_sakuga", source_version, counts)

    def load_bgm_web(
        self,
        *,
        staging_directory: Path,
        database_path: Path,
        source_version: str,
        workers: int = 1,
    ) -> SupplementaryLoadReport:
        """Load entity-anchored Bangumi community text into a fresh content build."""
        if workers <= 0:
            raise ValueError("workers must be greater than zero")
        counts = {
            "documents": 0,
            "chunks": 0,
            "subject_comments": 0,
            "episode_comments": 0,
            "reviews": 0,
            "dangling_subjects": 0,
            "dangling_episodes": 0,
        }
        documents_path = staging_directory / "documents.jsonl"
        with closing(sqlite3.connect(database_path)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = OFF")
            ingestion_id, asset_id = self._register_asset(
                db,
                source="bangumi_web",
                source_version=source_version,
                staging_directory=staging_directory,
                parser_version="bgm-web-loader-v1",
            )
            referenced_subjects: set[str] = set()
            referenced_episodes: set[str] = set()
            # Preparation can be CPU-parallel, but SQLite remains exclusively
            # owned by this parent process. The temporary pickle stream avoids
            # holding the corpus in memory or parsing documents.jsonl twice.
            with tempfile.TemporaryFile(mode="w+b") as prepared_stream:
                for prepared in _prepared_bgm_web_documents(documents_path, workers):
                    referenced_subjects.add(prepared.subject_id)
                    if prepared.episode_id is not None:
                        referenced_episodes.add(str(prepared.episode_id))
                    pickle.dump(prepared, prepared_stream, protocol=pickle.HIGHEST_PROTOCOL)

                valid_subjects: set[str] = set()
                for batch in _batched_strings(referenced_subjects):
                    placeholders = ",".join("?" for _ in batch)
                    valid_subjects.update(
                        str(row[0]) for row in db.execute(
                            f"SELECT entity_id FROM subjects WHERE entity_id IN ({placeholders})",
                            batch,
                        )
                    )
                valid_episodes: dict[str, str] = {}
                for batch in _batched_strings(referenced_episodes):
                    placeholders = ",".join("?" for _ in batch)
                    valid_episodes.update(
                        (str(row[0]), str(row[1])) for row in db.execute(
                            f"SELECT entity_id, subject_entity_id FROM episodes "
                            f"WHERE entity_id IN ({placeholders})",
                            batch,
                        )
                    )
                prepared_stream.seek(0)
                while True:
                    try:
                        prepared = pickle.load(prepared_stream)
                    except EOFError:
                        break
                    subject_id = prepared.subject_id
                    if subject_id not in valid_subjects:
                        counts["dangling_subjects"] += 1
                        continue
                    episode_id = prepared.episode_id
                    if (
                        episode_id is not None
                        and valid_episodes.get(str(episode_id)) != subject_id
                    ):
                        counts["dangling_episodes"] += 1
                        continue
                    db.execute(
                        """
                        INSERT INTO source_document(
                            id, source, source_record_id, title, url, content,
                            structured_json, content_hash, fetched_at, source_asset_id
                        ) VALUES (?, 'bangumi_web', ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prepared.document_id,
                            prepared.source_record_id,
                            prepared.title,
                            prepared.url,
                            prepared.raw_content,
                            prepared.structured_json,
                            prepared.content_hash,
                            prepared.fetched_at,
                            asset_id,
                        ),
                    )
                    db.execute(
                        """
                        INSERT INTO community_document(
                            document_id, source_kind, subject_entity_id, episode_entity_id,
                            author_hash, rating, posted_at_raw, license_id,
                            redistributable, attribution_required, commercial_use_allowed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prepared.document_id,
                            prepared.source_kind,
                            subject_id,
                            episode_id,
                            prepared.author_hash,
                            prepared.rating,
                            prepared.posted_at_raw,
                            prepared.license_id,
                            prepared.redistributable,
                            prepared.attribution_required,
                            prepared.commercial_use_allowed,
                        ),
                    )
                    for chunk in prepared.chunks:
                        db.execute(
                            """
                            INSERT INTO document_chunk(
                                id, document_id, chunk_index, char_start, char_end,
                                content, content_hash, source_asset_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (chunk[0], prepared.document_id, *chunk[1:], asset_id),
                        )
                        counts["chunks"] += 1
                    counts["documents"] += 1
                    counter = {
                        "bgm_subject_comment": "subject_comments",
                        "bgm_ep_comment": "episode_comments",
                        "bgm_review": "reviews",
                    }.get(prepared.source_kind)
                    if counter:
                        counts[counter] += 1
            db.execute(
                "UPDATE ingestion_run SET status='loaded', row_count=?, error_count=? WHERE id=?",
                (
                    counts["documents"],
                    counts["dangling_subjects"] + counts["dangling_episodes"],
                    ingestion_id,
                ),
            )
        return SupplementaryLoadReport("bangumi_web", source_version, counts)


def _batched_strings(values: set[str], size: int = 5_000) -> Iterator[tuple[str, ...]]:
    ordered = sorted(values)
    for start in range(0, len(ordered), size):
        yield tuple(ordered[start : start + size])


def _text_chunks(
    content: str,
    *,
    target_characters: int = 1_200,
    overlap_characters: int = 150,
) -> Iterator[tuple[int, int, int, str]]:
    if target_characters < 200 or overlap_characters < 0:
        raise ValueError("invalid chunk dimensions")
    if overlap_characters >= target_characters:
        raise ValueError("chunk overlap must be smaller than the target")
    if not content:
        return
    start = 0
    chunk_index = 0
    while start < len(content):
        end = min(len(content), start + target_characters)
        if end < len(content):
            boundary = content.rfind("\n", start + target_characters // 2, end)
            if boundary > start:
                end = boundary
        chunk_content = content[start:end].strip()
        if chunk_content:
            yield chunk_index, start, end, chunk_content
            chunk_index += 1
        if end >= len(content):
            break
        start = max(start + 1, end - overlap_characters)
