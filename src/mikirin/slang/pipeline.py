from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator

from mikirin.p1_build import ensure_p1_schema, stable_id, utc_now
from mikirin.slang.automaton import AhoCorasick, AliasRecord
from mikirin.slang.candidates import SLOT_PATTERNS, normalize_for_matching
from mikirin.slang.scoring import CorpusStats, ScoreBuckets, score_candidate
from mikirin.slang.linking import link_candidates


SLANG_PIPELINE_VERSION = "slang-discovery-v2-full-split-pmi"
DEFAULT_THRESHOLD_VERSION = "uncalibrated-v1"
_SOURCE_KINDS = ("bgm_ep_comment", "bgm_subject_comment", "bgm_review")
_ALIAS_DENYLIST = ("昵称", "日文名", "英文名", "中文名", "别名")
_ALIAS_SELECTION_CTE = """
WITH eligible AS (
    SELECT a.entity_id,a.alias_text,a.normalized_alias
    FROM entity_alias a JOIN p1_relevant_entity r ON r.entity_id=a.entity_id
    WHERE a.status IN ('active','confirmed')
      AND length(a.normalized_alias) BETWEEN ? AND ?
      AND a.alias_text NOT LIKE '%|'
      AND a.normalized_alias NOT IN (?,?,?,?,?)
), unambiguous AS (
    SELECT normalized_alias FROM eligible
    GROUP BY normalized_alias HAVING count(DISTINCT entity_id)=1
), selected AS (
    SELECT e.entity_id,e.alias_text,e.normalized_alias
    FROM eligible e JOIN unambiguous u USING(normalized_alias)
    GROUP BY e.entity_id,e.normalized_alias
)
"""
_MOJIBAKE = ("\ufffd", "锟斤拷", "Ã", "Â", "â€", "ï¿½")
_STOP_SINGLE = frozenset(
    "的一是了我不人在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从知全工想把情者最但很被呢啊哦吧呀嘛啦哈嗯哎诶喔么吗了着过"
)
_GENERIC_TERMS = frozenset(
    {
        "这个",
        "那个",
        "真的",
        "感觉",
        "剧情",
        "角色",
        "动画",
        "作品",
        "制作",
        "还是",
        "就是",
        "不是",
        "一个",
        "没有",
        "什么",
        "可以",
        "非常",
        "喜欢",
        "好看",
        "一集",
        "这集",
        "本集",
        "看到",
        "最后",
        "开始",
        "episode",
        "anime",
    }
)
_ANONYMOUS = frozenset({"", "anonymous", "anon", "匿名"})
_SLOT_REGEXES = tuple(
    (pattern.name, re.compile(pattern.expression, re.IGNORECASE | re.UNICODE))
    for pattern in SLOT_PATTERNS
)


@dataclass(frozen=True, slots=True)
class SlangPipelineConfig:
    minimum_alias_length: int = 2
    maximum_alias_length: int = 24
    maximum_entities_per_document: int = 12
    minimum_local_occurrences: int = 3
    minimum_local_documents: int = 2
    minimum_global_occurrences: int = 8
    minimum_global_documents: int = 4
    minimum_global_authors: int = 3
    candidate_score_threshold: float = 0.42
    single_char_score_threshold: float = 0.72
    candidate_limit: int = 5_000
    local_candidate_limit: int = 20_000
    evidence_limit: int = 5
    author_sample_limit: int = 2_000
    maximum_alias_count: int = 250_000
    maximum_normalized_alias_characters: int = 4_000_000
    threshold_version: str = DEFAULT_THRESHOLD_VERSION
    max_subjects: int | None = None
    max_documents_per_subject: int | None = None
    incremental_base_run_id: str | None = None
    vector_neighbors_by_candidate: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def validate(self) -> None:
        if not 1 <= self.minimum_alias_length <= self.maximum_alias_length:
            raise ValueError("alias length bounds are invalid")
        for value in (
            self.maximum_entities_per_document,
            self.minimum_local_occurrences,
            self.minimum_local_documents,
            self.minimum_global_occurrences,
            self.minimum_global_documents,
            self.minimum_global_authors,
            self.candidate_limit,
            self.local_candidate_limit,
            self.evidence_limit,
            self.author_sample_limit,
            self.maximum_alias_count,
            self.maximum_normalized_alias_characters,
        ):
            if value < 1:
                raise ValueError("slang pipeline count limits must be positive")
        if not 0 <= self.candidate_score_threshold <= 1:
            raise ValueError("candidate_score_threshold must be in [0,1]")
        if not 0 <= self.single_char_score_threshold <= 1:
            raise ValueError("single_char_score_threshold must be in [0,1]")
        if self.single_char_score_threshold < self.candidate_score_threshold:
            raise ValueError("single-character score threshold must be at least the general threshold")


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    subject_entity_id: str
    author_hash: str
    content: str


@dataclass(frozen=True, slots=True)
class _Span:
    text: str
    start: int
    end: int
    slot_names: tuple[str, ...] = ()


@dataclass(slots=True)
class _MutableStats:
    candidate_text: str
    occurrence_count: int = 0
    document_count: int = 0
    authors: set[str] = field(default_factory=set)
    left: Counter[str] = field(default_factory=Counter)
    right: Counter[str] = field(default_factory=Counter)
    slot_hits: int = 0
    bracketed: int = 0
    official_cooccurrence_documents: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _is_anonymous(value: str) -> bool:
    return value.strip().casefold() in _ANONYMOUS


def _valid_candidate(value: str) -> bool:
    if not value or len(value) > 32 or any(marker in value for marker in _MOJIBAKE):
        return False
    if "http" in value.casefold() or "www." in value.casefold():
        return False
    if any(unicodedata.category(char) in {"Cc", "Cs", "Co"} for char in value):
        return False
    compact = "".join(char for char in value if char.isalnum())
    if not compact or compact.isdigit():
        return False
    if any(char in value for char in "?？*\\/|<>{}[]=~`@#$%^"):
        return False
    normalized = normalize_for_matching(value)
    if normalized in _GENERIC_TERMS:
        return False
    if len(normalized) == 1 and normalized in _STOP_SINGLE:
        return False
    return True


def _meaningful_runs(text: str) -> Iterator[tuple[int, int]]:
    start: int | None = None
    for index, char in enumerate(text):
        if char.isalnum() or unicodedata.category(char).startswith("M"):
            if start is None:
                start = index
        elif start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(text)


def _candidate_spans(text: str) -> Iterator[_Span]:
    seen: dict[tuple[str, int, int], set[str]] = {}
    originals: dict[tuple[str, int, int], str] = {}
    for run_start, run_end in _meaningful_runs(text):
        for start in range(run_start, run_end):
            for width in range(1, 6):
                end = start + width
                if end > run_end:
                    break
                raw = text[start:end]
                normalized = normalize_for_matching(raw)
                if not _valid_candidate(normalized):
                    continue
                key = (normalized, start, end)
                seen.setdefault(key, set())
                originals.setdefault(key, raw)
        raw_run = text[run_start:run_end]
        normalized_run = normalize_for_matching(raw_run)
        if len(raw_run) > 5 and _valid_candidate(normalized_run):
            key = (normalized_run, run_start, run_end)
            seen.setdefault(key, set())
            originals.setdefault(key, raw_run)
    for name, expression in _SLOT_REGEXES:
        for match in expression.finditer(text):
            raw = match.groupdict().get("candidate")
            if not raw:
                continue
            start, end = match.span("candidate")
            stripped = raw.strip()
            left_trim = len(raw) - len(raw.lstrip())
            start += left_trim
            end = start + len(stripped)
            normalized = normalize_for_matching(stripped)
            if not _valid_candidate(normalized):
                continue
            key = (normalized, start, end)
            seen.setdefault(key, set()).add(name)
            originals.setdefault(key, stripped)
    for key in sorted(seen, key=lambda item: (item[1], item[2], item[0])):
        normalized, start, end = key
        yield _Span(originals[key], start, end, tuple(sorted(seen[key])))


def _count_corpus_ngrams(text: str, unigrams: Counter[str], bigrams: Counter[str],
                         ngrams: Counter[str]) -> int:
    count = 0
    normalized = normalize_for_matching(text)
    for start, end in _meaningful_runs(normalized):
        run = normalized[start:end]
        unigrams.update(run)
        bigrams.update(run[index : index + 2] for index in range(len(run) - 1))
        for width in range(1, 6):
            ngrams.update(run[index:index + width] for index in range(len(run) - width + 1))
        if 5 < len(run) <= 32:
            parts = {run, *(run[:split] for split in range(1, len(run))),
                     *(run[split:] for split in range(1, len(run)))}
            for part in parts:
                if len(part) > 5:
                    ngrams[part] += sum(run.startswith(part, index) for index in range(len(run)))
        count += len(run)
    return count


def _bracketed(text: str, start: int, end: int) -> bool:
    if start < 1 or end >= len(text):
        return False
    return (text[start - 1], text[end]) in {
        ("《", "》"),
        ("〈", "〉"),
        ("「", "」"),
        ("『", "』"),
        ("【", "】"),
        ("(", ")"),
        ("（", "）"),
        ("[", "]"),
    }


def _excerpt(text: str, start: int, end: int, radius: int = 120) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)].strip()


def _source_watermark(db: sqlite3.Connection) -> str:
    row = db.execute(
        """
        SELECT count(*),coalesce(max(d.rowid),0),coalesce(sum(length(d.content)),0)
        FROM community_document c JOIN source_document d ON d.id=c.document_id
        WHERE c.source_kind IN (?,?,?)
        """,
        _SOURCE_KINDS,
    ).fetchone()
    content_digest = hashlib.sha256()
    for document_id, source_kind, subject_entity_id, author_hash, content in db.execute(
        """
        SELECT d.id,c.source_kind,c.subject_entity_id,c.author_hash,d.content
        FROM community_document c JOIN source_document d ON d.id=c.document_id
        WHERE c.source_kind IN (?,?,?)
        ORDER BY d.id
        """,
        _SOURCE_KINDS,
    ):
        payload = json.dumps(
            [
                str(document_id),
                str(source_kind),
                str(subject_entity_id),
                str(author_hash or ""),
                str(content),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        content_digest.update(len(payload).to_bytes(8, "big"))
        content_digest.update(payload)
    return json.dumps(
        {
            "document_count": int(row[0]),
            "max_rowid": int(row[1]),
            "character_count": int(row[2]),
            "content_digest": content_digest.hexdigest(),
            "alias_digest": _alias_digest(db),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _alias_digest(db: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in db.execute("SELECT entity_id,alias_text,normalized_alias,status FROM entity_alias "
                          "WHERE status IN ('active','confirmed') ORDER BY entity_id,normalized_alias,alias_text,status"):
        digest.update((_canonical_json(list(row)) + "\n").encode("utf-8"))
    # These determine which aliases enter the relevant-entity automaton.
    for query in (
        "SELECT DISTINCT subject_entity_id FROM community_document ORDER BY subject_entity_id",
        "SELECT DISTINCT subject_id,character_id FROM rel_subject_character ORDER BY subject_id,character_id",
        "SELECT DISTINCT subject_id,person_id FROM rel_subject_person ORDER BY subject_id,person_id",
        "SELECT entity_id,subject_entity_id FROM episodes ORDER BY entity_id,subject_entity_id",
    ):
        digest.update(query.encode())
        for row in db.execute(query):
            digest.update((_canonical_json(list(row)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _documents_digest(documents: list[Document]) -> str:
    return stable_id("slang-subject-input", [(doc.document_id, doc.subject_entity_id,
                                             doc.author_hash, doc.content) for doc in documents])


def _source_watermark_payload(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("cannot resume slang run: invalid source watermark")
    if "source" in payload:
        payload = payload["source"]
    if not isinstance(payload, dict):
        raise ValueError("cannot resume slang run: invalid source watermark")
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_existing_run(
    row: sqlite3.Row | tuple[Any, ...],
    *,
    build_id: str,
    config: SlangPipelineConfig,
    source_watermark: str,
) -> tuple[str, str, str] | None:
    (
        stored_build_id,
        stored_processor_version,
        stored_threshold_version,
        stored_watermark,
        stored_input_digest,
        stored_config_json,
    ) = row
    if str(stored_build_id) != build_id:
        raise ValueError(
            f"cannot resume slang run: build_id mismatch "
            f"(stored {stored_build_id!r}, requested {build_id!r})"
        )
    if str(stored_processor_version) != SLANG_PIPELINE_VERSION:
        raise ValueError(
            f"cannot resume slang run: processor_version mismatch "
            f"(stored {stored_processor_version!r}, current {SLANG_PIPELINE_VERSION!r})"
        )
    if str(stored_threshold_version) != config.threshold_version:
        raise ValueError(
            f"cannot resume slang run: threshold_version mismatch "
            f"(stored {stored_threshold_version!r}, requested {config.threshold_version!r})"
        )
    try:
        stored_config = json.loads(str(stored_config_json))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot resume slang run: invalid stored config") from exc
    try:
        stored_source = _source_watermark_payload(str(stored_watermark))
        current_source = _source_watermark_payload(source_watermark)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot resume slang run: invalid source watermark") from exc
    expected_config = asdict(config)
    # These safety caps were added after the first real full scan had already
    # checkpointed hundreds of subjects.  Missing keys may be upgraded only
    # to their exact defaults; every older and user-controlled knob remains a
    # strict part of the run identity.
    compatible_config = dict(stored_config)
    for field in (
        "maximum_alias_count",
        "maximum_normalized_alias_characters",
    ):
        compatible_config.setdefault(field, getattr(SlangPipelineConfig(), field))
    if "content_digest" not in stored_source:
        legacy_fields = {"document_count", "max_rowid", "character_count"}
        if set(stored_source) != legacy_fields or any(
            stored_source[field] != current_source.get(field) for field in legacy_fields
        ):
            raise ValueError("cannot resume slang run: source watermark/content identity mismatch")
        if compatible_config != expected_config:
            raise ValueError("cannot resume slang run: config mismatch")
        legacy_watermark = _canonical_json(stored_source)
        expected_legacy_digest = stable_id("slang-input", legacy_watermark, stored_config)
        if str(stored_input_digest) != expected_legacy_digest:
            raise ValueError("cannot resume slang run: input/config digest mismatch")
        return (
            source_watermark,
            stable_id("slang-input", source_watermark, expected_config),
            _canonical_json(expected_config),
        )
    if stored_source != current_source:
        raise ValueError("cannot resume slang run: source watermark/content identity mismatch")
    if compatible_config != expected_config:
        raise ValueError("cannot resume slang run: config mismatch")
    expected_input_digest = stable_id("slang-input", source_watermark, expected_config)
    compatible_input_digest = stable_id("slang-input", source_watermark, stored_config)
    if str(stored_input_digest) not in {expected_input_digest, compatible_input_digest}:
        raise ValueError("cannot resume slang run: input/config digest mismatch")
    if stored_config != expected_config:
        return source_watermark, expected_input_digest, _canonical_json(expected_config)
    return None


def _load_alias_automaton(db: sqlite3.Connection, config: SlangPipelineConfig) -> AhoCorasick:
    db.execute("DROP TABLE IF EXISTS temp.p1_relevant_entity")
    db.execute("CREATE TEMP TABLE p1_relevant_entity(entity_id TEXT PRIMARY KEY) WITHOUT ROWID")
    db.executescript(
        """
        INSERT OR IGNORE INTO p1_relevant_entity
        SELECT DISTINCT subject_entity_id FROM community_document;
        INSERT OR IGNORE INTO p1_relevant_entity
        SELECT DISTINCT r.character_id FROM rel_subject_character r
        JOIN (SELECT DISTINCT subject_entity_id FROM community_document) c
          ON c.subject_entity_id=r.subject_id;
        INSERT OR IGNORE INTO p1_relevant_entity
        SELECT DISTINCT r.person_id FROM rel_subject_person r
        JOIN (SELECT DISTINCT subject_entity_id FROM community_document) c
          ON c.subject_entity_id=r.subject_id;
        INSERT OR IGNORE INTO p1_relevant_entity
        SELECT DISTINCT e.entity_id FROM episodes e
        JOIN (SELECT DISTINCT subject_entity_id FROM community_document) c
          ON c.subject_entity_id=e.subject_entity_id;
        """
    )
    alias_capacity = db.execute(
        """
        WITH eligible AS (
            SELECT a.entity_id,a.alias_text,a.normalized_alias
            FROM entity_alias a JOIN p1_relevant_entity r ON r.entity_id=a.entity_id
            WHERE a.status IN ('active','confirmed')
              AND length(a.normalized_alias) BETWEEN ? AND ?
              AND a.alias_text NOT LIKE '%|'
              AND a.normalized_alias NOT IN (?,?,?,?,?) /*
              AND a.normalized_alias NOT IN ('鏄电О','鏃ユ枃鍚?,'鑻辨枃鍚?,'涓枃鍚?,'鍒悕')
              */
        ), unambiguous AS (
            SELECT normalized_alias FROM eligible
            GROUP BY normalized_alias HAVING count(DISTINCT entity_id)=1
        ), selected AS (
            SELECT e.entity_id,e.alias_text,e.normalized_alias
            FROM eligible e JOIN unambiguous u USING(normalized_alias)
            GROUP BY e.entity_id,e.normalized_alias
        )
        SELECT count(*),coalesce(sum(length(normalized_alias)),0) FROM selected
        """,
        (
            config.minimum_alias_length,
            config.maximum_alias_length,
            "昵称",
            "日文名",
            "英文名",
            "中文名",
            "别名",
        ),
    ).fetchone()
    alias_count = int(alias_capacity[0])
    normalized_alias_characters = int(alias_capacity[1])
    if alias_count > config.maximum_alias_count or (
        normalized_alias_characters > config.maximum_normalized_alias_characters
    ):
        raise ValueError(
            "slang alias automaton capacity exceeded: "
            f"alias_count={alias_count} (limit={config.maximum_alias_count}), "
            "normalized_alias_characters="
            f"{normalized_alias_characters} "
            f"(limit={config.maximum_normalized_alias_characters})"
        )
    rows = db.execute(
        """
        WITH eligible AS (
            SELECT a.entity_id,a.alias_text,a.normalized_alias
            FROM entity_alias a JOIN p1_relevant_entity r ON r.entity_id=a.entity_id
            WHERE a.status IN ('active','confirmed')
              AND length(a.normalized_alias) BETWEEN ? AND ?
              AND a.alias_text NOT LIKE '%|'
              AND a.normalized_alias NOT IN ('昵称','日文名','英文名','中文名','别名')
        ), unambiguous AS (
            SELECT normalized_alias FROM eligible
            GROUP BY normalized_alias HAVING count(DISTINCT entity_id)=1
        )
        SELECT e.entity_id,min(e.alias_text),e.normalized_alias
        FROM eligible e JOIN unambiguous u USING(normalized_alias)
        GROUP BY e.entity_id,e.normalized_alias
        ORDER BY e.normalized_alias,e.entity_id,e.alias_text
        """,
        (config.minimum_alias_length, config.maximum_alias_length),
    )
    aliases = [AliasRecord(str(row[0]), str(row[1]), str(row[2])) for row in rows]
    db.execute("DROP TABLE p1_relevant_entity")
    return AhoCorasick(aliases)


def _subject_documents(
    db: sqlite3.Connection,
    subject_id: str,
    *,
    limit: int | None,
) -> list[Document]:
    sql = (
        "SELECT d.id,c.subject_entity_id,c.author_hash,d.content "
        "FROM community_document c JOIN source_document d ON d.id=c.document_id "
        "WHERE c.subject_entity_id=? AND c.source_kind IN (?,?,?) "
        "ORDER BY d.id"
    )
    parameters: list[Any] = [subject_id, *_SOURCE_KINDS]
    if limit is not None:
        sql += " LIMIT ?"
        parameters.append(limit)
    return [
        Document(str(row[0]), str(row[1]), str(row[2] or ""), str(row[3] or ""))
        for row in db.execute(sql, parameters)
    ]


def _store_cooccurrence(
    db: sqlite3.Connection,
    *,
    run_id: str,
    document: Document,
    matches: Iterable[Any],
    maximum_entities: int,
) -> int:
    mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    specificity: dict[str, int] = {}
    for match in matches:
        specificity[match.entity_id] = max(specificity.get(match.entity_id, 0), match.end - match.start)
        if len(mentions[match.entity_id]) < 8:
            mentions[match.entity_id].append(
                {"start": match.start, "end": match.end, "text": match.matched_text}
            )
    entity_ids = sorted(specificity, key=lambda item: (-specificity[item], item))[:maximum_entities]
    inserted = 0
    rows: list[tuple[Any, ...]] = []
    for index, left in enumerate(entity_ids):
        for right in entity_ids[index + 1 :]:
            src, dst = sorted((left, right))
            left_mentions = mentions[src]
            right_mentions = mentions[dst]
            bounds = [item["start"] for item in (*left_mentions, *right_mentions)] + [
                item["end"] for item in (*left_mentions, *right_mentions)
            ]
            start, end = min(bounds), max(bounds)
            rows.append(
                (
                    stable_id("cooccurrence", run_id, document.document_id, src, dst),
                    run_id,
                    document.document_id,
                    document.subject_entity_id,
                    src,
                    dst,
                    document.author_hash,
                    json.dumps(left_mentions, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(right_mentions, ensure_ascii=False, separators=(",", ":")),
                    _excerpt(document.content, start, end),
                )
            )
    if rows:
        before = db.total_changes
        db.executemany(
            "INSERT OR IGNORE INTO cooccurrence(" 
            "id,run_id,document_id,subject_entity_id,src_entity_id,dst_entity_id,author_hash," 
            "src_mentions_json,dst_mentions_json,evidence_text) VALUES(?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        inserted = db.total_changes - before
    return inserted


def _scan_subject(
    db: sqlite3.Connection,
    *,
    run_id: str,
    documents: list[Document],
    automaton: AhoCorasick,
    config: SlangPipelineConfig,
    alias_digest: str | None = None,
) -> dict[str, int]:
    stats: dict[str, _MutableStats] = {}
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    ngrams: Counter[str] = Counter()
    character_count = 0
    cooccurrence_count = 0
    for document in documents:
        scan = automaton.scan(document.content)
        cooccurrence_count += _store_cooccurrence(
            db,
            run_id=run_id,
            document=document,
            matches=scan.matches,
            maximum_entities=config.maximum_entities_per_document,
        )
        character_count += _count_corpus_ngrams(scan.erased_text, unigrams, bigrams, ngrams)
        seen_in_document: set[str] = set()
        for span in _candidate_spans(scan.erased_text):
            candidate = normalize_for_matching(span.text)
            value = stats.get(candidate)
            if value is None:
                value = _MutableStats(candidate_text=span.text.strip())
                stats[candidate] = value
            value.occurrence_count += 1
            value.left[document.content[span.start - 1] if span.start else "<BOS>"] += 1
            value.right[
                document.content[span.end] if span.end < len(document.content) else "<EOS>"
            ] += 1
            value.slot_hits += len(span.slot_names)
            value.bracketed += int(_bracketed(document.content, span.start, span.end))
            if candidate not in seen_in_document:
                value.document_count += 1
                value.official_cooccurrence_documents += int(document.subject_entity_id in scan.entity_ids)
                seen_in_document.add(candidate)
            if not _is_anonymous(document.author_hash):
                value.authors.add(document.author_hash.strip())
            if len(value.evidence) < config.evidence_limit:
                value.evidence.append(
                    {
                        "document_id": document.document_id,
                        "start": span.start,
                        "end": span.end,
                        "context": _excerpt(document.content, span.start, span.end),
                        "slot_names": list(span.slot_names),
                        "author_hash": document.author_hash,
                    }
                )
    eligible = [
        (candidate, value)
        for candidate, value in stats.items()
        if (
            value.slot_hits
            or (
                value.occurrence_count >= config.minimum_local_occurrences
                and value.document_count >= config.minimum_local_documents
            )
        )
    ]
    eligible.sort(
        key=lambda item: (
            -bool(item[1].slot_hits),
            -item[1].document_count,
            -len(item[1].authors),
            -item[1].occurrence_count,
            item[0],
        )
    )
    subject_id = documents[0].subject_entity_id if documents else ""
    for candidate, value in eligible[: config.local_candidate_limit]:
        stats_json = json.dumps(
            {
                "authors": sorted(value.authors),
                "left": dict(sorted(value.left.items())),
                "right": dict(sorted(value.right.items())),
                "slot_hits": value.slot_hits,
                "bracketed": value.bracketed,
                "official_cooccurrence_documents": value.official_cooccurrence_documents,
                "evidence": value.evidence,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        db.execute(
            """
            INSERT INTO slang_candidate_subject_stat(
                run_id,subject_entity_id,normalized_candidate,candidate_text,
                occurrence_count,document_count,author_count,stats_json
            ) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,subject_entity_id,normalized_candidate) DO UPDATE SET
                candidate_text=excluded.candidate_text,
                occurrence_count=excluded.occurrence_count,
                document_count=excluded.document_count,
                author_count=excluded.author_count,
                stats_json=excluded.stats_json
            """,
            (
                run_id,
                subject_id,
                candidate,
                value.candidate_text,
                value.occurrence_count,
                value.document_count,
                len(value.authors),
                stats_json,
            ),
        )
    ngram_json = json.dumps(
        {"unigrams": dict(sorted(unigrams.items())), "bigrams": dict(sorted(bigrams.items())),
         "ngrams": dict(sorted(ngrams.items())), "source_digest": _documents_digest(documents),
         "alias_digest": alias_digest},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if subject_id:
        db.execute(
            """
            INSERT INTO slang_scan_checkpoint(
                run_id,subject_entity_id,last_document_id,document_count,
                character_count,ngram_stats_json,completed_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(run_id,subject_entity_id) DO UPDATE SET
                last_document_id=excluded.last_document_id,
                document_count=excluded.document_count,
                character_count=excluded.character_count,
                ngram_stats_json=excluded.ngram_stats_json,
                completed_at=excluded.completed_at
            """,
            (
                run_id,
                subject_id,
                documents[-1].document_id if documents else "",
                len(documents),
                character_count,
                ngram_json,
                utc_now(),
            ),
        )
    return {
        "documents": len(documents),
        "characters": character_count,
        "local_candidates": min(len(eligible), config.local_candidate_limit),
        "cooccurrences": cooccurrence_count,
    }


def _candidate_existing_targets(
    db: sqlite3.Connection,
    candidate: str,
    subject_ids: list[str],
) -> list[str]:
    if not subject_ids:
        return []
    marks = ",".join("?" for _ in subject_ids)
    rows = db.execute(
        f"""
        SELECT DISTINCT a.entity_id
        FROM entity_alias a JOIN entity e ON e.id=a.entity_id
        WHERE a.normalized_alias=? AND a.status IN ('candidate','provisional') AND (
            (e.entity_type='subject' AND a.entity_id IN ({marks})) OR
            (e.entity_type='character' AND EXISTS(
                SELECT 1 FROM rel_subject_character r
                WHERE r.character_id=a.entity_id AND r.subject_id IN ({marks})
            )) OR
            (e.entity_type='person' AND EXISTS(
                SELECT 1 FROM rel_subject_person r
                WHERE r.person_id=a.entity_id AND r.subject_id IN ({marks})
            ))
        )
        ORDER BY a.entity_id LIMIT 12
        """,
        (candidate, *subject_ids, *subject_ids, *subject_ids),
    )
    return [str(row[0]) for row in rows]


def _aggregate_candidates(
    db: sqlite3.Connection,
    *,
    run_id: str,
    build_id: str,
    config: SlangPipelineConfig,
) -> dict[str, int]:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    ngrams: Counter[str] = Counter()
    total_characters = 0
    total_documents = 0
    for character_count, raw in db.execute(
        "SELECT character_count,ngram_stats_json FROM slang_scan_checkpoint WHERE run_id=?",
        (run_id,),
    ):
        payload = json.loads(str(raw))
        unigrams.update({str(key): int(value) for key, value in payload.get("unigrams", {}).items()})
        bigrams.update({str(key): int(value) for key, value in payload.get("bigrams", {}).items()})
        ngrams.update({str(key): int(value) for key, value in payload.get("ngrams", {}).items()})
        total_characters += int(character_count)
    total_documents = int(
        db.execute(
            "SELECT coalesce(sum(document_count),0) FROM slang_scan_checkpoint WHERE run_id=?",
            (run_id,),
        ).fetchone()[0]
    )
    total_authors = int(
        db.execute(
            "SELECT count(DISTINCT trim(author_hash)) FROM community_document "
            "WHERE lower(trim(coalesce(author_hash,''))) NOT IN ('','anon','anonymous','匿名')"
        ).fetchone()[0]
    )
    stat_rows = db.execute(
        """
        SELECT normalized_candidate,candidate_text,subject_entity_id,
               occurrence_count,document_count,author_count,stats_json
        FROM slang_candidate_subject_stat WHERE run_id=?
        ORDER BY normalized_candidate,subject_entity_id
        """,
        (run_id,),
    )
    ranked: list[dict[str, Any]] = []
    for candidate, group in groupby(stat_rows, key=lambda row: str(row[0])):
        rows = [tuple(row[1:]) for row in group]
        occurrence_count = sum(int(row[2]) for row in rows)
        document_count = sum(int(row[3]) for row in rows)
        if occurrence_count < config.minimum_global_occurrences or document_count < config.minimum_global_documents:
            continue
        authors: set[str] = set()
        left: Counter[str] = Counter()
        right: Counter[str] = Counter()
        contexts: dict[str, int] = {}
        slot_hits = 0
        bracketed = 0
        evidence: list[dict[str, Any]] = []
        subject_counts: list[tuple[str, int]] = []
        official_counts: dict[str, int] = {}
        display_text = str(rows[0][0])
        for row in rows:
            _, subject_id, _, subject_documents, _, raw_stats = row
            payload = json.loads(str(raw_stats))
            authors.update(str(value) for value in payload.get("authors", []))
            left.update({str(key): int(value) for key, value in payload.get("left", {}).items()})
            right.update({str(key): int(value) for key, value in payload.get("right", {}).items()})
            slot_hits += int(payload.get("slot_hits", 0))
            bracketed += int(payload.get("bracketed", 0))
            contexts[str(subject_id)] = int(subject_documents)
            subject_counts.append((str(subject_id), int(subject_documents)))
            official_counts[str(subject_id)] = int(payload.get("official_cooccurrence_documents", 0))
            for item in payload.get("evidence", []):
                if len(evidence) < config.evidence_limit:
                    evidence.append(dict(item))
        if len(authors) < config.minimum_global_authors:
            continue
        stats = CorpusStats(
            total_documents=total_documents,
            candidate_occurrences={candidate: occurrence_count},
            candidate_documents={candidate: document_count},
            left_neighbors={candidate: dict(left)},
            right_neighbors={candidate: dict(right)},
            authors={candidate: tuple(sorted(authors))},
            contexts={candidate: dict(sorted(contexts.items()))},
            slot_hits={candidate: slot_hits},
            bracketed_occurrences={candidate: bracketed},
            unigram_counts=unigrams,
            bigram_counts=bigrams,
            total_characters=total_characters,
            total_authors=total_authors,
            ngram_counts=ngrams,
        )
        score = score_candidate(candidate, stats, buckets=ScoreBuckets(low=0.25, medium=0.42, high=0.72))
        threshold = (
            config.single_char_score_threshold if len(candidate) == 1 else config.candidate_score_threshold
        )
        if score.score < threshold:
            continue
        subject_counts.sort(key=lambda item: (-item[1], item[0]))
        concentration = subject_counts[0][1] / max(1, document_count)
        adjusted = score.score * (0.75 + 0.25 * concentration)
        ranked.append(
            {
                "candidate": candidate,
                "display_text": display_text,
                "occurrence_count": occurrence_count,
                "document_count": document_count,
                "authors": authors,
                "score": score,
                "adjusted": adjusted,
                "threshold": threshold,
                "subject_counts": subject_counts,
                "official_counts": official_counts,
                "concentration": concentration,
                "evidence": evidence,
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["adjusted"],
            -item["document_count"],
            -len(item["authors"]),
            item["candidate"],
        )
    )
    db.execute("DELETE FROM slang_link_candidate WHERE candidate_id IN (SELECT id FROM slang_candidate WHERE run_id=?)", (run_id,))
    db.execute("DELETE FROM slang_evidence WHERE candidate_id IN (SELECT id FROM slang_candidate WHERE run_id=?)", (run_id,))
    db.execute("DELETE FROM slang_candidate WHERE run_id=?", (run_id,))
    candidate_count = 0
    link_count = 0
    for item in ranked[: config.candidate_limit]:
        candidate = item["candidate"]
        bucket = (
            "single_char"
            if len(candidate) == 1
            else ("self_reference" if item["concentration"] >= 0.5 else "cross_work")
        )
        candidate_id = stable_id("slang", run_id, candidate)
        feature_json = json.dumps(
            {
                **item["score"].features.as_dict(),
                "normalized_features": dict(item["score"].normalized_features),
                "weighted_components": dict(item["score"].weighted_components),
                "subject_concentration": item["concentration"],
                "subject_counts": item["subject_counts"][:20],
                "threshold_version": config.threshold_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        db.execute(
            "INSERT INTO slang_candidate VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                candidate_id,
                run_id,
                item["display_text"],
                candidate,
                bucket,
                item["occurrence_count"],
                item["document_count"],
                len(item["authors"]),
                feature_json,
                item["score"].score,
                item["threshold"],
                "shortlisted",
            ),
        )
        candidate_count += 1
        for evidence in item["evidence"]:
            db.execute(
                "INSERT OR IGNORE INTO slang_evidence VALUES(?,?,?,?,?,?,?,?)",
                (
                    stable_id(
                        "slang-evidence",
                        candidate_id,
                        evidence["document_id"],
                        evidence["start"],
                        evidence["end"],
                    ),
                    candidate_id,
                    evidence["document_id"],
                    evidence["start"],
                    evidence["end"],
                    evidence["context"],
                    evidence.get("author_hash", ""),
                    ",".join(evidence.get("slot_names", [])) or None,
                ),
            )
        top_subjects = [subject_id for subject_id, _ in item["subject_counts"][:3]]
        possible_targets = _candidate_existing_targets(db, candidate, top_subjects)
        official_names = {}
        for subject_id, _ in item["subject_counts"]:
            name = db.execute("SELECT canonical_name FROM entity WHERE id=?", (subject_id,)).fetchone()
            official_names[subject_id] = [str(name[0])] if name else []
        links = link_candidates(
            candidate,
            subject_anchors={subject: {"anchor_count": count, "document_count": count,
                                       "official_name_document_count": item["official_counts"][subject]}
                             for subject, count in item["subject_counts"]},
            official_names=official_names,
            vector_neighbors=config.vector_neighbors_by_candidate.get(candidate),
            author_breadth=len(item["authors"]),
        )
        link_by_id = {link.entity_id: link for link in links}
        for link in links:
            if db.execute("SELECT 1 FROM entity WHERE id=?", (link.entity_id,)).fetchone() is None:
                raise ValueError(f"frozen vector link references unknown entity: {link.entity_id}")
        targets = list(
            dict.fromkeys(
                [
                    *(link.entity_id for link in links),
                    *(possible_targets[:1]),
                    *(possible_targets[1:]),
                ]
            )
        )[:3]
        count_by_subject = dict(item["subject_counts"])
        for rank, entity_id in enumerate(targets, 1):
            if entity_id in count_by_subject:
                context_score = count_by_subject[entity_id] / max(1, item["document_count"])
            else:
                context_score = item["concentration"]
            link_score = min(1.0, item["score"].score * (0.65 + 0.35 * context_score))
            link = link_by_id.get(entity_id)
            if link is not None:
                link_score = link.score
            db.execute(
                """
                INSERT INTO slang_link_candidate(
                    id,candidate_id,entity_id,rank,link_score,link_bucket,
                    evidence_json,decision,decision_confidence
                ) VALUES(?,?,?,?,?,?,?,'pending',NULL)
                """,
                (
                    stable_id("slang-link", candidate_id, entity_id),
                    candidate_id,
                    entity_id,
                    rank,
                    link_score,
                    link.bucket if link is not None else bucket,
                    json.dumps(
                        {
                            "subject_counts": item["subject_counts"][:20],
                            "subject_concentration": item["concentration"],
                            "existing_candidate_alias": entity_id in possible_targets,
                            **(dict(link.evidence) if link is not None else {}),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            link_count += 1
    return {"candidates": candidate_count, "link_candidates": link_count}


def _reuse_subject_checkpoint(
    db: sqlite3.Connection, *, base_run_id: str, run_id: str,
    subject_id: str, documents: list[Document],
) -> bool:
    checkpoint = db.execute(
        "SELECT last_document_id,document_count,character_count,ngram_stats_json,completed_at "
        "FROM slang_scan_checkpoint WHERE run_id=? AND subject_entity_id=?",
        (base_run_id, subject_id),
    ).fetchone()
    if checkpoint is None or json.loads(checkpoint[3]).get("source_digest") != _documents_digest(documents):
        return False
    db.execute("INSERT INTO slang_scan_checkpoint VALUES(?,?,?,?,?,?,?)",
               (run_id, subject_id, *checkpoint))
    db.execute(
        "INSERT INTO slang_candidate_subject_stat "
        "SELECT ?,subject_entity_id,normalized_candidate,candidate_text,occurrence_count,"
        "document_count,author_count,stats_json FROM slang_candidate_subject_stat "
        "WHERE run_id=? AND subject_entity_id=?", (run_id, base_run_id, subject_id),
    )
    rows = db.execute(
        "SELECT document_id,subject_entity_id,src_entity_id,dst_entity_id,author_hash,"
        "src_mentions_json,dst_mentions_json,evidence_text FROM cooccurrence "
        "WHERE run_id=? AND subject_entity_id=? ORDER BY id", (base_run_id, subject_id),
    ).fetchall()
    db.executemany("INSERT INTO cooccurrence VALUES(?,?,?,?,?,?,?,?,?,?)", [
        (stable_id("cooccurrence", run_id, row[0], row[2], row[3]), run_id, *row) for row in rows
    ])
    return True


def discover_slang(
    database: Path,
    *,
    build_id: str,
    run_id: str,
    config: SlangPipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SlangPipelineConfig()
    cfg.validate()
    totals: Counter[str] = Counter()
    with closing(sqlite3.connect(database)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA cache_size=-262144")
        db.execute("PRAGMA temp_store=FILE")
        ensure_p1_schema(db)
        watermark = _source_watermark(db)
        reusable_base = None
        if cfg.incremental_base_run_id:
            if cfg.incremental_base_run_id == run_id:
                raise ValueError("incremental base must differ from target run")
            base = db.execute("SELECT config_json,source_watermark,processor_version,status "
                              "FROM slang_run WHERE id=?", (cfg.incremental_base_run_id,)).fetchone()
            if base is None or base[3] != "completed":
                raise ValueError("incremental base must be a completed run")
            old_config = json.loads(base[0])
            new_config = asdict(cfg)
            old_config.pop("incremental_base_run_id", None)
            new_config.pop("incremental_base_run_id", None)
            if old_config != new_config or base[2] != SLANG_PIPELINE_VERSION:
                raise ValueError("incremental base processor/config mismatch; use a fresh full run")
            if _source_watermark_payload(base[1]).get("alias_digest") == json.loads(watermark)["alias_digest"]:
                reusable_base = cfg.incremental_base_run_id
        existing = db.execute(
            """
            SELECT build_id,processor_version,threshold_version,source_watermark,
                   input_digest,config_json,status
            FROM slang_run WHERE id=?
            """,
            (run_id,),
        ).fetchone()
        if existing is None:
            db.execute(
                """
                INSERT INTO slang_run(
                    id,build_id,processor_version,threshold_version,source_watermark,
                    input_digest,config_json,started_at,status
                ) VALUES(?,?,?,?,?,?,?,?, 'building')
                """,
                (
                    run_id,
                    build_id,
                    SLANG_PIPELINE_VERSION,
                    cfg.threshold_version,
                    watermark,
                    stable_id("slang-input", watermark, asdict(cfg)),
                    json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
            db.commit()
        else:
            if "content_digest" not in _source_watermark_payload(str(existing[3])):
                # Count/length-only legacy watermarks cannot prove unchanged
                # inputs. Upgrade only if every saved subject has a matching
                # content-addressed checkpoint (never bless unknown old data).
                for subject_id, raw in db.execute(
                    "SELECT subject_entity_id,ngram_stats_json FROM slang_scan_checkpoint WHERE run_id=?",
                    (run_id,),
                ):
                    documents = _subject_documents(db, str(subject_id), limit=cfg.max_documents_per_subject)
                    checkpoint = json.loads(raw)
                    if (checkpoint.get("source_digest") != _documents_digest(documents)
                            or checkpoint.get("alias_digest") != json.loads(watermark)["alias_digest"]):
                        raise ValueError("cannot resume legacy watermark: subject content identity is unverified")
            migration = _validate_existing_run(
                existing[:6],
                build_id=build_id,
                config=cfg,
                source_watermark=watermark,
            )
            if migration is not None:
                upgraded_watermark, upgraded_input_digest, upgraded_config = migration
                db.execute(
                    "UPDATE slang_run SET source_watermark=?,input_digest=?,config_json=? WHERE id=?",
                    (upgraded_watermark, upgraded_input_digest, upgraded_config, run_id),
                )
                db.commit()
        if existing is not None and str(existing[6]) == "completed":
            return slang_run_report(database, run_id=run_id)
        automaton = _load_alias_automaton(db, cfg)
        completed = {
            str(row[0])
            for row in db.execute(
                "SELECT subject_entity_id FROM slang_scan_checkpoint WHERE run_id=?", (run_id,)
            )
        }
        all_subjects = [
            str(row[0])
            for row in db.execute(
                "SELECT DISTINCT subject_entity_id FROM community_document ORDER BY subject_entity_id"
            )
        ]
        if cfg.max_subjects is not None:
            all_subjects = all_subjects[: cfg.max_subjects]
        subjects = [subject for subject in all_subjects if subject not in completed]
        for index, subject_id in enumerate(subjects, 1):
            documents = _subject_documents(
                db,
                subject_id,
                limit=cfg.max_documents_per_subject,
            )
            if not documents:
                continue
            if reusable_base and _reuse_subject_checkpoint(
                db, base_run_id=reusable_base, run_id=run_id, subject_id=subject_id,
                documents=documents,
            ):
                totals["subjects_reused"] += 1
                db.commit()
                continue
            totals.update(
                _scan_subject(
                    db,
                    run_id=run_id,
                    documents=documents,
                    automaton=automaton,
                    config=cfg,
                    alias_digest=json.loads(watermark)["alias_digest"],
                )
            )
            db.execute(
                "UPDATE slang_run SET source_watermark=? WHERE id=?",
                (
                    json.dumps(
                        {
                            "source": json.loads(watermark),
                            "last_subject_entity_id": subject_id,
                            "subjects_completed": len(completed) + index,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    run_id,
                ),
            )
            db.commit()
        final = _aggregate_candidates(db, run_id=run_id, build_id=build_id, config=cfg)
        totals.update(final)
        output_digest = stable_id(
            "slang-output",
            run_id,
            *db.execute(
                "SELECT normalized_candidate,printf('%.12g',score),status "
                "FROM slang_candidate WHERE run_id=? ORDER BY normalized_candidate",
                (run_id,),
            ).fetchall(),
        )
        db.execute(
            "UPDATE slang_run SET output_digest=?,completed_at=?,status='completed' WHERE id=?",
            (output_digest, utc_now(), run_id),
        )
        db.commit()
    return {"run_id": run_id, "build_id": build_id, **dict(totals), **final}


def slang_run_report(database: Path, *, run_id: str) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        row = db.execute(
            "SELECT build_id,status,source_watermark,input_digest,output_digest,config_json "
            "FROM slang_run WHERE id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown slang run: {run_id}")
        counts = {
            "subjects_completed": int(
                db.execute(
                    "SELECT count(*) FROM slang_scan_checkpoint WHERE run_id=?", (run_id,)
                ).fetchone()[0]
            ),
            "subject_candidate_stats": int(
                db.execute(
                    "SELECT count(*) FROM slang_candidate_subject_stat WHERE run_id=?", (run_id,)
                ).fetchone()[0]
            ),
            "candidates": int(
                db.execute("SELECT count(*) FROM slang_candidate WHERE run_id=?", (run_id,)).fetchone()[0]
            ),
            "cooccurrences": int(
                db.execute("SELECT count(*) FROM cooccurrence WHERE run_id=?", (run_id,)).fetchone()[0]
            ),
            "link_candidates": int(
                db.execute(
                    "SELECT count(*) FROM slang_link_candidate WHERE candidate_id IN "
                    "(SELECT id FROM slang_candidate WHERE run_id=?)",
                    (run_id,),
                ).fetchone()[0]
            ),
        }
    return {
        "run_id": run_id,
        "build_id": str(row[0]),
        "status": str(row[1]),
        "source_watermark": json.loads(str(row[2])),
        "input_digest": str(row[3]),
        "output_digest": None if row[4] is None else str(row[4]),
        "config": json.loads(str(row[5])),
        **counts,
    }


__all__ = [
    "DEFAULT_THRESHOLD_VERSION",
    "SLANG_PIPELINE_VERSION",
    "SlangPipelineConfig",
    "discover_slang",
    "slang_run_report",
]
