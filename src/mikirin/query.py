from __future__ import annotations

import sqlite3
import json
import unicodedata
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mikirin.episode_ref import EpisodeTable
from mikirin.query_intent import ROUTE_QUOTAS, classify_query
from mikirin.sources.bgm_aliases import normalize_alias


def _read_connection(database: Path | str) -> sqlite3.Connection:
    """Enforce read-only queries for both native paths and file URI callers."""
    if isinstance(database, str) and database.startswith('file:'):
        parsed = urlsplit(database)
        parameters = dict(parse_qsl(parsed.query))
        parameters['mode'] = 'ro'
        uri = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(parameters), ''))
    else:
        uri = Path(database).resolve().as_uri() + '?mode=ro'
    return sqlite3.connect(uri, uri=True)


@dataclass(frozen=True, slots=True)
class EntityHit:
    entity_id: str
    entity_type: str
    canonical_name: str
    match_kind: str
    redirected_from: str | None = None
    retrieval_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeQueries:
    @staticmethod
    def search_entities(database: Path, query: str, *, limit: int = 10) -> list[EntityHit]:
        query = query.strip()
        if not query or limit < 1:
            return []
        normalized = normalize_alias(query)
        hits: list[EntityHit] = []
        seen: set[str] = set()
        with closing(_read_connection(database)) as db:
            exact_rows = db.execute(
                """
                SELECT e.id, e.entity_type, e.canonical_name,
                       CASE WHEN a.status='provisional' THEN 0.5 ELSE 1.0 END
                FROM entity_alias a
                JOIN entity e ON e.id=a.entity_id
                LEFT JOIN characters ch ON ch.entity_id=e.id
                LEFT JOIN score_snapshot sc ON sc.subject_entity_id=e.id
                WHERE a.normalized_alias=?
                  AND a.status NOT IN ('rejected', 'expired', 'invalidated')
                  AND (a.source_kind!='mined' OR a.status IN ('confirmed', 'provisional'))
                ORDER BY
                    CASE a.status
                        WHEN 'confirmed' THEN 0 WHEN 'active' THEN 1
                        WHEN 'provisional' THEN 2 WHEN 'candidate' THEN 3 ELSE 4
                    END,
                    CASE WHEN a.source_kind='mined' THEN 0 ELSE 1 END,
                    a.confidence * CASE WHEN a.status='provisional' THEN 0.5 ELSE 1.0 END DESC,
                    -- 同档别名歧义按人气裁决：炮姐的 8 个候选里御坂美琴
                    -- 收藏数碾压其余，人气是社区共识的免费代理
                    COALESCE(ch.collects, sc.score, 0) DESC,
                    CASE e.entity_type WHEN 'subject' THEN 0 WHEN 'character' THEN 1
                         WHEN 'person' THEN 2 ELSE 3 END,
                    e.id
                LIMIT ?
                """,
                (normalized, limit),
            )
            for entity_id, entity_type, name, weight in exact_rows:
                if entity_id in seen:
                    continue
                hits.append(EntityHit(entity_id, entity_type, name, "exact", retrieval_weight=weight))
                seen.add(entity_id)
            if len(hits) < limit:
                canonical_rows = db.execute(
                    """
                    SELECT id, entity_type, canonical_name
                    FROM entity
                    WHERE canonical_name=?
                    ORDER BY id LIMIT ?
                    """,
                    (query, limit),
                )
                for entity_id, entity_type, name in canonical_rows:
                    if entity_id in seen:
                        continue
                    hits.append(EntityHit(entity_id, entity_type, name, "canonical"))
                    seen.add(entity_id)
            if len(hits) < limit and len(normalized) >= 2:
                # Find aliases contained in a natural-language query with indexed equality
                # probes. A reverse `instr(query, alias)` scan takes tens of seconds on the
                # full alias table; longest-first probes stay sublinear and prefer specificity.
                max_width = min(len(normalized), 24)
                for width in range(max_width, 1, -1):
                    found_at_width = False
                    candidates = {
                        normalized[start : start + width]
                        for start in range(0, len(normalized) - width + 1)
                    }
                    for candidate in sorted(candidates):
                        contained_rows = db.execute(
                            """
                            SELECT e.id, e.entity_type, e.canonical_name,
                                   CASE WHEN a.status='provisional' THEN 0.5 ELSE 1.0 END
                            FROM entity_alias a JOIN entity e ON e.id=a.entity_id
                            WHERE a.normalized_alias=?
                              AND a.status NOT IN ('rejected', 'expired', 'invalidated')
                              AND (a.source_kind!='mined' OR a.status IN ('confirmed', 'provisional'))
                            ORDER BY
                              a.confidence * CASE WHEN a.status='provisional' THEN 0.5 ELSE 1.0 END DESC,
                              e.id LIMIT ?
                            """,
                            (candidate, limit * 2),
                        )
                        for entity_id, entity_type, name, weight in contained_rows:
                            if entity_id in seen:
                                continue
                            hits.append(
                                EntityHit(entity_id, entity_type, name, "contained_alias", retrieval_weight=weight)
                            )
                            seen.add(entity_id)
                            found_at_width = True
                            if len(hits) >= limit:
                                break
                        if len(hits) >= limit:
                            break
                    if not found_at_width and width >= 3:
                        for candidate in sorted(candidates):
                            phrase = '"' + candidate.replace('"', '""') + '"'
                            contained_fts_rows = db.execute(
                                """
                                SELECT entity_id, entity_type, canonical_name
                                FROM entity_fts
                                WHERE entity_fts MATCH ?
                                ORDER BY
                                    CASE entity_type WHEN 'subject' THEN 0 WHEN 'character' THEN 1
                                         WHEN 'person' THEN 1 ELSE 2 END,
                                    CASE WHEN entity_type='subject' THEN coalesce(
                                        (SELECT min(nullif(rank, 0)) FROM score_snapshot
                                         WHERE subject_entity_id=entity_fts.entity_id), 999999
                                    ) ELSE 0 END,
                                    bm25(entity_fts)
                                LIMIT ?
                                """,
                                (phrase, limit * 2),
                            )
                            for entity_id, entity_type, name in contained_fts_rows:
                                if entity_id in seen:
                                    continue
                                hits.append(
                                    EntityHit(entity_id, entity_type, name, "contained_fts")
                                )
                                seen.add(entity_id)
                                found_at_width = True
                                if len(hits) >= limit:
                                    break
                            if len(hits) >= limit:
                                break
                    if found_at_width or len(hits) >= limit:
                        break
            if len(hits) < limit and len(normalized) >= 3:
                phrase = '"' + query.replace('"', '""') + '"'
                fts_rows = db.execute(
                    """
                    SELECT entity_id, entity_type, canonical_name
                    FROM entity_fts
                    WHERE entity_fts MATCH ?
                    ORDER BY
                        CASE entity_type WHEN 'subject' THEN 0 WHEN 'character' THEN 1
                             WHEN 'person' THEN 1 ELSE 2 END,
                        CASE WHEN entity_type='subject' THEN coalesce(
                            (SELECT min(nullif(rank, 0)) FROM score_snapshot
                             WHERE subject_entity_id=entity_fts.entity_id), 999999
                        ) ELSE 0 END,
                        bm25(entity_fts)
                    LIMIT ?
                    """,
                    (phrase, limit * 2),
                )
                for entity_id, entity_type, name in fts_rows:
                    if entity_id in seen:
                        continue
                    hits.append(EntityHit(entity_id, entity_type, name, "fts"))
                    seen.add(entity_id)
                    if len(hits) >= limit:
                        break
            hits = _follow_redirects(db, hits)
        return hits

    @staticmethod
    def search_anime_context(
        database: Path,
        query: str,
        *,
        limit: int = 8,
        document_limit: int = 12,
        semantic: Any = None,
    ) -> dict[str, Any]:
        hits = KnowledgeQueries.search_entities(database, query, limit=limit)
        entities: list[dict[str, Any]] = []
        subject_ids: set[str] = set()
        with closing(_read_connection(database)) as db:
            for hit in hits:
                detail: dict[str, Any] = hit.to_dict()
                if hit.entity_type == "subject":
                    subject_ids.add(hit.entity_id)
                    row = db.execute(
                        """
                        SELECT s.name, s.name_cn, s.summary, s.date, s.platform,
                               s.tags_json, sc.score, sc.rank
                        FROM subjects s
                        LEFT JOIN score_snapshot sc ON sc.subject_entity_id=s.entity_id
                        WHERE s.entity_id=? ORDER BY sc.rowid DESC LIMIT 1
                        """,
                        (hit.entity_id,),
                    ).fetchone()
                    if row:
                        detail.update({
                            "name": row[0], "name_cn": row[1],
                            "summary": (row[2] or "")[:800], "date": row[3],
                            "platform": row[4], "tags": _safe_json(row[5], []),
                            "score": row[6], "rank": row[7],
                        })
                elif hit.entity_type == "character":
                    row = db.execute(
                        "SELECT summary FROM characters WHERE entity_id=?", (hit.entity_id,)
                    ).fetchone()
                    detail["summary"] = ((row[0] if row else "") or "")[:600]
                    works = db.execute(
                        """
                        SELECT e.id, e.canonical_name, r.role_type
                        FROM rel_subject_character r JOIN entity e ON e.id=r.subject_id
                        WHERE r.character_id=? ORDER BY r.role_type, e.canonical_name LIMIT 12
                        """,
                        (hit.entity_id,),
                    ).fetchall()
                    detail["works"] = [
                        {"subject_entity_id": row[0], "name": row[1], "role_type": row[2]}
                        for row in works
                    ]
                    subject_ids.update(str(row[0]) for row in works)
                elif hit.entity_type == "person":
                    row = db.execute(
                        "SELECT summary, career_json FROM persons WHERE entity_id=?", (hit.entity_id,)
                    ).fetchone()
                    if row:
                        detail["summary"] = (row[0] or "")[:600]
                        detail["career"] = _safe_json(row[1], [])
                    works = db.execute(
                        """
                        SELECT DISTINCT e.id, e.canonical_name, x.position
                        FROM rel_subject_person x JOIN entity e ON e.id=x.subject_id
                        WHERE x.person_id=? ORDER BY e.canonical_name LIMIT 12
                        """,
                        (hit.entity_id,),
                    ).fetchall()
                    detail["works"] = [
                        {"subject_entity_id": row[0], "name": row[1], "position": row[2]}
                        for row in works
                    ]
                    subject_ids.update(str(row[0]) for row in works)
                elif hit.entity_type == "episode":
                    row = db.execute(
                        """
                        SELECT ep.subject_entity_id, ep.sort, ep.type, ep.name, ep.name_cn,
                               e.canonical_name
                        FROM episodes ep JOIN entity e ON e.id=ep.subject_entity_id
                        WHERE ep.entity_id=?
                        """,
                        (hit.entity_id,),
                    ).fetchone()
                    if row:
                        subject_ids.add(str(row[0]))
                        detail.update({
                            "subject_entity_id": row[0], "episode_sort": row[1],
                            "episode_type": row[2], "name": row[3], "name_cn": row[4],
                            "subject_name": row[5],
                        })
                entities.append(detail)

            documents: list[dict[str, Any]] = []
            table_exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='community_document'"
            ).fetchone()
            chunk_table_exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_chunk'"
            ).fetchone()
            if table_exists and subject_ids and document_limit > 0:
                topic_fragments = _topic_fragments(query, entities)
                intent = classify_query(query)
                # 集数提及 → 作品内引用表映射（v4.1-2：不直接拿数字当 sort）
                resolved_episode_ids: set[str] = set()
                episode_mention_info: list[dict[str, Any]] = []
                for mention in intent.episode_mentions:
                    if mention.kind != "episode":
                        continue
                    for subject_id in subject_ids:
                        ref = EpisodeTable.load(db, subject_id).resolve_number(
                            mention.display_number
                        )
                        if ref is not None:
                            resolved_episode_ids.add(ref.episode_id)
                            episode_mention_info.append({
                                "display_number": mention.display_number,
                                "episode_entity_id": ref.episode_id,
                                "episode_sort": ref.sort,
                                "subject_entity_id": subject_id,
                            })
                # 语义兜底：锚点置信度判断（v4.1 路由：别名FTS 之后、LLM 之前）
                # 无任何 exact/canonical 命中 = 弱锚（垃圾实体比没有更糟）
                weak_hits = not any(
                    h.match_kind in ("exact", "canonical") for h in hits
                )
                semantic_hits: list[dict[str, Any]] = []
                if semantic is not None and getattr(semantic, "available", False):
                    need_fallback = weak_hits or (
                        intent.kind == "episode_reaction" and not episode_mention_info
                    )
                    if need_fallback:
                        for hit in semantic.search(query, top_k=6):
                            semantic_hits.append(hit.to_dict())
                        top = semantic_hits[0] if semantic_hits else None
                        if top and (not subject_ids or (weak_hits and top["score"] >= 0.45)):
                            # 弱锚且语义高分：用语义结果替换垃圾锚点
                            if weak_hits and top["score"] >= 0.45:
                                entities.clear()
                                subject_ids.clear()
                            subject_ids.add(top["subject_entity_id"])
                            resolved_episode_ids.add(top["episode_entity_id"])
                            episode_mention_info.append({
                                "display_number": None,
                                "episode_entity_id": top["episode_entity_id"],
                                "episode_sort": top["episode_sort"],
                                "subject_entity_id": top["subject_entity_id"],
                                "via": "semantic",
                            })
                placeholders = ",".join("?" for _ in subject_ids)
                candidates = db.execute(
                    f"""
                    SELECT d.id, c.source_kind, c.subject_entity_id, c.episode_entity_id,
                           ep.sort, d.title, d.content, d.url, c.rating,
                           c.posted_at_raw, c.attribution_required
                    FROM community_document c
                    JOIN source_document d ON d.id=c.document_id
                    LEFT JOIN episodes ep ON ep.entity_id=c.episode_entity_id
                    WHERE c.subject_entity_id IN ({placeholders})
                    ORDER BY
                        CASE c.source_kind
                            WHEN 'bgm_review' THEN 0
                            WHEN 'bgm_ep_comment' THEN 1
                            ELSE 2
                        END,
                        coalesce(c.rating, 0) DESC,
                        length(d.content) DESC
                    LIMIT ?
                    """,
                    (*sorted(subject_ids), max(document_limit * 20, 100)),
                ).fetchall()
                if resolved_episode_ids:
                    ep_marks = ",".join("?" for _ in resolved_episode_ids)
                    targeted = db.execute(
                        f"""
                        SELECT d.id, c.source_kind, c.subject_entity_id,
                               c.episode_entity_id, ep.sort, d.title, d.content,
                               d.url, c.rating, c.posted_at_raw, c.attribution_required
                        FROM community_document c
                        JOIN source_document d ON d.id=c.document_id
                        LEFT JOIN episodes ep ON ep.entity_id=c.episode_entity_id
                        WHERE c.episode_entity_id IN ({ep_marks})
                          AND c.source_kind='bgm_ep_comment'
                        ORDER BY coalesce(c.rating, 0) DESC, length(d.content) DESC
                        LIMIT ?
                        """,
                        (*sorted(resolved_episode_ids), max(document_limit * 10, 60)),
                    ).fetchall()
                    seen_ids = {row[0] for row in candidates}
                    candidates = list(candidates) + [
                        row for row in targeted if row[0] not in seen_ids
                    ]
                buckets: dict[str, list[dict[str, Any]]] = {
                    "bgm_review": [], "bgm_ep_comment": [], "bgm_subject_comment": []
                }
                for row in candidates:
                    full_content = str(row[6])
                    content = full_content[:1200]
                    if chunk_table_exists:
                        chunks = [str(value[0]) for value in db.execute(
                            "SELECT content FROM document_chunk WHERE document_id=? "
                            "ORDER BY chunk_index",
                            (row[0],),
                        )]
                        if chunks:
                            content = max(
                                chunks,
                                key=lambda value: sum(
                                    len(fragment) ** 2 for fragment in topic_fragments
                                    if fragment in _squash(value)
                                ),
                            )
                    searchable = _squash(f"{row[5] or ''} {full_content}")
                    lexical_score = sum(
                        len(fragment) ** 2 for fragment in topic_fragments
                        if fragment in searchable
                    )
                    item = {
                        "document_id": row[0], "source_kind": row[1],
                        "subject_entity_id": row[2], "episode_entity_id": row[3],
                        "episode_sort": row[4], "title": row[5],
                        "content": content, "truncated": len(full_content) > len(content),
                        "url": row[7], "rating": row[8], "posted_at_raw": row[9],
                        "attribution_required": bool(row[10]),
                        "lexical_score": lexical_score,
                    }
                    buckets.setdefault(str(row[1]), []).append(item)
                for bucket in buckets.values():
                    bucket.sort(
                        key=lambda item: (
                            int(item["lexical_score"]),
                            int(item["rating"] or 0),
                            len(str(item["content"])),
                        ),
                        reverse=True,
                    )
                # 语义锚定到具体集时按单集路由配额取料
                effective_kind = intent.kind
                if (
                    intent.kind == "general"
                    and any(m.get("via") == "semantic" for m in episode_mention_info)
                ):
                    effective_kind = "episode_reaction"
                quota_ep, quota_review, quota_subject = ROUTE_QUOTAS[effective_kind]
                scale = document_limit / (quota_ep + quota_review + quota_subject)
                quotas = {
                    "bgm_ep_comment": max(1, round(quota_ep * scale)),
                    "bgm_review": max(1, round(quota_review * scale)),
                    "bgm_subject_comment": max(1, round(quota_subject * scale)),
                }
                # 单集路由且已解析集数：该集的评论优先装满单集桶
                if resolved_episode_ids:
                    buckets["bgm_ep_comment"].sort(
                        key=lambda item: (
                            item["episode_entity_id"] not in resolved_episode_ids,
                            -int(item["lexical_score"]),
                            -int(item["rating"] or 0),
                        )
                    )
                for kind in ("bgm_ep_comment", "bgm_review", "bgm_subject_comment"):
                    documents.extend(buckets[kind][: quotas[kind]])
                if len(documents) < document_limit:
                    selected = {item["document_id"] for item in documents}
                    remaining = [
                        item for kind in buckets.values() for item in kind
                        if item["document_id"] not in selected
                    ]
                    documents.extend(remaining[: document_limit - len(documents)])

        return {
            "query": query,
            "intent": {
                "kind": intent.kind,
                "aspects": list(intent.aspects),
                "episode_mentions": episode_mention_info,
            },
            "entities": entities,
            "subject_entity_ids": sorted(subject_ids),
            "semantic_episodes": semantic_hits,
            "documents": documents,
        }


def _follow_redirects(db: sqlite3.Connection, hits: list[EntityHit]) -> list[EntityHit]:
    """命中被合并实体时跟随 entity_redirect 到规范实体（§1.4）。

    旧 UUID5 ID 直查返回规范实体并附 redirected_from；多个旧 ID 归一到同一
    规范实体时按首次出现去重。
    """
    redirect_map = {
        old: canonical
        for old, canonical in db.execute(
            "SELECT old_entity_id, canonical_entity_id FROM entity_redirect"
        )
    }
    if not redirect_map:
        return hits
    resolved: list[EntityHit] = []
    seen: set[str] = set()
    for hit in hits:
        terminal = hit.entity_id
        redirected_from: str | None = None
        if terminal in redirect_map:
            redirected_from = hit.entity_id
            guard: set[str] = set()
            while terminal in redirect_map and terminal not in guard:
                guard.add(terminal)
                terminal = redirect_map[terminal]
        if terminal in seen:
            continue
        seen.add(terminal)
        if redirected_from is None:
            resolved.append(hit)
            continue
        row = db.execute(
            "SELECT entity_type, canonical_name FROM entity WHERE id=?", (terminal,)
        ).fetchone()
        entity_type = str(row[0]) if row else hit.entity_type
        canonical_name = str(row[1]) if row else hit.canonical_name
        resolved.append(
            EntityHit(terminal, entity_type, canonical_name, hit.match_kind, redirected_from, hit.retrieval_weight)
        )
    return resolved


def _safe_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _squash(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _topic_fragments(query: str, entities: list[dict[str, Any]]) -> tuple[str, ...]:
    residual = _squash(query)
    names: set[str] = set()
    for entity in entities:
        for key in ("canonical_name", "name", "name_cn", "subject_name"):
            value = entity.get(key)
            if value:
                names.add(_squash(str(value)))
    for name in sorted(names, key=len, reverse=True):
        if len(name) >= 2:
            residual = residual.replace(name, "")
    for stop in (
        "怎么样", "怎么评价", "如何评价", "聊聊", "说说", "是什么", "是谁",
        "这个", "那个", "一下", "动画", "动漫", "作品", "角色", "的",
    ):
        residual = residual.replace(stop, "")
    if len(residual) < 2:
        return ()
    fragments = {residual} if len(residual) <= 12 else set()
    for width in (4, 3, 2):
        if len(residual) < width:
            continue
        fragments.update(
            residual[start : start + width]
            for start in range(0, len(residual) - width + 1)
        )
    return tuple(sorted(fragments, key=lambda value: (-len(value), value)))
