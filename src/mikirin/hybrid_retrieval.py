"""Opt-in A3.5 retrieval. No database writes and no implicit model construction.

Public modes: fts -> anchored -> evidence -> hybrid. These are experiments, not
an approval to replace KnowledgeQueries. Scores are ranking signals, not truth.
The optional disambiguator is callable(query, candidates) -> candidate ID | None;
it can only reorder the supplied closed set, never introduce evidence or scope.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mikirin.episode_ref import EpisodeTable
from mikirin.query import KnowledgeQueries, _follow_redirects, EntityHit, _topic_fragments
from mikirin.query_intent import classify_query
from mikirin.semantic_search import SemanticEpisodeSearch
from mikirin.vector_bundle import VectorBundlePublisher
from mikirin.vector_index import VectorIndex
from mikirin.vector_sources import COMMUNITY_CHUNK_SCHEMA_VERSION

RETRIEVAL_MODES = ("fts", "anchored", "evidence", "hybrid")
RETRIEVAL_VERSION = "a35-experimental-v1"
SCORE_WEIGHTS = {"episode_card": .50, "best_chunk": .30,
                 "top3_chunk_mean": .10, "lexical": .10}


def _unit(value: Any) -> float:
    value = float(value or 0)
    return max(0., min(1., value)) if math.isfinite(value) else 0.


def _open(database: Path) -> sqlite3.Connection:
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _has(db: sqlite3.Connection, table: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is not None


def _json(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


class HybridSemanticSearch(SemanticEpisodeSearch):
    """Validated card + chunk bundle; one query embedding, two scoped searches.

    Construct explicitly only for the experimental hybrid option. ``close()``
    releases both indexes. Missing/mismatched chunks disable the whole adapter.
    """

    def __init__(self, root: Path, database: Path) -> None:
        self.chunk_index: VectorIndex | None = None
        self.bundle_fingerprint: str | None = None
        super().__init__(Path(root), Path(database))
        if not self.available:
            return
        try:
            bundle = VectorBundlePublisher(Path(root) / "data/vector_index").current(
                expected_embedding_fingerprint=self.config.fingerprint())
            if bundle is None or not self._database_matches_bundle(bundle.source_database):
                raise ValueError("content-database-mismatch")
            # Guard against a pointer switch between parent and child initialization.
            if bundle.indexes["episode_cards"]["index_fingerprint"] != self.index.index_fingerprint:
                raise ValueError("bundle-changed")
            self.chunk_index = VectorIndex(bundle.directory / str(bundle.indexes["community_chunks"]["path"]))
            schemas = self.chunk_index.index_contract.get("schema_versions", {})
            if schemas.get("community_chunk") != COMMUNITY_CHUNK_SCHEMA_VERSION:
                raise ValueError("community-chunk-schema-mismatch")
            self.bundle_fingerprint = bundle.bundle_fingerprint
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            self.available = False
            self.fallback_reason = f"dual-index-unavailable:{type(error).__name__}"
            self.close()

    def status(self) -> dict[str, Any]:
        return {**super().status(), "bundle_fingerprint": self.bundle_fingerprint,
                "index_fingerprints": {
                    "episode_cards": self.index.index_fingerprint if self.index else None,
                    "community_chunks": self.chunk_index.index_fingerprint if self.chunk_index else None}}

    def search_dual(self, query: str, *, top_k: int = 40,
                    subject_entity_id: str | None = None,
                    episode_entity_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        if not self.available:
            return {"cards": [], "chunks": []}
        vector = self.client.embed_query(query)
        scope = {"subject_entity_id": subject_entity_id, "episode_entity_id": episode_entity_id}
        cards = self.index.search(vector, top=top_k, object_kinds={"episode_card"}, **scope)
        chunks = self.chunk_index.search(vector, top=top_k, object_kinds={"community_chunk"}, **scope)
        return {"cards": [{"episode_entity_id": key, "score": score} for _, key, score in cards],
                "chunks": [{"chunk_id": key, "score": score} for _, key, score in chunks]}

    def close(self) -> None:
        if self.chunk_index is not None:
            self.chunk_index.close()
            self.chunk_index = None
        super().close()
        self.index = None


def _scope_sql(subjects: set[str], episodes: set[str]) -> tuple[str, list[str]]:
    # Episode constraints are always intersected with subject constraints.
    clauses, args = [], []
    for column, values in (("subject_entity_id", subjects), ("episode_entity_id", episodes)):
        if values:
            clauses.append(f"c.{column} IN ({','.join('?' for _ in values)})")
            args.extend(sorted(values))
    return (" AND " + " AND ".join(clauses) if clauses else ""), args


_DOCUMENT_SELECT = """SELECT d.id AS document_id, d.title, d.content AS original,
    d.url, c.source_kind, c.subject_entity_id, c.episode_entity_id, c.author_hash,
    c.rating, c.attribution_required FROM source_document d
    JOIN community_document c ON c.document_id=d.id """


class HybridRetriever:
    """Read-only experimental retrieval, with explicitly injected model hooks.

    ``stance_classifier(query, candidates)`` returns a list of dictionaries with
    candidate_id, stance (supporting/dissenting/neutral), and claim_quote. The ID
    and quote are validated against that call's closed candidates. Callback
    labels always have model provenance, even if the response claims otherwise.
    The caller owns and closes the optional semantic adapter.
    """
    def __init__(self, database: Path | str, semantic: Any = None,
                 disambiguator: Any = None, *, stance_classifier: Any = None) -> None:
        self.database = Path(database).resolve()
        self.semantic = semantic
        self.disambiguator = disambiguator
        self.stance_classifier = stance_classifier

    def _entities(self, db: sqlite3.Connection, ids: list[str]) -> list[dict[str, Any]]:
        hits = []
        for key in ids:
            row = db.execute("SELECT id, entity_type, canonical_name FROM entity WHERE id=?", (key,)).fetchone()
            if row:
                hits.append(EntityHit(*row, "explicit"))
        return [h.to_dict() for h in _follow_redirects(db, hits)]

    def _parents(self, db: sqlite3.Connection, entities: list[dict[str, Any]]) -> tuple[set[str], set[str], dict[str, float]]:
        subjects, episodes, weights = set(), set(), {}
        for item in entities:
            key, kind = item["entity_id"], item["entity_type"]
            parents = []
            if kind == "subject":
                parents = [key]
            elif kind == "episode":
                row = db.execute("SELECT subject_entity_id FROM episodes WHERE entity_id=?", (key,)).fetchone()
                if row:
                    parents = [row[0]]
                    episodes.add(key)
            elif kind in ("character", "person"):
                table, column = (("rel_subject_character", "character_id") if kind == "character"
                                 else ("rel_subject_person", "person_id"))
                parents = [r[0] for r in db.execute(
                    f"SELECT DISTINCT subject_id FROM {table} WHERE {column}=? ORDER BY subject_id LIMIT 24", (key,))]
            for parent in parents:
                subjects.add(parent)
                weights[parent] = max(weights.get(parent, 0.), _unit(item.get("retrieval_weight", 1.)))
        return subjects, episodes, weights

    def _route(self, db: sqlite3.Connection, query: str, focus: Any,
               trace: list[dict[str, Any]], limit: int) -> tuple[list, set, set, dict, bool]:
        # Stable IDs, explicitly quoted titles and canonical names preceding an
        # episode/reaction expression are checked before conversational context.
        ids = re.findall(r"(?:bgm|uuid5):[\w-]+:[\w-]+", query)
        names = [query, *re.findall(r"[《「](.*?)[》」]", query)]
        names.append(re.split(r"第\s*[\d一二三四五六七八九十两]+[集话回]|\b[Ee][Pp]?\s*\d+|的|如何|怎么样|评价|口碑", query)[0].strip())
        for name in dict.fromkeys(names):
            ids.extend(r[0] for r in db.execute(
                "SELECT id FROM entity WHERE canonical_name=? ORDER BY id LIMIT ?", (name, limit)))
        entities = self._entities(db, list(dict.fromkeys(ids)))
        trace.append({"stage": "explicit", "entity_ids": [e["entity_id"] for e in entities]})
        subjects, episodes, weights = self._parents(db, entities)
        explicit = bool(entities)
        mentions = classify_query(query).episode_mentions
        f = asdict(focus) if is_dataclass(focus) else (focus or {})
        residual = query
        for mention in mentions:
            residual = residual.replace(mention.matched_text, "")
        bare_episode = bool(mentions) and not re.split(r"的|如何|怎么样|评价|标题|讲了|发生", residual)[0].strip(" ，,。？?")
        use_focus = (not explicit and _unit(f.get("confidence", 0)) >= .3
                     and f.get("turns_since_refresh", 0) <= 10
                     and (bare_episode or any(w in query for w in ("这部", "那部", "该部", "这集", "那集", "该集", "这一集", "那一集", "里面"))))
        if use_focus:
            entities = self._entities(db, [f.get("active_subject_id", "")])
            subjects, episodes, weights = self._parents(db, entities)
            episode_pronoun = any(w in query for w in ("这集", "那集", "该集", "这一集", "那一集"))
            if episode_pronoun and not mentions and subjects:
                row = db.execute("SELECT entity_id FROM episodes WHERE subject_entity_id=? AND "
                                 "(entity_id=? OR (? IS NULL AND sort=?)) ORDER BY type, entity_id LIMIT 1",
                                 (next(iter(subjects)), f.get("active_episode_id"),
                                  f.get("active_episode_id"), f.get("active_episode_sort"))).fetchone()
                if row:
                    episodes.add(row[0])
        trace.append({"stage": "focus", "used": bool(use_focus and subjects)})
        if not entities:
            entities = [h.to_dict() for h in KnowledgeQueries.search_entities(
                self.database, query, limit=limit)]
            subjects, episodes, weights = self._parents(db, entities)
        trace.append({"stage": "alias_fts", "entity_ids": [e["entity_id"] for e in entities]})
        if entities and all(e["match_kind"] in ("fts", "contained_fts") for e in entities):
            # Fuzzy names are candidates, not a hard anchor that may block the
            # global vector fallback for a remembered scene / "which show".
            subjects, episodes = set(), set()
            trace[-1]["weak_candidates_only"] = True
        unresolved = False
        if mentions and subjects:
            resolved = set()
            for subject in sorted(subjects):
                table = EpisodeTable.load(db, subject)
                for mention in mentions:
                    refs = ([table.resolve_number(mention.display_number)] if mention.kind == "episode"
                            else [ref for n, ref in table.main_by_number.items() if n <= mention.display_number])
                    resolved.update(ref.episode_id for ref in refs if ref is not None)
            episodes = resolved
            unresolved = not bool(resolved)
        elif mentions:
            unresolved = True
        return entities, subjects, episodes, weights, unresolved

    def _lexical(self, db: sqlite3.Connection, query: str, entities: list,
                 subjects: set, episodes: set, weights: dict, budget: int,
                 anchored: bool) -> list[dict[str, Any]]:
        scope, args = _scope_sql(subjects, episodes)
        fragments = _topic_fragments(query, entities)
        terms = [t for t in fragments if len(t) >= 3][:48]
        terms = list(dict.fromkeys([query, *terms]))
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms if len(term) >= 3)
        rows = []
        if expression and _has(db, "document_fts"):
            rows = db.execute(_DOCUMENT_SELECT + "JOIN document_fts f ON f.document_id=d.id "
                              "WHERE document_fts MATCH ?" + scope +
                              " ORDER BY bm25(document_fts), d.id LIMIT ?", (expression, *args, budget)).fetchall()
        result = []
        for rank, row in enumerate(rows):
            item = dict(row)
            item["lexical_score"] = (1. / (1. + rank / 8.)) * weights.get(item["subject_entity_id"], 1.)
            result.append(item)
        if anchored and subjects and len(result) < budget:
            seen = {d["document_id"] for d in result}
            rows = db.execute(_DOCUMENT_SELECT + "WHERE 1=1" + scope +
                              " ORDER BY d.id LIMIT ?", (*args, budget)).fetchall()
            for row in rows:
                if row["document_id"] not in seen:
                    result.append({**dict(row), "lexical_score": 0.})
        for item in result:
            content = item["original"]
            positions = [content.casefold().find(t.casefold()) for t in fragments if t.casefold() in content.casefold()]
            start = max(0, min(positions) - 80) if positions else 0
            item.update(content=content[start:start + 800], char_start=start,
                        char_end=min(len(content), start + 800), chunk_id=None)
        return result

    def _chunk(self, db: sqlite3.Connection, chunk_id: str, score: float) -> dict | None:
        row = db.execute("SELECT * FROM document_chunk WHERE id=?", (chunk_id,)).fetchone()
        if row is None:
            return None
        parent = db.execute(_DOCUMENT_SELECT + "WHERE d.id=?", (row["document_id"],)).fetchone()
        if parent is None:
            return None
        content, start, end = row["content"], row["char_start"], row["char_end"]
        if not content or parent["original"][start:end] != content:
            return None
        return {**dict(parent), "chunk_id": chunk_id, "content": content[:800],
                "char_start": start, "char_end": start + min(len(content), 800),
                "semantic_score": _unit(score), "lexical_score": 0.}

    def search(self, query: str, *, focus: Any = None, limit: int = 8,
               mode: str = "hybrid") -> dict[str, Any]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer in [1, 100]")
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("query must contain 1..2000 characters")
        trace, missing = [], []
        intent = classify_query(query)
        budget = min(400, max(40, limit * 8))
        with closing(_open(self.database)) as db:
            entities, subjects, episodes, weights, unresolved = [], set(), set(), {}, False
            if mode != "fts":
                entities, subjects, episodes, weights, unresolved = self._route(db, query, focus, trace, limit)
            else:
                trace.append({"stage": "flat_fts"})
            docs = [] if unresolved else self._lexical(db, query, entities, subjects, episodes, weights, budget, mode != "fts")
            if unresolved:
                missing.append("episode_scope_unresolved")
            cards, chunk_scores = {}, defaultdict(dict)
            semantic_status = {"available": False, "reason": "not-injected"}
            if self.semantic is not None:
                semantic_status = self.semantic.status() if hasattr(self.semantic, "status") else {
                    "available": getattr(self.semantic, "available", True), "reason": "injected-adapter"}
            ran_semantic = False
            direct_fact = bool(subjects) and bool(re.search(
                r"标题|播出日期|播出时间|什么时候播|叫什么|谁.{0,4}(分镜|导演)|(分镜|导演).{0,4}谁", query))
            if mode == "hybrid" and not unresolved and not direct_fact and semantic_status.get("available"):
                try:
                    if hasattr(self.semantic, "search_dual"):
                        # Multiple explicit works stay closed-scope; never let a
                        # global candidate budget crowd out one of their episodes.
                        scopes = [(s, e) for s in sorted(subjects) for e in sorted(episodes)] if episodes else [(s, None) for s in sorted(subjects)]
                        if not scopes:
                            scopes = [(None, None)]
                        batches = [self.semantic.search_dual(query, top_k=budget, subject_entity_id=s, episode_entity_id=e) for s, e in scopes]
                    else:
                        batches = [{"cards": [h.to_dict() if hasattr(h, "to_dict") else h for h in self.semantic.search(query, top_k=budget)], "chunks": []}]
                        missing.append("semantic_chunk_adapter_missing")
                    for batch in batches:
                        for hit in batch.get("cards", []):
                            key = hit["episode_entity_id"]
                            row = db.execute("SELECT subject_entity_id FROM episodes WHERE entity_id=?", (key,)).fetchone()
                            if row and (not subjects or row[0] in subjects) and (not episodes or key in episodes):
                                cards[key] = max(cards.get(key, 0.), _unit(hit["score"]))
                        for hit in batch.get("chunks", []):
                            item = self._chunk(db, hit["chunk_id"], hit["score"])
                            if item is None or (subjects and item["subject_entity_id"] not in subjects) or (episodes and item["episode_entity_id"] not in episodes):
                                continue
                            docs.append(item)
                            if item["episode_entity_id"]:
                                scores = chunk_scores[item["episode_entity_id"]]
                                scores[item["chunk_id"]] = max(scores.get(item["chunk_id"], 0.), item["semantic_score"])
                    ran_semantic = True
                except Exception as error:
                    # A remote failure must not discard valid local evidence.
                    missing.append(f"semantic_failed:{type(error).__name__}")
            if mode == "hybrid" and not ran_semantic and not direct_fact:
                missing.append("semantic_unavailable")
            trace.append({"stage": "semantic", "used": ran_semantic, "status": semantic_status,
                          "skip_reason": "anchored_factual_query" if direct_fact else None})
            lexical = defaultdict(float)
            for item in docs:
                if item["episode_entity_id"]:
                    lexical[item["episode_entity_id"]] = max(lexical[item["episode_entity_id"]], item["lexical_score"])
            ranked = []
            for key in sorted(set(cards) | set(chunk_scores) | set(lexical) | episodes):
                row = db.execute("SELECT ep.*, e.canonical_name AS subject_name FROM episodes ep JOIN entity e ON e.id=ep.subject_entity_id WHERE ep.entity_id=?", (key,)).fetchone()
                if row is None:
                    continue
                scores = sorted(chunk_scores[key].values(), reverse=True)[:3]
                components = {"episode_card": cards.get(key, 0.), "best_chunk": max(scores, default=0.),
                              "top3_chunk_mean": sum(scores) / len(scores) if scores else 0.,
                              "lexical": lexical[key]}
                score = sum(SCORE_WEIGHTS[k] * v for k, v in components.items()) if mode == "hybrid" else components["lexical"]
                ranked.append({"episode_entity_id": key, "subject_entity_id": row["subject_entity_id"],
                               "subject_name": row["subject_name"], "episode_sort": row["sort"],
                               "episode_name": row["name_cn"] or row["name"], "score": score,
                               "components": components, "matching_description": row["description"] or row["name_cn"] or row["name"] or ""})
            ranked.sort(key=lambda r: (-r["score"], r["episode_entity_id"]))
            disambiguation = {"stage": "disambiguation", "used": False}
            if mode == "hybrid" and self.disambiguator is not None and len(ranked) > 1:
                candidates = copy.deepcopy(ranked[:limit])
                try:
                    selected = self.disambiguator(query, candidates)
                    allowed = {c["episode_entity_id"] for c in ranked[:limit]}
                    if isinstance(selected, str) and selected in allowed:
                        ranked.sort(key=lambda r: r["episode_entity_id"] != selected)
                        disambiguation.update(used=True, selected=selected)
                    elif selected is not None:
                        missing.append("disambiguator_outside_closed_candidates")
                except Exception as error:
                    missing.append(f"disambiguator_failed:{type(error).__name__}")
            trace.append(disambiguation)
            # Preserve score order, prefer semantic excerpts over duplicate full
            # document excerpts, and never count repeated index hits as top-three.
            docs.sort(key=lambda d: (-max(d.get("semantic_score", 0.), d["lexical_score"]), d["document_id"], d.get("chunk_id") or ""))
            unique = {}
            for item in docs:
                identity = (item["document_id"], item["content"])
                unique.setdefault(identity, item)
            docs = list(unique.values())
            scope_subjects = subjects or {r["subject_entity_id"] for r in ranked[:limit]} or {d["subject_entity_id"] for d in docs[:limit]}
            pack = self._evidence(db, docs, scope_subjects, episodes, query, limit) if mode in ("evidence", "hybrid") else None
            selected_docs = pack.pop("_documents") if pack is not None else docs[:limit]
            clean_docs = [{k: v for k, v in d.items() if k != "original"} for d in selected_docs]
            by_subject = {}
            for ep in ranked:
                ep["supporting"] = [d for d in clean_docs if d["episode_entity_id"] == ep["episode_entity_id"]]
                subject = by_subject.setdefault(ep["subject_entity_id"], {"subject_entity_id": ep["subject_entity_id"], "subject_name": ep["subject_name"], "score": 0., "episodes": []})
                subject["episodes"].append(ep)
                subject["score"] = max(subject["score"], ep["score"])
            # Subject-only comments/reviews remain evidence, never fake episodes.
            for key in sorted(scope_subjects):
                if key not in by_subject:
                    row = db.execute("SELECT canonical_name FROM entity WHERE id=?", (key,)).fetchone()
                    by_subject[key] = {"subject_entity_id": key, "subject_name": row[0] if row else "", "score": max((max(d.get("semantic_score", 0.), d["lexical_score"]) for d in docs if d["subject_entity_id"] == key), default=0.), "episodes": []}
            subject_results = sorted(by_subject.values(), key=lambda s: (-s["score"], s["subject_entity_id"]))[:limit]
            if disambiguation["used"] and ranked:
                subject_results.sort(key=lambda s: s["subject_entity_id"] != ranked[0]["subject_entity_id"])
            for subject in subject_results:
                subject["episodes"] = subject["episodes"][:limit]
            if not entities:
                entities = self._entities(db, [s["subject_entity_id"] for s in subject_results])
            # Side-builds can retain their source's historical build rows. SQL
            # ordering (including timestamps) is not a publication pointer.
            builds = [dict(row) for row in db.execute("SELECT * FROM build ORDER BY id")] if _has(db, "build") else []
            schema_metadata = dict(db.execute("SELECT key, value FROM schema_metadata ORDER BY key")) if _has(db, "schema_metadata") else {}
            return {"query": query, "mode": mode, "experimental": True, "human_approved": False,
                    "intent": {"kind": intent.kind, "aspects": list(intent.aspects)},
                    "entities": entities, "documents": clean_docs, "episodes": ranked[:limit],
                    "semantic_episodes": ranked[:limit] if ran_semantic else [],
                    "subjects": subject_results, "subject_entity_ids": sorted(scope_subjects),
                    "evidence_pack": pack, "trace": trace, "missing": missing,
                    "provenance": {"retrieval_version": RETRIEVAL_VERSION, "database": str(self.database),
                                   "builds": builds,
                                   "build": builds[0] if len(builds) == 1 else None,
                                   "current_build_id": None,
                                   "build_identity_policy": "all-recorded-builds; current published build is not inferred; legacy build field is populated only for a single recorded row",
                                   "schema_metadata": schema_metadata,
                                   "sqlite_user_version": db.execute("PRAGMA user_version").fetchone()[0],
                                   "sqlite_schema_version": db.execute("PRAGMA schema_version").fetchone()[0],
                                   "semantic": semantic_status,
                                   "score_weights": SCORE_WEIGHTS.copy(), "subject_aggregation": "max_episode_score"}}

    def _evidence(self, db: sqlite3.Connection, docs: list[dict], subjects: set,
                  episodes: set, query: str, limit: int) -> dict[str, Any]:
        pack = {key: [] for key in ("official-facts", "scope-summary", "supporting", "dissenting", "deep-analysis")}
        missing = []
        evaluation = any(w in query for w in ("评价", "口碑", "争议", "分歧", "好看", "怎么样", "如何", "翻车", "差", "烂"))
        candidates = [d for d in docs if d.get("url") and d.get("author_hash") and d["content"] and d["original"][d["char_start"]:d["char_end"]] == d["content"]]
        if len(candidates) < len(docs):
            missing.append("uncitable_or_unknown_author_candidates_excluded")
        candidates = [dict(d, candidate_id=("chunk:" + d["chunk_id"] if d.get("chunk_id") else
                       f"document:{d['document_id']}:{d['char_start']}:{d['char_end']}")) for d in candidates]
        labels = {}
        if self.stance_classifier is not None and candidates:
            closed = {d["candidate_id"]: d for d in candidates[:40]}
            payload = [{k: d.get(k) for k in ("candidate_id", "document_id", "chunk_id", "content", "url", "author_hash")}
                       for d in closed.values()]
            try:
                response = self.stance_classifier(query, copy.deepcopy(payload))
                if not isinstance(response, list):
                    raise ValueError("stance labels must be a list")
                conflicts = set()
                for label in response:
                    if not isinstance(label, dict):
                        missing.append("invalid_stance_label")
                        continue
                    key, stance, quote = label.get("candidate_id"), label.get("stance"), label.get("claim_quote")
                    if not isinstance(key, str) or key not in closed or stance not in ("supporting", "dissenting", "neutral") or not isinstance(quote, str) or not quote.strip() or quote not in closed[key]["content"]:
                        missing.append("invalid_stance_label")
                        continue
                    normalized = {"stance": stance, "claim_quote": quote, "label_source": "model",
                                  "quote_verified": True, "human_verified": False}
                    if key in labels and labels[key] != normalized:
                        conflicts.add(key)
                    else:
                        labels[key] = normalized
                for key in conflicts:
                    labels.pop(key, None)
                if conflicts:
                    missing.append("conflicting_stance_labels")
            except Exception as error:
                missing.append(f"stance_classifier_failed:{type(error).__name__}")
        if evaluation:
            candidates.sort(key=lambda d: 0 if labels.get(d["candidate_id"], {}).get("stance") == "dissenting" else 1)
        chosen = []
        # Search downward for a feasible *actual* final size. A capacity-based
        # quota alone would incorrectly allow one review to be 100% of a pack.
        for size in range(min(limit, len(candidates)), 0, -1):
            authors, reviews, identities = Counter(), Counter(), set()
            trial = []
            for d in candidates:
                identity = (d["document_id"], d["content"])
                if identity in identities or authors[d["author_hash"]] >= 2:
                    continue
                if d["source_kind"] == "bgm_review" and reviews[d["document_id"]] + 1 > math.floor(.4 * size):
                    continue
                trial.append(d)
                identities.add(identity)
                authors[d["author_hash"]] += 1
                if d["source_kind"] == "bgm_review":
                    reviews[d["document_id"]] += 1
                if len(trial) == size:
                    break
            if len(trial) == size:
                chosen = trial
                break
        if len(chosen) < min(limit, len(candidates)):
            missing.append("insufficient_diverse_evidence_for_author_or_review_caps")
        for d in chosen:
            item = {k: v for k, v in d.items() if k != "original"}
            item.update(quote=d["content"], citation_verified=True)
            label = labels.get(d["candidate_id"])
            if label:
                item.update(label)
                d.update(label)
            bucket = "supporting"
            if label and label["stance"] == "dissenting":
                bucket = "dissenting"
            elif d["source_kind"] == "bgm_review":
                bucket = "deep-analysis"
            elif label and label["stance"] == "neutral":
                bucket = "neutral"
                pack.setdefault(bucket, [])
            pack[bucket].append(item)
        if evaluation:
            if not pack["dissenting"]:
                missing.append("claim_level_dissent_missing")
                if any(label["stance"] == "dissenting" for label in labels.values()):
                    missing.append("dissent_unavailable_under_evidence_caps")
        for key in sorted(episodes or subjects):
            row = db.execute("SELECT canonical_name FROM entity WHERE id=?", (key,)).fetchone()
            source = db.execute("SELECT source_url FROM source_entity WHERE entity_id=? AND source_url IS NOT NULL AND source_url!='' ORDER BY source, id LIMIT 1", (key,)).fetchone()
            if row and source:
                table, columns = (("episodes", "name,name_cn,sort,type,airdate,description") if key in episodes
                                  else ("subjects", "name,name_cn,date,platform,summary"))
                facts = db.execute(f"SELECT {columns} FROM {table} WHERE entity_id=?", (key,)).fetchone()
                pack["official-facts"].append({"entity_id": key, "text": row[0],
                    "fields": dict(facts) if facts else {}, "url": source[0], "author_hash": "",
                    "author_status": "not_recorded_for_official_source", "kind": "catalog_fact"})
            if _has(db, "entity_fact"):
                for fact in db.execute("SELECT f.fact_kind, f.value_json, a.url FROM entity_fact f JOIN source_asset a ON a.id=f.source_asset_id WHERE f.entity_id=? AND a.url IS NOT NULL AND a.url!='' ORDER BY f.id LIMIT 8", (key,)):
                    pack["official-facts"].append({"entity_id": key, "fact_kind": fact[0], "value": _json(fact[1], fact[1]), "url": fact[2], "author_hash": "", "author_status": "not_recorded_for_official_source", "kind": "sourced_fact"})
        for subject in sorted(subjects):
            if _has(db, "community_report"):
                report = db.execute("SELECT reception_summary, source_digest, build_id FROM community_report WHERE subject_entity_id=?", (subject,)).fetchone()
                if report:
                    # Reports have no per-claim source mapping: expose explicitly
                    # as uncitable navigation metadata, never a source quotation.
                    pack.setdefault("report_signposts", []).append({"subject_entity_id": subject, "summary": report[0], "source_digest": report[1], "build_id": report[2], "citable": False})
            if _has(db, "discussion_theme"):
                clauses = " AND episode_entity_id IN (" + ",".join("?" for _ in episodes) + ")" if episodes else ""
                for theme in db.execute("SELECT theme_text, quotes_json FROM discussion_theme WHERE subject_entity_id=?" + clauses + " ORDER BY id LIMIT 12", (subject, *sorted(episodes))):
                    refs = _json(theme[1], [])
                    if not isinstance(refs, list):
                        continue
                    for ref in refs:
                        if not isinstance(ref, dict) or not isinstance(ref.get("quote"), str) or not ref["quote"]:
                            continue
                        match = next((d for d in chosen if d["document_id"] == ref.get("document_id") and ref["quote"] in d["content"]), None)
                        if match:
                            # Reference existing selected evidence; don't silently
                            # add more author/review contributions to the pack.
                            pack["scope-summary"].append({"theme": theme[0], "document_id": match["document_id"], "quote": ref["quote"], "url": match["url"], "author_hash": match["author_hash"], "evidence_reference_only": True})
        for section in ("official-facts", "scope-summary", "supporting", "deep-analysis"):
            if not pack[section]:
                missing.append(f"{section}_missing")
        pack.update(missing=missing, constraints={"author_max": 2, "single_review_max_fraction": .4,
                    "denominator": "selected_community_excerpts", "dissent_required": evaluation,
                    "stance_source": "model" if self.stance_classifier is not None else None,
                    "human_verified": False}, _documents=chosen)
        return pack
