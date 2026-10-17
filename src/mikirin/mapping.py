from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json
from mikirin.organize import ResolutionEvidence, load_resolution_evidence
from mikirin.sources.bgm_aliases import normalize_alias


DETERMINISTIC_MAPPING_VERSION = "deterministic-mapping-v1"


def _fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _work_key(value: str) -> str:
    result = _fingerprint(value)
    for suffix in ("系列角色", "系列", "动画角色", "游戏角色", "角色", "日本作品", "作品"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    return result


def deterministic_character_decision(
    evidence: ResolutionEvidence,
) -> dict[str, Any] | None:
    searchable = _fingerprint(
        " ".join((evidence.bangumi_name, evidence.bangumi_infobox, evidence.bangumi_summary))
    )
    names = {
        _fingerprint(value)
        for value in (evidence.moegirl_name, *evidence.moegirl_aliases)
        if len(_fingerprint(value)) >= 2
    }
    name_match = any(name in searchable for name in names)
    left_works = {
        _work_key(str(value))
        for value in evidence.moegirl_facts.get("subjects", [])
        if len(_work_key(str(value))) >= 3
    }
    right_works = {
        _work_key(value) for value in evidence.bangumi_subjects if len(_work_key(value)) >= 3
    }
    work_match = any(
        left in right or right in left for left in left_works for right in right_works
    )
    voice_actors = {
        _fingerprint(str(value))
        for value in evidence.moegirl_facts.get("voice_actors", [])
        if len(_fingerprint(str(value))) >= 2
    }
    voice_actor_match = any(value in searchable for value in voice_actors)
    if not work_match or not (name_match or voice_actor_match):
        return None
    matched = ["same_work"]
    if name_match:
        matched.append("same_full_name")
    if voice_actor_match:
        matched.append("same_voice_actor")
    confidence = 0.99 if name_match and voice_actor_match else (0.97 if name_match else 0.96)
    return {
        "candidate_id": evidence.candidate_id,
        "left_entity_id": evidence.left_entity_id,
        "right_entity_id": evidence.right_entity_id,
        "decision": "same",
        "confidence": confidence,
        "matched_evidence": matched,
        "conflicting_evidence": [],
        "reason": "唯一候选同时命中作品，并由姓名或声优提供独立证据",
        "decision_source": DETERMINISTIC_MAPPING_VERSION,
        "review_required": False,
    }


@dataclass(frozen=True, slots=True)
class DeterministicMappingReport:
    run_id: str
    moegirl_unique_candidates: int
    moegirl_auto_accepted: int
    moegirl_deferred: int
    atwiki_pages: int
    atwiki_exact_unique: int
    atwiki_ambiguous: int
    atwiki_unresolved: int
    output_directory: str


class DeterministicMapper:
    def __init__(self, *, database: Path, staging_root: Path) -> None:
        self.database = database
        self.staging_root = staging_root

    def run(self, *, run_id: str, limit: int | None = None) -> DeterministicMappingReport:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
            raise ValueError("invalid mapping run id")
        output = self.staging_root / "entity_mapping" / run_id
        if output.exists():
            raise FileExistsError(f"mapping run already exists: {output}")
        output.mkdir(parents=True)
        try:
            unique_ids = self._unique_moegirl_candidate_ids(limit=limit)
            evidence = load_resolution_evidence(
                self.database, limit=len(unique_ids) or 1, candidate_ids=unique_ids
            ) if unique_ids else []
            accepted = [
                decision for item in evidence
                if (decision := deterministic_character_decision(item)) is not None
            ]
            self._write_jsonl(output / "moegirl_decisions.jsonl", accepted)
            atwiki = self._map_atwiki(limit=limit)
            self._write_jsonl(output / "atwiki_links.jsonl", atwiki["accepted"])
            self._write_jsonl(output / "atwiki_ambiguous.jsonl", atwiki["ambiguous"])
            report = DeterministicMappingReport(
                run_id=run_id,
                moegirl_unique_candidates=len(evidence),
                moegirl_auto_accepted=len(accepted),
                moegirl_deferred=len(evidence) - len(accepted),
                atwiki_pages=atwiki["pages"],
                atwiki_exact_unique=len(atwiki["accepted"]),
                atwiki_ambiguous=len(atwiki["ambiguous"]),
                atwiki_unresolved=atwiki["unresolved"],
                output_directory=str(output),
            )
            atomic_write_json(output / "manifest.json", {
                **asdict(report),
                "mapping_version": DETERMINISTIC_MAPPING_VERSION,
                "database": str(self.database.resolve()),
                "created_at": datetime.now(UTC).isoformat(),
                "canonical_database_modified": False,
            })
            return report
        except BaseException:
            # Keep partial evidence files for diagnosis, but mark the run unmistakably incomplete.
            atomic_write_json(output / "FAILED.json", {
                "run_id": run_id,
                "mapping_version": DETERMINISTIC_MAPPING_VERSION,
                "failed_at": datetime.now(UTC).isoformat(),
            })
            raise

    def _unique_moegirl_candidate_ids(self, *, limit: int | None) -> list[str]:
        sql = """
            SELECT min(id)
            FROM resolution_candidate
            WHERE status='candidate' AND relation_kind='same_character'
            GROUP BY left_entity_id
            HAVING count(*)=1
            ORDER BY min(id)
        """
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            if limit < 1:
                raise ValueError("mapping limit must be positive")
            sql += " LIMIT ?"
            parameters = (limit,)
        uri = self.database.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            return [str(row[0]) for row in db.execute(sql, parameters)]

    def _map_atwiki(self, *, limit: int | None) -> dict[str, Any]:
        uri = self.database.resolve().as_uri() + "?mode=ro"
        accepted: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        unresolved = 0
        with sqlite3.connect(uri, uri=True) as db:
            sql = "SELECT id, title, url FROM source_document WHERE source='atwiki_sakuga' ORDER BY id"
            parameters: tuple[Any, ...] = ()
            if limit is not None:
                sql += " LIMIT ?"
                parameters = (limit,)
            pages = db.execute(sql, parameters).fetchall()
            for document_id, title, url in pages:
                hits = db.execute(
                    """
                    SELECT DISTINCT e.id, e.canonical_name
                    FROM entity_alias a JOIN entity e ON e.id=a.entity_id
                    WHERE a.normalized_alias=? AND e.entity_type='subject'
                    ORDER BY e.id LIMIT 20
                    """,
                    (normalize_alias(str(title)),),
                ).fetchall()
                base = {
                    "document_id": str(document_id),
                    "title": str(title),
                    "url": url,
                    "mapping_version": DETERMINISTIC_MAPPING_VERSION,
                }
                if len(hits) == 1:
                    accepted.append({
                        **base,
                        "subject_entity_id": str(hits[0][0]),
                        "subject_name": str(hits[0][1]),
                        "match_kind": "exact_unique_alias",
                        "confidence": 1.0,
                    })
                elif hits:
                    ambiguous.append({
                        **base,
                        "candidates": [
                            {"subject_entity_id": str(row[0]), "subject_name": str(row[1])}
                            for row in hits
                        ],
                        "match_kind": "exact_ambiguous_alias",
                    })
                else:
                    unresolved += 1
        return {
            "pages": len(pages),
            "accepted": accepted,
            "ambiguous": ambiguous,
            "unresolved": unresolved,
        }

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
