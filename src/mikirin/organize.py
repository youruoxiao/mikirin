from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from mikirin.atomic import atomic_write_json
from mikirin.cache import ModelCache, ModelInvocation


ORGANIZER_VERSION = "character-resolution-v2"
MATCH_CODES = {
    "same_work", "same_franchise_continuity", "same_full_name", "alias_match",
    "same_voice_actor", "same_gender", "same_birthday", "same_biography", "same_role",
}
CONFLICT_CODES = {
    "different_work", "different_adaptation", "different_franchise", "different_gender",
    "different_voice_actor", "different_birthday", "different_biography", "generic_name",
    "placeholder_name", "insufficient_work_evidence",
}
WORK_MATCH_CODES = {"same_work", "same_franchise_continuity"}
SYSTEM_PROMPT = """You are a conservative database entity-resolution reviewer for an anime knowledge base.
Judge whether each Moegirl record and Bangumi record represent the exact same fictional-character incarnation.
Use only supplied evidence. Never fill missing evidence with outside knowledge.

Identity rules:
1. The same mythological, historical, literary, or franchise archetype is not enough.
2. Characters from different adaptations, timelines, games, or unrelated works are different entities unless supplied evidence explicitly establishes character continuity.
3. Shared short names, surnames, titles, generic labels, and placeholders such as 主人公, 田中, ？, or ？？？ are never sufficient for same.
4. Compare work/franchise identity first, then full name/aliases, voice actor, gender, birthday, biography, and role.
5. If work evidence explicitly conflicts, choose different. If work identity cannot be established, choose uncertain.
6. A same decision requires a work match (same_work or same_franchise_continuity), at least one additional independent match, and no conflicting evidence.
7. Crossovers, sequels, and spinoffs count as continuity only when supplied evidence identifies the same character, not merely the same name.
8. Return exactly one decision for every candidate ID.
9. Candidates sharing the same left_entity_id are rival matches for one Moegirl record: choose at most one same per group, preferring the candidate whose work exactly matches the Moegirl record's work. Treat spinoff or alternate-incarnation lookalikes as different unless the evidence shows the Moegirl record specifically covers that incarnation.

Valid decisions: same, different, uncertain. Confidence is a JSON number from 0 to 1.
matched_evidence values: same_work, same_franchise_continuity, same_full_name, alias_match, same_voice_actor, same_gender, same_birthday, same_biography, same_role.
conflicting_evidence values: different_work, different_adaptation, different_franchise, different_gender, different_voice_actor, different_birthday, different_biography, generic_name, placeholder_name, insufficient_work_evidence.
Keep reason under 100 Chinese characters. Return one valid json object with no markdown.
Required json shape:
{"decisions":[{"candidate_id":"candidate:...","decision":"same","confidence":0.95,"matched_evidence":["same_work","same_full_name"],"conflicting_evidence":[],"reason":"简短的证据理由"}]}"""


class JsonCompleter(Protocol):
    model: str

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        reasoning_effort: str = "high",
    ) -> dict[str, Any]: ...

    def response_content(self, payload: dict[str, Any]) -> str: ...


@dataclass(frozen=True, slots=True)
class ResolutionEvidence:
    candidate_id: str
    left_entity_id: str
    right_entity_id: str
    source_confidence: float
    moegirl_name: str
    moegirl_aliases: list[str]
    moegirl_facts: dict[str, Any]
    bangumi_name: str
    bangumi_infobox: str
    bangumi_summary: str
    bangumi_subjects: list[str]

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore and dash")
    return run_id


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%SZ")


def load_resolution_evidence(
    database: Path, *, limit: int, candidate_ids: list[str] | None = None
) -> list[ResolutionEvidence]:
    if limit < 1:
        raise ValueError("limit must be positive")
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        candidates = []
        id_chunks: list[list[str | None]] = [None]  # None 表示不过滤
        if candidate_ids:
            id_chunks = [
                candidate_ids[index : index + 500]
                for index in range(0, len(candidate_ids), 500)
            ]
        remaining = limit
        for chunk in id_chunks:
            candidate_filter = ""
            parameters: list[Any] = []
            if chunk is not None:
                candidate_filter = " AND rc.id IN (" + ",".join("?" for _ in chunk) + ")"
                parameters.extend(chunk)
            parameters.append(remaining)
            rows = db.execute(
                f"""
                SELECT rc.id, rc.left_entity_id, rc.right_entity_id, rc.confidence,
                       left_entity.canonical_name AS left_name,
                       right_entity.canonical_name AS right_name,
                       right_character.infobox_raw, right_character.summary
                FROM resolution_candidate rc
                JOIN entity left_entity ON left_entity.id = rc.left_entity_id
                JOIN entity right_entity ON right_entity.id = rc.right_entity_id
                LEFT JOIN characters right_character ON right_character.entity_id = rc.right_entity_id
                WHERE rc.status = 'candidate' AND rc.relation_kind = 'same_character'
                {candidate_filter}
                ORDER BY rc.id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            candidates.extend(rows)
            remaining -= len(rows)
            if remaining <= 0:
                break
        if candidate_ids:
            order = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
            candidates.sort(key=lambda row: order[row["id"]])
        aliases_by_entity: dict[str, list[str]] = {}
        facts_by_entity: dict[str, dict[str, Any]] = {}
        subjects_by_entity: dict[str, list[str]] = {}

        def groups(values: list[str], size: int = 400) -> list[list[str]]:
            unique = list(dict.fromkeys(values))
            return [unique[index : index + size] for index in range(0, len(unique), size)]

        left_ids = [row["left_entity_id"] for row in candidates]
        right_ids = [row["right_entity_id"] for row in candidates]
        for entity_ids in groups(left_ids):
            marks = ",".join("?" for _ in entity_ids)
            for entity_id, alias_text in db.execute(
                f"""
                SELECT entity_id, alias_text FROM entity_alias
                WHERE entity_id IN ({marks})
                ORDER BY entity_id, confidence DESC, alias_text
                """,
                entity_ids,
            ):
                bucket = aliases_by_entity.setdefault(entity_id, [])
                if len(bucket) < 30:
                    bucket.append(alias_text)
            for entity_id, kind, raw_value in db.execute(
                f"""
                SELECT entity_id, fact_kind, value_json FROM entity_fact
                WHERE entity_id IN ({marks})
                """,
                entity_ids,
            ):
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                facts_by_entity.setdefault(entity_id, {})[kind] = value
        for entity_ids in groups(right_ids):
            marks = ",".join("?" for _ in entity_ids)
            for entity_id, subject_name in db.execute(
                f"""
                SELECT rel.character_id, COALESCE(NULLIF(s.name_cn, ''), s.name)
                FROM rel_subject_character rel
                JOIN subjects s ON s.entity_id = rel.subject_id
                WHERE rel.character_id IN ({marks})
                ORDER BY rel.character_id, s.entity_id
                """,
                entity_ids,
            ):
                bucket = subjects_by_entity.setdefault(entity_id, [])
                if len(bucket) < 20:
                    bucket.append(subject_name)

        result: list[ResolutionEvidence] = []
        for row in candidates:
            result.append(
                ResolutionEvidence(
                    candidate_id=row["id"],
                    left_entity_id=row["left_entity_id"],
                    right_entity_id=row["right_entity_id"],
                    source_confidence=float(row["confidence"]),
                    moegirl_name=row["left_name"],
                    moegirl_aliases=aliases_by_entity.get(row["left_entity_id"], []),
                    moegirl_facts=facts_by_entity.get(row["left_entity_id"], {}),
                    bangumi_name=row["right_name"],
                    bangumi_infobox=(row["infobox_raw"] or "")[:4_000],
                    bangumi_summary=(row["summary"] or "")[:2_000],
                    bangumi_subjects=subjects_by_entity.get(row["right_entity_id"], []),
                )
            )
    return result


def validate_decisions(payload: Any, expected_ids: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError("model JSON must contain a decisions array")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload["decisions"]:
        if not isinstance(item, dict):
            raise ValueError("every decision must be an object")
        candidate_id = item.get("candidate_id")
        decision = item.get("decision")
        confidence = item.get("confidence")
        reason = item.get("reason")
        matched = item.get("matched_evidence")
        conflicting = item.get("conflicting_evidence")
        if candidate_id not in expected_ids or candidate_id in seen:
            raise ValueError(f"unexpected or duplicate candidate_id: {candidate_id!r}")
        if decision not in {"same", "different", "uncertain"}:
            raise ValueError(f"invalid decision for {candidate_id}: {decision!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence is not numeric for {candidate_id}")
        if not 0 <= float(confidence) <= 1:
            raise ValueError(f"confidence is outside [0, 1] for {candidate_id}")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise ValueError(f"reason is missing or too long for {candidate_id}")
        if not isinstance(matched, list) or not all(isinstance(code, str) for code in matched):
            raise ValueError(f"matched_evidence must be a string array for {candidate_id}")
        if not isinstance(conflicting, list) or not all(
            isinstance(code, str) for code in conflicting
        ):
            raise ValueError(f"conflicting_evidence must be a string array for {candidate_id}")
        unknown_matched = [code for code in matched if code not in MATCH_CODES]
        unknown_conflicting = [code for code in conflicting if code not in CONFLICT_CODES]
        matched = list(dict.fromkeys(
            code if code in MATCH_CODES else "other_match" for code in matched
        ))
        conflicting = list(dict.fromkeys(
            code if code in CONFLICT_CODES else "other_conflict" for code in conflicting
        ))
        if decision == "same":
            if not WORK_MATCH_CODES.intersection(matched) or len(matched) < 2:
                raise ValueError(f"same decision lacks two-part identity evidence: {candidate_id}")
            if conflicting:
                raise ValueError(f"same decision contains conflicting evidence: {candidate_id}")
        if decision == "different" and not conflicting:
            raise ValueError(f"different decision lacks conflict evidence: {candidate_id}")
        seen.add(candidate_id)
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "confidence": float(confidence),
                "matched_evidence": matched,
                "conflicting_evidence": conflicting,
                "unrecognized_evidence": list(dict.fromkeys(
                    [*unknown_matched, *unknown_conflicting]
                )),
                "reason": reason.strip(),
            }
        )
    missing = set(expected_ids) - seen
    if missing:
        raise ValueError(f"model omitted candidate IDs: {sorted(missing)}")
    decisions.sort(key=lambda item: expected_ids.index(item["candidate_id"]))
    return {"decisions": decisions}


def review_policy(
    decision: dict[str, Any], evidence: ResolutionEvidence
) -> tuple[bool, list[str]]:
    flags: list[str] = []
    left = evidence.moegirl_name.strip()
    right = evidence.bangumi_name.strip()
    placeholders = {"?", "？", "??", "？？", "???", "？？？", "主人公", "主角"}
    if left in placeholders or right in placeholders:
        flags.append("placeholder_or_generic_name")
    if len(left) <= 1 or len(right) <= 1:
        flags.append("very_short_name")
    if decision["decision"] == "uncertain":
        flags.append("model_uncertain")
    if decision["confidence"] < 0.9:
        flags.append("low_confidence")
    if decision["decision"] == "same" and not WORK_MATCH_CODES.intersection(
        decision["matched_evidence"]
    ):
        flags.append("same_without_work_match")
    if decision["unrecognized_evidence"]:
        flags.append("unrecognized_evidence_code")
    return bool(flags), flags


@dataclass(slots=True)
class ResolutionOrganizer:
    database: Path
    staging_root: Path
    cache: ModelCache
    client: JsonCompleter | None
    model_name: str
    batch_size: int = 5
    max_tokens: int = 4096
    reasoning_effort: str = "high"
    workers: int = 1

    def run(
        self,
        *,
        limit: int,
        run_id: str,
        dry_run: bool = False,
        candidate_ids: list[str] | None = None,
        persons_per_batch: int | None = None,
    ) -> dict[str, Any]:
        run_id = _safe_run_id(run_id)
        if self.batch_size < 1 or self.batch_size > 25:
            raise ValueError("batch_size must be between 1 and 25")
        evidence = load_resolution_evidence(
            self.database, limit=limit, candidate_ids=candidate_ids
        )
        batches: list[list[ResolutionEvidence]] = []
        if persons_per_batch:
            if persons_per_batch < 1 or persons_per_batch > 25:
                raise ValueError("persons_per_batch must be between 1 and 25")
            by_person: dict[str, list[ResolutionEvidence]] = {}
            person_order: list[str] = []
            for item in evidence:
                if item.left_entity_id not in by_person:
                    by_person[item.left_entity_id] = []
                    person_order.append(item.left_entity_id)
                by_person[item.left_entity_id].append(item)
            max_pairs = 20  # 防超长输出截断；个别人的撞名候选可达 70+
            current: list[ResolutionEvidence] = []
            current_persons = 0
            for person_id in person_order:
                items = by_person[person_id]
                if len(items) > max_pairs:
                    if current:
                        batches.append(current)
                        current, current_persons = [], 0
                    for offset in range(0, len(items), max_pairs):
                        batches.append(items[offset : offset + max_pairs])
                    continue
                if current_persons >= persons_per_batch or len(current) + len(items) > max_pairs:
                    batches.append(current)
                    current, current_persons = [], 0
                current.extend(items)
                current_persons += 1
            if current:
                batches.append(current)
        else:
            batches = [
                evidence[offset : offset + self.batch_size]
                for offset in range(0, len(evidence), self.batch_size)
            ]
        run_dir = self.staging_root / "deepseek_resolution" / run_id
        batch_dir = run_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        all_decisions: list[dict[str, Any]] = []
        cache_hits = 0
        completed_batches = 0
        failed_batches: list[int] = []
        def process_batch(
            batch_number: int, batch: list[ResolutionEvidence]
        ) -> tuple[int, list[dict[str, Any]], bool] | None:
            expected_ids = [item.candidate_id for item in batch]
            user_payload = {"candidates": [item.to_prompt_dict() for item in batch]}
            user_prompt = (
                "Review the following candidates and return the required json object.\n"
                + _canonical_json(user_payload)
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            invocation = ModelInvocation(
                kind="character_entity_resolution",
                input_hash=_sha256_text(_canonical_json(user_payload)),
                prompt_hash=_sha256_text(SYSTEM_PROMPT),
                model_name=self.model_name,
                model_revision="api-current",
                parameters={
                    "response_format": "json_object",
                    "thinking": "enabled",
                    "reasoning_effort": self.reasoning_effort,
                    "max_tokens": self.max_tokens,
                },
                code_version=ORGANIZER_VERSION,
                parser_version="decision-json-v2",
            )
            plan = {
                "batch_number": batch_number,
                "invocation": asdict(invocation),
                "messages": messages,
                "expected_candidate_ids": expected_ids,
            }
            atomic_write_json(batch_dir / f"{batch_number:05d}.request.json", plan)
            if dry_run:
                return batch_number, [], True
            cached = self.cache.get(invocation)
            cache_hit = cached is not None
            if cache_hit:
                validated = validate_decisions(cached["validated_response"], expected_ids)
            else:
                if self.client is None:
                    raise RuntimeError("DeepSeek client is required unless --dry-run is used")
                validated = None
                last_error: Exception | None = None
                for attempt in range(2):
                    try:
                        raw_response = self.client.complete_json(
                            messages,
                            max_tokens=self.max_tokens,
                            reasoning_effort=self.reasoning_effort,
                        )
                        atomic_write_json(
                            batch_dir / f"{batch_number:05d}.raw-response.json", raw_response
                        )
                        content = self.client.response_content(raw_response)
                        parsed = json.loads(content)
                        validated = validate_decisions(parsed, expected_ids)
                        break
                    except Exception as error:  # API 级错误也记批次错误，不拖垮整跑
                        last_error = error
                if validated is None:
                    atomic_write_json(
                        batch_dir / f"{batch_number:05d}.error.json",
                        {
                            "batch_number": batch_number,
                            "expected_candidate_ids": expected_ids,
                            "error": str(last_error),
                        },
                    )
                    return None
                self.cache.put(
                    invocation,
                    raw_response=raw_response,
                    validated_response=validated,
                )
            evidence_by_id = {item.candidate_id: item for item in batch}
            for decision in validated["decisions"]:
                review_required, review_flags = review_policy(
                    decision, evidence_by_id[decision["candidate_id"]]
                )
                decision["review_required"] = review_required
                decision["review_flags"] = review_flags
            atomic_write_json(
                batch_dir / f"{batch_number:05d}.result.json",
                {"batch_number": batch_number, **validated},
            )
            return batch_number, validated["decisions"], cache_hit

        max_error_batches = 30  # 熔断阈值：余额耗尽/鉴权失败时快速止损
        if dry_run or self.workers <= 1:
            outcomes = []
            for number, batch in enumerate(batches, start=1):
                outcomes.append(process_batch(number, batch))
                if (
                    sum(1 for o in outcomes if o is None) >= max_error_batches
                    and sum(1 for o in outcomes if o is not None) == 0
                ):
                    break
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                outcomes = list(
                    pool.map(
                        lambda item: process_batch(item[0], item[1]),
                        enumerate(batches, start=1),
                    )
                )
        for outcome in outcomes:
            if outcome is None:
                continue
            batch_number, decisions, cache_hit = outcome
            if decisions:
                all_decisions.extend(decisions)
            if cache_hit:
                cache_hits += 1
            completed_batches += 1
        failed_batches = [
            number
            for number, batch in enumerate(batches, start=1)
            if (batch_dir / f"{number:05d}.error.json").exists()
        ]
        counts = {name: 0 for name in ("same", "different", "uncertain")}
        review_required_count = 0
        for decision in all_decisions:
            counts[decision["decision"]] += 1
            review_required_count += int(decision["review_required"])
        manifest = {
            "run_id": run_id,
            "organizer_version": ORGANIZER_VERSION,
            "database": str(self.database.resolve()),
            "database_size": self.database.stat().st_size,
            "model": self.model_name,
            "dry_run": dry_run,
            "candidate_count": len(evidence),
            "candidate_selection": "explicit_review_queue" if candidate_ids else "first_n",
            "persons_per_batch": persons_per_batch,
            "batch_size": self.batch_size,
            "planned_batches": len(batches),
            "completed_batches": completed_batches,
            "failed_batches": failed_batches,
            "cache_hits": cache_hits,
            "decision_counts": counts,
            "review_required_count": review_required_count,
            "canonical_database_modified": False,
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
        if not dry_run:
            atomic_write_json(run_dir / "decisions.json", {"decisions": all_decisions})
        return {**manifest, "run_directory": str(run_dir)}
