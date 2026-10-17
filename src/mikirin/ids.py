from __future__ import annotations

import re
import unicodedata
import uuid
from urllib.parse import quote


MIKIRIN_ENTITY_NAMESPACE = uuid.UUID("267a8d51-7b64-5ed9-9518-c9058ddf470d")
_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _namespace_segment(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not _SEGMENT_RE.fullmatch(normalized):
        raise ValueError(f"{label} 必须匹配 {_SEGMENT_RE.pattern!r}: {value!r}")
    return normalized


def authoritative_entity_id(source: str, entity_type: str, source_id: str | int) -> str:
    """Create a stable ID for an entity owned by an authoritative source."""
    source_part = _namespace_segment(source, "source")
    type_part = _namespace_segment(entity_type, "entity_type")
    raw_id = str(source_id).strip()
    if not raw_id:
        raise ValueError("source_id 不能为空")
    return f"{source_part}:{type_part}:{quote(raw_id, safe='-._~')}"


def provisional_entity_id(
    *,
    source: str,
    entity_type: str,
    source_record_id: str,
    record_locator: str,
    raw_name: str,
) -> str:
    """Create a stable provisional ID without ever merging records by name alone."""
    source_part = _namespace_segment(source, "source")
    type_part = _namespace_segment(entity_type, "entity_type")
    record = unicodedata.normalize("NFKC", source_record_id).strip()
    locator = unicodedata.normalize("NFKC", record_locator).strip()
    name = unicodedata.normalize("NFKC", raw_name).strip()
    if not record or not locator or not name:
        raise ValueError("source_record_id、record_locator 和 raw_name 均不能为空")
    seed = "\x1f".join((source_part, type_part, record, locator, name))
    return f"uuid5:{type_part}:{uuid.uuid5(MIKIRIN_ENTITY_NAMESPACE, seed)}"

