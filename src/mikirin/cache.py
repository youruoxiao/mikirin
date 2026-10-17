from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    kind: str
    input_hash: str
    prompt_hash: str
    model_name: str
    model_revision: str
    parameters: dict[str, Any]
    code_version: str
    parser_version: str

    def cache_key(self) -> str:
        canonical = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ModelCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, invocation: ModelInvocation) -> Path:
        key = invocation.cache_key()
        return self.root / key[:2] / f"{key}.json"

    def get(self, invocation: ModelInvocation) -> dict[str, Any] | None:
        path = self.path_for(invocation)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def put(
        self,
        invocation: ModelInvocation,
        *,
        raw_response: Any,
        validated_response: Any,
    ) -> Path:
        path = self.path_for(invocation)
        payload = {
            "cache_key": invocation.cache_key(),
            "invocation": asdict(invocation),
            "raw_response": raw_response,
            "validated_response": validated_response,
        }
        if path.exists():
            existing = self.get(invocation)
            if existing != payload:
                raise RuntimeError(f"缓存键冲突，拒绝覆盖不可变产物: {path}")
            return path
        atomic_write_json(path, payload)
        return path

