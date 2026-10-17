from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ByteFragment:
    start: int
    end: int
    path: Path

    @property
    def expected_size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class AssemblyPlan:
    expected_size: int
    expected_sha256: str
    fragments: tuple[ByteFragment, ...]

    @classmethod
    def from_json(cls, path: Path, *, root: Path | None = None) -> "AssemblyPlan":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        base = (root or path.parent).resolve()
        fragments = tuple(
            ByteFragment(
                start=int(item["start"]),
                end=int(item["end"]),
                path=(base / item["path"]).resolve(),
            )
            for item in payload["fragments"]
        )
        return cls(
            expected_size=int(payload["expected_size"]),
            expected_sha256=str(payload["expected_sha256"]).removeprefix("sha256:"),
            fragments=fragments,
        )

    def validate_fragments(self) -> tuple[ByteFragment, ...]:
        if self.expected_size <= 0:
            raise ValueError("expected_size 必须大于 0")
        ordered = tuple(sorted(self.fragments, key=lambda fragment: fragment.start))
        cursor = 0
        for fragment in ordered:
            if fragment.start != cursor:
                relation = "缺口" if fragment.start > cursor else "重叠"
                raise ValueError(
                    f"分段存在{relation}: 期望从 {cursor} 开始，实际为 {fragment.start}: {fragment.path}"
                )
            if fragment.end < fragment.start:
                raise ValueError(f"分段结束位置早于开始位置: {fragment}")
            actual_size = fragment.path.stat().st_size
            if actual_size != fragment.expected_size:
                raise ValueError(
                    f"分段长度不符: {fragment.path}: "
                    f"期望 {fragment.expected_size}，实际 {actual_size}"
                )
            cursor = fragment.end + 1
        if cursor != self.expected_size:
            raise ValueError(f"分段总长度不符: 期望 {self.expected_size}，实际覆盖到 {cursor}")
        return ordered

    def assemble(self, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
        fragments = self.validate_fragments()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".assembling", dir=destination.parent
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as output:
                for fragment in fragments:
                    with fragment.path.open("rb") as source:
                        while chunk := source.read(chunk_size):
                            output.write(chunk)
                            digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            actual_hash = digest.hexdigest()
            if actual_hash != self.expected_sha256:
                raise ValueError(
                    f"合并文件 SHA-256 不符: 期望 {self.expected_sha256}，实际 {actual_hash}"
                )
            os.replace(temp_path, destination)
            return destination
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

