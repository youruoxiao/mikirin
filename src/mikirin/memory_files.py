from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mikirin.atomic import atomic_write_text
from mikirin.run_lock import exclusive_process_lock


OperationKind = Literal["add", "replace", "remove"]


@dataclass(frozen=True, slots=True)
class TextOperation:
    op: OperationKind
    text: str = ""
    old: str = ""
    new: str = ""


@dataclass(frozen=True, slots=True)
class EditReport:
    characters: int
    remaining: int
    operations: int


class MemoryDriftError(RuntimeError):
    pass


class BudgetedTextFile:
    def __init__(self, path: Path, *, limit: int, initial_text: str = "") -> None:
        if len(initial_text) > limit:
            raise ValueError(f"memory budget exceeded: {len(initial_text)} > {limit}; merge old entries first")
        self.path = path
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            atomic_write_text(self.path, initial_text)
        initial = self.path.read_text(encoding="utf-8")
        self._known_hash = self._hash(initial)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        self._known_hash = self._hash(text)
        return text

    @staticmethod
    def _apply_operation(text: str, operation: TextOperation) -> str:
        if operation.op == "add":
            addition = operation.text.strip()
            if not addition:
                raise ValueError("add operation text cannot be empty")
            separator = "" if not text or text.endswith("\n") else "\n"
            return text + separator + addition + "\n"
        if operation.op == "replace":
            if not operation.old or operation.old not in text:
                raise ValueError("replace operation old text was not found")
            if text.count(operation.old) != 1:
                raise ValueError("replace operation old text must match exactly once")
            return text.replace(operation.old, operation.new, 1)
        if operation.op == "remove":
            target = operation.text or operation.old
            if not target or target not in text:
                raise ValueError("remove operation text was not found")
            if text.count(target) != 1:
                raise ValueError("remove operation text must match exactly once")
            return text.replace(target, "", 1)
        raise ValueError(f"unknown operation: {operation.op}")

    def apply_batch(self, operations: list[TextOperation]) -> EditReport:
        if not operations:
            raise ValueError("operations cannot be empty")
        with exclusive_process_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            current = self.path.read_text(encoding="utf-8")
            current_hash = self._hash(current)
            if current_hash != self._known_hash:
                timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                backup = self.path.with_name(f"{self.path.name}.{timestamp}.bak")
                shutil.copy2(self.path, backup)
                raise MemoryDriftError(f"external modification detected; backup saved to {backup}")
            candidate = current
            for operation in operations:
                candidate = self._apply_operation(candidate, operation)
            if len(candidate) > self.limit:
                raise ValueError(
                    f"memory budget exceeded: {len(candidate)} > {self.limit}; merge old entries first"
                )
            atomic_write_text(self.path, candidate)
            self._known_hash = self._hash(candidate)
        return EditReport(len(candidate), self.limit - len(candidate), len(operations))

    def add(self, text: str) -> EditReport:
        return self.apply_batch([TextOperation("add", text=text)])

    def replace(self, old: str, new: str) -> EditReport:
        return self.apply_batch([TextOperation("replace", old=old, new=new)])

    def remove(self, text: str) -> EditReport:
        return self.apply_batch([TextOperation("remove", text=text)])
