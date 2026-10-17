from __future__ import annotations

import re
import unicodedata
from typing import Any


_ALIAS_KEYS = {
    "中文名", "简体中文名", "繁体中文名", "別名", "别名", "原名",
    "日文名", "英文名", "罗马字", "羅馬字", "alias", "aliases",
}
_NORMALIZED_ALIAS_KEYS = {
    unicodedata.normalize("NFKC", item).strip().casefold().replace(" ", "")
    for item in _ALIAS_KEYS
}
_FIELD = re.compile(r"^\|\s*([^=]+?)\s*=\s*(.*)$")
_BRACKET = re.compile(r"^\[([^\[\]]+)\]$")


def normalize_alias(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def extract_infobox_aliases(infobox: Any) -> list[str]:
    if not isinstance(infobox, str) or not infobox.strip():
        return []
    lines = infobox.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    aliases: list[str] = []
    index = 0
    while index < len(lines):
        match = _FIELD.match(lines[index].strip())
        index += 1
        if not match:
            continue
        key = normalize_alias(match.group(1)).replace(" ", "")
        value = match.group(2).strip()
        if key not in _NORMALIZED_ALIAS_KEYS:
            continue
        values: list[str] = []
        if value == "{":
            while index < len(lines):
                item = lines[index].strip()
                index += 1
                if item == "}":
                    break
                if item:
                    values.append(item)
        elif value:
            values.append(value)
        for item in values:
            bracket = _BRACKET.match(item)
            if bracket:
                item = bracket.group(1).strip()
            if not item or item.startswith(("http://", "https://")):
                continue
            if item not in aliases:
                aliases.append(item)
    return aliases


def record_aliases(record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for value in (
        record.get("name"),
        record.get("name_cn"),
        *extract_infobox_aliases(record.get("infobox")),
    ):
        if value is None:
            continue
        alias = str(value).strip()
        if alias and alias not in result:
            result.append(alias)
    return result
