from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


_BLOCK_TAGS = {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "section"}
_CELL_TAGS = {"td", "th"}
_ROLES = tuple(
    sorted(
        {
            "キャラクターデザイン",
            "総作画監督",
            "作画監督補佐",
            "作画監督補",
            "作画監督",
            "レイアウト協力",
            "レイアウト",
            "シリーズ構成",
            "第二原画",
            "絵コンテ",
            "プロデューサー",
            "アニメーション制作",
            "脚本",
            "演出",
            "原画",
            "動画",
            "監督",
        },
        key=len,
        reverse=True,
    )
)


class WikibodyTextExtractor(HTMLParser):
    """Extract human page content while excluding atwiki controls and ads."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside = False
        self._finished = False
        self._wikibody_depth = 0
        self._pieces: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if not self._inside and tag == "div" and attributes.get("id") == "wikibody":
            self._inside = True
            self._wikibody_depth = 1
            return
        if not self._inside or self._finished:
            return
        if tag == "div":
            if attributes.get("id") == "atwiki-page-tags":
                self._finished = True
                self._inside = False
                return
            self._wikibody_depth += 1
        if tag == "br" or tag in _BLOCK_TAGS:
            self._pieces.append("\n")
        elif tag in _CELL_TAGS:
            self._pieces.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if not self._inside or self._finished:
            return
        if tag == "div":
            self._wikibody_depth -= 1
            if self._wikibody_depth <= 0:
                self._inside = False
                self._finished = True
                return
        if tag in _BLOCK_TAGS:
            self._pieces.append("\n")
        elif tag in _CELL_TAGS:
            self._pieces.append("\t")

    def handle_data(self, data: str) -> None:
        if self._inside and not self._finished:
            self._pieces.append(data)

    def text(self) -> str:
        joined = "".join(self._pieces).replace("\u3000", " ")
        lines = []
        for raw_line in joined.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RoleSection:
    role: str
    lines: tuple[str, ...]


def extract_wikibody_text(html: str) -> str:
    parser = WikibodyTextExtractor()
    parser.feed(html)
    return parser.text()


def split_role_sections(text: str) -> tuple[RoleSection, ...]:
    sections: list[RoleSection] = []
    active_role: str | None = None
    active_lines: list[str] = []

    def flush() -> None:
        nonlocal active_role, active_lines
        if active_role is not None:
            sections.append(RoleSection(active_role, tuple(active_lines)))
        active_role = None
        active_lines = []

    for line in text.splitlines():
        matched_role = None
        remainder = ""
        for role in _ROLES:
            if line == role:
                matched_role = role
                break
            if line.startswith(role):
                candidate = line[len(role) :]
                if not candidate or candidate[0].isspace() or candidate[0] in "：:・.　":
                    matched_role = role
                    remainder = candidate.lstrip(" ：:・.　")
                    break
        if matched_role is not None:
            flush()
            active_role = matched_role
            if remainder:
                active_lines.append(remainder)
        elif active_role is not None:
            active_lines.append(line)
    flush()
    return tuple(sections)

