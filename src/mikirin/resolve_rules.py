"""萌百↔BGM 角色候选的确定性预裁决（三层漏斗的前两层）。

设计目标：把证据明显充分的候选用可解释规则直接裁决（auto_same /
auto_different），只有灰色地带（needs_llm）才进入 LLM 仲裁。

规则层不修改正式库；输出落 staging，与 LLM 决策一样是可审计的构建输入。
阈值用历史 LLM 标注（data/staging/deepseek_resolution/*/decisions.json）
离线校准，auto_same 精确率目标 >=99%，auto_different >=97%。
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 归一化时剥掉的引号/括号/装饰字符
_STRIP_CHARS = "\"'“”‘’「」『』【】[]()（）<>《》〈〉・· 　\t\n"
_BIRTHDAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")

# 特征投票权重（用户设计：作品/名字/声优等都是角色包的外化特征，无高低，
# 命中够多即判）。阈值经 500 条去重标注校准：
# >=3.5 判 same（标注集 precision 0.994），<=0.5 判 different（0.981）。
SCORE_WEIGHTS = {
    "work_exact": 2.0, "work_fuzzy": 1.0, "name": 2.0, "cv": 2.0,
    "gender": 1.0, "birthday": 1.5, "generic": -1.0,
    "gender_conflict": -2.0, "cv_conflict": -1.5, "birthday_conflict": -1.5,
}
AUTO_SAME_THRESHOLD = 3.5
AUTO_DIFFERENT_THRESHOLD = 0.5


_GENDER_MAP = {
    "男": "male", "男性": "male", "male": "male",
    "女": "female", "女性": "female", "female": "female",
}

# 萌百 subjects 里的分类/导航条目，不是具体作品，不能参与锚定
_GENERIC_SUBJECT_RE = re.compile(r"(作品|系列|系列角色|人物|角色|角色列表)$")

# 模糊标题匹配时剥掉的后缀（"合金弹头系列" → "合金弹头"）
_STRIP_SUFFIX_RE = re.compile(r"(系列角色|系列|作品|人物|角色|角色列表)$")


def _strip_suffix(title: str) -> str:
    previous = None
    while previous != title:
        previous = title
        title = _STRIP_SUFFIX_RE.sub("", title)
    return title


def _bigrams(title: str) -> set[str]:
    return {title[i : i + 2] for i in range(len(title) - 1)} if len(title) >= 2 else set()


def fuzzy_title_match(left_title: str, right_title: str) -> bool:
    """归一化标题的模糊相等：剥后缀后相等、互为子串（短者 >=3 字）。

    用于作品证据的弱形式；标注集验证命中后 same 占 42/43。
    """
    left, right = _strip_suffix(left_title), _strip_suffix(right_title)
    if len(left) < 2 or len(right) < 2:
        return False
    if left == right:
        return True
    if len(left) >= 3 and left in right:
        return True
    if len(right) >= 3 and right in left:
        return True
    return False


def normalize(text: Any) -> str:
    """NFKC + 小写 + 去引号括号空白，用于名称/标题比较。"""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if ch not in _STRIP_CHARS and not ch.isspace())


def alias_variants(text: Any) -> set[str]:
    """把 infobox 来源的脏别名拆成可比较的变体。

    处理三类噪声：
    - 键名前缀：'罗马字|fuchigamimai' → 'fuchigamimai'（'|键名' 后无内容则丢弃）
    - 复合串：'川岛梨乃=かわしまりの=渦井ゆきの' → 三个独立变体
    - 并列列表：'ちょこたん、ウサたん' → 两个独立变体
    """
    raw = normalize(text)
    if not raw:
        return set()
    variants: set[str] = set()
    for segment in re.split(r"[=]", raw):
        tail = segment.split("|")[-1]
        for piece in re.split(r"[、,，/]", tail):
            piece = piece.strip()
            if piece:
                variants.add(piece)
    return variants


def _parse_birthday(value: Any) -> tuple[int, int] | None:
    """萌百 [null, 5, 11] 或 BGM '5月11日' 文本统一成 (月, 日)。"""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            month, day = int(value[1]), int(value[2])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return (month, day)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = _BIRTHDAY_RE.search(value)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    return None


def _parse_infobox(raw: str | None) -> dict[str, str]:
    """从 BGM infobox wiki 串提取顶层字段（|键= 值）。"""
    fields: dict[str, str] = {}
    if not raw:
        return fields
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("|") and "=" in line:
            key, _, value = line[1:].partition("=")
            fields[key.strip()] = value.strip()
    return fields


def _gender_of(value: Any) -> str | None:
    if isinstance(value, str):
        return _GENDER_MAP.get(value.strip())
    return None


@dataclass
class CandidateFeatures:
    candidate_id: str
    left_entity_id: str
    right_entity_id: str
    left_names: set[str] = field(default_factory=set)      # 归一化后的名字集合
    right_names: set[str] = field(default_factory=set)
    left_canonical: str = ""                               # 萌百 canonical（归一化）
    right_canonical: str = ""                              # BGM canonical（归一化）
    left_subjects: set[str] = field(default_factory=set)   # 归一化后的作品标题
    right_subjects: set[str] = field(default_factory=set)
    left_gender: str | None = None
    right_gender: str | None = None
    left_birthday: tuple[int, int] | None = None
    right_birthday: tuple[int, int] | None = None
    left_voice_actors: set[str] = field(default_factory=set)
    right_voice_actors: set[str] = field(default_factory=set)
    anchored_work: bool = False  # 萌百作品标题映射到 BGM 条目，且右侧角色在该条目角色表中
    weighted_work: float = 0.0   # 双侧作品标题的 IDF 加权 bigram 最大重合度
    variant_name_match: bool = False  # 仅差字体变体的名字一致（島↔岛等，视为同一名字）
    right_collects: int = 0      # BGM 角色收藏数（人气/路人程度代理指标）
    right_comments: int = 0      # BGM 角色评论数
    right_has_summary: bool = False

    @property
    def name_match(self) -> bool:
        return bool(self.left_names & self.right_names)

    @property
    def work_matches(self) -> set[str]:
        return self.left_subjects & self.right_subjects

    @property
    def fuzzy_work_matches(self) -> set[tuple[str, str]]:
        """模糊匹配命中的 (萌百标题, BGM标题) 对，不含精确重合。"""
        pairs: set[tuple[str, str]] = set()
        exact = self.work_matches
        for left in self.left_subjects:
            for right in self.right_subjects:
                if left in exact or right in exact:
                    continue
                if fuzzy_title_match(left, right):
                    pairs.add((left, right))
        return pairs

    @property
    def left_has_subjects(self) -> bool:
        return bool(self.left_subjects)

    @property
    def generic_name(self) -> bool:
        """最短名字 <=1 个字符视为通用名（单字黑话/常见字），证据强度降级。"""
        shortest = min((len(n) for n in self.left_names if n), default=99)
        return shortest <= 1

    @property
    def is_sparse(self) -> bool:
        """萌百侧几乎无数据（无作品/声优/性别/生日）的路人角色。

        这类角色即使判对也几乎没有知识库价值，按用户决策直接丢弃，
        不占用 LLM 额度。
        """
        return (
            not self.left_subjects
            and not self.left_voice_actors
            and self.left_gender is None
            and self.left_birthday is None
        )

    @property
    def right_is_nobody(self) -> bool:
        """BGM 侧是路人：零收藏且零评论（社区无存在感）。

        这种候选无论真假都没有消歧价值——真对的主角不可能零收藏零评论，
        假对更不需要判。
        """
        return self.right_collects == 0 and self.right_comments == 0

    @property
    def gender_state(self) -> str:
        if not self.left_gender or not self.right_gender:
            return "unknown"
        return "match" if self.left_gender == self.right_gender else "conflict"

    @property
    def birthday_state(self) -> str:
        if not self.left_birthday or not self.right_birthday:
            return "unknown"
        return "match" if self.left_birthday == self.right_birthday else "conflict"

    @property
    def voice_actor_state(self) -> str:
        if not self.left_voice_actors or not self.right_voice_actors:
            return "unknown"
        return "match" if self.left_voice_actors & self.right_voice_actors else "conflict"

    def corroborations(self) -> list[str]:
        """作品+名字之外的独立佐证。"""
        codes: list[str] = []
        if self.gender_state == "match":
            codes.append("same_gender")
        if self.birthday_state == "match":
            codes.append("same_birthday")
        if self.voice_actor_state == "match":
            codes.append("same_voice_actor")
        return codes

    def conflicts(self) -> list[str]:
        codes: list[str] = []
        if self.gender_state == "conflict":
            codes.append("different_gender")
        if self.birthday_state == "conflict":
            codes.append("different_birthday")
        if self.voice_actor_state == "conflict":
            codes.append("different_voice_actor")
        return codes


@dataclass
class RuleDecision:
    candidate_id: str
    decision: str               # auto_same / auto_different / needs_llm
    matched_evidence: list[str]
    conflicting_evidence: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "matched_evidence": self.matched_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "reason": self.reason,
            "decision_source": "resolve_rules",
        }


def _chunked(values: Iterable[str], size: int = 400) -> Iterable[list[str]]:
    unique = list(dict.fromkeys(values))
    for index in range(0, len(unique), size):
        yield unique[index : index + size]


def load_candidate_features(database: Path) -> list[CandidateFeatures]:
    """从发布库装载全部候选并计算确定性特征（只读）。"""
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT rc.id, rc.left_entity_id, rc.right_entity_id,
                   le.canonical_name AS left_name, re.canonical_name AS right_name,
                   rc_char.infobox_raw, rc_char.collects, rc_char.comments,
                   rc_char.summary
            FROM resolution_candidate rc
            JOIN entity le ON le.id = rc.left_entity_id
            JOIN entity re ON re.id = rc.right_entity_id
            LEFT JOIN characters rc_char ON rc_char.entity_id = rc.right_entity_id
            WHERE rc.status = 'candidate' AND rc.relation_kind = 'same_character'
            ORDER BY rc.id
            """
        ).fetchall()

        features: list[CandidateFeatures] = []
        left_index: dict[str, list[CandidateFeatures]] = {}
        right_index: dict[str, list[CandidateFeatures]] = {}
        for row in rows:
            feat = CandidateFeatures(
                candidate_id=row["id"],
                left_entity_id=row["left_entity_id"],
                right_entity_id=row["right_entity_id"],
            )
            feat.left_canonical = normalize(row["left_name"])
            feat.right_canonical = normalize(row["right_name"])
            for names, normalized in ((feat.left_names, feat.left_canonical),
                                      (feat.right_names, feat.right_canonical)):
                if normalized:
                    names.add(normalized)
            infobox = _parse_infobox(row["infobox_raw"])
            feat.right_gender = _gender_of(infobox.get("性别"))
            feat.right_birthday = _parse_birthday(infobox.get("生日"))
            feat.right_collects = row["collects"] or 0
            feat.right_comments = row["comments"] or 0
            feat.right_has_summary = bool((row["summary"] or "").strip())
            features.append(feat)
            left_index.setdefault(feat.left_entity_id, []).append(feat)
            right_index.setdefault(feat.right_entity_id, []).append(feat)

        def all_entity_ids() -> Iterable[str]:
            yield from left_index
            yield from right_index

        # 两侧别名（infobox 来源可能带键名/复合串，拆变体）
        for entity_ids in _chunked(all_entity_ids()):
            marks = ",".join("?" for _ in entity_ids)
            for entity_id, alias_text in db.execute(
                f"SELECT entity_id, alias_text FROM entity_alias WHERE entity_id IN ({marks})",
                entity_ids,
            ):
                for alias in alias_variants(alias_text):
                    for feat in left_index.get(entity_id, ()):
                        feat.left_names.add(alias)
                    for feat in right_index.get(entity_id, ()):
                        feat.right_names.add(alias)

        # 萌百侧事实（fields/gender/voice_actors/subjects）
        for entity_ids in _chunked(left_index):
            marks = ",".join("?" for _ in entity_ids)
            for entity_id, kind, raw_value in db.execute(
                f"SELECT entity_id, fact_kind, value_json FROM entity_fact WHERE entity_id IN ({marks})",
                entity_ids,
            ):
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                for feat in left_index.get(entity_id, ()):
                    if kind == "gender":
                        feat.left_gender = _gender_of(value)
                    elif kind == "voice_actors" and isinstance(value, list):
                        feat.left_voice_actors.update(
                            n for n in (normalize(v) for v in value) if n
                        )
                    elif kind == "subjects" and isinstance(value, list):
                        feat.left_subjects.update(
                            t for t in (normalize(v) for v in value) if t
                        )
                    elif kind == "fields" and isinstance(value, dict):
                        for name in value.get("本名") or []:
                            normalized = normalize(name)
                            if normalized:
                                feat.left_names.add(normalized)
                        if feat.left_birthday is None:
                            feat.left_birthday = _parse_birthday(value.get("生日"))

        # BGM 侧作品标题（本名+中文名），并记录 subject→characters 反查
        subject_to_characters: dict[str, set[str]] = {}
        for entity_ids in _chunked(right_index):
            marks = ",".join("?" for _ in entity_ids)
            for character_id, subject_id, name, name_cn in db.execute(
                f"""
                SELECT rel.character_id, rel.subject_id, s.name, s.name_cn
                FROM rel_subject_character rel
                JOIN subjects s ON s.entity_id = rel.subject_id
                WHERE rel.character_id IN ({marks})
                """,
                entity_ids,
            ):
                subject_to_characters.setdefault(subject_id, set()).add(character_id)
                for feat in right_index.get(character_id, ()):
                    for title in (name, name_cn):
                        normalized = normalize(title)
                        if normalized:
                            feat.right_subjects.add(normalized)

        # 作品别名补充标题（提高与萌百中文标题的命中率）
        for subject_group in _chunked(subject_to_characters):
            marks = ",".join("?" for _ in subject_group)
            for entity_id, alias_text in db.execute(
                f"SELECT entity_id, alias_text FROM entity_alias WHERE entity_id IN ({marks})",
                subject_group,
            ):
                for variant in alias_variants(alias_text):
                    for character_id in subject_to_characters.get(entity_id, ()):
                        for feat in right_index.get(character_id, ()):
                            feat.right_subjects.add(variant)

        # BGM 侧声优（人名原文 + 人物别名，兼容中日写法差异）
        character_persons: dict[str, set[str]] = {}
        for entity_ids in _chunked(right_index):
            marks = ",".join("?" for _ in entity_ids)
            for character_id, person_id, person_name in db.execute(
                f"""
                SELECT rpc.character_id, rpc.person_id, p.name
                FROM rel_person_character rpc
                JOIN persons p ON p.entity_id = rpc.person_id
                WHERE rpc.character_id IN ({marks})
                """,
                entity_ids,
            ):
                bucket = character_persons.setdefault(character_id, set())
                normalized = normalize(person_name)
                if normalized:
                    bucket.add(normalized)
                bucket.add(f"__person__{person_id}")
        person_ids = [
            key.removeprefix("__person__")
            for names in character_persons.values()
            for key in names
            if key.startswith("__person__")
        ]
        person_aliases: dict[str, set[str]] = {}
        for person_group in _chunked(person_ids):
            marks = ",".join("?" for _ in person_group)
            for entity_id, alias_text in db.execute(
                f"SELECT entity_id, alias_text FROM entity_alias WHERE entity_id IN ({marks})",
                person_group,
            ):
                for alias in alias_variants(alias_text):
                    person_aliases.setdefault(entity_id, set()).add(alias)
        for character_id, names in character_persons.items():
            resolved = {n for n in names if not n.startswith("__person__")}
            for key in names:
                if key.startswith("__person__"):
                    resolved.update(person_aliases.get(key.removeprefix("__person__"), ()))
            for feat in right_index.get(character_id, ()):
                feat.right_voice_actors.update(resolved)

        # 锚定作品匹配：全库标题/别名 → subject 索引 + 各 subject 角色表
        title_to_subject: dict[str, set[str]] = {}
        for row in db.execute("SELECT entity_id, name, name_cn FROM subjects"):
            for title in (row[1], row[2]):
                normalized = normalize(title)
                if normalized:
                    title_to_subject.setdefault(normalized, set()).add(row[0])
        for row in db.execute(
            "SELECT entity_id, alias_text FROM entity_alias WHERE entity_id LIKE 'bgm:subject:%'"
        ):
            normalized = normalize(row[1])
            if normalized:
                title_to_subject.setdefault(normalized, set()).add(row[0])
        subject_characters: dict[str, set[str]] = {}
        for row in db.execute("SELECT subject_id, character_id FROM rel_subject_character"):
            subject_characters.setdefault(row[0], set()).add(row[1])

        anchored_by_left: dict[str, set[str]] = {}
        for left_id, group_feats in left_index.items():
            mapped: set[str] = set()
            for title in group_feats[0].left_subjects:
                if _GENERIC_SUBJECT_RE.search(title) or len(title) < 2:
                    continue
                mapped.update(title_to_subject.get(title, ()))
            anchored_by_left[left_id] = mapped
        for feat in features:
            for subject_id in anchored_by_left.get(feat.left_entity_id, ()):
                if feat.right_entity_id in subject_characters.get(subject_id, ()):
                    feat.anchored_work = True
                    break

        # 字体变体对挖掘：同一 BGM 角色的多个别名按位对齐，
        # 仅差一个字符的对记为变体（島↔岛、風↔风），支持度 >=20 采用。
        aliases_by_char: dict[str, set[str]] = {}
        for row in db.execute(
            "SELECT entity_id, alias_text FROM entity_alias WHERE entity_id LIKE 'bgm:character:%'"
        ):
            normalized = normalize(row[1])
            if normalized and 2 <= len(normalized) <= 8 and "|" not in row[1]:
                aliases_by_char.setdefault(row[0], set()).add(normalized)
        pair_counts: dict[tuple[str, str], int] = {}
        for names in aliases_by_char.values():
            ordered = sorted(names)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a, b = ordered[i], ordered[j]
                    if len(a) != len(b):
                        continue
                    diffs = tuple((x, y) for x, y in zip(a, b) if x != y)
                    if len(diffs) == 1:
                        pair = tuple(sorted(diffs[0]))
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        variants = {pair for pair, count in pair_counts.items() if count >= 20}

        def _variant_equal(a: str, b: str) -> bool:
            if len(a) != len(b):
                return False
            return all(
                x == y or (x, y) in variants or (y, x) in variants
                for x, y in zip(a, b)
            )

        for feat in features:
            if feat.name_match:
                continue
            feat.variant_name_match = any(
                _variant_equal(left, right)
                for left in feat.left_names
                for right in feat.right_names
            )

        # IDF 加权 bigram 作品重合（用户提议的部分命中匹配）：
        # 系列通用词（"少女""传说"）被 IDF 自然降权，稀有重合词才是有效信号。
        document_freq: dict[str, int] = {}
        title_count = 0
        seen_titles: set[str] = set()
        for feat in features:
            for title in feat.left_subjects | feat.right_subjects:
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                title_count += 1
                for bg in _bigrams(title):
                    document_freq[bg] = document_freq.get(bg, 0) + 1
        import math

        def _idf(bigram: str) -> float:
            return math.log(max(title_count, 1) / (1 + document_freq[bigram]))

        def _weighted_jaccard(left_title: str, right_title: str) -> float:
            lb, rb = _bigrams(left_title), _bigrams(right_title)
            if not lb or not rb:
                return 0.0
            inter = sum(_idf(x) for x in lb & rb)
            union = sum(_idf(x) for x in lb | rb)
            return inter / union if union else 0.0

        for feat in features:
            if feat.work_matches or feat.anchored_work or feat.fuzzy_work_matches:
                continue
            feat.weighted_work = max(
                (
                    _weighted_jaccard(lt, rt)
                    for lt in feat.left_subjects
                    for rt in feat.right_subjects
                ),
                default=0.0,
            )

        return features


def feature_score(feat: CandidateFeatures) -> tuple[float, list[str], list[str]]:
    """特征投票打分：返回 (分数, 命中特征, 冲突特征)。

    用户设计：作品/名字/声优/性别/生日都是角色包的外化特征，无高低之分，
    命中够多即判对。阈值经 500 条去重标注校准。
    """
    matched: list[str] = []
    conflicting: list[str] = []
    score = 0.0
    if feat.work_matches or feat.anchored_work:
        score += SCORE_WEIGHTS["work_exact"]
        matched.append("same_work")
    elif feat.fuzzy_work_matches or feat.weighted_work >= 0.2:
        score += SCORE_WEIGHTS["work_fuzzy"]
        matched.append(
            "fuzzy_work" if feat.fuzzy_work_matches else "weighted_work"
        )
    if feat.name_match:
        score += SCORE_WEIGHTS["name"]
        matched.append("same_full_name")
    elif feat.variant_name_match:
        score += SCORE_WEIGHTS["name"]
        matched.append("same_variant_name")
    if feat.voice_actor_state == "match":
        score += SCORE_WEIGHTS["cv"]
        matched.append("same_voice_actor")
    elif feat.voice_actor_state == "conflict":
        score += SCORE_WEIGHTS["cv_conflict"]
        conflicting.append("different_voice_actor")
    if feat.gender_state == "match":
        score += SCORE_WEIGHTS["gender"]
        matched.append("same_gender")
    elif feat.gender_state == "conflict":
        score += SCORE_WEIGHTS["gender_conflict"]
        conflicting.append("different_gender")
    if feat.birthday_state == "match":
        score += SCORE_WEIGHTS["birthday"]
        matched.append("same_birthday")
    elif feat.birthday_state == "conflict":
        score += SCORE_WEIGHTS["birthday_conflict"]
        conflicting.append("different_birthday")
    if feat.generic_name:
        score += SCORE_WEIGHTS["generic"]
        conflicting.append("generic_name")
    return score, matched, conflicting


def decide_pair(feat: CandidateFeatures) -> RuleDecision:
    """特征投票裁决：>=3.5 判 same，<=0.5 判 different，其余交 LLM。"""
    score, matched, conflicting = feature_score(feat)
    if score >= AUTO_SAME_THRESHOLD:
        return RuleDecision(
            feat.candidate_id,
            "auto_same",
            matched,
            conflicting,
            f"特征投票 {score:.1f} 分（>= {AUTO_SAME_THRESHOLD}）",
        )
    if score <= AUTO_DIFFERENT_THRESHOLD:
        return RuleDecision(
            feat.candidate_id,
            "auto_different",
            matched,
            conflicting,
            f"特征投票 {score:.1f} 分（<= {AUTO_DIFFERENT_THRESHOLD}）",
        )
    return RuleDecision(
        feat.candidate_id,
        "needs_llm",
        matched,
        conflicting,
        f"特征投票 {score:.1f} 分，落在灰区",
    )


def decide_all(features: list[CandidateFeatures]) -> list[RuleDecision]:
    """逐对打分裁决 + 组内一致性检查 + 路人丢弃。

    组内约束：同一萌百角色最多合并到一个 BGM 角色。若多条候选都判
    auto_same（典型：正作与外传 incarnation 双双高分），只保留分数最高的，
    其余降级 needs_llm 交 LLM 区分；平分则全部交 LLM。
    """
    decisions: dict[str, RuleDecision] = {}
    groups: dict[str, list[CandidateFeatures]] = {}
    for feat in features:
        groups.setdefault(feat.left_entity_id, []).append(feat)
        decisions[feat.candidate_id] = decide_pair(feat)

    for group in groups.values():
        if len(group) < 2:
            continue
        same_feats = [
            feat for feat in group
            if decisions[feat.candidate_id].decision == "auto_same"
        ]
        if len(same_feats) <= 1:
            continue
        # 多条候选达 same 阈值：大多是 BGM 内部重复/异名形态条目
        # （盖鲁德×2、傀影 vs 酒神、假日威龙陈 vs 陈）。
        # 选正主：两侧 canonical 全等优先，其次收藏+评论数（用户建议：
        # 真身挂着更丰富的作品关系，人气代理同样有效）。
        def canonical_rank(feat: CandidateFeatures) -> tuple[int, int]:
            exact = int(
                bool(feat.left_canonical)
                and feat.left_canonical == feat.right_canonical
            )
            return (exact, feat.right_collects + feat.right_comments)
        best_feat = max(same_feats, key=canonical_rank)
        tied = [
            feat for feat in same_feats
            if canonical_rank(feat) == canonical_rank(best_feat)
        ]
        for feat in same_feats:
            if len(tied) == 1 and feat is best_feat:
                decisions[feat.candidate_id] = RuleDecision(
                    feat.candidate_id,
                    "auto_same",
                    decisions[feat.candidate_id].matched_evidence,
                    decisions[feat.candidate_id].conflicting_evidence,
                    "组内多候选达标，按全等名字/收藏评论数选为正主（BGM 内部重复或异名形态）",
                )
                continue
            original = decisions[feat.candidate_id]
            decisions[feat.candidate_id] = RuleDecision(
                feat.candidate_id,
                "needs_llm",
                original.matched_evidence,
                original.conflicting_evidence,
                "组内多条候选达到 same 阈值（衍生 incarnation/重复条目歧义），交 LLM",
            )

    for feat in features:
        decision = decisions[feat.candidate_id]
        if decision.decision == "needs_llm" and (feat.is_sparse or feat.right_is_nobody):
            reason = (
                "萌百侧无作品/声优/性别/生日数据，路人角色直接丢弃"
                if feat.is_sparse
                else "BGM 侧零收藏且零评论，路人角色直接丢弃"
            )
            decisions[feat.candidate_id] = RuleDecision(
                feat.candidate_id,
                "dropped_sparse",
                decision.matched_evidence,
                decision.conflicting_evidence,
                reason,
            )
    return [decisions[feat.candidate_id] for feat in features]


def load_labeled_decisions(staging_root: Path) -> dict[str, str]:
    """汇总历史 LLM 标注（candidate_id -> same/different/uncertain，多数决）。"""
    votes: dict[str, dict[str, int]] = {}
    for path in sorted(staging_root.glob("deepseek_resolution/*/decisions.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("decisions", []):
            candidate_id = item.get("candidate_id")
            decision = item.get("decision")
            if not candidate_id or decision not in {"same", "different", "uncertain"}:
                continue
            bucket = votes.setdefault(candidate_id, {})
            bucket[decision] = bucket.get(decision, 0) + 1
    return {
        candidate_id: max(counts, key=lambda key: (counts[key], key))
        for candidate_id, counts in votes.items()
    }


def evaluate(decisions: list[RuleDecision], labels: dict[str, str]) -> dict[str, Any]:
    """对照标注计算各类裁决的精确率与覆盖率。"""
    report: dict[str, Any] = {}
    labeled = [d for d in decisions if d.candidate_id in labels]
    report["labeled_total"] = len(labeled)
    for kind in ("auto_same", "auto_different", "dropped_sparse", "needs_llm"):
        subset = [d for d in labeled if d.decision == kind]
        entry: dict[str, Any] = {"count": len(subset)}
        if subset and kind in ("auto_same", "auto_different"):
            expected = "same" if kind == "auto_same" else "different"
            agree = sum(1 for d in subset if labels[d.candidate_id] == expected)
            entry["precision"] = round(agree / len(subset), 4)
            entry["mistakes"] = [
                {
                    "candidate_id": d.candidate_id,
                    "label": labels[d.candidate_id],
                    "reason": d.reason,
                }
                for d in subset
                if labels[d.candidate_id] != expected
            ][:20]
        if kind in ("dropped_sparse", "needs_llm"):
            entry["label_breakdown"] = {
                label: sum(1 for d in subset if labels[d.candidate_id] == label)
                for label in ("same", "different", "uncertain")
            }
        report[kind] = entry
    auto_total = sum(
        1 for d in labeled if d.decision in ("auto_same", "auto_different")
    )
    report["rule_coverage"] = round(auto_total / len(labeled), 4) if labeled else 0.0
    return report
