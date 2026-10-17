"""汇总规则层与 LLM 裁决为统一的实体解析构建输入。

三层漏斗的最后一环：
- 规则层（resolve-rules）的 auto_same / auto_different
- LLM（organize-resolution）的 same / different / uncertain
合并为 resolution-compiled.jsonl，作为后续实体合并构建（entity_redirect）
的输入。只写 staging，不修改发布库；需要人工判断的进 review-queue.jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# LLM same 直接采纳的最低置信度；低于此值进人工队列
LLM_SAME_MIN_CONFIDENCE = 0.8


def load_rule_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            decisions[item["candidate_id"]] = item
    return decisions


def load_llm_decisions(run_dir: Path) -> dict[str, dict[str, Any]]:
    """读取一次 organize-resolution 跑批的全部批次结果（跳过 error 批次）。"""
    decisions: dict[str, dict[str, Any]] = {}
    for batch_path in sorted(run_dir.glob("batches/*.result.json")):
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        for item in payload.get("decisions", []):
            decisions[item["candidate_id"]] = item
    return decisions


def compile_resolution(
    rule_decisions: dict[str, dict[str, Any]],
    llm_decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """合并两类裁决。

    返回 (compiled, review_queue, summary)。同一候选被两类裁决覆盖时
    （历史上 LLM 跑过规则层之前的全量候选），以规则层为准并记录冲突。
    """
    compiled: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    conflicts = 0

    candidate_ids = sorted(set(rule_decisions) | set(llm_decisions))
    for candidate_id in candidate_ids:
        rule = rule_decisions.get(candidate_id)
        llm = llm_decisions.get(candidate_id)

        if rule and rule["decision"] in ("auto_same", "auto_different", "dropped_sparse"):
            final = rule["decision"]
            source = "rule"
            confidence = 1.0 if final != "dropped_sparse" else None
            if llm and llm.get("decision") in ("same", "different"):
                expected = "same" if final == "auto_same" else "different"
                if final != "dropped_sparse" and llm["decision"] != expected:
                    conflicts += 1
                    review_queue.append(
                        {
                            "candidate_id": candidate_id,
                            "issue": "rule_llm_conflict",
                            "rule_decision": final,
                            "llm_decision": llm["decision"],
                            "llm_confidence": llm.get("confidence"),
                            "llm_reason": llm.get("reason"),
                        }
                    )
        elif llm:
            llm_decision = llm["decision"]
            confidence = float(llm.get("confidence", 0.0))
            if (
                llm_decision == "same"
                and confidence >= LLM_SAME_MIN_CONFIDENCE
                and not llm.get("review_required")
            ):
                final, source = "auto_same", "llm"
            elif llm_decision == "different" and not llm.get("review_required"):
                final, source = "auto_different", "llm"
            else:
                final, source = "review", "llm"
                review_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "issue": "llm_low_confidence_or_flagged",
                        "llm_decision": llm_decision,
                        "llm_confidence": confidence,
                        "review_flags": llm.get("review_flags", []),
                        "llm_reason": llm.get("reason"),
                    }
                )
        else:
            final, source, confidence = "review", "none", None
            review_queue.append(
                {"candidate_id": candidate_id, "issue": "no_decision"}
            )

        compiled.append(
            {
                "candidate_id": candidate_id,
                "final_decision": final,
                "decision_source": source,
                "confidence": confidence,
                "matched_evidence": (rule or llm or {}).get("matched_evidence", []),
                "conflicting_evidence": (rule or llm or {}).get(
                    "conflicting_evidence", []
                ),
            }
        )

    summary: dict[str, Any] = {
        "candidate_total": len(compiled),
        "final_counts": {},
        "rule_llm_conflicts": conflicts,
        "review_queue_size": len(review_queue),
    }
    for item in compiled:
        key = item["final_decision"]
        summary["final_counts"][key] = summary["final_counts"].get(key, 0) + 1
    return compiled, review_queue, summary
