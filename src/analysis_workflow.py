"""自主分析循环的纯逻辑辅助函数。

这些函数不依赖 LangGraph 或 LLM，便于单元测试分析深度、重复追问和证据快照。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence


CAUSAL_QUESTION_RE = re.compile(
    r"为什么|为何|原因|归因|驱动|下降|下滑|增长|上涨|异常|波动|变化"
)


def needs_broad_schema(question: str) -> bool:
    """归因类问题需要更完整的 schema，避免首轮检索限制后续下钻。"""
    return bool(CAUSAL_QUESTION_RE.search(question))


def active_question(state: Mapping[str, Any]) -> str:
    """返回当前 SQL 应回答的问题：优先使用 Analyzer 生成的追问。"""
    followup = str(state.get("followup_question") or "").strip()
    return followup or str(state.get("question") or "").strip()


def snapshot_evidence(
    *,
    question: str,
    sql: str,
    rows: Sequence[Dict[str, Any]] | None,
    columns: Sequence[str] | None,
    depth: int,
    row_limit: int = 50,
) -> Dict[str, Any]:
    """保存一轮可审计证据；仅保留样例行以控制状态和 prompt 大小。"""
    materialized_rows = list(rows or [])
    return {
        "round": depth + 1,
        "question": question,
        "sql": sql,
        "columns": list(columns or []),
        "row_count": len(materialized_rows),
        "rows": materialized_rows[:row_limit],
        "truncated": len(materialized_rows) > row_limit,
    }


def format_evidence(evidence: Sequence[Dict[str, Any]], row_limit: int = 30) -> str:
    """把多轮证据压缩成适合传给 LLM 的 JSON。"""
    compact: List[Dict[str, Any]] = []
    for item in evidence:
        compact.append({
            "round": item.get("round"),
            "question": item.get("question"),
            "sql": item.get("sql"),
            "columns": item.get("columns") or [],
            "row_count": item.get("row_count", 0),
            "rows": list(item.get("rows") or [])[:row_limit],
            "truncated": bool(item.get("truncated")),
        })
    return json.dumps(compact, ensure_ascii=False)


def normalize_analysis_plan(
    raw_plan: Any,
    *,
    original_question: str,
    evidence: Sequence[Dict[str, Any]],
    depth: int,
    max_depth: int,
) -> Dict[str, Any]:
    """校验 LLM 规划输出，并确定是否允许进入下一轮查询。"""
    if depth >= max_depth:
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": f"已达到最大深挖轮数 {max_depth}",
            "analysis_type": "depth_limit",
            "expected_insight": "",
        }

    if not isinstance(raw_plan, dict):
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": "Analyzer 规划输出无法解析，使用现有证据总结",
            "analysis_type": "parse_fallback",
            "expected_insight": "",
        }

    needs_followup = raw_plan.get("needs_followup") is True
    followup_question = str(raw_plan.get("followup_question") or "").strip()
    rationale = str(raw_plan.get("rationale") or "").strip()
    analysis_type = str(raw_plan.get("analysis_type") or "direct").strip()
    expected_insight = str(raw_plan.get("expected_insight") or "").strip()

    if not needs_followup:
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": rationale or "现有证据足以回答用户问题",
            "analysis_type": analysis_type,
            "expected_insight": expected_insight,
        }

    if not followup_question:
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": "Analyzer 未给出有效追问，使用现有证据总结",
            "analysis_type": "invalid_followup",
            "expected_insight": "",
        }

    previous_questions = {
        str(item.get("question") or "").strip().casefold()
        for item in evidence
        if item.get("question")
    }
    previous_questions.add(original_question.strip().casefold())
    if followup_question.casefold() in previous_questions:
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": "Analyzer 生成了重复问题，停止循环并使用现有证据总结",
            "analysis_type": "duplicate_guard",
            "expected_insight": expected_insight,
        }

    return {
        "needs_followup": True,
        "followup_question": followup_question,
        "rationale": rationale or "需要补充一轮数据以形成可靠结论",
        "analysis_type": analysis_type,
        "expected_insight": expected_insight,
    }

