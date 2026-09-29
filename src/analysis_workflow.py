"""自主分析循环的纯逻辑辅助函数。

这些函数不依赖 LangGraph 或 LLM，便于单元测试分析深度、重复追问和证据快照。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Sequence


CAUSAL_QUESTION_RE = re.compile(
    r"为什么|为何|原因|归因|驱动|下降|下滑|增长|上涨|异常|波动|变化|"
    r"分群|画像|漏斗|留存|转化|流失"
)


def needs_broad_schema(question: str) -> bool:
    """归因类问题需要更完整的 schema，避免首轮检索限制后续下钻。"""
    return bool(CAUSAL_QUESTION_RE.search(question))


EVENT_STEP_RE = re.compile(r"流量|访客|访问量|曝光|加购|浏览|点击|会话|购买转化率|支付转化率")
EVENT_SCHEMA_RE = re.compile(
    r"visits?|pageviews?|events?|impressions?|exposures?|add_to_cart|"
    r"cart_events?|sessions?|浏览|访问|曝光|加购",
    re.IGNORECASE,
)


DEVICE_STEP_RE = re.compile(
    r"设备|终端|iPad\s*用户|iPhone\s*用户|Android\s*用户", re.IGNORECASE
)
DEVICE_SCHEMA_RE = re.compile(
    r"\b(device|device_type|platform|user_agent|operating_system)\b",
    re.IGNORECASE,
)


def schema_structure_text(schema: str) -> str:
    """只查看字段定义，不把样例值误当成可用维度。"""
    return re.sub(r"### 示例数据\s*```json.*?```", "", schema, flags=re.DOTALL)


def unsupported_followup_reason(
    question: str, schema: str, required_columns: Sequence[str] = ()
) -> str:
    """对明显缺字段的候选下钻做确定性拦截。"""
    structure = schema_structure_text(schema)
    if not structure:
        return ""
    if EVENT_STEP_RE.search(question) and not EVENT_SCHEMA_RE.search(structure):
        return "schema 缺少流量、浏览或加购事件"
    if DEVICE_STEP_RE.search(question) and not DEVICE_SCHEMA_RE.search(structure):
        return "schema 缺少设备或终端字段"
    for column in required_columns:
        name = str(column).strip()
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            return f"无效的依赖字段：{name}"
        if not re.search(rf"\b{re.escape(name)}\b", structure, re.IGNORECASE):
            return f"schema 缺少依赖字段：{name}"
    return ""


QUESTION_TYPES = {
    "metric_query",
    "trend_analysis",
    "anomaly_diagnosis",
    "attribution_analysis",
    "user_segmentation",
    "funnel_retention",
}


def infer_question_type(question: str) -> str:
    """模型输出不可用时，按用户目标给出保守的分析类型。"""
    if re.search(r"漏斗|留存|复购|首购|转化", question):
        return "funnel_retention"
    if re.search(r"分群|画像|人群|客群|客户群|用户群", question):
        return "user_segmentation"
    if re.search(r"为什么|为何|原因|归因|贡献|驱动", question):
        return "attribution_analysis"
    if re.search(r"异常|突增|突降|下滑|下跌", question):
        return "anomaly_diagnosis"
    if re.search(r"趋势|走势|逐月|每月|每日|环比|同比|变化", question):
        return "trend_analysis"
    return "metric_query"


def normalize_initial_plan(
    raw_plan: Any,
    *,
    question: str,
    max_followups: int,
    schema: str = "",
) -> Dict[str, Any]:
    """把查询前的模型规划压缩成可执行、有限且可审计的步骤。"""
    data = raw_plan if isinstance(raw_plan, dict) else {}
    question_type = str(data.get("question_type") or infer_question_type(question)).strip()
    if question_type not in QUESTION_TYPES or not isinstance(raw_plan, dict):
        question_type = infer_question_type(question)

    steps: List[Dict[str, Any]] = []
    seen = set()
    dropped_reasons: set[str] = set()
    proposed = data.get("steps")
    if not isinstance(proposed, list):
        proposed = []
    for item in proposed:
        if not isinstance(item, dict):
            continue
        step_question = str(item.get("question") or "").strip()
        key = " ".join(step_question.casefold().split())
        reason = unsupported_followup_reason(step_question, schema)
        if reason:
            dropped_reasons.add(reason)
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        steps.append({
            "id": f"step_{len(steps) + 1}",
            "question": step_question,
            "purpose": str(item.get("purpose") or "").strip(),
            "status": "pending",
        })
        if len(steps) >= max(1, max_followups + 1):
            break
    if dropped_reasons:
        limitations = data.get("limitations")
        if not isinstance(limitations, list):
            limitations = []
        data = {
            **data,
            "limitations": [
                *limitations,
                *sorted(dropped_reasons),
            ],
        }
    if not steps:
        steps = [{
            "id": "step_1",
            "question": (
                "按订单状态统计订单数（已完成、待支付、已取消）"
                if dropped_reasons else question.strip()
            ),
            "purpose": "回答用户原始问题",
            "status": "pending",
        }]

    limitations = data.get("limitations")
    if not isinstance(limitations, list):
        limitations = []
    return {
        "question_type": question_type,
        "metric_definition": str(data.get("metric_definition") or "").strip(),
        "decomposition": str(data.get("decomposition") or "").strip(),
        "limitations": [str(value).strip() for value in limitations if str(value).strip()][:5],
        "steps": steps,
    }


def complete_plan_step(
    steps: Sequence[Dict[str, Any]], question: str
) -> List[Dict[str, Any]]:
    """只将当前成功取证的计划步骤标记为完成。"""
    updated = [dict(step) for step in steps]
    for step in updated:
        if step.get("question") == question and step.get("status") == "pending":
            step["status"] = "completed"
            break
    return updated


def next_plan_step(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    return next((step for step in steps if step.get("status") == "pending"), None)


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



def choose_drilldown(
    raw_plan: Any,
    *,
    original_question: str,
    evidence: Sequence[Dict[str, Any]],
    depth: int,
    max_depth: int,
    schema: str = "",
    pending_step: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """从候选下钻中选最高信息增益的一项，同时保留提前停止与深度保护。"""
    if depth >= max_depth:
        return normalize_analysis_plan(
            raw_plan, original_question=original_question,
            evidence=evidence, depth=depth, max_depth=max_depth,
        )

    data = raw_plan if isinstance(raw_plan, dict) else {}
    sufficient = data.get("answer_sufficient")
    if sufficient is True or (
        sufficient is not False and data.get("needs_followup") is False
    ):
        return {
            "needs_followup": False,
            "followup_question": "",
            "rationale": str(
                data.get("sufficiency_reason") or data.get("rationale")
                or "现有证据足以回答用户问题"
            ),
            "analysis_type": "sufficient",
            "expected_insight": "",
            "answer_sufficient": True,
            "considered_candidates": [],
        }

    candidates = data.get("candidates")
    candidates = (
        [item for item in candidates if isinstance(item, dict)][:3]
        if isinstance(candidates, list) else []
    )
    if data.get("needs_followup") is True and data.get("followup_question"):
        candidates.append({
            "question": data["followup_question"],
            "rationale": data.get("rationale", ""),
            "analysis_type": data.get("analysis_type", "direct"),
            "expected_insight": data.get("expected_insight", ""),
            "information_gain": data.get("information_gain", 3),
        })
    if pending_step and pending_step.get("question"):
        candidates.append({
            "question": pending_step["question"],
            "rationale": "模型输出不可用或其他候选不可执行时，继续查询前计划",
            "analysis_type": "planned_followup",
            "expected_insight": pending_step.get("purpose", ""),
            "information_gain": 0,
        })

    def score(item: Mapping[str, Any]) -> float:
        try:
            value = float(item.get("information_gain", 0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(5.0, value)) if math.isfinite(value) else 0.0

    considered: List[Dict[str, Any]] = []
    seen = set()
    selected_plan: Dict[str, Any] | None = None
    selected_gain = 0.0
    for item in sorted(candidates, key=score, reverse=True):
        question = str(item.get("question") or "").strip()
        key = " ".join(question.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        gain = score(item)
        required = item.get("required_columns")
        required = required if isinstance(required, list) else []
        reason = unsupported_followup_reason(question, schema, required)
        if not reason:
            plan = normalize_analysis_plan(
                {
                    "needs_followup": True,
                    "followup_question": question,
                    "rationale": item.get("rationale") or data.get("sufficiency_reason") or "",
                    "analysis_type": item.get("analysis_type") or "anomaly_drilldown",
                    "expected_insight": item.get("expected_insight") or "",
                },
                original_question=original_question,
                evidence=evidence,
                depth=depth,
                max_depth=max_depth,
            )
            if plan["needs_followup"]:
                if selected_plan is None:
                    selected_plan, selected_gain = plan, gain
                    considered.append({
                        "question": question, "information_gain": gain, "selected": True
                    })
                else:
                    considered.append({
                        "question": question, "information_gain": gain,
                        "selected": False, "reason": "信息增益低于已选方向",
                    })
                continue
            reason = plan["rationale"]
        considered.append({
            "question": question,
            "information_gain": gain,
            "selected": False,
            "reason": reason,
        })

    if selected_plan is not None:
        return {
            **selected_plan,
            "answer_sufficient": False,
            "information_gain": selected_gain,
            "considered_candidates": considered,
        }

    return {
        "needs_followup": False,
        "followup_question": "",
        "rationale": (
            "没有不重复且可由当前 schema 执行的有效下钻问题；使用已有证据总结"
            if candidates else
            str(data.get("sufficiency_reason") or data.get("rationale") or
                "Analyzer 没有给出下一步，使用已有证据总结")
        ),
        "analysis_type": "no_valid_followup" if candidates else "parse_fallback",
        "expected_insight": "",
        "answer_sufficient": sufficient is True,
        "considered_candidates": considered,
    }
