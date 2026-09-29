"""LangGraph SQL Agent 的核心工作流。

流程：
    用户问题 → 问题分类与初始计划 → 核心 SQL → SQL Reviewer → 执行 → Result Validator
        → Analyzer 判断证据是否充足并比较候选下钻方向
        →（需要深挖）选择下一问题 → SQL Reviewer → 执行 → Result Validator
        →（证据充足或达到上限）多轮证据归因 → 可视化 → 回答

每一轮 SQL 都经过安全、编译、语义和结果合理性检查；后续查询失败时，
保留已成功获得的证据并降级总结。
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_FALLBACK_API_KEY,
    LLM_FALLBACK_BASE_URL,
    LLM_FALLBACK_MODEL,
    LLM_KEY_HINT,
    LLM_MODEL,
    LLM_PROXY,
    LLM_TEMPERATURE,
    MAX_ANALYSIS_DEPTH,
    MAX_RETRIES,
    RESULT_MAX_ROWS,
)
from .analysis_workflow import (
    active_question,
    complete_plan_step,
    format_evidence,
    needs_broad_schema,
    next_plan_step,
    choose_drilldown,
    normalize_initial_plan,
    snapshot_evidence,
)
from .db import get_connection, get_relevant_schema, list_tables
from .result_validator import format_validation_feedback, validate_result
from .state import AgentState, default_state
from .viz import parse_chart_spec

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_http_clients():
    """构建显式 httpx 客户端，默认绕过环境代理。

    很多开发机配置了 socks 代理（如 Clash 的 socks://127.0.0.1:7897），
    但 httpx/openai 客户端不支持 socks 协议，会导致 ChatOpenAI 初始化失败。
    当前模型服务可按需直连，因此默认 trust_env=False 忽略环境代理；
    如需代理，在 .env 里配置 LLM_PROXY=http://... 即可。
    """
    if LLM_PROXY:
        client = httpx.Client(proxy=LLM_PROXY, trust_env=False)
        async_client = httpx.AsyncClient(proxy=LLM_PROXY, trust_env=False)
    else:
        client = httpx.Client(trust_env=False)
        async_client = httpx.AsyncClient(trust_env=False)
    return client, async_client


def _get_llm(model: str | None = None) -> Runnable:
    """优先 GLM；调用异常时以相同消息自动重试 DeepSeek。"""
    if not LLM_API_KEY and not LLM_FALLBACK_API_KEY:
        raise RuntimeError(
            f"未配置模型 API Key：请设置 {LLM_KEY_HINT} 或 DEEPSEEK_API_KEY"
        )
    http_client, http_async_client = _build_http_clients()

    def build_client(api_key: str, base_url: str, model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=LLM_TEMPERATURE,
            http_client=http_client,
            http_async_client=http_async_client,
            http_socket_options=(),
        )

    if LLM_API_KEY:
        primary = build_client(LLM_API_KEY, LLM_BASE_URL, model or LLM_MODEL)
        if LLM_FALLBACK_API_KEY:
            secondary = build_client(
                LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL
            )
            return primary.with_fallbacks([secondary])
        return primary
    return build_client(
        LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL,
        model or LLM_FALLBACK_MODEL,
    )


def _extract_sql(text: str) -> str:
    """从 LLM 输出中剥离 markdown 代码块与多余分号。"""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return text.rstrip(";").strip()


def _parse_json_object(text: str) -> Any:
    """尽力从 LLM 文本中解析出 JSON 对象，失败返回 None。"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0) if m else text)
    except Exception:  # noqa: BLE001
        return None


def _format_rows(rows: List[Dict[str, Any]], limit: int = 50) -> str:
    return json.dumps(rows[:limit], ensure_ascii=False)


FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "trigger",
    "grant", "revoke", "truncate",
)


def check_sql_safe(sql: str) -> Tuple[bool, str]:
    """SQL 安全校验：只允许单条 SELECT 查询。"""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "SQL 为空"
    parts = [p for p in s.split(";") if p.strip()]
    if len(parts) > 1:
        return False, "一次只允许执行一条 SQL"
    s = parts[0].strip() if parts else s
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "仅允许 SELECT 查询"
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", low):
            return False, f"检测到禁用关键字: {kw}"
    return True, ""


def compile_check(sql: str) -> Tuple[bool, str]:
    """用 EXPLAIN 做确定性编译校验：验证语法 / 表名 / 列名，不真正执行。"""
    conn = get_connection()
    try:
        conn.execute("EXPLAIN " + sql)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> Dict[str, Any]:
    """节点 1：意图识别。"""
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.INTENT_SYSTEM),
        HumanMessage(content=state["question"]),
    ])
    text = resp.content.strip() if isinstance(resp.content, str) else ""
    intent = "database"  # 默认走数据库流程
    data = _parse_json_object(text)
    if isinstance(data, dict) and data.get("intent") in ("database", "other"):
        intent = data["intent"]
    return {"intent": intent}


def direct_answer(state: AgentState) -> Dict[str, Any]:
    """节点 1b：非数据库问题直接回答。"""
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.DIRECT_ANSWER_SYSTEM),
        HumanMessage(content=state["question"]),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"analysis": text, "answer": text}


def retrieve_schema(state: AgentState) -> Dict[str, Any]:
    """节点 2：schema 检索。"""
    schema_question = "" if needs_broad_schema(state["question"]) else state["question"]
    schema = get_relevant_schema(schema_question)
    tables = list_tables()
    return {"schema": schema, "tables": tables}


def create_analysis_plan(state: AgentState) -> Dict[str, Any]:
    """查询前分类并规划可执行步骤；解析失败时退化为单步核心查询。"""
    prompt = prompts.INITIAL_PLAN_TEMPLATE.format(
        today=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        question=state["question"],
        schema=state.get("schema", ""),
        max_steps=MAX_ANALYSIS_DEPTH + 1,
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.INITIAL_PLAN_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    plan = normalize_initial_plan(
        _parse_json_object(text),
        question=state["question"],
        max_followups=MAX_ANALYSIS_DEPTH,
        schema=state.get("schema", ""),
    )
    return {
        "initial_plan": plan,
        "plan_steps": plan["steps"],
        "followup_question": plan["steps"][0]["question"],
    }


def generate_sql(state: AgentState) -> Dict[str, Any]:
    """节点 3：为核心问题或 Analyzer 追问生成 SQL。"""
    feedback = ""
    if state.get("feedback"):
        feedback = (
            "上一次生成的 SQL 存在以下问题，请针对性修复后重新生成：\n"
            f"{state['feedback']}\n\n"
        )
    current_question = active_question(state)
    prompt = prompts.GENERATE_SQL_TEMPLATE.format(
        schema=state.get("schema", ""),
        original_question=state["question"],
        initial_plan=json.dumps(state.get("initial_plan") or {}, ensure_ascii=False),
        question=current_question,
        analysis_context=format_evidence(state.get("analysis_evidence") or []),
        error_feedback=feedback,
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.GENERATE_SQL_SYSTEM),
        HumanMessage(content=prompt),
    ])
    sql = _extract_sql(resp.content if isinstance(resp.content, str) else str(resp.content))
    return {"sql_query": sql, "feedback": None}


def review_sql(state: AgentState) -> Dict[str, Any]:
    """节点 4：SQL 审查 = 安全校验 + 编译校验 + 语义审查。"""
    sql = state.get("sql_query", "")
    current_question = active_question(state)

    # 1. 安全校验（确定性）
    ok, reason = check_sql_safe(sql)
    if not ok:
        return {"review_issues": [reason], "feedback": reason}

    # 2. 编译校验：语法 / 表名 / 列名（确定性，EXPLAIN 不真正执行）
    ok, err = compile_check(sql)
    if not ok:
        msg = f"SQL 无法编译（表/列不存在或语法错误）：{err}"
        return {"review_issues": [msg], "feedback": msg}

    # 3. 语义审查（LLM）：COUNT/DISTINCT、JOIN 膨胀、聚合粒度、时间窗口等
    issues = _llm_review_sql(current_question, state.get("schema", ""), sql)
    if issues:
        feedback = "SQL 语义审查发现问题，请针对性修复：\n" + "\n".join(f"- {i}" for i in issues)
        return {"review_issues": issues, "feedback": feedback}

    return {"review_issues": [], "feedback": None}


def _llm_review_sql(question: str, schema: str, sql: str) -> List[str]:
    """LLM 语义审查，返回问题列表；解析失败则放行（fail-open）。"""
    prompt = prompts.REVIEW_TEMPLATE.format(question=question, schema=schema, sql=sql)
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.REVIEW_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    data = _parse_json_object(text)
    if not isinstance(data, dict):
        return []  # 解析失败放行，避免审查器自身不稳定卡死流程
    if not data.get("approved", True):
        return [str(i) for i in (data.get("issues") or [])]
    return []


def execute_sql(state: AgentState) -> Dict[str, Any]:
    """节点 5：执行 SQL。"""
    sql = state.get("sql_query", "")
    conn = get_connection()
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchmany(RESULT_MAX_ROWS + 1)]
        columns = [d[0] for d in cur.description] if cur.description else []
        return {"query_result": rows, "columns": columns, "feedback": None}
    except Exception as exc:  # noqa: BLE001
        return {"query_result": None, "columns": None, "feedback": f"SQL 执行报错：{exc}"}
    finally:
        conn.close()


def check_results(state: AgentState) -> Dict[str, Any]:
    """节点 6：确定性规则检查 + LLM 业务语义复核。"""
    rows = state.get("query_result") or []
    columns = state.get("columns") or []
    current_question = active_question(state)

    # 第一层：范围、重复、加总等高置信度规则，异常直接回溯 SQL。
    report = validate_result(
        rows, columns, question=current_question, max_rows=RESULT_MAX_ROWS
    )
    history = list(state.get("validation_history") or [])
    if not report["ok"]:
        history.append({
            "sql": state.get("sql_query", ""),
            "question": current_question,
            "analysis_depth": state.get("analysis_depth", 0),
            "retry": state.get("retry_count", 0),
            **report,
        })
        issue_messages = [str(item["message"]) for item in report["issues"]]
        return {
            "result_validation": report,
            "validation_history": history,
            "review_issues": issue_messages,
            "feedback": format_validation_feedback(report),
        }

    # 第二层：粒度、时间窗口、业务量级等需要上下文的异常由 LLM 复核。
    prompt = prompts.CHECK_RESULT_TEMPLATE.format(
        question=current_question,
        sql=state.get("sql_query", ""),
        columns=", ".join(columns),
        row_count=len(rows),
        sample=_format_rows(rows, 30),
        deterministic_report=json.dumps(report, ensure_ascii=False),
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.CHECK_RESULT_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    data = _parse_json_object(text)
    if isinstance(data, dict) and data.get("ok") is False:
        issues = [str(i) for i in (data.get("issues") or [])]
        if issues:
            llm_report = {
                **report,
                "ok": False,
                "issues": [
                    {"code": "semantic_anomaly", "severity": "error", "message": issue}
                    for issue in issues
                ],
            }
            history.append({
                "sql": state.get("sql_query", ""),
                "question": current_question,
                "analysis_depth": state.get("analysis_depth", 0),
                "retry": state.get("retry_count", 0),
                **llm_report,
            })
            return {
                "result_validation": llm_report,
                "validation_history": history,
                "review_issues": issues,
                "feedback": "执行结果语义质检发现异常：\n"
                + "\n".join(f"- {issue}" for issue in issues),
            }
    return {"result_validation": report, "review_issues": [], "feedback": None}


def repair_sql(state: AgentState) -> Dict[str, Any]:
    """当前查询失败后计数并回退到 SQL 生成。"""
    return {"retry_count": state.get("retry_count", 0) + 1}


def plan_analysis(state: AgentState) -> Dict[str, Any]:
    """判断是否需要基于当前结果继续提出一个可查询的分析问题。"""
    depth = state.get("analysis_depth", 0)
    current_question = active_question(state)
    evidence = list(state.get("analysis_evidence") or [])
    steps = complete_plan_step(state.get("plan_steps") or [], current_question)
    pending_step = next_plan_step(steps)
    evidence.append(snapshot_evidence(
        question=current_question,
        sql=state.get("sql_query", ""),
        rows=state.get("query_result"),
        columns=state.get("columns"),
        depth=depth,
    ))

    raw_plan: Any = None
    if depth < MAX_ANALYSIS_DEPTH:
        prompt = prompts.ANALYSIS_PLANNER_TEMPLATE.format(
            question=state["question"],
            schema=state.get("schema", ""),
            depth=depth,
            max_depth=MAX_ANALYSIS_DEPTH,
            initial_plan=json.dumps(
                {**(state.get("initial_plan") or {}), "steps": steps},
                ensure_ascii=False,
            ),
            next_step=json.dumps(pending_step or {}, ensure_ascii=False),
            evidence=format_evidence(evidence),
        )
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content=prompts.ANALYSIS_PLANNER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw_plan = _parse_json_object(text)

    plan = choose_drilldown(
        raw_plan,
        original_question=state["question"],
        evidence=evidence,
        depth=depth,
        max_depth=MAX_ANALYSIS_DEPTH,
        schema=state.get("schema", ""),
        pending_step=pending_step,
    )
    if plan["needs_followup"]:
        followup = plan["followup_question"]
        if pending_step and pending_step["question"] != followup:
            for step in steps:
                if step["id"] == pending_step["id"]:
                    step["status"] = "superseded"
                    break
        if not any(step.get("question") == followup for step in steps):
            steps.append({
                "id": f"step_{len(steps) + 1}",
                "question": followup,
                "purpose": plan.get("expected_insight", ""),
                "status": "pending",
            })
    else:
        for step in steps:
            if step.get("status") == "pending":
                step["status"] = "skipped"
    trace = list(state.get("analysis_trace") or [])
    trace.append({
        "round": depth + 1,
        "source_question": current_question,
        **plan,
    })

    updates: Dict[str, Any] = {
        "plan_steps": steps,
        "analysis_evidence": evidence,
        "analysis_plan": plan,
        "analysis_trace": trace,
    }
    if plan["needs_followup"]:
        updates.update({
            "followup_question": plan["followup_question"],
            "analysis_depth": depth + 1,
            "retry_count": 0,
            "feedback": None,
            "review_issues": [],
            "sql_query": "",
            "query_result": None,
            "columns": None,
            "result_validation": None,
        })
    else:
        updates["followup_question"] = None
    return updates


def abandon_followup(state: AgentState) -> Dict[str, Any]:
    """后续查询重试耗尽时，保留既有证据并降级完成分析。"""
    trace = list(state.get("analysis_trace") or [])
    trace.append({
        "round": state.get("analysis_depth", 0) + 1,
        "source_question": active_question(state),
        "needs_followup": False,
        "followup_question": "",
        "rationale": f"后续查询多次失败，已用先前证据降级总结：{state.get('feedback')}",
        "analysis_type": "query_failure_fallback",
        "expected_insight": "",
    })
    steps = [dict(step) for step in state.get("plan_steps") or []]
    for step in steps:
        if step.get("question") == active_question(state) and step.get("status") == "pending":
            step["status"] = "failed"
            break
    return {
        "plan_steps": steps,
        "followup_question": None,
        "analysis_plan": {
            "needs_followup": False,
            "rationale": "后续查询失败，使用已有证据总结",
            "analysis_type": "query_failure_fallback",
        },
        "analysis_trace": trace,
        "feedback": None,
    }


def analyze(state: AgentState) -> Dict[str, Any]:
    """基于完整 SQL 证据链做归因总结。"""
    evidence = list(state.get("analysis_evidence") or [])
    if not evidence and state.get("query_result") is not None:
        evidence.append(snapshot_evidence(
            question=active_question(state),
            sql=state.get("sql_query", ""),
            rows=state.get("query_result"),
            columns=state.get("columns"),
            depth=state.get("analysis_depth", 0),
        ))
    prompt = prompts.ANALYZE_TEMPLATE.format(
        question=state["question"],
        initial_plan=json.dumps(
            {**(state.get("initial_plan") or {}), "steps": state.get("plan_steps") or []},
            ensure_ascii=False,
        ),
        analysis_trace=json.dumps(state.get("analysis_trace") or [], ensure_ascii=False),
        evidence=format_evidence(evidence, row_limit=50),
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.ANALYZE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"analysis": text, "analysis_evidence": evidence}


def decide_chart(state: AgentState) -> Dict[str, Any]:
    """节点 9：可视化决策。"""
    rows = state.get("query_result") or []
    columns = state.get("columns") or []
    if (not rows or not columns) and state.get("analysis_evidence"):
        last_evidence = state["analysis_evidence"][-1]
        rows = last_evidence.get("rows") or []
        columns = last_evidence.get("columns") or []
    if not rows or not columns:
        return {"chart": {"should_chart": False}}

    df = pd.DataFrame(rows, columns=columns)
    sample = df.head(30).to_string(index=False)
    prompt = prompts.CHART_TEMPLATE.format(
        question=state["question"],
        columns=", ".join(columns),
        sample=sample,
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.CHART_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"chart": parse_chart_spec(text, df)}


def finalize(state: AgentState) -> Dict[str, Any]:
    """节点 10：汇总最终回答。"""
    return {"answer": state.get("analysis", "")}


def finalize_error(state: AgentState) -> Dict[str, Any]:
    """重试耗尽后的兜底回答。"""
    return {"answer": f"抱歉，多次尝试后仍无法得到正确结果。最后反馈：{state.get('feedback')}"}


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    return "direct_answer" if state.get("intent") == "other" else "retrieve_schema"


def route_after_review(state: AgentState) -> str:
    return "repair_sql" if state.get("feedback") else "execute_sql"


def route_after_execute(state: AgentState) -> str:
    return "repair_sql" if state.get("feedback") else "check_results"


def route_after_check(state: AgentState) -> str:
    return "repair_sql" if state.get("feedback") else "plan_analysis"


def route_after_analysis_plan(state: AgentState) -> str:
    plan = state.get("analysis_plan") or {}
    return "generate_sql" if plan.get("needs_followup") else "analyze"


def route_after_repair(state: AgentState) -> str:
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "generate_sql"
    if state.get("analysis_evidence") and state.get("analysis_depth", 0) > 0:
        return "abandon_followup"
    return "finalize_error"


# ---------------------------------------------------------------------------
# 组装图
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("direct_answer", direct_answer)
    g.add_node("retrieve_schema", retrieve_schema)
    g.add_node("create_analysis_plan", create_analysis_plan)
    g.add_node("generate_sql", generate_sql)
    g.add_node("review_sql", review_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("check_results", check_results)
    g.add_node("repair_sql", repair_sql)
    g.add_node("plan_analysis", plan_analysis)
    g.add_node("abandon_followup", abandon_followup)
    g.add_node("analyze", analyze)
    g.add_node("decide_chart", decide_chart)
    g.add_node("finalize", finalize)
    g.add_node("finalize_error", finalize_error)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent", route_after_intent,
        {"direct_answer": "direct_answer", "retrieve_schema": "retrieve_schema"},
    )
    g.add_edge("direct_answer", END)

    g.add_edge("retrieve_schema", "create_analysis_plan")
    g.add_edge("create_analysis_plan", "generate_sql")
    g.add_edge("generate_sql", "review_sql")
    g.add_conditional_edges(
        "review_sql", route_after_review,
        {"repair_sql": "repair_sql", "execute_sql": "execute_sql"},
    )
    g.add_conditional_edges(
        "execute_sql", route_after_execute,
        {"repair_sql": "repair_sql", "check_results": "check_results"},
    )
    g.add_conditional_edges(
        "check_results", route_after_check,
        {"repair_sql": "repair_sql", "plan_analysis": "plan_analysis"},
    )
    g.add_conditional_edges(
        "repair_sql", route_after_repair,
        {
            "generate_sql": "generate_sql",
            "abandon_followup": "abandon_followup",
            "finalize_error": "finalize_error",
        },
    )

    g.add_conditional_edges(
        "plan_analysis", route_after_analysis_plan,
        {"generate_sql": "generate_sql", "analyze": "analyze"},
    )
    g.add_edge("abandon_followup", "analyze")

    g.add_edge("analyze", "decide_chart")
    g.add_edge("decide_chart", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("finalize_error", END)

    return g.compile()


_graph = None


def run_agent(question: str) -> Tuple[AgentState, List[str]]:
    """执行一次 Agent 工作流，返回 (最终状态, 节点执行顺序)。"""
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial = default_state(question)
    state = dict(initial)
    steps: List[str] = []
    for chunk in _graph.stream(
        initial,
        stream_mode="updates",
        config={"recursion_limit": 25 + (MAX_ANALYSIS_DEPTH + 1) * (MAX_RETRIES + 5)},
    ):
        for node, update in chunk.items():
            steps.append(node)
            if update:
                state.update(update)
    return state, steps
