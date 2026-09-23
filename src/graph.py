"""LangGraph SQL Agent 的核心工作流。

流程（可视化）：
    用户问题 → 意图识别 → Schema检索 → SQL生成 → SQL审查(SQL Reviewer)
        → 执行 → 执行结果质检 →(异常→修复→重新生成) 分析 → 可视化 → 回答

SQL 审查分三层：
    1. 安全校验（只允许 SELECT、单语句）—— 确定性
    2. 编译校验（EXPLAIN 验证语法/表名/列名，不真正执行）—— 确定性
    3. 语义审查（LLM 检查 COUNT/DISTINCT、JOIN 膨胀、聚合粒度、时间窗口等）—— LLM
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import httpx
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROXY,
    LLM_TEMPERATURE,
    MAX_RETRIES,
)
from .db import get_connection, get_relevant_schema, list_tables
from .state import AgentState, default_state
from .viz import parse_chart_spec

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_http_clients():
    """构建显式 httpx 客户端，默认绕过环境代理。

    很多开发机配置了 socks 代理（如 Clash 的 socks://127.0.0.1:7897），
    但 httpx/openai 客户端不支持 socks 协议，会导致 ChatOpenAI 初始化失败。
    DeepSeek 国内直连无需代理，因此默认 trust_env=False 忽略环境代理；
    如需代理，在 .env 里配置 LLM_PROXY=http://... 即可。
    """
    if LLM_PROXY:
        client = httpx.Client(proxy=LLM_PROXY, trust_env=False)
        async_client = httpx.AsyncClient(proxy=LLM_PROXY, trust_env=False)
    else:
        client = httpx.Client(trust_env=False)
        async_client = httpx.AsyncClient(trust_env=False)
    return client, async_client


def _get_llm() -> ChatOpenAI:
    http_client, http_async_client = _build_http_clients()
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        http_client=http_client,
        http_async_client=http_async_client,
        http_socket_options=(),
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
    schema = get_relevant_schema(state["question"])
    tables = list_tables()
    return {"schema": schema, "tables": tables}


def generate_sql(state: AgentState) -> Dict[str, Any]:
    """节点 3：SQL 生成（修复时携带上次的反馈信息）。"""
    feedback = ""
    if state.get("feedback"):
        feedback = (
            "上一次生成的 SQL 存在以下问题，请针对性修复后重新生成：\n"
            f"{state['feedback']}\n\n"
        )
    prompt = prompts.GENERATE_SQL_TEMPLATE.format(
        schema=state.get("schema", ""),
        question=state["question"],
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
    issues = _llm_review_sql(state["question"], state.get("schema", ""), sql)
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
        rows = [dict(r) for r in cur.fetchall()]
        columns = [d[0] for d in cur.description] if cur.description else []
        return {"query_result": rows, "columns": columns, "feedback": None}
    except Exception as exc:  # noqa: BLE001
        return {"query_result": None, "columns": None, "feedback": f"SQL 执行报错：{exc}"}
    finally:
        conn.close()


def check_results(state: AgentState) -> Dict[str, Any]:
    """节点 6：执行结果质检（成功执行不代表结果正确）。"""
    rows = state.get("query_result") or []
    columns = state.get("columns") or []
    prompt = prompts.CHECK_RESULT_TEMPLATE.format(
        question=state["question"],
        sql=state.get("sql_query", ""),
        columns=", ".join(columns),
        row_count=len(rows),
        sample=_format_rows(rows, 30),
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
            return {"review_issues": issues, "feedback": "执行结果质检发现异常：" + "; ".join(issues)}
    return {"review_issues": [], "feedback": None}


def repair_sql(state: AgentState) -> Dict[str, Any]:
    """节点 7：失败修复（计数 + 回退到 SQL 生成）。"""
    return {"retry_count": state.get("retry_count", 0) + 1}


def analyze(state: AgentState) -> Dict[str, Any]:
    """节点 8：结果分析。"""
    rows = state.get("query_result") or []
    columns = state.get("columns") or []
    prompt = prompts.ANALYZE_TEMPLATE.format(
        question=state["question"],
        sql=state.get("sql_query", ""),
        columns=", ".join(columns),
        row_count=len(rows),
        sample=_format_rows(rows),
    )
    llm = _get_llm()
    resp = llm.invoke([
        SystemMessage(content=prompts.ANALYZE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"analysis": text}


def decide_chart(state: AgentState) -> Dict[str, Any]:
    """节点 9：可视化决策。"""
    rows = state.get("query_result") or []
    columns = state.get("columns") or []
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
    return "repair_sql" if state.get("feedback") else "analyze"


def route_after_repair(state: AgentState) -> str:
    return "generate_sql" if state.get("retry_count", 0) < MAX_RETRIES else "finalize_error"


# ---------------------------------------------------------------------------
# 组装图
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("direct_answer", direct_answer)
    g.add_node("retrieve_schema", retrieve_schema)
    g.add_node("generate_sql", generate_sql)
    g.add_node("review_sql", review_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("check_results", check_results)
    g.add_node("repair_sql", repair_sql)
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

    g.add_edge("retrieve_schema", "generate_sql")
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
        {"repair_sql": "repair_sql", "analyze": "analyze"},
    )
    g.add_conditional_edges(
        "repair_sql", route_after_repair,
        {"generate_sql": "generate_sql", "finalize_error": "finalize_error"},
    )

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
    for chunk in _graph.stream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            steps.append(node)
            if update:
                state.update(update)
    return state, steps
