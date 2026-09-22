"""LangGraph SQL Agent 的核心工作流。

流程（可视化）：
    用户问题 → 意图识别 → Schema检索 → SQL生成 → SQL检查
                → 执行 →(失败→修复→重新生成) 分析 → 可视化 → 回答
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import httpx
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


def _format_rows(rows: List[Dict[str, Any]], limit: int = 50) -> str:
    return json.dumps(rows[:limit], ensure_ascii=False)


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
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if data.get("intent") in ("database", "other"):
            intent = data["intent"]
    except Exception:
        intent = "database"
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
    """节点 3：SQL 生成（含修复时携带上次错误信息）。"""
    feedback = ""
    if state.get("sql_error"):
        feedback = (
            "上一次生成的 SQL 执行/校验失败，请根据错误信息修复：\n"
            f"{state['sql_error']}\n\n"
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
    return {"sql_query": sql, "sql_error": None}


def validate_sql(state: AgentState) -> Dict[str, Any]:
    """节点 4：SQL 检查。"""
    ok, reason = check_sql_safe(state.get("sql_query", ""))
    if not ok:
        return {"sql_error": reason}
    return {"sql_error": None}


def execute_sql(state: AgentState) -> Dict[str, Any]:
    """节点 5：执行 SQL。"""
    sql = state.get("sql_query", "")
    conn = get_connection()
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        columns = [d[0] for d in cur.description] if cur.description else []
        return {"query_result": rows, "columns": columns, "sql_error": None}
    except Exception as exc:  # noqa: BLE001
        return {"query_result": None, "columns": None, "sql_error": str(exc)}
    finally:
        conn.close()


def repair_sql(state: AgentState) -> Dict[str, Any]:
    """节点 6：失败修复（计数 + 回退到 SQL 生成）。"""
    return {"retry_count": state.get("retry_count", 0) + 1}


def analyze(state: AgentState) -> Dict[str, Any]:
    """节点 7：结果分析。"""
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
    """节点 8：可视化决策。"""
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
    """节点 9：汇总最终回答。"""
    return {"answer": state.get("analysis", "")}


def finalize_error(state: AgentState) -> Dict[str, Any]:
    """重试耗尽后的兜底回答。"""
    return {"answer": f"抱歉，多次尝试后仍无法生成可执行的 SQL。错误信息：{state.get('sql_error')}"}


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    return "direct_answer" if state.get("intent") == "other" else "retrieve_schema"


def route_after_validate(state: AgentState) -> str:
    return "repair_sql" if state.get("sql_error") else "execute_sql"


def route_after_execute(state: AgentState) -> str:
    return "repair_sql" if state.get("sql_error") else "analyze"


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
    g.add_node("validate_sql", validate_sql)
    g.add_node("execute_sql", execute_sql)
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
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges(
        "validate_sql", route_after_validate,
        {"repair_sql": "repair_sql", "execute_sql": "execute_sql"},
    )
    g.add_conditional_edges(
        "execute_sql", route_after_execute,
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
