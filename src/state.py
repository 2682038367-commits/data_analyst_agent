"""Agent 状态定义。

LangGraph 各节点通过读写这个共享状态对象来协作。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # 输入
    question: str

    # 意图识别
    intent: str  # "database" | "other"

    # schema 检索
    schema: str
    tables: List[str]

    # SQL 生成 / 校验 / 执行
    sql_query: str
    sql_error: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]
    columns: Optional[List[str]]
    retry_count: int

    # 分析 / 可视化 / 回答
    analysis: str
    chart: Optional[Dict[str, Any]]
    answer: str


def default_state(question: str = "") -> AgentState:
    """返回一个带默认值的初始状态。"""
    return AgentState(
        question=question,
        intent="",
        schema="",
        tables=[],
        sql_query="",
        sql_error=None,
        query_result=None,
        columns=None,
        retry_count=0,
        analysis="",
        chart=None,
        answer="",
    )
