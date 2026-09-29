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
    # 查询前分析计划及各步骤执行状态
    initial_plan: Optional[Dict[str, Any]]
    plan_steps: List[Dict[str, Any]]

    # SQL 生成 / 审查 / 执行
    sql_query: str
    # 修复反馈：语义审查问题 / 执行错误 / 结果异常，统一作为「回炉重造」的信号
    feedback: Optional[str]
    # 审查/质检发现的问题清单（供展示）
    review_issues: Optional[List[str]]
    query_result: Optional[List[Dict[str, Any]]]
    columns: Optional[List[str]]
    # 当前结果的结构化合理性检查报告，以及每次回溯的审计历史
    result_validation: Optional[Dict[str, Any]]
    validation_history: List[Dict[str, Any]]
    retry_count: int

    # 自主分析循环：当前追问、深挖轮数、规划决策和多轮证据链
    followup_question: Optional[str]
    analysis_depth: int
    analysis_plan: Optional[Dict[str, Any]]
    analysis_trace: List[Dict[str, Any]]
    analysis_evidence: List[Dict[str, Any]]

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
        initial_plan=None,
        plan_steps=[],
        sql_query="",
        feedback=None,
        review_issues=None,
        query_result=None,
        columns=None,
        result_validation=None,
        validation_history=[],
        retry_count=0,
        followup_question=None,
        analysis_depth=0,
        analysis_plan=None,
        analysis_trace=[],
        analysis_evidence=[],
        analysis="",
        chart=None,
        answer="",
    )
