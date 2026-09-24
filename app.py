"""Streamlit 交互界面：输入问题 → 展示 SQL / 结果 / 图表 / 回答。"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.config import DB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.db import get_table_schema, init_db, list_tables
from src.graph import run_agent
from src.viz import build_figure

st.set_page_config(page_title="LangGraph SQL Agent", page_icon="🛢️", layout="wide")

# 首次启动自动建表 + 灌入示例数据
init_db()


def render_result(result: Dict[str, Any], steps: List[str]) -> None:
    with st.expander("🔍 Agent 执行流程", expanded=False):
        st.write(" → ".join(steps))
        issues = result.get("review_issues")
        if issues:
            st.markdown("**审查/质检发现的问题：**")
            for i in issues:
                st.markdown(f"- {i}")
        history = result.get("validation_history") or []
        if history:
            st.markdown("**Result Validator 回溯历史：**")
            for attempt in history:
                messages = "；".join(
                    str(item.get("message", "未知异常"))
                    for item in attempt.get("issues", [])
                )
                st.markdown(f"- 第 {attempt.get('retry', 0) + 1} 次结果：{messages}")

    trace = result.get("analysis_trace") or []
    if trace:
        with st.expander("🧠 Analyzer 决策轨迹", expanded=False):
            for decision in trace:
                followup = decision.get("followup_question")
                st.markdown(
                    f"**第 {decision.get('round')} 轮**："
                    f"{decision.get('rationale', '')}"
                )
                if followup:
                    st.markdown(f"下一步查询：{followup}")

    evidence = result.get("analysis_evidence") or []
    display_rows = result.get("query_result")
    display_columns = result.get("columns") or []
    if evidence:
        st.markdown("**🧭 自主分析证据链**")
        for item in evidence:
            with st.expander(
                f"第 {item.get('round')} 轮：{item.get('question', '')}",
                expanded=len(evidence) == 1,
            ):
                st.code(item.get("sql") or "", language="sql")
                item_rows = item.get("rows") or []
                item_columns = item.get("columns") or []
                if item_rows and item_columns:
                    st.dataframe(
                        pd.DataFrame(item_rows, columns=item_columns),
                        use_container_width=True,
                    )
                if item.get("truncated"):
                    st.caption(f"仅展示前 {len(item_rows)} 行，共 {item.get('row_count')} 行")
        display_rows = evidence[-1].get("rows") or []
        display_columns = evidence[-1].get("columns") or []
    else:
        sql = result.get("sql_query")
        if sql:
            st.markdown("**🧩 生成的 SQL**")
            st.code(sql, language="sql")
        if display_rows is not None and display_columns:
            st.markdown("**📋 查询结果**")
            st.dataframe(
                pd.DataFrame(display_rows, columns=display_columns),
                use_container_width=True,
            )

    chart = result.get("chart")
    if chart and chart.get("should_chart") and display_rows and display_columns:
        try:
            fig = build_figure(
                chart,
                pd.DataFrame(display_rows, columns=display_columns),
            )
            st.markdown("**📊 可视化**")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"图表生成失败：{exc}")

    st.markdown("**💬 回答**")
    st.markdown(result.get("answer") or "(无回答)")


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 配置")
    st.caption(f"模型：`{LLM_MODEL}`")
    st.caption(f"接口：`{LLM_BASE_URL}`")
    st.caption(f"数据库：`{DB_PATH}`")
    st.divider()
    st.subheader("🗄️ 数据表")
    for t in list_tables():
        with st.expander(t):
            st.code(get_table_schema(t), language="text")

# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------
st.title("🛢️ LangGraph SQL Agent")
st.caption("用自然语言查询数据库，自动完成 意图识别 → Schema 检索 → SQL 生成 → SQL 语义审查 → 执行 → 结果质检 → Analyzer 自主下钻 → 多轮归因 → 可视化 → 回答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("输入你的问题，例如：2024 年每月销售额是多少？"):
    if not LLM_API_KEY:
        st.error("未检测到 API Key。请复制 `.env.example` 为 `.env` 并填入 `DEEPSEEK_API_KEY`。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent 工作中…"):
                try:
                    result, steps = run_agent(prompt)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"运行出错：{exc}")
                    st.session_state.messages.append({"role": "assistant", "content": f"运行出错：{exc}"})
                else:
                    render_result(result, steps)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.get("answer", "")}
                    )
