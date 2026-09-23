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

    sql = result.get("sql_query")
    if sql:
        st.markdown("**🧩 生成的 SQL**")
        st.code(sql, language="sql")

    rows = result.get("query_result")
    columns = result.get("columns")
    if rows is not None and columns:
        df = pd.DataFrame(rows, columns=columns)
        st.markdown("**📋 查询结果**")
        st.dataframe(df, use_container_width=True)

        chart = result.get("chart")
        if chart and chart.get("should_chart"):
            try:
                fig = build_figure(chart, df)
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
st.caption("用自然语言查询数据库，自动完成 意图识别 → Schema 检索 → SQL 生成 → SQL 语义审查 → 执行 → 结果质检 → 分析 → 可视化 → 回答")

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
