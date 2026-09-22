"""图表生成：图表配置解析、启发式推荐、Plotly 图表构建。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go

ALLOWED_CHART_TYPES = {"bar", "line", "pie", "scatter"}


def _numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _looks_datetime(df: pd.DataFrame, col: str) -> bool:
    name = str(col).lower()
    if any(k in name for k in ("date", "日期", "时间", "month", "月", "year", "年",
                               "day", "周", "quarter", "季度")):
        return True
    return pd.api.types.is_datetime64_any_dtype(df[col])


def suggest_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """无 LLM 时的启发式图表推荐（作为兜底）。"""
    if df.empty or df.shape[1] < 2:
        return {"should_chart": False}

    num = _numeric_cols(df)
    if not num:
        return {"should_chart": False}

    cat = [c for c in df.columns if c not in num]
    dt = [c for c in df.columns if _looks_datetime(df, c)]

    if dt:
        x, y = dt[0], num[0]
        return {"should_chart": True, "chart_type": "line", "x": x, "y": y,
                "title": f"{y} 随 {x} 的变化趋势"}

    if cat:
        x, y = cat[0], num[0]
        if df[x].nunique() <= 8:
            return {"should_chart": True, "chart_type": "pie", "x": x, "y": y,
                    "title": f"{y} 按 {x} 的占比"}
        return {"should_chart": True, "chart_type": "bar", "x": x, "y": y,
                "title": f"{y} 按 {x} 的分布"}

    if len(num) >= 2:
        return {"should_chart": True, "chart_type": "scatter", "x": num[0], "y": num[1],
                "title": f"{num[1]} 与 {num[0]} 的关系"}

    return {"should_chart": False}


def parse_chart_spec(text: str, df: pd.DataFrame) -> Dict[str, Any]:
    """解析 LLM 返回的图表配置；解析失败或非法时回退到启发式。"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        raw = m.group(0) if m else text
        spec = json.loads(raw)
        ct = spec.get("chart_type")
        if ct not in ALLOWED_CHART_TYPES:
            return suggest_chart(df)
        x, y = spec.get("x"), spec.get("y")
        if x not in df.columns or (y is not None and y not in df.columns):
            return suggest_chart(df)
        return {
            "should_chart": bool(spec.get("should_chart", True)),
            "chart_type": ct,
            "x": x,
            "y": y,
            "title": spec.get("title") or "",
        }
    except Exception:
        return suggest_chart(df)


def build_figure(spec: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    """根据图表配置构建 Plotly Figure。"""
    ct = spec["chart_type"]
    x = spec["x"]
    y = spec.get("y")
    title = spec.get("title") or ""

    d = df.copy()
    # 类别过多时只取 Top 30，避免图表拥挤
    if y and x in d.columns and d[x].nunique() > 30 and pd.api.types.is_numeric_dtype(d[y]):
        d = d.sort_values(y, ascending=False).head(30)

    if ct == "pie":
        fig = go.Figure(go.Pie(labels=d[x], values=d[y], hole=0.35))
    elif ct == "line":
        fig = go.Figure(go.Scatter(x=d[x], y=d[y], mode="lines+markers"))
    elif ct == "scatter":
        fig = go.Figure(go.Scatter(x=d[x], y=d[y], mode="markers"))
    else:  # bar
        fig = go.Figure(go.Bar(x=d[x], y=d[y]))

    fig.update_layout(title=title, template="plotly_white", height=420, margin=dict(t=60, b=40))
    return fig
