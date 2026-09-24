"""SQL 查询结果的确定性合理性校验。"""
from __future__ import annotations

import math
import re
from collections import Counter
from numbers import Number
from typing import Any, Dict, List, Sequence


TOTAL_LABELS = {"总计", "合计", "总和", "全部", "整体", "total", "grand total"}
RETENTION_RE = re.compile(r"留存|retention", re.IGNORECASE)
RATE_RE = re.compile(r"率|占比|比例|百分比|percent|percentage|ratio|rate", re.IGNORECASE)
AMOUNT_RE = re.compile(
    r"金额|销售额|营收|收入|流水|成交额|客单价|单价|售价|成本|amount|revenue|sales|price|cost",
    re.IGNORECASE,
)
# 这些指标出现负值可能正是业务含义，不能按“异常金额”拦截。
SIGNED_AMOUNT_RE = re.compile(
    r"利润|亏损|损益|差额|变化|增长|退款|退货|折扣|净额|profit|loss|delta|change|refund|discount|net",
    re.IGNORECASE,
)
NON_ADDITIVE_RE = re.compile(
    r"(^|_)id$|编号|序号|排名|rank|年份|月份|日期|year|month|date|率|占比|比例|平均|均值|avg|average",
    re.IGNORECASE,
)
TOP_N_RE = re.compile(r"(?:前\s*|top\s*|最多\s*)(\d+)\s*(?:名|个|条|项)?", re.IGNORECASE)


def _issue(
    code: str,
    message: str,
    *,
    column: str | None = None,
    observed: Any = None,
    expected: str | None = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {"code": code, "severity": "error", "message": message}
    if column is not None:
        item["column"] = column
    if observed is not None:
        item["observed"] = observed
    if expected is not None:
        item["expected"] = expected
    return item


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _numeric_values(rows: Sequence[Dict[str, Any]], column: str) -> List[float]:
    return [float(row[column]) for row in rows if _is_number(row.get(column))]


def _stable_value(value: Any) -> Any:
    """把常见的不可哈希值变成可比较形式，供整行重复检查使用。"""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _stable_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_stable_value(v) for v in value), key=repr))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _total_row_indices(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> List[int]:
    indices: List[int] = []
    for index, row in enumerate(rows):
        labels = [
            str(row.get(column)).strip().lower()
            for column in columns
            if row.get(column) is not None and not _is_number(row.get(column))
        ]
        if labels and all(label in TOTAL_LABELS for label in labels):
            indices.append(index)
    return indices


def validate_result(
    rows: Sequence[Dict[str, Any]] | None,
    columns: Sequence[str] | None,
    *,
    question: str = "",
    max_rows: int = 10_000,
) -> Dict[str, Any]:
    """返回可序列化报告；``ok=False`` 表示应回溯 SQL。"""
    materialized_rows = list(rows or [])
    materialized_columns = list(columns or [])
    issues: List[Dict[str, Any]] = []
    row_count = len(materialized_rows)

    if not materialized_rows:
        issues.append(_issue(
            "empty_result", "查询结果为空，可能是时间、状态或关联条件过严",
            observed=0, expected="至少返回 1 行",
        ))
        return {"ok": False, "row_count": 0, "duplicate_count": 0, "issues": issues}

    if row_count > max_rows:
        issues.append(_issue(
            "excessive_row_count",
            f"返回 {row_count} 行，超过合理性检查阈值 {max_rows}，可能缺少聚合或过滤",
            observed=row_count, expected=f"不超过 {max_rows} 行",
        ))

    top_n = TOP_N_RE.search(question)
    if top_n and row_count > int(top_n.group(1)):
        expected_count = int(top_n.group(1))
        issues.append(_issue(
            "top_n_row_count_mismatch",
            f"问题要求前 {expected_count} 项，但结果返回 {row_count} 行",
            observed=row_count, expected=f"不超过 {expected_count} 行",
        ))

    signatures = [
        tuple(_stable_value(row.get(column)) for column in materialized_columns)
        for row in materialized_rows
    ]
    duplicate_count = sum(count - 1 for count in Counter(signatures).values() if count > 1)
    if duplicate_count:
        issues.append(_issue(
            "duplicate_rows",
            f"结果中存在 {duplicate_count} 条完全重复记录，可能由 JOIN 膨胀或分组遗漏导致",
            observed=duplicate_count, expected="无完全重复记录",
        ))

    for column in materialized_columns:
        values = _numeric_values(materialized_rows, column)
        if not values:
            continue
        non_finite = [value for value in values if not math.isfinite(value)]
        if non_finite:
            issues.append(_issue(
                "non_finite_number", f"列「{column}」包含 NaN 或无穷大",
                column=column, observed=non_finite[:3], expected="有限数值",
            ))
            continue

        if RETENTION_RE.search(column):
            invalid = [value for value in values if value < 0 or value > 1]
            if invalid:
                issues.append(_issue(
                    "retention_out_of_range", f"留存率列「{column}」存在超出 [0, 1] 的值",
                    column=column, observed=invalid[:3], expected="0 <= 留存率 <= 1",
                ))
        elif RATE_RE.search(column):
            invalid = [value for value in values if value < 0 or value > 100]
            if invalid:
                issues.append(_issue(
                    "rate_out_of_range", f"比率列「{column}」存在负值或超过 100% 的值",
                    column=column, observed=invalid[:3], expected="0 <= 比率 <= 100%",
                ))

        if AMOUNT_RE.search(column) and not SIGNED_AMOUNT_RE.search(column):
            negative = [value for value in values if value < 0]
            if negative:
                issues.append(_issue(
                    "unexpected_negative_amount", f"金额列「{column}」存在明显异常负值",
                    column=column, observed=negative[:3], expected="金额不小于 0",
                ))

    total_indices = _total_row_indices(materialized_rows, materialized_columns)
    if len(total_indices) == 1 and row_count > 1:
        total_index = total_indices[0]
        detail_rows = [row for index, row in enumerate(materialized_rows) if index != total_index]
        total_row = materialized_rows[total_index]
        for column in materialized_columns:
            total_value = total_row.get(column)
            if not _is_number(total_value) or NON_ADDITIVE_RE.search(column):
                continue
            detail_values = _numeric_values(detail_rows, column)
            if len(detail_values) != len(detail_rows):
                continue
            detail_sum = math.fsum(detail_values)
            tolerance = max(1e-6, abs(float(total_value)) * 1e-6)
            if not math.isclose(float(total_value), detail_sum, rel_tol=1e-6, abs_tol=tolerance):
                issues.append(_issue(
                    "total_mismatch",
                    f"列「{column}」的总量 {total_value} 与分组加总 {detail_sum:g} 不一致",
                    column=column,
                    observed={"total": float(total_value), "group_sum": detail_sum},
                    expected="总量与分组加总一致",
                ))

    return {
        "ok": not issues,
        "row_count": row_count,
        "duplicate_count": duplicate_count,
        "issues": issues,
    }


def format_validation_feedback(report: Dict[str, Any]) -> str:
    """把结构化报告转为 SQL 生成节点可直接使用的修复反馈。"""
    messages = [str(item.get("message", "未知异常")) for item in report.get("issues", [])]
    return "执行结果确定性校验发现异常，请回溯并修复 SQL：\n" + "\n".join(
        f"- {message}" for message in messages
    )

