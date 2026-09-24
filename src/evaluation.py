"""Benchmark 加载、结果等价性判定与评估指标汇总。"""
from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def load_benchmark(path: Path | str) -> List[Dict[str, Any]]:
    """加载 JSONL benchmark，并检查必要字段与重复 ID。"""
    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = {"id", "category", "question", "gold_sql"} - set(case)
            if missing:
                raise ValueError(f"benchmark 第 {line_number} 行缺少字段: {sorted(missing)}")
            if case["id"] in seen_ids:
                raise ValueError(f"benchmark ID 重复: {case['id']}")
            seen_ids.add(case["id"])
            cases.append(case)
    return cases


def _normalize_value(value: Any) -> Tuple[str, Any]:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return ("number", str(number))
        return ("number", round(number, 6))
    return ("text", str(value).strip())


def _canonical_rows(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
) -> Counter:
    return Counter(
        tuple(_normalize_value(row.get(column)) for column in columns)
        for row in rows
    )


def results_equivalent(
    expected_rows: Sequence[Dict[str, Any]],
    expected_columns: Sequence[str],
    actual_rows: Sequence[Dict[str, Any]],
    actual_columns: Sequence[str],
    *,
    max_permutation_columns: int = 6,
) -> Tuple[bool, str]:
    """比较两个结果集，忽略行顺序和列别名，并允许全局列顺序不同。

    这比“SQL 能执行”更接近业务语义正确性。列数较少时会尝试全局列排列，
    因而模型使用不同别名或交换 SELECT 列顺序不会被误判。
    """
    expected_columns = list(expected_columns)
    actual_columns = list(actual_columns)
    if len(expected_rows) != len(actual_rows):
        return False, f"行数不一致: expected={len(expected_rows)}, actual={len(actual_rows)}"
    if len(expected_columns) != len(actual_columns):
        return False, (
            f"列数不一致: expected={len(expected_columns)}, actual={len(actual_columns)}"
        )

    expected = _canonical_rows(expected_rows, expected_columns)
    if set(expected_columns) == set(actual_columns):
        actual = _canonical_rows(actual_rows, expected_columns)
        if expected == actual:
            return True, "结果集等价（按同名列对齐）"

    column_count = len(actual_columns)
    if column_count > max_permutation_columns:
        return False, "列名不同且列数过多，未进行排列匹配"

    for permutation in itertools.permutations(actual_columns):
        if expected == _canonical_rows(actual_rows, permutation):
            return True, "结果集等价（忽略列别名/列顺序）"
    return False, "结果值与 gold SQL 不等价"


def _wilson_interval(successes: int, total: int) -> List[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _metric(successes: int, total: int) -> Dict[str, Any]:
    return {
        "correct": successes,
        "total": total,
        "accuracy": round(successes / total, 4) if total else None,
        "ci95": _wilson_interval(successes, total),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 4)


def build_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总三项核心准确率、置信区间、延迟和分类表现。"""
    items = list(records)
    execution_correct = sum(bool(item.get("execution_success")) for item in items)
    semantic_correct = sum(bool(item.get("semantic_correct")) for item in items)
    answer_items = [item for item in items if item.get("answer_correct") is not None]
    answer_correct = sum(bool(item.get("answer_correct")) for item in answer_items)
    latencies = [
        float(item["latency_seconds"])
        for item in items
        if item.get("latency_seconds") is not None
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("category", "unknown"))].append(item)

    by_category: Dict[str, Any] = {}
    for category, category_items in sorted(grouped.items()):
        judged = [item for item in category_items if item.get("answer_correct") is not None]
        by_category[category] = {
            "count": len(category_items),
            "execution_accuracy": _metric(
                sum(bool(item.get("execution_success")) for item in category_items),
                len(category_items),
            )["accuracy"],
            "semantic_accuracy": _metric(
                sum(bool(item.get("semantic_correct")) for item in category_items),
                len(category_items),
            )["accuracy"],
            "answer_accuracy": _metric(
                sum(bool(item.get("answer_correct")) for item in judged),
                len(judged),
            )["accuracy"],
        }

    return {
        "case_count": len(items),
        "execution_accuracy": _metric(execution_correct, len(items)),
        "semantic_accuracy": _metric(semantic_correct, len(items)),
        "answer_accuracy": _metric(answer_correct, len(answer_items)),
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "by_category": by_category,
    }

