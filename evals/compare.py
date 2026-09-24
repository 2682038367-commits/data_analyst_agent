#!/usr/bin/env python3
"""合并多种模式的 JSONL 明细，生成 Baseline / Reviewer / Full 对比报告。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import build_summary  # noqa: E402


def load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _accuracy(summary: Dict[str, Any], name: str) -> str:
    value = summary[name]["accuracy"]
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _exact_mcnemar_p(improved: int, regressed: int) -> float | None:
    """双侧 exact McNemar/binomial p-value，仅用于同一题集的配对比较。"""
    discordant = improved + regressed
    if not discordant:
        return None
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(improved, regressed) + 1)
    ) / (2 ** discordant)
    return round(min(1.0, 2 * tail), 6)


def paired_delta(
    baseline: List[Dict[str, Any]],
    candidate: List[Dict[str, Any]],
    field: str,
) -> Dict[str, Any]:
    left = {item["id"]: item for item in baseline}
    right = {item["id"]: item for item in candidate}
    shared = sorted(set(left) & set(right))
    if field == "answer_correct":
        shared = [
            item_id for item_id in shared
            if left[item_id].get(field) is not None
            and right[item_id].get(field) is not None
        ]
    improved = sum(
        not bool(left[item_id].get(field)) and bool(right[item_id].get(field))
        for item_id in shared
    )
    regressed = sum(
        bool(left[item_id].get(field)) and not bool(right[item_id].get(field))
        for item_id in shared
    )
    unchanged = len(shared) - improved - regressed
    baseline_accuracy = (
        sum(bool(left[item_id].get(field)) for item_id in shared) / len(shared)
        if shared else None
    )
    candidate_accuracy = (
        sum(bool(right[item_id].get(field)) for item_id in shared) / len(shared)
        if shared else None
    )
    return {
        "paired_cases": len(shared),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta": (
            round(candidate_accuracy - baseline_accuracy, 4)
            if shared else None
        ),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "mcnemar_exact_p": _exact_mcnemar_p(improved, regressed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Evaluation 模式对比报告")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "results" / "comparison.md",
    )
    args = parser.parse_args()

    records = load_records(args.results)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["mode"])].append(record)
    summaries = {mode: build_summary(items) for mode, items in grouped.items()}

    lines = [
        "# SQL Agent Evaluation Comparison",
        "",
        "| Mode | N | Execution Accuracy | SQL Semantic Accuracy | Answer Accuracy | p50 Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    preferred_order = ["baseline", "reviewer", "full"]
    modes = [mode for mode in preferred_order if mode in summaries]
    modes.extend(sorted(set(summaries) - set(modes)))
    for mode in modes:
        summary = summaries[mode]
        lines.append(
            f"| {mode} | {summary['case_count']} | "
            f"{_accuracy(summary, 'execution_accuracy')} | "
            f"{_accuracy(summary, 'semantic_accuracy')} | "
            f"{_accuracy(summary, 'answer_accuracy')} | "
            f"{summary['latency_seconds']['p50']}s |"
        )

    comparisons: Dict[str, Any] = {}
    if "baseline" in grouped:
        for mode in modes:
            if mode == "baseline":
                continue
            lines += ["", f"## baseline → {mode}", ""]
            comparisons[mode] = {}
            for field, label in (
                ("execution_success", "Execution Accuracy"),
                ("semantic_correct", "SQL Semantic Accuracy"),
                ("answer_correct", "Answer Accuracy"),
            ):
                delta = paired_delta(grouped["baseline"], grouped[mode], field)
                comparisons[mode][field] = delta
                delta_text = (
                    "N/A" if delta["delta"] is None else f"{delta['delta'] * 100:+.1f}pp"
                )
                lines.append(
                    f"- {label}: {delta_text}; improved={delta['improved']}, "
                    f"regressed={delta['regressed']}, "
                    f"McNemar exact p={delta['mcnemar_exact_p']}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"summaries": summaries, "paired_comparisons": comparisons},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"对比报告: {args.output}")


if __name__ == "__main__":
    main()

