"""命令行快速测试：python cli.py "你的问题"。"""
from __future__ import annotations

import sys

from src.db import init_db
from src.graph import run_agent


def main() -> None:
    init_db()
    question = sys.argv[1] if len(sys.argv) > 1 else "2024 年每月销售额是多少？"
    result, steps = run_agent(question)

    print("流程:", " -> ".join(steps))
    print("\nSQL:\n", result.get("sql_query"))
    trace = result.get("analysis_trace") or []
    if trace:
        print("\nAnalyzer 决策:")
        for decision in trace:
            print(
                f"- 第 {decision.get('round')} 轮: {decision.get('rationale', '')}"
                + (
                    f" -> {decision.get('followup_question')}"
                    if decision.get("followup_question")
                    else ""
                )
            )
    print("\n回答:\n", result.get("answer"))
    print("\n图表配置:", result.get("chart"))


if __name__ == "__main__":
    main()
