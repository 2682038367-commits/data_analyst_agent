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
    plan = result.get("initial_plan") or {}
    if plan:
        print("\n问题类型:", plan.get("question_type"))
        print("指标口径:", plan.get("metric_definition"))
        print("计划步骤:")
        for step in result.get("plan_steps") or plan.get("steps") or []:
            print(f"  {step.get('id')} [{step.get('status')}] {step.get('question')}")
        for limitation in plan.get("limitations") or []:
            print("数据限制:", limitation)
    print("\nSQL:\n", result.get("sql_query"))
    trace = result.get("analysis_trace") or []
    if trace:
        print("\nAnalyzer 决策:")
        for decision in trace:
            gain = decision.get("information_gain")
            gain_label = f" (相对信息增益 {gain}/5)" if gain is not None else ""
            followup = decision.get("followup_question")
            print(
                f"- 第 {decision.get('round')} 轮: {decision.get('rationale', '')}"
                + (f" -> {followup}" if followup else "")
                + gain_label
            )
    print("\n回答:\n", result.get("answer"))
    print("\n图表配置:", result.get("chart"))


if __name__ == "__main__":
    main()
