from __future__ import annotations

import json
import unittest

from src.analysis_workflow import (
    active_question,
    format_evidence,
    needs_broad_schema,
    normalize_analysis_plan,
    snapshot_evidence,
)


class AnalysisWorkflowTests(unittest.TestCase):
    def test_causal_questions_request_broad_schema(self) -> None:
        self.assertTrue(needs_broad_schema("为什么这个月 GMV 下降了？"))
        self.assertTrue(needs_broad_schema("分析销售额异常波动的原因"))
        self.assertFalse(needs_broad_schema("销售额最高的前 5 个品类"))


    def test_followup_question_becomes_active(self) -> None:
        state = {"question": "为什么 GMV 下降？", "followup_question": "按渠道拆解 GMV 变化"}
        self.assertEqual(active_question(state), "按渠道拆解 GMV 变化")
        self.assertEqual(active_question({"question": "原始问题"}), "原始问题")

    def test_snapshot_limits_rows_but_preserves_total_count(self) -> None:
        rows = [{"月份": str(index), "GMV": index} for index in range(5)]
        item = snapshot_evidence(
            question="月度 GMV",
            sql="SELECT ...",
            rows=rows,
            columns=["月份", "GMV"],
            depth=0,
            row_limit=2,
        )
        self.assertEqual(item["row_count"], 5)
        self.assertEqual(len(item["rows"]), 2)
        self.assertTrue(item["truncated"])

    def test_allows_a_new_high_value_followup(self) -> None:
        evidence = [{"question": "为什么 GMV 下降？"}]
        plan = normalize_analysis_plan(
            {
                "needs_followup": True,
                "followup_question": "对比本月和上月的订单数、每单件数及平均成交单价",
                "rationale": "先拆解 GMV 驱动",
                "analysis_type": "driver_decomposition",
                "expected_insight": "识别影响最大的乘法因子",
            },
            original_question="为什么 GMV 下降？",
            evidence=evidence,
            depth=0,
            max_depth=3,
        )
        self.assertTrue(plan["needs_followup"])
        self.assertEqual(plan["analysis_type"], "driver_decomposition")

    def test_stops_duplicate_followup(self) -> None:
        evidence = [
            {"question": "为什么 GMV 下降？"},
            {"question": "按渠道拆解 GMV 变化"},
        ]
        plan = normalize_analysis_plan(
            {
                "needs_followup": True,
                "followup_question": "按渠道拆解 GMV 变化",
                "rationale": "继续查询",
            },
            original_question="为什么 GMV 下降？",
            evidence=evidence,
            depth=1,
            max_depth=3,
        )
        self.assertFalse(plan["needs_followup"])
        self.assertEqual(plan["analysis_type"], "duplicate_guard")

    def test_stops_at_depth_limit(self) -> None:
        plan = normalize_analysis_plan(
            {"needs_followup": True, "followup_question": "继续拆解"},
            original_question="为什么下降？",
            evidence=[],
            depth=3,
            max_depth=3,
        )
        self.assertFalse(plan["needs_followup"])
        self.assertEqual(plan["analysis_type"], "depth_limit")

    def test_unparseable_plan_fails_open_to_summary(self) -> None:
        plan = normalize_analysis_plan(
            "not-json",
            original_question="问题",
            evidence=[],
            depth=0,
            max_depth=3,
        )
        self.assertFalse(plan["needs_followup"])
        self.assertEqual(plan["analysis_type"], "parse_fallback")

    def test_evidence_prompt_is_valid_json(self) -> None:
        evidence = [{"round": 1, "question": "问题", "rows": [{"值": 1}], "row_count": 1}]
        parsed = json.loads(format_evidence(evidence))
        self.assertEqual(parsed[0]["rows"], [{"值": 1}])


if __name__ == "__main__":
    unittest.main()

