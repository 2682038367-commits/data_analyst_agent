from __future__ import annotations

import json
import unittest

from src.analysis_workflow import (
    active_question,
    complete_plan_step,
    choose_drilldown,
    schema_structure_text,
    unsupported_followup_reason,
    infer_question_type,
    next_plan_step,
    normalize_initial_plan,
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


    def test_question_types_and_bounded_initial_plan(self) -> None:
        examples = {
            "今年销售额是多少": "metric_query",
            "每月销售额走势": "trend_analysis",
            "本周 GMV 突降": "anomaly_diagnosis",
            "为什么本周 GMV 下滑": "attribution_analysis",
            "按客户群分析消费": "user_segmentation",
            "新用户次日留存": "funnel_retention",
        }
        for question, expected in examples.items():
            self.assertEqual(infer_question_type(question), expected)
        raw = {
            "question_type": "attribution_analysis",
            "metric_definition": "已完成订单 GMV",
            "decomposition": "订单数 × 每单件数 × 平均成交单价",
            "steps": [
                {"question": "对比本周与上周 GMV", "purpose": "确认降幅"},
                {"question": "对比两周订单数和客单价", "purpose": "拆解驱动"},
                {"question": "按渠道比较两周 GMV", "purpose": "定位渠道"},
                {"question": "按渠道比较两周 GMV", "purpose": "重复项"},
                {"question": "按新老用户比较两周 GMV", "purpose": "定位人群"},
            ],
        }
        plan = normalize_initial_plan(
            raw, question="为什么本周 GMV 下滑", max_followups=2
        )
        self.assertEqual(plan["question_type"], "attribution_analysis")
        self.assertEqual(len(plan["steps"]), 3)
        self.assertEqual(plan["steps"][0]["question"], "对比本周与上周 GMV")
        updated = complete_plan_step(plan["steps"], "对比本周与上周 GMV")
        self.assertEqual(updated[0]["status"], "completed")
        self.assertEqual(next_plan_step(updated)["question"], "对比两周订单数和客单价")

    def test_missing_event_data_removes_unanswerable_funnel_steps(self) -> None:
        plan = normalize_initial_plan(
            {
                "question_type": "funnel_retention",
                "steps": [
                    {"question": "计算浏览到加购转化率"},
                    {"question": "按订单状态统计订单数"},
                ],
            },
            question="浏览到支付的漏斗如何",
            max_followups=3,
            schema="CREATE TABLE orders (order_id INTEGER, status TEXT)",
        )
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["question"], "按订单状态统计订单数")
        self.assertTrue(plan["limitations"])
        unsupported_only = normalize_initial_plan(
            {
                "question_type": "funnel_retention",
                "steps": [{"question": "浏览到加购的转化率"}],
            },
            question="浏览到加购转化率是多少",
            max_followups=2,
            schema="CREATE TABLE orders (order_id INTEGER, status TEXT)",
        )
        self.assertEqual(
            unsupported_only["steps"][0]["question"],
            "按订单状态统计订单数（已完成、待支付、已取消）",
        )
        self.assertEqual(
            normalize_initial_plan(None, question="为什么 GMV 下滑", max_followups=2)[
                "question_type"
            ],
            "attribution_analysis",
        )

    def test_selects_highest_valid_information_gain(self) -> None:
        schema = "### 列\norder_id, channel\n### 示例数据\n```json\n{\"device\": \"iPad\"}\n```"
        decision = choose_drilldown(
            {
                "answer_sufficient": False,
                "sufficiency_reason": "渠道差异尚未解释",
                "candidates": [
                    {"question": "按设备分析 iPad 用户 GMV", "information_gain": 5,
                     "required_columns": ["device"]},
                    {"question": "按渠道分析 GMV 变化", "information_gain": 4,
                     "required_columns": ["channel"]},
                    {"question": "重复整体 GMV", "information_gain": 3},
                ],
            },
            original_question="为什么 GMV 下降",
            evidence=[{"question": "重复整体 GMV"}],
            depth=1, max_depth=3, schema=schema,
        )
        self.assertTrue(decision["needs_followup"])
        self.assertEqual(decision["followup_question"], "按渠道分析 GMV 变化")
        self.assertEqual(decision["information_gain"], 4)
        self.assertIn("设备", decision["considered_candidates"][0]["reason"])
        self.assertNotIn("device", schema_structure_text(schema))

    def test_sufficiency_stops_before_pending_step(self) -> None:
        decision = choose_drilldown(
            {"answer_sufficient": True, "sufficiency_reason": "直接指标已回答",
             "candidates": [{"question": "无关追问", "information_gain": 5}]},
            original_question="总销售额", evidence=[{"question": "总销售额"}],
            depth=0, max_depth=3,
            pending_step={"question": "按品类继续拆解"},
        )
        self.assertFalse(decision["needs_followup"])
        self.assertEqual(decision["analysis_type"], "sufficient")

    def test_duplicate_candidates_fall_back_to_valid_plan(self) -> None:
        decision = choose_drilldown(
            {"answer_sufficient": False, "candidates": [
                {"question": "整体 GMV", "information_gain": 5},
                {"question": "按设备分析 GMV", "information_gain": 4},
            ]},
            original_question="为什么下降",
            evidence=[{"question": "整体 GMV"}],
            depth=0, max_depth=3,
            schema="orders(order_id, channel)",
            pending_step={"question": "按渠道分析 GMV", "purpose": "定位渠道"},
        )
        self.assertEqual(decision["followup_question"], "按渠道分析 GMV")
        self.assertEqual(len(decision["considered_candidates"]), 3)
        self.assertEqual(
            unsupported_followup_reason("按设备分析 GMV", "orders(order_id, channel)"),
            "schema 缺少设备或终端字段",
        )

    def test_drilldown_depth_guard_overrides_candidates(self) -> None:
        decision = choose_drilldown(
            {"answer_sufficient": False, "candidates": [
                {"question": "继续拆解", "information_gain": 5}
            ]},
            original_question="为什么下降",
            evidence=[], depth=3, max_depth=3,
        )
        self.assertFalse(decision["needs_followup"])
        self.assertEqual(decision["analysis_type"], "depth_limit")

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


    def test_graph_executes_planned_steps_in_order(self) -> None:
        import sqlite3
        from types import SimpleNamespace
        from unittest.mock import patch

        from src import graph
        from src.state import default_state

        class FakeLLM:
            def invoke(self, messages):
                system = messages[0].content
                user = messages[1].content
                if "意图识别器" in system:
                    content = '{"intent":"database"}'
                elif "数据分析规划师" in system:
                    content = json.dumps({
                        "question_type": "attribution_analysis",
                        "metric_definition": "已完成订单 GMV",
                        "decomposition": "订单数 × 客单价",
                        "limitations": [],
                        "steps": [
                            {"question": "比较 2023 与 2024 年 GMV", "purpose": "确认变化"},
                            {"question": "按渠道比较两年 GMV", "purpose": "定位渠道"},
                        ],
                    }, ensure_ascii=False)
                elif "SQL 专家" in system:
                    if "当前这一轮需要回答的问题：按渠道比较两年 GMV" in user:
                        content = "SELECT '线上' AS channel, 20 AS gmv"
                    else:
                        content = "SELECT 2023 AS year, 100 AS gmv UNION ALL SELECT 2024, 80"
                elif "SQL 审查员" in system:
                    content = '{"approved":true,"issues":[]}'
                elif "执行结果的质检员" in system:
                    content = '{"ok":true,"issues":[]}'
                elif "自主数据分析师中的 Analyzer" in system:
                    if "当前深挖轮数：0/" in user:
                        content = '{"needs_followup":true,"followup_question":"","rationale":"继续拆渠道"}'
                    else:
                        content = '{"needs_followup":false,"followup_question":"","rationale":"证据足够"}'
                elif "数据可视化专家" in system:
                    content = '{"should_chart":false}'
                else:
                    content = "渠道变化是主要线索。"
                return SimpleNamespace(content=content)

        def connection():
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            return conn

        with (
            patch.object(graph, "_get_llm", return_value=FakeLLM()),
            patch.object(graph, "get_relevant_schema", return_value="orders(order_id, channel)"),
            patch.object(graph, "list_tables", return_value=["orders"]),
            patch.object(graph, "get_connection", side_effect=connection),
        ):
            state = dict(default_state("为什么 2024 年 GMV 变化？"))
            executed = []
            for chunk in graph.build_graph().stream(
                state, stream_mode="updates", config={"recursion_limit": 80}
            ):
                for node, update in chunk.items():
                    executed.append(node)
                    state.update(update)

        self.assertLess(executed.index("create_analysis_plan"), executed.index("generate_sql"))
        self.assertEqual([item["status"] for item in state["plan_steps"]], ["completed", "completed"])
        self.assertEqual(len(state["analysis_evidence"]), 2)
        self.assertEqual(state["analysis_evidence"][1]["question"], "按渠道比较两年 GMV")


    def test_graph_replans_from_evidence_across_three_queries(self) -> None:
        import sqlite3
        from types import SimpleNamespace
        from unittest.mock import patch

        from src import graph
        from src.state import default_state

        class FakeLLM:
            def invoke(self, messages):
                system = messages[0].content
                user = messages[1].content
                if "意图识别器" in system:
                    content = '{"intent":"database"}'
                elif "数据分析规划师" in system:
                    content = json.dumps({
                        "question_type": "attribution_analysis",
                        "steps": [
                            {"question": "比较两期 GMV", "purpose": "确认降幅"},
                            {"question": "按品类比较 GMV", "purpose": "原计划维度"},
                        ],
                    }, ensure_ascii=False)
                elif "SQL 专家" in system:
                    if "当前这一轮需要回答的问题：拆解订单数与客单价" in user:
                        content = "SELECT 90 AS order_count, 8 AS average_amount"
                    elif "当前这一轮需要回答的问题：按渠道比较 GMV" in user:
                        content = "SELECT '广告' AS channel, 72 AS gmv"
                    else:
                        content = "SELECT 100 AS previous_gmv, 72 AS current_gmv"
                elif "SQL 审查员" in system:
                    content = '{"approved":true,"issues":[]}'
                elif "执行结果的质检员" in system:
                    content = '{"ok":true,"issues":[]}'
                elif "自主数据分析师中的 Analyzer" in system:
                    if "当前深挖轮数：0/" in user:
                        content = json.dumps({
                            "answer_sufficient": False,
                            "candidates": [
                                {"question": "按设备分析 iPad 用户 GMV",
                                 "information_gain": 5, "required_columns": ["device"]},
                                {"question": "拆解订单数与客单价",
                                 "information_gain": 4, "required_columns": ["order_id"]},
                            ],
                        }, ensure_ascii=False)
                    elif "当前深挖轮数：1/" in user:
                        content = json.dumps({
                            "answer_sufficient": False,
                            "candidates": [
                                {"question": "按渠道比较 GMV",
                                 "information_gain": 5, "required_columns": ["channel"]},
                            ],
                        }, ensure_ascii=False)
                    else:
                        content = '{"answer_sufficient":true,"sufficiency_reason":"驱动与渠道证据已足够","candidates":[]}'
                elif "数据可视化专家" in system:
                    content = '{"should_chart":false}'
                else:
                    content = "GMV 下降与订单数变化及广告渠道有关。"
                return SimpleNamespace(content=content)

        def connection():
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            return conn

        with (
            patch.object(graph, "_get_llm", return_value=FakeLLM()),
            patch.object(graph, "get_relevant_schema",
                         return_value="orders(order_id, channel, status)"),
            patch.object(graph, "list_tables", return_value=["orders"]),
            patch.object(graph, "get_connection", side_effect=connection),
        ):
            state = dict(default_state("为什么 GMV 下滑？"))
            nodes = []
            for chunk in graph.build_graph().stream(
                state, stream_mode="updates", config={"recursion_limit": 80}
            ):
                for node, update in chunk.items():
                    nodes.append(node)
                    state.update(update)

        self.assertEqual(nodes.count("generate_sql"), 3)
        self.assertEqual(
            [item["question"] for item in state["analysis_evidence"]],
            ["比较两期 GMV", "拆解订单数与客单价", "按渠道比较 GMV"],
        )
        self.assertEqual(state["plan_steps"][1]["status"], "superseded")
        self.assertEqual(state["analysis_trace"][0]["information_gain"], 4)
        self.assertEqual(state["analysis_trace"][-1]["analysis_type"], "sufficient")

if __name__ == "__main__":
    unittest.main()

