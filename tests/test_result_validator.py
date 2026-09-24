from __future__ import annotations

import unittest

from src.result_validator import validate_result


class ResultValidatorTests(unittest.TestCase):
    def test_empty_result(self) -> None:
        report = validate_result([], ["销售额"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["issues"][0]["code"], "empty_result")

    def test_detects_retention_out_of_range(self) -> None:
        report = validate_result([{"月份": "2024-01", "次月留存率": 1.2}], ["月份", "次月留存率"])
        self.assertIn("retention_out_of_range", {item["code"] for item in report["issues"]})

    def test_detects_ratio_over_one_hundred_percent(self) -> None:
        report = validate_result([{"渠道": "线上", "销售占比": 101}], ["渠道", "销售占比"])
        self.assertIn("rate_out_of_range", {item["code"] for item in report["issues"]})

    def test_detects_negative_amount_but_allows_negative_profit(self) -> None:
        bad = validate_result([{"销售额": -1}], ["销售额"])
        allowed = validate_result([{"利润": -1}], ["利润"])
        self.assertIn("unexpected_negative_amount", {item["code"] for item in bad["issues"]})
        self.assertTrue(allowed["ok"])

    def test_detects_total_mismatch(self) -> None:
        rows = [
            {"品类": "A", "销售额": 30},
            {"品类": "B", "销售额": 40},
            {"品类": "合计", "销售额": 100},
        ]
        report = validate_result(rows, ["品类", "销售额"])
        self.assertIn("total_mismatch", {item["code"] for item in report["issues"]})

    def test_accepts_consistent_total(self) -> None:
        rows = [
            {"品类": "A", "销售额": 30},
            {"品类": "B", "销售额": 70},
            {"品类": "合计", "销售额": 100},
        ]
        self.assertTrue(validate_result(rows, ["品类", "销售额"])["ok"])

    def test_detects_exact_duplicates(self) -> None:
        rows = [{"城市": "上海", "订单数": 2}, {"城市": "上海", "订单数": 2}]
        report = validate_result(rows, ["城市", "订单数"])
        self.assertIn("duplicate_rows", {item["code"] for item in report["issues"]})

    def test_detects_top_n_mismatch(self) -> None:
        rows = [{"客户": str(index)} for index in range(6)]
        report = validate_result(rows, ["客户"], question="消费最高的前 5 名客户")
        self.assertIn("top_n_row_count_mismatch", {item["code"] for item in report["issues"]})


    def test_detects_excessive_row_count(self) -> None:
        rows = [{"订单": index} for index in range(3)]
        report = validate_result(rows, ["订单"], max_rows=2)
        self.assertIn("excessive_row_count", {item["code"] for item in report["issues"]})


if __name__ == "__main__":
    unittest.main()

