from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluation import build_summary, load_benchmark, results_equivalent


class EvaluationTests(unittest.TestCase):
    def test_results_equivalent_ignores_row_order_and_aliases(self) -> None:
        expected = [
            {"category": "A", "gmv": 10.0},
            {"category": "B", "gmv": 20.0},
        ]
        actual = [
            {"销售额": 20, "品类": "B"},
            {"销售额": 10.0, "品类": "A"},
        ]
        equivalent, reason = results_equivalent(
            expected,
            ["category", "gmv"],
            actual,
            ["销售额", "品类"],
        )
        self.assertTrue(equivalent, reason)

    def test_results_equivalent_detects_semantic_error(self) -> None:
        equivalent, _ = results_equivalent(
            [{"channel": "线上", "users": 3}],
            ["channel", "users"],
            [{"channel": "线上", "users": 4}],
            ["channel", "users"],
        )
        self.assertFalse(equivalent)

    def test_summary_keeps_execution_and_semantic_accuracy_separate(self) -> None:
        records = [
            {
                "category": "join",
                "execution_success": True,
                "semantic_correct": False,
                "answer_correct": False,
                "latency_seconds": 1.0,
            },
            {
                "category": "join",
                "execution_success": True,
                "semantic_correct": True,
                "answer_correct": True,
                "latency_seconds": 3.0,
            },
        ]
        summary = build_summary(records)
        self.assertEqual(summary["execution_accuracy"]["accuracy"], 1.0)
        self.assertEqual(summary["semantic_accuracy"]["accuracy"], 0.5)
        self.assertEqual(summary["answer_accuracy"]["accuracy"], 0.5)
        self.assertEqual(summary["latency_seconds"]["p50"], 2.0)

    def test_benchmark_loader_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(
                '{"id":"x","category":"a","question":"q","gold_sql":"SELECT 1"}\n'
                '{"id":"x","category":"a","question":"q2","gold_sql":"SELECT 2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复"):
                load_benchmark(path)

    def test_project_benchmark_has_fifty_unique_cases(self) -> None:
        benchmark = Path(__file__).resolve().parents[1] / "evals" / "benchmark.jsonl"
        cases = load_benchmark(benchmark)
        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case["id"] for case in cases}), 50)
        self.assertGreaterEqual(len({case["category"] for case in cases}), 8)


if __name__ == "__main__":
    unittest.main()

