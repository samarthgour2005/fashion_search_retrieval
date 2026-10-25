"""
tests/test_evaluation_metrics.py
---------------------------------
Unit tests for evaluation/metrics.py — no GPU / index required.
"""
import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    compute_aggregate_metrics,
)


class TestPrecisionAtK(unittest.TestCase):
    def test_all_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3], {1, 2, 3}, k=3), 1.0)

    def test_none_relevant(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3], {4, 5}, k=3), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3, 4], {1, 3}, k=4), 0.5)

    def test_k_zero(self):
        self.assertEqual(precision_at_k([1, 2], {1}, k=0), 0.0)


class TestRecallAtK(unittest.TestCase):
    def test_all_found(self):
        self.assertAlmostEqual(recall_at_k([1, 2, 3], {1, 2}, k=3), 1.0)

    def test_none_found(self):
        self.assertAlmostEqual(recall_at_k([4, 5], {1, 2}, k=2), 0.0)

    def test_empty_relevant(self):
        self.assertEqual(recall_at_k([1, 2], set(), k=2), 0.0)


class TestAveragePrecision(unittest.TestCase):
    def test_perfect(self):
        # ranked = [r, r, r], relevant = {1,2,3}  -> AP = 1.0
        self.assertAlmostEqual(average_precision([1, 2, 3], {1, 2, 3}), 1.0)

    def test_first_hit_at_rank2(self):
        # ranked=[0, 1], relevant={1} -> hits@2 => P@2=0.5 -> AP = 0.5/1 = 0.5
        self.assertAlmostEqual(average_precision([0, 1], {1}), 0.5)

    def test_no_hits(self):
        self.assertAlmostEqual(average_precision([1, 2], {3, 4}), 0.0)

    def test_empty_relevant(self):
        self.assertEqual(average_precision([1, 2], set()), 0.0)


class TestNDCGAtK(unittest.TestCase):
    def test_perfect(self):
        # Ideal ranking = [1, 2, 3], so NDCG@3 should be 1.0
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3), 1.0)

    def test_worst(self):
        # No relevant item in top K
        self.assertAlmostEqual(ndcg_at_k([4, 5, 6], {1, 2}, k=3), 0.0)

    def test_k_zero(self):
        self.assertEqual(ndcg_at_k([1, 2], {1}, k=0), 0.0)

    def test_single_hit_at_rank1(self):
        dcg = 1.0 / math.log2(2)   # = 1.0
        idcg = 1.0 / math.log2(2)  # = 1.0
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1}, k=3), dcg / idcg)


class TestReciprocalRank(unittest.TestCase):
    def test_first_hit(self):
        self.assertAlmostEqual(reciprocal_rank([1, 2, 3], {1}), 1.0)

    def test_second_hit(self):
        self.assertAlmostEqual(reciprocal_rank([0, 1, 2], {1}), 0.5)

    def test_no_hit(self):
        self.assertAlmostEqual(reciprocal_rank([4, 5], {1}), 0.0)


class TestHitRateAtK(unittest.TestCase):
    def test_hit(self):
        self.assertEqual(hit_rate_at_k([1, 2, 3], {3}, k=3), 1.0)

    def test_miss(self):
        self.assertEqual(hit_rate_at_k([1, 2], {5}, k=2), 0.0)

    def test_hit_beyond_k(self):
        self.assertEqual(hit_rate_at_k([1, 2, 3], {3}, k=2), 0.0)


class TestAggregateMetrics(unittest.TestCase):
    def test_basic_aggregate(self):
        query_results = [
            {"query_id": "Q1", "ranked_ids": [1, 2, 3, 4, 5], "relevant_ids": {1, 3}},
            {"query_id": "Q2", "ranked_ids": [5, 4, 3, 2, 1], "relevant_ids": {2, 4}},
        ]
        out = compute_aggregate_metrics(query_results, k_values=[1, 5])
        summary = out["summary"]

        self.assertEqual(summary["n_queries"], 2)
        self.assertIn("MAP", summary)
        self.assertIn("MRR", summary)
        self.assertIn("P@1", summary)
        self.assertIn("P@5", summary)
        self.assertIn("NDCG@5", summary)

    def test_empty_input(self):
        self.assertEqual(compute_aggregate_metrics([]), {})

    def test_per_query_length(self):
        query_results = [
            {"query_id": "Q1", "ranked_ids": [1], "relevant_ids": {1}},
            {"query_id": "Q2", "ranked_ids": [2], "relevant_ids": {2}},
        ]
        out = compute_aggregate_metrics(query_results, k_values=[1])
        self.assertEqual(len(out["per_query"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
