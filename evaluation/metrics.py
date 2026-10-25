"""
evaluation/metrics.py
---------------------
Standard information-retrieval metrics for the fashion retrieval system.

Metrics implemented
-------------------
* Precision@K          – fraction of top-K results that are relevant
* Recall@K             – fraction of relevant items found in top-K
* F1@K                 – harmonic mean of Precision@K and Recall@K
* Average Precision    – area under the precision-recall curve (per query)
* Mean Average Precision (MAP)
* NDCG@K               – Normalized Discounted Cumulative Gain
* Mean Reciprocal Rank (MRR)
* Hit Rate@K           – 1 if any relevant item appears in top-K, else 0

All functions take a anked_ids list (system output, ordered best-first)
and a elevant_ids set (ground-truth), so they are agnostic to how the
system produced the ranking.
"""

from __future__ import annotations
import math
from typing import Sequence, Set, Any

# ---------------------------------------------------------------------------
# Low-level per-query helpers
# ---------------------------------------------------------------------------

def precision_at_k(ranked_ids: Sequence, relevant_ids: Set, k: int) -> float:
    """Fraction of top-K results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: Sequence, relevant_ids: Set, k: int) -> float:
    """Fraction of all relevant items found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids: Sequence, relevant_ids: Set, k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(ranked_ids: Sequence, relevant_ids: Set) -> float:
    """
    Computes the Average Precision (AP) for a single query.
    AP = (1 / |R|) * sum_k [ P@k * rel(k) ]
    where rel(k) = 1 if the item at rank k is relevant, else 0.
    Returns 0 if there are no relevant items.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant_ids)


def ndcg_at_k(ranked_ids: Sequence, relevant_ids: Set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K.
    Graded relevance: 1 if relevant, 0 otherwise (binary relevance).
    """
    if not relevant_ids or k <= 0:
        return 0.0

    def dcg(ids):
        gain = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            if doc_id in relevant_ids:
                gain += 1.0 / math.log2(i + 1)
        return gain

    actual_dcg = dcg(ranked_ids)
    # Ideal: put all relevant items first (up to k)
    ideal_ids = list(relevant_ids)[:k]
    ideal_dcg = dcg(ideal_ids)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def reciprocal_rank(ranked_ids: Sequence, relevant_ids: Set) -> float:
    """Reciprocal rank of the first relevant item (0 if none found)."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(ranked_ids: Sequence, relevant_ids: Set, k: int) -> float:
    """1 if at least one relevant item is in the top-K, else 0."""
    return float(any(r in relevant_ids for r in ranked_ids[:k]))


# ---------------------------------------------------------------------------
# Aggregate over a collection of queries
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(
    query_results: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """
    Compute aggregate metrics over all queries.

    Parameters
    ----------
    query_results : list of dicts, each with keys:
        - 'query_id'      : identifier for the query
        - 'ranked_ids'    : list of retrieved image_ids, best-first
        - 'relevant_ids'  : set (or list) of ground-truth relevant image_ids
    k_values : list of K thresholds (default [1, 5, 10])

    Returns
    -------
    dict with per-K and aggregate metric values.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    n = len(query_results)
    if n == 0:
        return {}

    # Containers
    per_k: dict[int, dict[str, list]] = {
        k: {"precision": [], "recall": [], "f1": [], "ndcg": [], "hit_rate": []}
        for k in k_values
    }
    aps: list[float] = []
    rrs: list[float] = []
    per_query_rows: list[dict] = []

    for entry in query_results:
        ranked = entry["ranked_ids"]
        relevant = set(entry["relevant_ids"])

        ap = average_precision(ranked, relevant)
        rr = reciprocal_rank(ranked, relevant)
        aps.append(ap)
        rrs.append(rr)

        row = {
            "query_id": entry.get("query_id", ""),
            "query_text": entry.get("query_text", ""),
            "n_relevant": len(relevant),
            "ap": round(ap, 4),
            "rr": round(rr, 4),
        }

        for k in k_values:
            p = precision_at_k(ranked, relevant, k)
            r = recall_at_k(ranked, relevant, k)
            f = f1_at_k(ranked, relevant, k)
            n_val = ndcg_at_k(ranked, relevant, k)
            h = hit_rate_at_k(ranked, relevant, k)
            per_k[k]["precision"].append(p)
            per_k[k]["recall"].append(r)
            per_k[k]["f1"].append(f)
            per_k[k]["ndcg"].append(n_val)
            per_k[k]["hit_rate"].append(h)
            row[f"P@{k}"] = round(p, 4)
            row[f"R@{k}"] = round(r, 4)
            row[f"F1@{k}"] = round(f, 4)
            row[f"NDCG@{k}"] = round(n_val, 4)
            row[f"Hit@{k}"] = round(h, 4)

        per_query_rows.append(row)

    # Build aggregate summary
    summary: dict[str, Any] = {
        "n_queries": n,
        "MAP": round(sum(aps) / n, 4),
        "MRR": round(sum(rrs) / n, 4),
    }
    for k in k_values:
        for metric_name, values in per_k[k].items():
            key = f"{metric_name.upper()}@{k}"
            if metric_name == "precision":
                key = f"P@{k}"
            elif metric_name == "recall":
                key = f"R@{k}"
            elif metric_name == "f1":
                key = f"F1@{k}"
            elif metric_name == "ndcg":
                key = f"NDCG@{k}"
            elif metric_name == "hit_rate":
                key = f"HitRate@{k}"
            summary[key] = round(sum(values) / n, 4)

    return {"summary": summary, "per_query": per_query_rows}
