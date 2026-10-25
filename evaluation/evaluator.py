"""
evaluation/evaluator.py
-----------------------
Runs the full evaluation pipeline and produces a structured report.

Usage
-----
    # from repo root:
    python -m evaluation.evaluator

    # with options:
    python -m evaluation.evaluator --top-k 10 --pool 200 --n-image-queries 30 \
        --output results/eval_report.json --modes text image both --compare-baseline

Metrics computed (for each mode: text / image / both)
------------------------------------------------------
  P@K, R@K, F1@K, NDCG@K, Hit-Rate@K  (K = 1, 5, 10 by default)
  MAP, MRR

Two systems are always compared:
  - *Reranked* : FashionSearchEngine (CLIP + attribute re-rank)
  - *Baseline* : CLIP-only (no re-rank) via search_baseline()
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make sure repo root is on the path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retriever.rerank_search import FashionSearchEngine
from evaluation.metrics import compute_aggregate_metrics
from evaluation.benchmark import GroundTruthBuilder


# ---------------------------------------------------------------------------
# Helper: run text queries through the system
# ---------------------------------------------------------------------------

def evaluate_text_queries(
    engine: FashionSearchEngine,
    benchmark: list[dict],
    top_k: int,
    candidate_pool: int,
    k_values: list[int],
    compare_baseline: bool,
) -> dict:
    """
    Runs every text query through the engine and collects ranked_ids.
    Returns metrics for reranked and (optionally) baseline systems.
    """
    reranked_results: list[dict] = []
    baseline_results: list[dict] = []

    print(f"\n[Text] Evaluating {len(benchmark)} queries …")
    for i, entry in enumerate(benchmark, 1):
        query_text = entry["query_text"]
        relevant = entry["relevant_ids"]

        # Skip queries with no relevant items (would give 0 on all metrics)
        if not relevant:
            print(f"  [{i}/{len(benchmark)}] '{query_text}' — skipped (0 relevant items)")
            continue

        # Reranked search
        try:
            hits, _ = engine.search(query=query_text, top_k=top_k, candidate_pool=candidate_pool)
            ranked_ids = [h["image_id"] for h in hits]
        except Exception as e:
            print(f"  [{i}/{len(benchmark)}] '{query_text}' — search error: {e}")
            ranked_ids = []

        reranked_results.append({
            "query_id": entry["query_id"],
            "query_text": query_text,
            "ranked_ids": ranked_ids,
            "relevant_ids": relevant,
        })

        # Baseline search
        if compare_baseline:
            try:
                b_hits = engine.search_baseline(query=query_text, top_k=top_k)
                b_ranked = [h["image_id"] for h in b_hits]
            except Exception as e:
                b_ranked = []
            baseline_results.append({
                "query_id": entry["query_id"],
                "query_text": query_text,
                "ranked_ids": b_ranked,
                "relevant_ids": relevant,
            })

        print(f"  [{i}/{len(benchmark)}] '{query_text}' — {len(relevant)} relevant, "
              f"retrieved {len(ranked_ids)}, "
              f"hits={sum(1 for r in ranked_ids if r in relevant)}")

    out = {"reranked": compute_aggregate_metrics(reranked_results, k_values)}
    if compare_baseline:
        out["baseline"] = compute_aggregate_metrics(baseline_results, k_values)
    return out


# ---------------------------------------------------------------------------
# Helper: run image queries through the system
# ---------------------------------------------------------------------------

def evaluate_image_queries(
    engine: FashionSearchEngine,
    benchmark: list[dict],
    top_k: int,
    candidate_pool: int,
    k_values: list[int],
    compare_baseline: bool,
) -> dict:
    """
    Runs image-based queries. Uses the stored embeddings in the FAISS index
    directly via a nearest-neighbour lookup (avoids re-encoding).
    """
    import faiss  # noqa: F401 — already imported via FashionSearchEngine

    print(f"\n[Image] Evaluating {len(benchmark)} queries …")
    reranked_results: list[dict] = []
    baseline_results: list[dict] = []

    # Load raw fused embeddings to use as query vectors
    # (The index is IndexFlatIP so we can search with any normalised vector.)
    # We extract the query vector by reconstructing it from the FAISS index.
    index = engine.index

    for i, entry in enumerate(benchmark, 1):
        img_idx = entry["query_image_index"]
        relevant = entry["relevant_ids"]

        if not relevant:
            print(f"  [{i}/{len(benchmark)}] {entry['query_text']} — skipped (0 relevant)")
            continue

        # Reconstruct the stored embedding for this image
        try:
            query_vec = np.zeros((1, index.d), dtype="float32")
            index.reconstruct(img_idx, query_vec[0])
        except Exception:
            # IndexFlatIP supports reconstruct; fall back to encode_image_query
            # if the index type doesn't (e.g. IVF).
            query_vec = None

        if query_vec is not None:
            # Reranked: use _rerank_candidates directly
            try:
                query_attrs = engine.garment_colors_per_image[img_idx]
                hits = engine._rerank_candidates(
                    query_vec, query_attrs, top_k, candidate_pool,
                    attribute_weight=config.ATTRIBUTE_WEIGHT,
                )
                ranked_ids = [h["image_id"] for h in hits]
                # exclude the query image itself
                ranked_ids = [r for r in ranked_ids if r != engine.image_ids[img_idx]]
            except Exception as e:
                print(f"  [{i}] rerank error: {e}")
                ranked_ids = []

            if compare_baseline:
                try:
                    scores, indices = index.search(query_vec, top_k + 1)
                    b_ranked = [
                        engine.image_ids[int(j)]
                        for j in indices[0]
                        if j >= 0 and engine.image_ids[int(j)] != engine.image_ids[img_idx]
                    ][:top_k]
                except Exception as e:
                    b_ranked = []
        else:
            ranked_ids = []
            b_ranked = []

        reranked_results.append({
            "query_id": entry["query_id"],
            "query_text": entry["query_text"],
            "ranked_ids": ranked_ids,
            "relevant_ids": relevant,
        })
        if compare_baseline:
            baseline_results.append({
                "query_id": entry["query_id"],
                "query_text": entry["query_text"],
                "ranked_ids": b_ranked,
                "relevant_ids": relevant,
            })

        hits_in_top = sum(1 for r in ranked_ids if r in relevant)
        print(f"  [{i}/{len(benchmark)}] {entry['query_text']} — "
              f"{len(relevant)} relevant, hits@{top_k}={hits_in_top}")

    out = {"reranked": compute_aggregate_metrics(reranked_results, k_values)}
    if compare_baseline:
        out["baseline"] = compute_aggregate_metrics(baseline_results, k_values)
    return out


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_summary_table(title: str, summary: dict):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Queries evaluated : {summary.get('n_queries', '?')}")
    print(f"  MAP               : {summary.get('MAP', 'N/A')}")
    print(f"  MRR               : {summary.get('MRR', 'N/A')}")
    # Print K-level metrics in a table
    k_metrics = ["P", "R", "F1", "NDCG", "HitRate"]
    ks = sorted({int(key.split("@")[1]) for key in summary if "@" in key})
    header = f"  {'Metric':<12}" + "".join(f"  @{k:<6}" for k in ks)
    print(header)
    print("  " + "-" * (12 + 9 * len(ks)))
    for prefix in k_metrics:
        row = f"  {prefix:<12}"
        for k in ks:
            val = summary.get(f"{prefix}@{k}", "N/A")
            row += f"  {val:<8}" if isinstance(val, float) else f"  {'N/A':<8}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fashion retrieval system with standard IR metrics."
    )
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to retrieve per query (default 10)")
    parser.add_argument("--pool", type=int, default=config.RERANK_CANDIDATE_POOL,
                        help="Candidate pool size for re-ranking (default from config)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10],
                        help="K thresholds for P/R/F1/NDCG/HitRate (default: 1 5 10)")
    parser.add_argument("--modes", nargs="+", choices=["text", "image", "both"],
                        default=["text", "image"],
                        help="Which evaluation modes to run (default: text image)")
    parser.add_argument("--n-text-queries", type=int, default=len([], ),
                        help="Number of text queries (default: all predefined)")
    parser.add_argument("--n-image-queries", type=int, default=20,
                        help="Number of image queries to sample (default: 20)")
    parser.add_argument("--min-relevant", type=int, default=1,
                        help="Min relevant items required for a query to be included (default: 1)")
    parser.add_argument("--image-min-shared-attrs", type=int, default=1,
                        help="Minimum shared garment+color attributes for image GT (default: 1)")
    parser.add_argument("--compare-baseline", action="store_true", default=True,
                        help="Also evaluate CLIP-only baseline (default: True)")
    parser.add_argument("--no-baseline", dest="compare_baseline", action="store_false")
    parser.add_argument("--output", type=str, default="results/eval_report.json",
                        help="Path to save JSON report (default: results/eval_report.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image query sampling (default: 42)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Fashion Retrieval — Evaluation Suite")
    print("=" * 60)
    print(f"  Top-K         : {args.top_k}")
    print(f"  Candidate pool: {args.pool}")
    print(f"  K values      : {args.k_values}")
    print(f"  Modes         : {args.modes}")
    print(f"  Compare base  : {args.compare_baseline}")

    # Load engine (this reads FAISS index + metadata)
    print("\n[Setup] Loading FashionSearchEngine …")
    t0 = time.time()
    engine = FashionSearchEngine()
    print(f"  Engine ready in {time.time()-t0:.1f}s  "
          f"({engine.index.ntotal} vectors in index)")

    # Build ground-truth
    print("[Setup] Building ground-truth …")
    gt_builder = GroundTruthBuilder()

    report: dict[str, Any] = {
        "config": {
            "top_k": args.top_k,
            "candidate_pool": args.pool,
            "k_values": args.k_values,
            "modes": args.modes,
            "compare_baseline": args.compare_baseline,
            "n_index": engine.index.ntotal,
        }
    }

    # -----------------------------------------------------------------------
    # TEXT evaluation
    # -----------------------------------------------------------------------
    if "text" in args.modes or "both" in args.modes:
        from evaluation.benchmark import TEXT_QUERIES
        queries = TEXT_QUERIES
        text_bench = gt_builder.build_text_benchmark(queries, min_attr_match=1)
        # Filter out queries with no relevant items
        text_bench = [q for q in text_bench if len(q["relevant_ids"]) >= args.min_relevant]
        print(f"[Text] {len(text_bench)} queries with >={args.min_relevant} relevant items")

        text_metrics = evaluate_text_queries(
            engine, text_bench,
            top_k=args.top_k,
            candidate_pool=args.pool,
            k_values=args.k_values,
            compare_baseline=args.compare_baseline,
        )
        report["text"] = text_metrics

        print_summary_table("TEXT QUERIES — Reranked System",
                            text_metrics["reranked"]["summary"])
        if args.compare_baseline and "baseline" in text_metrics:
            print_summary_table("TEXT QUERIES — CLIP Baseline",
                                text_metrics["baseline"]["summary"])

    # -----------------------------------------------------------------------
    # IMAGE evaluation
    # -----------------------------------------------------------------------
    if "image" in args.modes or "both" in args.modes:
        image_bench = gt_builder.build_image_benchmark(
            n_query_images=args.n_image_queries,
            seed=args.seed,
            min_shared_attrs=args.image_min_shared_attrs,
        )
        image_bench = [q for q in image_bench if len(q["relevant_ids"]) >= args.min_relevant]
        print(f"[Image] {len(image_bench)} image queries with >={args.min_relevant} relevant items")

        image_metrics = evaluate_image_queries(
            engine, image_bench,
            top_k=args.top_k,
            candidate_pool=args.pool,
            k_values=args.k_values,
            compare_baseline=args.compare_baseline,
        )
        report["image"] = image_metrics

        print_summary_table("IMAGE QUERIES — Reranked System",
                            image_metrics["reranked"]["summary"])
        if args.compare_baseline and "baseline" in image_metrics:
            print_summary_table("IMAGE QUERIES — CLIP Baseline",
                                image_metrics["baseline"]["summary"])

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert sets to lists for JSON serialisation
    def _jsonify(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(x) for x in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_jsonify(report), f, indent=2)

    print(f"\n[Done] Full report saved to: {out_path.resolve()}")
    print(f"       Total wall time: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
