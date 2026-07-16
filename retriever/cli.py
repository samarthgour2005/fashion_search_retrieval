"""
Retriever entrypoint (Part B of the assignment).

Usage:
    python -m retriever.cli "A red tie and a white shirt in a formal setting"
    python -m retriever.cli "A red tie and a white shirt in a formal setting" --top_k 10
    python -m retriever.cli "casual weekend outfit" --baseline   # CLIP-only, no re-ranking
"""

# retriever/cli.py
import os
import argparse

from retriever.rerank_search import FashionSearchEngine
from indexer.data_loader import load_or_build_subset


def main():
    parser = argparse.ArgumentParser(description="Fashion text-to-image search")
    parser.add_argument("query", type=str)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--save_images", type=str, default=None,
                         help="Directory to save the retrieved images into, e.g. ./results")
    args = parser.parse_args()

    engine = FashionSearchEngine()

    if args.baseline:
        results = engine.search_baseline(args.query, top_k=args.top_k)
    else:
        results, query_attrs = engine.search(args.query, top_k=args.top_k)
        print(f"Parsed query attributes: {query_attrs}\n")

    for r in results:
        print(r)

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        subset, _ = load_or_build_subset()  # reloads from data/ cache, no re-download
        for rank, r in enumerate(results, start=1):
            img = subset[r["index"]]["image"].convert("RGB")
            path = os.path.join(args.save_images, f"rank{rank}_img{r['image_id']}.jpg")
            img.save(path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
