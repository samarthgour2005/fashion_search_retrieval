"""
Two-stage retrieval (Part B of the assignment).

Stage 1 (recall): FAISS retrieves a wide candidate pool by fused CLIP
similarity.
Stage 2 (precision): candidates are re-scored by how well their precomputed
per-garment colors match the query's parsed {garment: color} attributes, and
re-sorted.

If the query has no color attributes, the attribute score is 0 for every
candidate, so ranking falls back to pure CLIP similarity -- Stage 2 never
hurts attribute-free queries.
"""

import pickle

import clip
import torch

import config
from indexer.embed_index import load_clip, load_faiss_index
from retriever.query_parser import QueryParser


class FashionSearchEngine:
    def __init__(self):
        self.model, self.preprocess = load_clip()
        self.index = load_faiss_index()
        with open(config.METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        self.image_ids = metadata["image_ids"]
        self.descriptions = metadata["descriptions"]
        self.garment_colors_per_image = metadata["garment_colors_per_image"]
        self.parser = QueryParser()

    def encode_query(self, query: str):
        with torch.no_grad():
            tokens = clip.tokenize([query], truncate=True).to(config.DEVICE)
            text_embed = self.model.encode_text(tokens)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed.cpu().numpy().astype("float32")

    @staticmethod
    def attribute_match_score(query_attrs: dict, image_attrs: dict) -> float:
        if not query_attrs:
            return 0.0  # nothing to check against -> defer entirely to CLIP similarity
        matches = sum(1 for g, c in query_attrs.items() if image_attrs.get(g) == c)
        return matches / len(query_attrs)

    def search(self, query: str, top_k=5, candidate_pool=config.RERANK_CANDIDATE_POOL,
               attribute_weight=config.ATTRIBUTE_WEIGHT):
        query_embed = self.encode_query(query)
        scores, indices = self.index.search(query_embed, candidate_pool)
        query_attrs = self.parser.parse(query)

        candidates = []
        for idx, clip_score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            idx = int(idx)
            image_attrs = self.garment_colors_per_image[idx]
            attr_score = self.attribute_match_score(query_attrs, image_attrs)
            final_score = (1 - attribute_weight) * float(clip_score) + attribute_weight * attr_score
            candidates.append({
                "index": idx,
                "image_id": self.image_ids[idx],
                "description": self.descriptions[idx],
                "clip_score": float(clip_score),
                "attr_score": attr_score,
                "final_score": final_score,
                "garment_colors": image_attrs,
            })

        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        return candidates[:top_k], query_attrs

    def search_baseline(self, query: str, top_k=5):
        """CLIP-only search, no re-ranking -- useful for comparison/ablation."""
        query_embed = self.encode_query(query)
        scores, indices = self.index.search(query_embed, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            idx = int(idx)
            results.append({
                "index": idx,
                "image_id": self.image_ids[idx],
                "description": self.descriptions[idx],
                "score": float(score),
            })
        return results
