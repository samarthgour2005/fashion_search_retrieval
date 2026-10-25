"""
Two-stage retrieval with multimodal CLIP query fusion.

Stage 1 (recall): FAISS retrieves a wide candidate pool by fused CLIP
similarity.
Stage 2 (precision): candidates are re-scored by how well their precomputed
per-garment colors match the query's parsed {garment: color} attributes, and
re-sorted.

If the query has no color attributes, the attribute score is 0 for every
candidate, so ranking falls back to pure CLIP similarity — Stage 2 never
hurts attribute-free queries.
"""

import pickle

import clip
import numpy as np
import torch
from PIL import Image

import config
from indexer.embed_index import load_clip, load_faiss_index
from retriever.query_parser import QueryParser




# ---------------------------------------------------------------------------
# Main search engine
# ---------------------------------------------------------------------------

class FashionSearchEngine:
    def __init__(self):
        self.model, self.preprocess = load_clip()
        self.index = load_faiss_index()
        with open(config.METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        self.image_ids = metadata["image_ids"]
        self.descriptions = metadata["descriptions"]
        self.garment_colors_per_image = metadata["garment_colors_per_image"]
        self._parser = None

    @property
    def parser(self):
        if self._parser is None:
            print("Lazy loading QueryParser model...")
            self._parser = QueryParser()
        return self._parser

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        with torch.no_grad():
            tokens = clip.tokenize([query], truncate=True).to(config.DEVICE)
            text_embed = self.model.encode_text(tokens)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed.cpu().numpy().astype("float32")

    def encode_image_query(self, image_input) -> np.ndarray:
        """image_input can be a PIL Image object or a string path to an image."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path string or PIL Image object.")

        with torch.no_grad():
            img_tensor = self.preprocess(image).unsqueeze(0).to(config.DEVICE)
            img_embed = self.model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed.cpu().numpy().astype("float32")

    def encode_search_query(
        self,
        query: str = None,
        image_input=None,
        fuse_alpha: float = config.FUSE_ALPHA,
    ) -> np.ndarray:
        if query is None and image_input is None:
            raise ValueError("At least one of `query` or `image_input` must be provided.")

        text_embed = self.encode_query(query) if query else None
        img_embed = self.encode_image_query(image_input) if image_input else None

        if text_embed is not None and img_embed is not None:
            fused = fuse_alpha * img_embed + (1 - fuse_alpha) * text_embed
            fused = fused / np.linalg.norm(fused, axis=-1, keepdims=True)
            return fused.astype("float32")
        elif text_embed is not None:
            return text_embed
        else:
            return img_embed

    # ------------------------------------------------------------------
    # v2: Multimodal fusion (image + optional text)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Attribute scoring
    # ------------------------------------------------------------------

    @staticmethod
    def attribute_match_score(query_attrs: dict, image_attrs: dict) -> float:
        if not query_attrs:
            return 0.0  # nothing to check against -> defer entirely to CLIP similarity
        matches = sum(1 for g, c in query_attrs.items() if image_attrs.get(g) == c)
        return matches / len(query_attrs)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _rerank_candidates(
        self,
        query_embed: np.ndarray,
        query_attrs: dict,
        top_k: int,
        candidate_pool: int,
        attribute_weight: float,
    ) -> list[dict]:
        scores, indices = self.index.search(query_embed, candidate_pool)

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
        return candidates[:top_k]

    def search(
        self,
        query: str = None,
        image_input=None,
        top_k: int = 5,
        candidate_pool: int = config.RERANK_CANDIDATE_POOL,
        attribute_weight: float = config.ATTRIBUTE_WEIGHT,
        query_attrs: dict = None,
    ) -> tuple[list[dict], dict]:
        """Standard two-stage search (text, image, or fused query)."""
        query_embed = self.encode_search_query(query=query, image_input=image_input)
        if query_attrs is None:
            query_attrs = self.parser.parse(query) if query else {}
        results = self._rerank_candidates(query_embed, query_attrs, top_k, candidate_pool, attribute_weight)
        return results, query_attrs

    def search_baseline(
        self,
        query: str = None,
        image_input=None,
        top_k: int = 5,
    ) -> list[dict]:
        """CLIP-only search (text, image, or multimodal), no re-ranking -- useful for comparison/ablation."""
        query_embed = self.encode_search_query(query=query, image_input=image_input)
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
