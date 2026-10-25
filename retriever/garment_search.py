"""
Segment-level fashion image retrieval.

``GarmentSearchEngine`` answers queries of the form:
    "Find images where the *top/pants/dress/...* looks like this."

Instead of searching the full-image FAISS index, it:
1. Segments the query image with SegFormer to isolate the target garment.
2. CLIP-encodes the extracted garment crop.
3. Searches the per-garment FAISS index (built by
   ``python -m indexer.build_garment_index``).
4. De-duplicates by image_id and returns the top-k images ranked by their
   best per-garment similarity score.

Graceful degradation
--------------------
If the garment FAISS index files don't exist, ``GarmentSearchEngine`` raises
``GarmentIndexNotFoundError`` on construction, so callers can detect this and
fall back gracefully (e.g., the Streamlit app disables this tab).

If the target garment is not found in the query image, the engine falls back
to full-image CLIP search using the full query-image embedding.
"""

from __future__ import annotations

import pickle

import clip
import faiss
import numpy as np
import torch
from PIL import Image

import config
from indexer.embed_index import load_clip
from indexer.segformer_segmenter import GarmentSegmenter


class GarmentIndexNotFoundError(FileNotFoundError):
    """Raised when the per-garment FAISS index hasn't been built yet."""


class GarmentSearchEngine:
    """Segment-level retrieval: finds images by their per-garment appearance."""

    def __init__(self):
        # Guard: fail early if index hasn't been built
        import os
        if not os.path.exists(config.GARMENT_FAISS_INDEX_PATH):
            raise GarmentIndexNotFoundError(
                f"Garment FAISS index not found at {config.GARMENT_FAISS_INDEX_PATH}.\n"
                "Build it first with: python -m indexer.build_garment_index"
            )
        if not os.path.exists(config.GARMENT_EMBEDDINGS_META_PATH):
            raise GarmentIndexNotFoundError(
                f"Garment metadata not found at {config.GARMENT_EMBEDDINGS_META_PATH}.\n"
                "Build it first with: python -m indexer.build_garment_index"
            )

        self.clip_model, self.preprocess = load_clip()
        self.segmenter = GarmentSegmenter()

        self.garment_index: faiss.Index = faiss.read_index(config.GARMENT_FAISS_INDEX_PATH)

        # Set nprobe for IVF indexes (no-op for flat indexes)
        if hasattr(self.garment_index, "nprobe"):
            self.garment_index.nprobe = config.GARMENT_FAISS_NPROBE

        # Load metadata — handle both old list format and new dict-with-header format
        with open(config.GARMENT_EMBEDDINGS_META_PATH, "rb") as f:
            raw_meta = pickle.load(f)

        if isinstance(raw_meta, dict) and "entries" in raw_meta:
            self.garment_meta: list[dict] = raw_meta["entries"]
            self._index_dim: int = raw_meta.get("embedding_dim", self.garment_index.d)
        else:
            # Legacy list format from v2
            self.garment_meta = raw_meta
            self._index_dim = self.garment_index.d

        # Load full-image metadata for descriptions / fallback
        with open(config.METADATA_PATH, "rb") as f:
            meta = pickle.load(f)
        self.image_ids: list = meta["image_ids"]
        self.descriptions: list[str] = meta["descriptions"]
        self.garment_colors: list[dict] = meta["garment_colors_per_image"]

    # ------------------------------------------------------------------
    # Internal encoding
    # ------------------------------------------------------------------

    def _encode_crop(self, crop_pil: Image.Image) -> np.ndarray:
        """CLIP-encode a single garment crop → (1, D) float32."""
        with torch.no_grad():
            tensor = self.preprocess(crop_pil.convert("RGB")).unsqueeze(0).to(config.DEVICE)
            embed = self.clip_model.encode_image(tensor)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().astype("float32")

    def _encode_full_image(self, image_pil: Image.Image) -> np.ndarray:
        """CLIP-encode the whole image (fallback when garment not found)."""
        with torch.no_grad():
            tensor = self.preprocess(image_pil.convert("RGB")).unsqueeze(0).to(config.DEVICE)
            embed = self.clip_model.encode_image(tensor)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().astype("float32")

    def _encode_text(self, prompt: str) -> np.ndarray:
        """CLIP-encode a text prompt → (1, D) float32."""
        with torch.no_grad():
            tokens = clip.tokenize([prompt], truncate=True).to(config.DEVICE)
            embed = self.clip_model.encode_text(tokens)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed.cpu().numpy().astype("float32")

    def _faiss_search_and_dedup(
        self,
        query_embed: np.ndarray,
        candidate_pool: int,
        top_k: int,
    ) -> list[dict]:
        """Search the garment FAISS index and de-duplicate by image."""
        search_k = min(candidate_pool, self.garment_index.ntotal)
        distances, indices = self.garment_index.search(query_embed, search_k)

        best_per_image: dict[int, dict] = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            entry = self.garment_meta[idx]
            img_idx = entry["image_index"]
            if img_idx not in best_per_image or score > best_per_image[img_idx]["score"]:
                best_per_image[img_idx] = {
                    "index": img_idx,
                    "image_id": entry["image_id"],
                    "garment_label": entry["garment_label"],
                    "description": self.descriptions[img_idx],
                    "garment_colors": self.garment_colors[img_idx],
                    "score": float(score),
                }

        return sorted(best_per_image.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(
        self,
        image_input,
        target_garment: str,
        top_k: int = 5,
        candidate_pool: int = config.RERANK_CANDIDATE_POOL,
    ) -> dict:
        """Find images whose ``target_garment`` visually matches the query.

        Parameters
        ----------
        image_input:
            PIL Image or file path to the reference / query image.
        target_garment:
            Garment label to search by.  Must be one of the keys in
            ``config.SEGFORMER_LABEL_TO_GARMENT.values()``, e.g.
            ``"top"``, ``"pants"``, ``"dress"``.
        top_k:
            Number of final results to return.
        candidate_pool:
            How many garment-level FAISS hits to retrieve before de-dup.

        Returns
        -------
        dict with keys:
          ``results``        — list of result dicts (image_id, description, score, …)
          ``query_crop``     — PIL crop used as query, or None if fallback was used
          ``fallback``       — True if full-image CLIP was used (garment not found)
          ``found_garment``  — actual garment label found in query image
          ``segment_info``   — list of all detected segment summary dicts (garment + area_pct)
        """
        # Load image
        if isinstance(image_input, str):
            image_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image_pil = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image.")

        # Segment and find target garment — use full mask data for richer output
        segments = self.segmenter.segment_garments_with_masks(image_pil)
        segment_info = [
            {"garment": s.garment, "area_pct": round(s.area_frac * 100, 1)}
            for s in segments
        ]

        target_seg = next((s for s in segments if s.garment == target_garment), None)
        fallback = target_seg is None

        if not fallback:
            query_embed = self._encode_crop(target_seg.crop)
            query_crop = target_seg.crop
            found_garment = target_garment
        else:
            query_embed = self._encode_full_image(image_pil)
            query_crop = None
            found_garment = None

        results = self._faiss_search_and_dedup(query_embed, candidate_pool, top_k)
        return {
            "results": results,
            "query_crop": query_crop,
            "fallback": fallback,
            "found_garment": found_garment,
            "segment_info": segment_info,
        }

    def available_garments(self) -> list[str]:
        """Return sorted list of garment labels present in the index."""
        labels = {entry["garment_label"] for entry in self.garment_meta}
        return sorted(labels)
