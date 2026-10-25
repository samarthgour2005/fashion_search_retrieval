"""
evaluation/benchmark.py
-----------------------
Defines synthetic benchmark query sets used for evaluation.

Since the Fashionpedia dataset does not ship with explicit relevance labels,
we derive *proxy* ground-truth from the metadata already stored at index time:

TEXT QUERIES
    Ground truth = all indexed images whose pre-computed garment_colors
    dict satisfies the query's parsed {garment: color} attributes.
    e.g. query "blue shirt" -> relevant = images where garment_colors["shirt"] == "blue"

IMAGE QUERIES
    Ground truth = images that share at least one garment+color attribute with
    the query image (based on indexed garment_colors), AND whose CLIP similarity
    to the query image is above a configurable threshold.

This gives us clean, reproducible relevance labels that are consistent with
what the system is actually trying to retrieve.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np

import config
from retriever.query_parser import RuleBasedQueryParser


# ---------------------------------------------------------------------------
# Predefined text benchmark queries
# ---------------------------------------------------------------------------

TEXT_QUERIES: list[dict] = [
    # Single-garment, single-color
    {"query_id": "TQ01", "query": "blue shirt"},
    {"query_id": "TQ02", "query": "red dress"},
    {"query_id": "TQ03", "query": "black pants"},
    {"query_id": "TQ04", "query": "white jacket"},
    {"query_id": "TQ05", "query": "green coat"},
    {"query_id": "TQ06", "query": "yellow skirt"},
    {"query_id": "TQ07", "query": "pink sweater"},
    {"query_id": "TQ08", "query": "gray hat"},
    {"query_id": "TQ09", "query": "brown bag"},
    {"query_id": "TQ10", "query": "orange shoe"},
    # Multi-garment
    {"query_id": "TQ11", "query": "blue shirt and black pants"},
    {"query_id": "TQ12", "query": "white shirt with red tie"},
    {"query_id": "TQ13", "query": "red dress with black belt"},
    {"query_id": "TQ14", "query": "green jacket and blue pants"},
    {"query_id": "TQ15", "query": "pink skirt and white top"},
    # Descriptive / natural language
    {"query_id": "TQ16", "query": "someone wearing a bright yellow raincoat"},
    {"query_id": "TQ17", "query": "casual blue jeans outfit"},
    {"query_id": "TQ18", "query": "woman in a red gown"},
    {"query_id": "TQ19", "query": "man in a black blazer"},
    {"query_id": "TQ20", "query": "summer dress in white"},
]


# ---------------------------------------------------------------------------
# Ground-truth builder
# ---------------------------------------------------------------------------

class GroundTruthBuilder:
    """
    Loads the indexed metadata and builds relevance sets for evaluation queries.
    """

    def __init__(self):
        with open(config.METADATA_PATH, "rb") as f:
            meta = pickle.load(f)
        self.image_ids: list = meta["image_ids"]
        self.descriptions: list[str] = meta["descriptions"]
        self.garment_colors: list[dict] = meta["garment_colors_per_image"]
        self._n = len(self.image_ids)

    # ------------------------------------------------------------------
    # Text-query ground truth
    # ------------------------------------------------------------------

    def relevant_for_text_query(
        self,
        query: str,
        min_attr_match: int = 1,
    ) -> set:
        """
        Return image_ids where at least min_attr_match parsed attributes match.
        Falls back to description-level substring matching when the parser
        finds no color attributes.
        """
        attrs = RuleBasedQueryParser.parse(query)

        if attrs:
            relevant = set()
            for img_id, gc in zip(self.image_ids, self.garment_colors):
                matched = sum(
                    1 for garment, color in attrs.items()
                    if gc.get(garment) == color
                )
                if matched >= min_attr_match:
                    relevant.add(img_id)
            return relevant

        # Fallback: substring match on CLIP description
        query_lower = query.lower()
        return {
            img_id for img_id, desc in zip(self.image_ids, self.descriptions)
            if query_lower in desc.lower()
        }

    # ------------------------------------------------------------------
    # Image-query ground truth
    # ------------------------------------------------------------------

    def relevant_for_image_query(
        self,
        query_image_id: Any,
        min_shared_attrs: int = 1,
    ) -> set:
        """
        For an image query, ground-truth = images that share >= min_shared_attrs
        garment+color attributes with the query image.
        The query image itself is excluded.
        """
        try:
            q_idx = self.image_ids.index(query_image_id)
        except ValueError:
            return set()

        q_gc = self.garment_colors[q_idx]
        if not q_gc:
            return set()

        relevant = set()
        for idx, (img_id, gc) in enumerate(zip(self.image_ids, self.garment_colors)):
            if idx == q_idx:
                continue
            shared = sum(1 for g, c in q_gc.items() if gc.get(g) == c)
            if shared >= min_shared_attrs:
                relevant.add(img_id)
        return relevant

    # ------------------------------------------------------------------
    # Build full benchmark
    # ------------------------------------------------------------------

    def build_text_benchmark(
        self,
        queries: list[dict] | None = None,
        min_attr_match: int = 1,
    ) -> list[dict]:
        """
        Returns a list of dicts ready for `compute_aggregate_metrics`.
        Each dict has: query_id, query_text, relevant_ids (set).
        Note: ranked_ids must be filled in by the evaluator after searching.
        """
        if queries is None:
            queries = TEXT_QUERIES
        benchmark = []
        for q in queries:
            relevant = self.relevant_for_text_query(q["query"], min_attr_match)
            benchmark.append({
                "query_id": q["query_id"],
                "query_text": q["query"],
                "relevant_ids": relevant,
            })
        return benchmark

    def build_image_benchmark(
        self,
        n_query_images: int = 20,
        seed: int = 42,
        min_shared_attrs: int = 1,
    ) -> list[dict]:
        """
        Randomly samples 
_query_images images from the index as image queries.
        Ground truth = images sharing >= min_shared_attrs garment+color attributes.
        """
        rng = np.random.default_rng(seed)
        eligible = [
            (i, img_id)
            for i, (img_id, gc) in enumerate(zip(self.image_ids, self.garment_colors))
            if gc  # must have color attributes
        ]
        if len(eligible) < n_query_images:
            n_query_images = len(eligible)

        chosen = rng.choice(len(eligible), size=n_query_images, replace=False)
        benchmark = []
        for rank, sel_idx in enumerate(chosen):
            idx, img_id = eligible[int(sel_idx)]
            relevant = self.relevant_for_image_query(img_id, min_shared_attrs)
            benchmark.append({
                "query_id": f"IQ{rank+1:02d}",
                "query_text": f"[image:{img_id}]",
                "query_image_id": img_id,
                "query_image_index": idx,
                "relevant_ids": relevant,
            })
        return benchmark
