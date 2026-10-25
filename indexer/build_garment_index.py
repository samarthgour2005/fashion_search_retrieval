"""
Per-garment FAISS index builder (v2/v3 extension).

Run this once after ``python -m indexer.build_index`` has built the main index:

    python -m indexer.build_garment_index

What it does
------------
1. Loads the cached Fashionpedia subset from disk.
2. For each image, runs SegFormer to detect garment regions.
3. CLIP-encodes each garment crop (independently of the full-image embedding).
4. Accumulates (image_index, image_id, garment_label, crop_embedding) records.
5. Builds a FAISS index over all garment-crop embeddings.
6. Saves the FAISS index to ``config.GARMENT_FAISS_INDEX_PATH`` and the
   per-entry metadata (image_index, image_id, garment_label) to
   ``config.GARMENT_EMBEDDINGS_META_PATH``.

After building, ``retriever.garment_search.GarmentSearchEngine`` loads both
files at query time to enable segment-level retrieval.

Index types
-----------
- **Flat (default)**: ``IndexFlatIP`` — exact inner-product search.  Accurate
  but O(N) at query time.
- **IVF (--use-ivf)**: ``IndexIVFFlat`` — approximate search via Voronoi
  partitioning.  ~8–20× faster at query time for large corpora; recall stays
  >95% with the default nprobe setting.  Requires training on the full
  embedding matrix before adding vectors.

CLI options
-----------
--max_images N    Only index the first N images (useful for a dry-run).
--batch_size N    CLIP encoding batch size (default: config.BATCH_SIZE).
--use-ivf         Build an IVFFlat index instead of an exact FlatIP index.
--nlist N         Number of Voronoi cells for IVF (default: config.GARMENT_FAISS_NLIST).
--resume          Skip images already present in a partial metadata file and
                  append new embeddings to it (useful after an interrupted run).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import clip
import faiss
import numpy as np
import torch
from tqdm.auto import tqdm

import config
from indexer.data_loader import load_or_build_subset
from indexer.embed_index import load_clip
from indexer.segformer_segmenter import GarmentSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_crops_batch(
    model,
    preprocess,
    crops: list,
) -> np.ndarray:
    """CLIP-encode a list of PIL crop images, returns (N, D) float32 array."""
    with torch.no_grad():
        tensors = torch.stack([preprocess(c.convert("RGB")) for c in crops]).to(config.DEVICE)
        embeds = model.encode_image(tensors)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.cpu().numpy().astype("float32")


def _build_flat_index(dim: int) -> faiss.Index:
    """Return an exact IndexFlatIP index, optionally GPU-accelerated."""
    index = faiss.IndexFlatIP(dim)
    if hasattr(faiss, "StandardGpuResources") and config.DEVICE == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index


def _build_ivf_index(dim: int, embeddings: np.ndarray, nlist: int) -> faiss.Index:
    """Train and return an IndexIVFFlat index on the provided embeddings."""
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"Training IVF index with nlist={nlist} on {len(embeddings)} vectors…")
    index.train(embeddings)
    index.nprobe = config.GARMENT_FAISS_NPROBE
    return index


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_garment_index(
    max_images: int | None = None,
    batch_size: int = config.BATCH_SIZE,
    use_ivf: bool = False,
    nlist: int = config.GARMENT_FAISS_NLIST,
    resume: bool = False,
) -> None:
    print(f"Device: {config.DEVICE}")

    # ---- Load dataset subset -----------------------------------------------
    subset, _ = load_or_build_subset()
    n = len(subset) if max_images is None else min(max_images, len(subset))
    print(f"Building garment index for {n} images…")

    # ---- Resume: load existing partial metadata ----------------------------
    already_indexed: set[int] = set()
    existing_embeddings: list[np.ndarray] = []
    existing_meta: list[dict] = []

    if resume and Path(config.GARMENT_EMBEDDINGS_META_PATH).exists():
        with open(config.GARMENT_EMBEDDINGS_META_PATH, "rb") as f:
            existing_meta = pickle.load(f)
        already_indexed = {m["image_index"] for m in existing_meta}
        print(f"Resume mode: {len(already_indexed)} images already indexed, skipping.")

        # If an existing FAISS index exists, reload its vectors
        if Path(config.GARMENT_FAISS_INDEX_PATH).exists():
            old_idx = faiss.read_index(config.GARMENT_FAISS_INDEX_PATH)
            if old_idx.ntotal > 0:
                existing_embeddings_arr = np.zeros(
                    (old_idx.ntotal, old_idx.d), dtype="float32"
                )
                faiss.extract_index_ivf(old_idx) if hasattr(faiss, "extract_index_ivf") else None
                # Use reconstruct_n only for Flat-compatible indexes
                try:
                    old_idx.reconstruct_n(0, old_idx.ntotal, existing_embeddings_arr)
                    existing_embeddings = [existing_embeddings_arr]
                except Exception:
                    print("  [warn] Could not reconstruct existing vectors; starting fresh embeddings.")

    # ---- Load models --------------------------------------------------------
    clip_model, preprocess = load_clip()
    segmenter = GarmentSegmenter()

    # ---- Segment + encode ---------------------------------------------------
    all_embeddings: list[np.ndarray] = existing_embeddings.copy()
    garment_meta: list[dict] = existing_meta.copy()

    # Accumulated crops buffer for batched CLIP encoding
    crop_buffer: list = []
    meta_buffer: list[dict] = []

    def flush_buffer():
        if not crop_buffer:
            return
        batch_embs = encode_crops_batch(clip_model, preprocess, crop_buffer)
        all_embeddings.append(batch_embs)
        garment_meta.extend(meta_buffer)
        crop_buffer.clear()
        meta_buffer.clear()

    for i in tqdm(range(n), desc="Segmenting + encoding garments"):
        if i in already_indexed:
            continue

        example = subset[i]
        image_pil = example["image"].convert("RGB")
        image_id = example["image_id"]

        try:
            segments = segmenter.segment_garments(image_pil)
        except Exception as exc:
            # Gracefully skip problematic images
            print(f"  [warn] image {image_id} segmentation failed: {exc}")
            continue

        for garment_label, crop in segments:
            crop_buffer.append(crop)
            meta_buffer.append({
                "image_index": i,
                "image_id": image_id,
                "garment_label": garment_label,
            })
            if len(crop_buffer) >= batch_size:
                flush_buffer()

    flush_buffer()  # encode any remaining crops

    if not all_embeddings:
        print("No garment segments found. Aborting.")
        return

    all_embs = np.vstack(all_embeddings)  # (total_garments, D)
    print(f"Total garment embeddings: {all_embs.shape}")

    # ---- Build FAISS index --------------------------------------------------
    dim = all_embs.shape[1]

    if use_ivf:
        actual_nlist = min(nlist, max(1, all_embs.shape[0] // 4))
        index = _build_ivf_index(dim, all_embs, actual_nlist)
    else:
        index = _build_flat_index(dim)

    index.add(all_embs)
    print(f"FAISS garment index ({('IVF' if use_ivf else 'Flat')}): {index.ntotal} vectors")

    # ---- Save ---------------------------------------------------------------
    # For GPU-based flat indexes, move to CPU before writing
    use_gpu = hasattr(faiss, "StandardGpuResources") and config.DEVICE == "cuda" and not use_ivf
    cpu_index = faiss.index_gpu_to_cpu(index) if use_gpu else index

    faiss.write_index(cpu_index, config.GARMENT_FAISS_INDEX_PATH)
    print(f"Saved garment FAISS index → {config.GARMENT_FAISS_INDEX_PATH}")

    # Store embedding dim in metadata header for validation at load time
    meta_with_header = {
        "entries": garment_meta,
        "embedding_dim": dim,
        "use_ivf": use_ivf,
        "nlist": nlist if use_ivf else None,
    }
    with open(config.GARMENT_EMBEDDINGS_META_PATH, "wb") as f:
        pickle.dump(meta_with_header, f)
    print(f"Saved garment metadata ({len(garment_meta)} entries) → {config.GARMENT_EMBEDDINGS_META_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build per-garment FAISS index with SegFormer")
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit to first N images (useful for a quick dry-run).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help=f"CLIP encoding batch size (default: {config.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--use-ivf", action="store_true",
        help="Build an approximate IVFFlat index instead of an exact FlatIP index.",
    )
    parser.add_argument(
        "--nlist", type=int, default=config.GARMENT_FAISS_NLIST,
        help=f"IVF Voronoi cells (only relevant with --use-ivf; default: {config.GARMENT_FAISS_NLIST}).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted indexing run — skip already-indexed images.",
    )
    args = parser.parse_args()
    build_garment_index(
        max_images=args.max_images,
        batch_size=args.batch_size,
        use_ivf=args.use_ivf,
        nlist=args.nlist,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
