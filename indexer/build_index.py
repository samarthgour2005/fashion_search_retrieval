"""
Indexer entrypoint (Part A of the assignment).

Run this once to build the searchable index:

    python -m indexer.build_index

It will:
  1. Sample a non-IID subset of Fashionpedia (or reuse a cached one)
  2. Synthesize a text description per image
  3. Encode images+text with CLIP, fuse into one embedding per image
  4. Extract per-garment dominant colors (cached separately)
  5. Build and save a FAISS index + metadata to disk (config.VECTORSTORE_DIR)
"""

import pickle

import config
from indexer.data_loader import load_or_build_subset, build_description
from indexer.color_labeling import extract_garment_colors
from indexer.embed_index import (
    load_clip, build_fused_embeddings, build_faiss_index, save_faiss_index,
)


def main():
    print(f"Device: {config.DEVICE}")

    subset, category_names = load_or_build_subset()
    n = len(subset)
    print(f"Indexing {n} images")

    descriptions = [build_description(subset[i], category_names) for i in range(n)]

    model, preprocess = load_clip()
    fused_embeddings = build_fused_embeddings(model, preprocess, subset, descriptions)
    print("Fused embeddings shape:", fused_embeddings.shape)

    garment_colors_per_image = extract_garment_colors(subset, category_names)

    index, use_gpu_index = build_faiss_index(fused_embeddings)
    save_faiss_index(index, use_gpu_index)
    print(f"Saved FAISS index to {config.FAISS_INDEX_PATH} ({index.ntotal} vectors)")

    metadata = {
        "image_ids": [subset[i]["image_id"] for i in range(n)],
        "descriptions": descriptions,
        "garment_colors_per_image": garment_colors_per_image,
        "category_to_garment": config.CATEGORY_TO_GARMENT,
    }
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {config.METADATA_PATH}")


if __name__ == "__main__":
    main()
