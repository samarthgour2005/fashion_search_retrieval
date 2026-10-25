"""
Central configuration for the indexing and retrieval pipelines.

Everything a person would want to tune (sampling, fusion weight, re-rank
weight, model names, file paths) lives here — indexer/ and retriever/ import
from this module rather than hard-coding values, so changing an experiment
setting means editing one file, not hunting through the codebase.

Extensions:
 - SegFormer-based per-garment segmentation stage (indexer/segformer_segmenter.py)
 - Per-garment FAISS index (indexer/build_garment_index.py)
 - GarmentSearchEngine for segment-level retrieval (retriever/garment_search.py)
 - IVF FAISS garment index for scalable approximate search (build_garment_index --use-ivf)
 - Embedding Explorer and HTML export in the Streamlit app
"""

import os
import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Dataset sampling (indexer/data_loader.py)
# ---------------------------------------------------------------------------
DATASET_NAME = "detection-datasets/fashionpedia"
DATASET_SPLIT = "train"

SEED = 42
TOTAL_IMAGES = 30_000       # how many images to sample into the working subset
ALPHA = 0.3                 # Dirichlet concentration: lower = more class-skewed (non-IID)

# ---------------------------------------------------------------------------
# CLIP embedding + fusion (indexer/embed_index.py)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-B/32"
FUSE_ALPHA = 0.7            # weight on image embedding; (1 - FUSE_ALPHA) on text embedding
BATCH_SIZE = 64              # safe batch size for ViT-B/32 on a T4 (16GB)

# ---------------------------------------------------------------------------
# Color labeling (indexer/color_labeling.py)
# ---------------------------------------------------------------------------
# raw Fashionpedia category name -> simplified single-word garment name.
# Shared vocabulary between color labeling (indexer) and query parsing (retriever).
CATEGORY_TO_GARMENT = {
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "top",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "vest": "vest",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat": "coat",              # also stands in for "raincoat" (no separate category)
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "cape": "cape",
    "tie": "tie",
    "scarf": "scarf",
    "hat": "hat",
    "bag, wallet": "bag",
    "belt": "belt",
    "shoe": "shoe",
}

KNOWN_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "white", "black", "gray", "brown",
}

# ---------------------------------------------------------------------------
# Query parser (retriever/query_parser.py)
# ---------------------------------------------------------------------------
# Set to False to use the fast, zero-RAM rule-based regex parser (recommended on CPU).
# Set to True to use the Hugging Face LLM parser.
USE_LLM_QUERY_PARSER = False
PARSER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PARSER_MAX_NEW_TOKENS = 40

# ---------------------------------------------------------------------------
# Two-stage retrieval (retriever/rerank_search.py)
# ---------------------------------------------------------------------------
RERANK_CANDIDATE_POOL = 100   # Stage 1 recall size before re-ranking
ATTRIBUTE_WEIGHT = 0.4        # weight on attribute-match score; (1 - w) on CLIP similarity


# ---------------------------------------------------------------------------
# SegFormer garment segmentation (indexer/segformer_segmenter.py)
# ---------------------------------------------------------------------------
# Fashion-specific SegFormer fine-tuned on clothes segmentation.
# Labels: background, hat, hair, sunglasses, upper-clothes, skirt, pants,
#         dress, belt, left-shoe, right-shoe, face, left-leg, right-leg,
#         left-arm, right-arm, bag, scarf
SEGFORMER_MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

# Minimum fraction of total image area a segment must cover to be indexed.
# Filters out tiny artifacts and partial occlusions.
SEG_MIN_AREA_FRAC = 0.01

# Mapping from SegFormer label names -> our simplified garment vocabulary.
# Only labels listed here will be indexed as per-garment embeddings.
SEGFORMER_LABEL_TO_GARMENT = {
    "upper-clothes": "top",
    "skirt": "skirt",
    "pants": "pants",
    "dress": "dress",
    "belt": "belt",
    "left-shoe": "shoe",
    "right-shoe": "shoe",
    "bag": "bag",
    "scarf": "scarf",
    "hat": "hat",
}

# ---------------------------------------------------------------------------
# Per-garment IVF FAISS index (v3 — scalable approximate search)
# ---------------------------------------------------------------------------
# Number of Voronoi cells (partitions) used when building an IVFFlat index.
# A good rule of thumb: nlist ≈ sqrt(N) for N garment embeddings.
# Has no effect when --use-ivf is not passed to build_garment_index.
GARMENT_FAISS_NLIST = 64

# Number of cells to probe at query time.  Higher = better recall, slower search.
# Typically nprobe ≈ nlist / 8 gives a good recall/speed trade-off.
GARMENT_FAISS_NPROBE = 8

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
# Point these at a persistent volume (e.g. a mounted Google Drive path in Colab,
# or any durable disk path in a normal environment) if you need the artifacts
# to survive an ephemeral runtime being torn down.
BASE_DIR = os.environ.get("FASHION_SEARCH_HOME", os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

SUBSET_PATH = os.path.join(DATA_DIR, "fashionpedia_subset")            # HF datasets save_to_disk dir
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "fashionpedia_fused_index.faiss")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "fashionpedia_metadata.pkl")
GARMENT_COLORS_CACHE_PATH = os.path.join(VECTORSTORE_DIR, "garment_colors_per_image.pkl")

# Per-garment FAISS index (built by indexer/build_garment_index.py)
GARMENT_FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "fashionpedia_garment_index.faiss")
GARMENT_EMBEDDINGS_META_PATH = os.path.join(VECTORSTORE_DIR, "fashionpedia_garment_meta.pkl")

# Optional: redirect the Hugging Face cache itself (CLIP/Qwen/dataset downloads)
# to a persistent path so repeated sessions don't re-download from the Hub.
HF_CACHE_DIR = os.environ.get("HF_HOME")  # set this env var externally if desired

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
