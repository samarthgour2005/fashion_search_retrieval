"""
Central configuration for the indexing and retrieval pipelines.

Everything a person would want to tune (sampling, fusion weight, re-rank
weight, model names, file paths) lives here — indexer/ and retriever/ import
from this module rather than hard-coding values, so changing an experiment
setting means editing one file, not hunting through the codebase.
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
# Query parser LLM (retriever/query_parser.py)
# ---------------------------------------------------------------------------
PARSER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PARSER_MAX_NEW_TOKENS = 40

# ---------------------------------------------------------------------------
# Two-stage retrieval (retriever/rerank_search.py)
# ---------------------------------------------------------------------------
RERANK_CANDIDATE_POOL = 100   # Stage 1 recall size before re-ranking
ATTRIBUTE_WEIGHT = 0.4        # weight on attribute-match score; (1 - w) on CLIP similarity

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

# Optional: redirect the Hugging Face cache itself (CLIP/Qwen/dataset downloads)
# to a persistent path so repeated sessions don't re-download from the Hub.
HF_CACHE_DIR = os.environ.get("HF_HOME")  # set this env var externally if desired

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
