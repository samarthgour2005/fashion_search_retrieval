# 🚀 Developer Guide — Fashion Search Studio

A complete setup, configuration, and development guide for contributors.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Building the Indices (Required Before Running)](#building-the-indices)
5. [Running the Application](#running-the-application)
6. [Configuration Reference](#configuration-reference)
7. [CLI Usage](#cli-usage)
8. [Running Tests](#running-tests)
9. [Development Workflow](#development-workflow)
10. [Troubleshooting](#troubleshooting)
11. [Performance Tuning](#performance-tuning)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.13 | Managed via .python-version |
| PyTorch | >= 2.14 | CUDA optional but recommended |
| GPU VRAM | >= 6GB | Required for comfortable SegFormer + CLIP inference |
| Disk space | ~10 GB | Dataset subset + model caches + indices |
| RAM | >= 16 GB | 30k dataset subset + Qwen 1.5B on CPU |

---

## Project Structure

`
fashion-retrieval/
├── config.py                   # All parameters (single source of truth)
├── app.py                      # Streamlit Fashion Search Studio (v3, 1461 lines)
├── pyproject.toml              # Project metadata + dependencies
├── requirements.txt            # pip-compatible dependency list
├── hello.py                    # Sanity check / scratch file
│
├── indexer/                    # Offline indexing pipeline (run once)
│   ├── __init__.py
│   ├── data_loader.py          # Fashionpedia download + non-IID sampling
│   ├── color_labeling.py       # Garment color extraction (HSV buckets)
│   ├── embed_index.py          # CLIP encoding + FAISS index utilities
│   ├── segformer_segmenter.py  # SegFormer garment segmentation
│   ├── build_index.py          # Entrypoint: build full-image index
│   └── build_garment_index.py  # Entrypoint: build per-garment index
│
├── retriever/                  # Online retrieval engine
│   ├── __init__.py
│   ├── query_parser.py         # Qwen 2.5-1.5B LLM query parser
│   ├── rerank_search.py        # FashionSearchEngine (two-stage + fusion)
│   ├── garment_search.py       # GarmentSearchEngine (segment-level)
│   └── cli.py                  # Command-line interface
│
├── data/                       # Cached dataset subset (created at runtime)
│   └── fashionpedia_subset/    # HuggingFace Dataset saved to disk
│
├── vectorstore/                # FAISS indices + metadata (created at runtime)
│   ├── fashionpedia_fused_index.faiss      # Full-image index (~60 MB)
│   ├── fashionpedia_metadata.pkl           # image_ids, descriptions, colors
│   ├── garment_colors_per_image.pkl        # Color cache
│   ├── fashionpedia_garment_index.faiss    # Per-garment index (optional)
│   └── fashionpedia_garment_meta.pkl       # Garment metadata
│
├── results/                    # CLI output images (if --save_images used)
├── tests/                      # Unit tests
│   └── test_retriever_and_indexer.py
└── docs/                       # This documentation
    ├── ARCHITECTURE.md
    ├── API_REFERENCE.md
    ├── DEVELOPER_GUIDE.md      (this file)
    └── SEARCH_MODES.md
`

---

## Environment Setup

### Option A: Using uv (recommended)

`ash
# Install uv (fast Python package manager)
pip install uv

# Create virtual environment and install dependencies
uv sync

# Activate the environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/macOS
`

### Option B: Using pip

`ash
pip install -r requirements.txt
`

> **Note:** CLIP is installed from the OpenAI GitHub repository (not PyPI).
> pyproject.toml configures this automatically via [tool.uv.sources].

### Environment Variables (Optional)

Set these before running to redirect storage to a persistent volume:

`ash
# Windows PowerShell
 = "D:\fashion_data"
 = "D:\fashion_data\hf_cache"

# Linux / macOS
export FASHION_SEARCH_HOME="/mnt/data/fashion_data"
export HF_HOME="/mnt/data/fashion_data/hf_cache"
`

FASHION_SEARCH_HOME overrides where data/ and ectorstore/ are created.
HF_HOME overrides where HuggingFace downloads models and datasets.

---

## Building the Indices

Indices must be built **once** before running the application. Building requires internet access to download the Fashionpedia dataset and model weights.

### Step 1: Build the Full-Image Index

`ash
python -m indexer.build_index
`

This will:
1. Download detection-datasets/fashionpedia (~8 GB) if not cached
2. Sample 30,000 images with non-IID Dirichlet distribution
3. Synthesize text descriptions per image
4. Download CLIP ViT-B/32 (~350 MB) if not cached
5. Encode all 30,000 images in batches (takes 15-45 min on GPU, longer on CPU)
6. Extract dominant garment colors from bbox annotations
7. Build and save FAISS IndexFlatIP (~60 MB)

**Output files:**
- ectorstore/fashionpedia_fused_index.faiss
- ectorstore/fashionpedia_metadata.pkl
- ectorstore/garment_colors_per_image.pkl
- data/fashionpedia_subset/ (HF Dataset)

### Step 2: Build the Per-Garment Index

`ash
# Exact FlatIP index (smaller corpus or high accuracy needed)
python -m indexer.build_garment_index

# Approximate IVFFlat index (faster at query time, recommended for scale)
python -m indexer.build_garment_index --use-ivf

# Dry-run: index only first 500 images
python -m indexer.build_garment_index --max_images 500

# Resume an interrupted run
python -m indexer.build_garment_index --resume
`

This will:
1. Load the subset from disk (Step 1 required)
2. Download SegFormer-B2 (mattmdjaga/segformer_b2_clothes, ~350 MB) if not cached
3. Run SegFormer segmentation on each image (slow — ~5-15 sec/image on CPU)
4. CLIP-encode each garment crop in batches
5. Build and save the garment FAISS index

**Output files:**
- ectorstore/fashionpedia_garment_index.faiss
- ectorstore/fashionpedia_garment_meta.pkl

> **Tip:** The Garment Segment Search tab in the Streamlit app will be disabled (with a clear error message) if this index hasn't been built yet.

---

## Running the Application

### Streamlit Web UI

`ash
streamlit run app.py
`

Opens at http://localhost:8501 by default.

The app caches heavy resources (CLIP model, FAISS index, SegFormer) on first load.
Subsequent searches are fast (milliseconds to low seconds depending on mode).

### Configuration via Sidebar

The sidebar exposes runtime controls without restarting the app:

| Control | Default | Description |
|---|---|---|
| Results (top-k) | 6 | Number of images returned |
| FAISS candidate pool | 100 | Stage 1 recall breadth |
| Attribute weight | 0.4 | Balance between CLIP and attribute scoring |
| Override blend alpha | 0.35 | Override shift strength |
| SLERP interpolation | Off | Smoother embedding interpolation |
| Image weight in fusion | 0.7 | Image vs text in multimodal fusion |
| Results per row | 5 | Grid layout |
| Show baseline comparison | On | Side-by-side with CLIP-only results |

---

## Configuration Reference

All tunable parameters live in config.py. Edit this file to:

- Change the CLIP model: CLIP_MODEL_NAME = "ViT-L/14"
- Adjust sampling size: TOTAL_IMAGES = 10_000
- Change the LLM parser: PARSER_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
- Tighten color segmentation: SEG_MIN_AREA_FRAC = 0.05
- Tune IVF parameters: GARMENT_FAISS_NLIST = 128

See [API_REFERENCE.md](API_REFERENCE.md) for the full parameter table.

---

## CLI Usage

The CLI provides a quick way to run searches without the Streamlit UI.

`ash
# Text search with attribute re-ranking (default)
python -m retriever.cli "A red tie and a white shirt in a formal setting"

# Adjust top-k results
python -m retriever.cli "casual weekend outfit" --top_k 10

# CLIP-only baseline (no LLM parsing, no attribute re-ranking)
python -m retriever.cli "casual weekend outfit" --baseline

# Image-only search
python -m retriever.cli --image path/to/reference.jpg

# Multimodal: image + text
python -m retriever.cli "summer outfit" --image path/to/reference.jpg

# Save retrieved images to disk
python -m retriever.cli "blue dress" --save_images ./results
`

---

## Running Tests

`ash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/ -k TestQueryParserJSONExtraction
`

### Test Coverage

| Test Class | What It Tests |
|---|---|
| TestQueryParserJSONExtraction | JSON extraction from LLM output (standard, nested, markdown code blocks, invalid JSON) |
| TestFashionSearchEngineLazyLoading | QueryParser is not instantiated at init; lazily loads on first use |
| TestFAISSIndexBuilding | IndexFlatIP builds correctly on CPU with correct vector count |

---

## Development Workflow

### Adding a New Search Mode

1. Add encoding logic to etriever/rerank_search.py (new method on FashionSearchEngine)
2. Add a new tab to pp.py using st.tabs()
3. Update config.py if new hyperparameters are needed
4. Add tests in 	ests/

### Adding a New Color

Add the color name string to config.KNOWN_COLORS and update the HSV bucket logic in indexer/color_labeling.py::hsv_to_color_name(). Rebuild the index.

### Adding a New Garment Type

1. Add to config.CATEGORY_TO_GARMENT (for color labeling)
2. Add to config.SEGFORMER_LABEL_TO_GARMENT if SegFormer supports it
3. Rebuild both indices

### Changing Embedding Dimension

If you switch CLIP models (e.g., to ViT-L/14 which outputs 768-dim), you **must** rebuild both FAISS indices. The dimension is inferred automatically at index build time.

---

## Troubleshooting

### GarmentIndexNotFoundError

The garment segment search index hasn't been built. Run:
`ash
python -m indexer.build_garment_index
`

### FileNotFoundError for FAISS index or metadata

The main index hasn't been built. Run:
`ash
python -m indexer.build_index
`

### CUDA out of memory

Reduce BATCH_SIZE in config.py (try 32 or 16). The model is already loaded in eval mode but some GPUs can't handle the default 64.

### Slow CPU inference

- SegFormer segmentation takes 5-15 seconds per image on CPU. This is expected.
- CLIP encoding is faster (~0.1s/image on CPU).
- Consider running uild_garment_index on a machine with GPU access, then copying the resulting files.

### Qwen LLM takes too long

The LLM parser runs per query (not per image), so it's a fixed cost per search (~2-8 seconds on CPU). To disable parsing, use search_baseline() or pass --baseline in the CLI.

### Subset not downloading

Check that HF_HOME is set correctly or has enough disk space (~8 GB for Fashionpedia).

### Import errors

Ensure you're running from the project root (where config.py lives). The modules use relative imports via import config.

---

## Performance Tuning

### Speed

| Bottleneck | Solution |
|---|---|
| Stage 1 FAISS search (exact) | Use IVF index for garment search (--use-ivf) |
| Qwen LLM parsing | Reduce PARSER_MAX_NEW_TOKENS to 20 |
| SegFormer segmentation | Reduce input image resolution in preprocessor |
| CLIP encoding at index time | Increase BATCH_SIZE (if VRAM allows) |

### Recall Quality

| Goal | Solution |
|---|---|
| Better attribute matching | Increase ATTRIBUTE_WEIGHT (up to 0.6) |
| Richer candidate pool | Increase RERANK_CANDIDATE_POOL (e.g., 200) |
| Smoother override interpolation | Enable SLERP_BLEND = True |
| Stronger override effect | Increase OVERRIDE_BLEND_ALPHA (e.g., 0.5) |

### Memory

| Resource | Size | Reduction Strategy |
|---|---|---|
| CLIP ViT-B/32 | ~150 MB | Use smaller model (ViT-S) |
| SegFormer-B2 | ~100 MB | Use SegFormer-B0 |
| Qwen 2.5-1.5B | ~3 GB on CPU | Use quantized version |
| FAISS 30k x 512 | ~60 MB | Reduce subset size |

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details and [API_REFERENCE.md](API_REFERENCE.md) for full function signatures.*
