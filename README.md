# 👗 Fashion Search Studio

> **Version 3** | Multimodal fashion image retrieval powered by CLIP, SegFormer, FAISS, and LLM query parsing

An end-to-end fashion image retrieval engine that solves CLIP's compositional weakness (e.g., distinguishing *"red shirt, blue pants"* from *"blue shirt, red pants"*) through a **two-stage attribute-aware re-ranking architecture**, per-garment segmentation indexing, and flexible embedding fusion.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **6 Search Modes** | Text, Image, Multimodal+Overrides, Full Triangulation, Garment Segment, Embedding Explorer |
| **Two-Stage Retrieval** | FAISS cosine recall + LLM-parsed attribute re-ranking |
| **SegFormer Segmentation** | Per-garment pixel crops from full outfit photos (18 semantic classes) |
| **SLERP Fusion** | Spherical Linear Interpolation for smooth embedding blending |
| **IVF FAISS** | Scalable approximate search for large garment corpora |
| **Interactive UI** | Streamlit app with glassmorphism dark theme, real-time metrics, HTML export |
| **Evaluation Suite** | Full IR benchmark (MAP, MRR, NDCG, P/R/F1, HitRate) with PDF report generation |

---

## 🤖 Models Used

| Model | Identifier | Role |
|---|---|---|
| **CLIP ViT-B/32** | openai/clip-vit-base-patch32 | 512-dim multimodal embeddings for images and text |
| **SegFormer-B2** | [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) | Pixel-level garment segmentation (18 classes) |
| **Qwen 2.5-1.5B** | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | LLM for parsing {garment: color} attribute pairs |
| **FAISS** | IndexFlatIP / IndexIVFFlat | High-performance vector similarity search |

---

## 📊 Benchmark Results

Evaluation run on **30,000 Fashionpedia images**, **Top-K = 10**, candidate pool = 100. All metrics averaged over **20 queries** each for text and image modalities. Two systems compared — *Reranked* (CLIP + attribute re-rank) vs. *Baseline* (CLIP-only).

### Text Queries — Summary

| Metric | Reranked ✅ | CLIP Baseline |
|---|---|---|
| **MAP** | **0.0403** | 0.0092 |
| **MRR** | **1.0000** | 0.5575 |
| **P@1** | **1.00** | 0.35 |
| **NDCG@1** | **1.0000** | 0.3500 |
| **HitRate@1** | **1.00** | 0.35 |
| **P@5** | **1.00** | 0.49 |
| **NDCG@5** | **1.0000** | 0.4634 |
| **HitRate@5** | **1.00** | 0.90 |
| **P@10** | **0.98** | 0.46 |
| **NDCG@10** | **0.9864** | 0.4512 |
| **HitRate@10** | **1.00** | 0.90 |

> 💡 The two-stage attribute re-ranking boosts MRR by **+79.3%** and NDCG@10 by **+118.7%** over raw CLIP retrieval on color+garment queries.

### Image Queries — Summary (Reranked)

| Metric | @1 | @5 | @10 |
|---|---|---|---|
| **P** | 1.00 | 1.00 | 0.90 |
| **R** | 0.0004 | 0.0022 | 0.0040 |
| **NDCG** | 1.0000 | 1.0000 | 0.9364 |
| **HitRate** | 1.00 | 1.00 | 1.00 |
| **MAP** | — | — | 0.0040 |
| **MRR** | — | — | 1.0000 |

> ℹ️ Recall values are low by construction — Fashionpedia ground-truth pools are large (avg ~5,000 relevant items per image query), making recall at small K inherently bounded.

### Sample Query Breakdown

| Query | # Relevant | Re-ranked NDCG@10 | Baseline NDCG@10 |
|---|---|---|---|
| "blue shirt" | 1,155 | 1.000 | 0.305 |
| "red dress" | 995 | 1.000 | 1.000 |
| "green coat" | 51 | 1.000 | 0.498 |
| "yellow skirt" | 88 | 1.000 | 0.520 |
| "pink sweater" | 63 | 1.000 | 0.325 |
| "white shirt with red tie" | 652 | 1.000 | 0.347 |
| "man in a black blazer" | 899 | 1.000 | 0.490 |
| "someone wearing a bright yellow raincoat" | 52 | 0.727 | 0.000 |

Full per-query breakdown: [`results/eval_report.json`](results/eval_report.json) · PDF: [`results/Fashion_Retrieval_Progress_Report.pdf`](results/Fashion_Retrieval_Progress_Report.pdf)

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
# Using uv (recommended)
pip install uv && uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Build the Indices (Run Once)

```bash
# Full-image CLIP index + color metadata (~15-45 min on GPU)
python -m indexer.build_index

# Per-garment SegFormer index (required for Tab 5)
python -m indexer.build_garment_index

# Optional: faster approximate IVFFlat garment index
python -m indexer.build_garment_index --use-ivf
```

### 3. Launch the App

```bash
streamlit run app.py
```

### 4. Or Use the CLI

```bash
# Text search with attribute re-ranking
python -m retriever.cli "a red tie with a white shirt"

# CLIP-only baseline
python -m retriever.cli "casual weekend outfit" --baseline

# Image search
python -m retriever.cli --image path/to/outfit.jpg

# Save results to disk
python -m retriever.cli "blue dress" --save_images ./results
```

### 5. Run the Evaluation Suite

```bash
# Full evaluation (text + image, with baseline comparison)
python -m evaluation.evaluator --top-k 10 --pool 100 --modes text image --compare-baseline

# Generate PDF report from results
python -m evaluation.generate_pdf_report
```

---

## 🏗️ Architecture

The system is built in two phases:

### Offline Indexing (Run Once)
```
Fashionpedia (30k images)
    → Non-IID sampling (Dirichlet α=0.3)
    → CLIP encode: fused(image, text) → FAISS IndexFlatIP  [full-image index]
    → SegFormer-B2 segmentation → CLIP encode crops → FAISS [garment index]
    → HSV color extraction per garment bbox                  [color cache]
```

### Online Retrieval (Per Query)
```
Query (text / image / overrides)
    → CLIP encode → Fused embedding (SLERP/NLerp for overrides)
    → Stage 1: FAISS cosine similarity → candidate pool (N=100)
    → Qwen 2.5-1.5B / Regex parser: parse {garment: color} attributes
    → Stage 2: re-rank by (1-w)*clip + w*attr_match
    → Top-K results
```

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed diagrams.

---

## 📱 Search Modes

### 1. 🔍 Text Search
Natural language queries. LLM extracts garment-color attributes for Stage 2 re-ranking.

### 2. 🖼️ Image Search
Upload a reference outfit photo. Optionally add text context for multimodal fusion.
```
fused = α·E_image + (1-α)·E_text
```

### 3. ✨ Multimodal + Attribute Overrides (v2)
Same style, different colors. Each override shifts the image embedding:
```
accumulated = Blend(E_image, CLIP("a {color} {garment}"), α_override)
```

### 4. 🔀 Full Triangulation (v3)
Fuse text + image + overrides simultaneously:
```
embed = Blend(α_img·E_image + α_txt·E_text, overrides)
```

### 5. 🧩 Garment Segment Search
SegFormer isolates a specific garment crop from your image, CLIP-encodes it, and searches a dedicated per-garment FAISS index. Includes live segmentation overlay visualization.

### 6. 🗺️ Embedding Explorer
Interactive PCA/UMAP visualization of 512-dim FAISS embeddings colored by garment category. Upload a query to see where it lands in embedding space.

See **[docs/SEARCH_MODES.md](docs/SEARCH_MODES.md)** for detailed guides.

---

## 📁 Project Structure

```
fashion-retrieval/
├── config.py                   # All parameters (single source of truth)
├── app.py                      # Streamlit Fashion Search Studio (6 tabs)
├── pyproject.toml              # Project metadata + dependencies
├── requirements.txt            # pip dependency list
│
├── indexer/                    # Offline indexing pipeline
│   ├── data_loader.py          # Fashionpedia + non-IID sampling
│   ├── color_labeling.py       # HSV dominant color extraction
│   ├── embed_index.py          # CLIP encoding + FAISS utilities
│   ├── segformer_segmenter.py  # SegFormer-B2 garment segmentation
│   ├── build_index.py          # Entrypoint: full-image index
│   └── build_garment_index.py  # Entrypoint: per-garment index
│
├── retriever/                  # Online retrieval engine
│   ├── query_parser.py         # Qwen 2.5-1.5B LLM + regex query parsing
│   ├── rerank_search.py        # FashionSearchEngine (two-stage + fusion)
│   ├── garment_search.py       # GarmentSearchEngine (segment-level)
│   └── cli.py                  # CLI search interface
│
├── evaluation/                 # 📏 Evaluation & benchmarking
│   ├── benchmark.py            # Ground-truth builder (text + image queries)
│   ├── evaluator.py            # Full eval pipeline (P/R/F1/NDCG/MAP/MRR)
│   ├── metrics.py              # IR metrics implementation
│   └── generate_pdf_report.py  # PDF report generator
│
├── results/                    # Evaluation outputs
│   ├── eval_report.json        # Full benchmark results (text + image)
│   └── Fashion_Retrieval_Progress_Report.pdf
│
├── data/                       # Cached dataset subset (auto-created)
├── vectorstore/                # FAISS indices + metadata (auto-created)
├── tests/                      # Unit + integration tests
│   ├── test_retriever_and_indexer.py
│   └── test_evaluation_metrics.py
└── docs/                       # 📖 Full documentation
    ├── README.md               # Documentation index
    ├── ARCHITECTURE.md         # System diagrams + data flow
    ├── API_REFERENCE.md        # Full API reference
    ├── DEVELOPER_GUIDE.md      # Setup + config + troubleshooting
    ├── SEARCH_MODES.md         # End-user search mode guide
    └── CONTRIBUTING.md         # Contribution guidelines
```

---

## ⚙️ Configuration

All parameters live in [config.py](config.py). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_IMAGES` | 30,000 | Dataset subset size |
| `CLIP_MODEL_NAME` | `"ViT-B/32"` | CLIP model variant |
| `FUSE_ALPHA` | 0.7 | Image weight in text+image fusion |
| `RERANK_CANDIDATE_POOL` | 100 | Stage 1 recall breadth |
| `ATTRIBUTE_WEIGHT` | 0.4 | Stage 2 color-attribute weight |
| `OVERRIDE_BLEND_ALPHA` | 0.35 | Attribute override shift strength |
| `SLERP_BLEND` | `False` | Use SLERP vs NLerp for blending |
| `USE_LLM_QUERY_PARSER` | `False` | Use Qwen LLM vs fast regex parser |
| `SEGFORMER_MODEL_NAME` | `"mattmdjaga/segformer_b2_clothes"` | SegFormer model |
| `SEG_MIN_AREA_FRAC` | 0.01 | Minimum garment segment area |
| `GARMENT_FAISS_NLIST` | 64 | IVFFlat Voronoi cells |
| `GARMENT_FAISS_NPROBE` | 8 | IVFFlat cells probed per query |

### Persistent Storage

```bash
# Redirect storage to a persistent volume (e.g., in Colab)
export FASHION_SEARCH_HOME="/content/drive/MyDrive/fashion_data"
export HF_HOME="/content/drive/MyDrive/fashion_data/hf_cache"
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- JSON extraction from LLM output (standard, nested, markdown code blocks, invalid)
- `QueryParser` lazy loading behavior
- FAISS index building on CPU
- IR metrics correctness (MAP, MRR, NDCG, P/R/F1, HitRate)
- Retriever and indexer integration

---

## 📖 Documentation

Full documentation is in the [docs/](docs/) directory:

| Doc | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System overview, pipeline diagrams, module dependency graph |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API for all classes and functions |
| [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Setup, indexing, config, CLI, troubleshooting, performance tuning |
| [SEARCH_MODES.md](docs/SEARCH_MODES.md) | End-user guide to all 6 search modes with examples |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contribution guidelines and development workflow |

---

## 🗂️ Data

| Asset | Source | Size |
|---|---|---|
| Fashionpedia dataset | [HuggingFace](https://huggingface.co/datasets/detection-datasets/fashionpedia) | ~8 GB |
| Full-image FAISS index (30k) | Built locally | ~60 MB |
| Full-image metadata | Built locally | ~3.5 MB |
| Garment color cache | Built locally | ~460 KB |
| Eval report (JSON) | Built locally | ~46 KB |

---

## 🗺️ Project Status

| Milestone | Status |
|---|---|
| Dataset loading + Non-IID sampling | ✅ Done |
| CLIP full-image indexing (30k) | ✅ Done |
| HSV color extraction + garment labeling | ✅ Done |
| Two-stage text retrieval (FAISS + re-rank) | ✅ Done |
| LLM query parser (Qwen 2.5-1.5B) | ✅ Done |
| Regex query parser (fast, CPU-only) | ✅ Done |
| **SegFormer-B2 garment segmentation** | ✅ **Done** |
| **Per-garment FAISS index (flat + IVF)** | ✅ **Done** |
| **Garment Segment Search engine** | ✅ **Done** |
| SLERP / NLerp embedding fusion | ✅ Done |
| Multimodal + Attribute Override search | ✅ Done |
| Full Triangulation search (v3) | ✅ Done |
| Streamlit app (6 tabs) | ✅ Done |
| **Evaluation suite (MAP / MRR / NDCG)** | ✅ **Done** |
| **Benchmark results gathered** | ✅ **Done** |
| **PDF progress report** | ✅ **Done** |
| CLI interface | ✅ Done |
| Unit + integration tests | ✅ Done |

---

## 📄 License

This project is for research and educational purposes. Models used are subject to their respective licenses:
- CLIP: [OpenAI Research License](https://github.com/openai/CLIP/blob/main/LICENSE)
- SegFormer: [Apache 2.0](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
- Qwen: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Fashionpedia: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
