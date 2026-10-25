# 📖 Documentation Index — Fashion Search Studio

Welcome to the documentation for **Fashion Search Studio v3**, an end-to-end multimodal fashion image retrieval engine.

---

## Documentation Structure

| Document | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow diagrams, and module dependency graph |
| [API_REFERENCE.md](API_REFERENCE.md) | Full API reference for all classes and functions |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Setup, indexing, configuration, and troubleshooting |
| [SEARCH_MODES.md](SEARCH_MODES.md) | End-user guide to all 6 search modes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Guidelines for contributing to the codebase |

---

## Quick Navigation

### For Users

- [How to set up and run the app](DEVELOPER_GUIDE.md#environment-setup)
- [Build the search indices](DEVELOPER_GUIDE.md#building-the-indices)
- [Understand each search mode](SEARCH_MODES.md)
- [CLI reference](DEVELOPER_GUIDE.md#cli-usage)

### For Developers

- [System architecture overview](ARCHITECTURE.md#1-high-level-system-overview)
- [Two-stage retrieval pipeline](ARCHITECTURE.md#3-two-stage-retrieval-architecture)
- [Embedding fusion modes](ARCHITECTURE.md#5-embedding-fusion-modes)
- [FashionSearchEngine API](API_REFERENCE.md#class-fashionsearchengine)
- [GarmentSearchEngine API](API_REFERENCE.md#class-garmentsearchengine)
- [config.py parameters](API_REFERENCE.md#configpy)
- [Adding a new search mode](CONTRIBUTING.md#adding-a-new-search-mode)
- [Performance tuning](DEVELOPER_GUIDE.md#performance-tuning)

---

## System at a Glance

`
Fashionpedia Dataset (30k images)
           |
           |  Offline indexing (run once)
           v
┌──────────────────────────────────────────┐
│  indexer/                                │
│  ├── data_loader.py    Non-IID sampling  │
│  ├── color_labeling.py HSV colors        │
│  ├── embed_index.py    CLIP + FAISS      │
│  ├── segformer_segmenter.py SegFormer    │
│  ├── build_index.py    Full-image index  │
│  └── build_garment_index.py Garment idx │
└──────────────────────────────────────────┘
           |
           |  Persisted to vectorstore/
           v
┌──────────────────────────────────────────┐
│  vectorstore/                            │
│  ├── fashionpedia_fused_index.faiss      │
│  ├── fashionpedia_metadata.pkl           │
│  ├── garment_colors_per_image.pkl        │
│  ├── fashionpedia_garment_index.faiss    │
│  └── fashionpedia_garment_meta.pkl       │
└──────────────────────────────────────────┘
           |
           |  Online retrieval (per query)
           v
┌──────────────────────────────────────────┐
│  retriever/                              │
│  ├── query_parser.py   Qwen 2.5-1.5B    │
│  ├── rerank_search.py  FashionEngine     │
│  ├── garment_search.py GarmentEngine     │
│  └── cli.py            CLI interface     │
└──────────────────────────────────────────┘
           |
           v
   Streamlit app.py (6 search tabs)
`

---

## Key Technologies

| Technology | Version | Role |
|---|---|---|
| **CLIP ViT-B/32** | OpenAI | 512-dim multimodal embedding |
| **SegFormer-B2** | mattmdjaga/segformer_b2_clothes | Garment pixel segmentation |
| **Qwen 2.5-1.5B-Instruct** | Alibaba | LLM query attribute parsing |
| **FAISS IndexFlatIP** | Meta AI | Exact cosine similarity search |
| **FAISS IVFFlat** | Meta AI | Approximate cosine similarity (scalable) |
| **Fashionpedia** | Detection Datasets | 30k fashion image corpus |
| **Streamlit** | >= 1.63 | Interactive web UI |

---

*For the main project README, see the [repository root](../README.md).*
