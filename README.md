# Fashionpedia Multimodal Text-to-Image Search

Two-stage fashion image retrieval: CLIP for broad semantic recall, plus a
color-attribute re-ranking layer that fixes CLIP's known weakness on
compositional queries (e.g. "red shirt, blue pants" vs "blue shirt, red
pants"). See `docs/` (or the submitted PDF) for the full approaches writeup.

## Structure

```
fashion-retrieval/
├── config.py               # all tunable parameters + file paths, single source of truth
├── requirements.txt
├── indexer/                # Part A — builds the searchable index (offline, run once)
│   ├── data_loader.py       # Fashionpedia loading + non-IID subset sampling
│   ├── color_labeling.py    # per-garment dominant color extraction (bbox -> HSV bucket)
│   ├── embed_index.py       # CLIP embedding, image/text fusion, FAISS index build
│   └── build_index.py       # entrypoint: python -m indexer.build_index
├── retriever/               # Part B — answers natural-language queries (online)
│   ├── query_parser.py      # LLM parses query -> {garment: color} pairs
│   ├── rerank_search.py     # two-stage search: FAISS recall + attribute re-rank
│   └── cli.py                # entrypoint: python -m retriever.cli "<query>"
├── data/                    # cached dataset subset (generated, not committed)
└── vectorstore/             # FAISS index + metadata + color cache (generated, not committed)
```

Indexing logic (`indexer/`) and retrieval logic (`retriever/`) are fully
separated modules — no shared mutable state, both read the same `config.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Build the index (run once)

```bash
python -m indexer.build_index
```

This samples a non-IID Fashionpedia subset, embeds it with CLIP, extracts
per-garment colors, and writes the FAISS index + metadata to
`vectorstore/`. Re-running reuses the cached subset (`data/`) and cached
color labels (`vectorstore/garment_colors_per_image.pkl`) instead of
recomputing them.

## Query the index

```bash
python -m retriever.cli "A red tie and a white shirt in a formal setting"

# CLIP-only, no re-ranking (for comparison)
python -m retriever.cli "casual weekend outfit for a city walk" --baseline

# more results
python -m retriever.cli "a bright yellow raincoat" --top_k 10
```

## Configuration

Every tunable value — dataset sample size, non-IID skew (`ALPHA`), CLIP
fusion weight (`FUSE_ALPHA`), re-rank candidate pool size, attribute weight,
model names, and all file paths — lives in `config.py`. Nothing is
hard-coded elsewhere; change an experiment setting in one place.


