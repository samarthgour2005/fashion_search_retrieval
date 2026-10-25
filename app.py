"""
Fashion Search Studio — Streamlit App
======================================

Four search modes:
  1. 🔍 Text Search          — natural-language query with attribute parsing
  2. 🖼️ Image Search         — upload a reference image (+ optional text)
  3. 🧩 Garment Segment Search — search by a specific garment crop
  4. 🗺️ Embedding Explorer    — visualise query position in 2D embedding space

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import pickle
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Fashion Search Studio",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS (Light Luxury Theme) ─────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables (Pure Light Theme) ── */
:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --accent: #4f46e5;
    --accent-hover: #4338ca;
    --accent-glow: rgba(79, 70, 229, 0.15);
    --accent-alt: #ec4899;
    --accent-green: #059669;
    --accent-yellow: #d97706;
    --accent-blue: #0284c7;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --shadow-soft: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
    --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    --shadow-hover: 0 12px 24px -4px rgba(79, 70, 229, 0.12), 0 4px 8px -2px rgba(0, 0, 0, 0.04);
    --radius: 12px;
    --radius-sm: 8px;
}

/* ── Global Styles ── */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit default decoration header */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem; max-width: 1520px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 16px rgba(15, 23, 42, 0.03) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
}

/* ── Headings ── */
h1, h2, h3, h4 { color: var(--text-primary) !important; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif !important; }
h1 { font-size: 2.1rem !important; font-weight: 800 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.42) !important;
    color: #ffffff !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs / Text Area / Selects ── */
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.16) !important;
}
.stSlider [data-baseweb="slider"] { background: transparent !important; }

/* ── File Uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #ffffff !important;
    border: 2px dashed var(--border-strong) !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
    background: #f0f4ff !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: var(--shadow-soft) !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: var(--accent) !important;
}

/* ── Image Display Cards ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-soft) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
[data-testid="stImage"] img:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.1) !important;
}

/* ── Score Badges (Light Fashion Palette) ── */
.score-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 16px;
    font-size: 0.73rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
    letter-spacing: 0.01em;
}
.badge-clip  { background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }
.badge-attr  { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.badge-final { background: #ffe4e6; color: #be123c; border: 1px solid #fecdd3; }
.badge-seg   { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.badge-tri   { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }

/* ── Divider ── */
.divider { border-top: 1px solid var(--border); margin: 1.3rem 0; }

/* ── Section Header ── */
.section-header {
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.5rem 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.01em;
}

/* ── Info / Alert Boxes ── */
.info-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 1.15rem;
    line-height: 1.6;
    box-shadow: var(--shadow-soft);
}
.v3-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.86rem;
    color: #166534;
    margin-bottom: 1rem;
}
.disabled-hint {
    background: #fff1f2;
    border: 1px dashed #fca5a5;
    border-left: 4px solid #e11d48;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #9f1239;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* ── Chip Tags ── */
.chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2px 10px;
    font-size: 0.74rem;
    color: var(--text-primary);
    margin: 2px;
    font-weight: 600;
}

/* ── Tabs (Chic Light Switcher) ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 6px !important;
    gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Checkbox ── */
.stCheckbox label { color: var(--text-secondary) !important; font-weight: 500 !important; }

/* ── Hero Banner (Animated Light Luxury Border) ── */
@keyframes heroGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-banner {
    background: linear-gradient(270deg, #4f46e5, #ec4899, #06b6d4, #10b981, #4f46e5);
    background-size: 300% 300%;
    animation: heroGlow 10s ease infinite;
    border-radius: 16px;
    padding: 2px;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.12);
}
.hero-inner {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.6rem 2.2rem;
}

/* ── Feature Pills in Hero ── */
.pill-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 4px 6px 0 0;
}

/* ── History Card ── */
.hist-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.65rem 0.95rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    cursor: default;
    transition: all 0.2s ease;
    box-shadow: var(--shadow-soft);
}
.hist-card:hover {
    border-color: var(--accent);
    color: var(--text-primary);
    transform: translateX(2px);
    box-shadow: var(--shadow-card);
}

/* ── Performance Breakdown Bar ── */
.perf-bar {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent-alt));
    margin: 2px 0 6px;
    transition: width 0.5s ease;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────
import config

ALL_GARMENTS = sorted(set(config.CATEGORY_TO_GARMENT.values()))
SEG_GARMENTS = sorted(set(config.SEGFORMER_LABEL_TO_GARMENT.values()))

GARMENT_ICONS = {
    "shirt": "👔",
    "pants": "👖",
    "dress": "👗",
    "coat": "🧥",
    "jacket": "🧥",
    "skirt": "👗",
    "sweater": "🧶",
    "suit": "🤵",
    "cape": "🧣",
    "tie": "👔",
    "scarf": "🧣",
    "hat": "👒",
    "bag": "👜",
    "belt": "🎗️",
    "shoe": "👟",
}

# ── Session state init ─────────────────────────────────────────────────────
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "export_html" not in st.session_state:
    st.session_state.export_html = None


# ── Helper: add to search history ──────────────────────────────────────────
def _add_history(mode: str, query: str):
    entry = {
        "mode": mode,
        "query": query,
        "ts": time.strftime("%H:%M:%S"),
    }
    hist = st.session_state.search_history
    if not hist or hist[-1]["query"] != query:
        hist.append(entry)
    if len(hist) > 8:
        hist.pop(0)


# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset subset…")
def get_subset():
    from indexer.data_loader import load_or_build_subset
    subset, _ = load_or_build_subset()
    return subset


@st.cache_resource(show_spinner="Loading CLIP + FAISS index…")
def get_search_engine():
    from retriever.rerank_search import FashionSearchEngine
    return FashionSearchEngine()


@st.cache_resource(show_spinner="Loading garment FAISS index…")
def get_garment_engine():
    try:
        from retriever.garment_search import GarmentSearchEngine
        return GarmentSearchEngine(), None
    except Exception as e:
        return None, str(e)


# ── Image from dataset ─────────────────────────────────────────────────────
def get_image_from_index(subset, idx: int) -> Image.Image:
    try:
        return subset[idx]["image"].convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (241, 245, 249))


# ── PIL → bytes ────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Render results grid ────────────────────────────────────────────────────
def render_results_grid(
    results: list[dict],
    subset,
    mode: str = "rerank",
    cols_per_row: int = 5,
    label: str = "",
):
    if not results:
        st.warning("No results returned.")
        return

    if label:
        st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

    cols = st.columns(cols_per_row)
    for i, r in enumerate(results):
        col = cols[i % cols_per_row]
        with col:
            img = get_image_from_index(subset, r["index"])
            st.image(img, use_container_width=True)

            rank_badge = f'<span class="score-badge badge-clip">#{i+1}</span> '
            if mode == "rerank":
                badges = (
                    rank_badge
                    + f'<span class="score-badge badge-clip">CLIP {r["clip_score"]:.3f}</span> '
                    + f'<span class="score-badge badge-attr">Attr {r["attr_score"]:.2f}</span> '
                    + f'<span class="score-badge badge-final">★ {r["final_score"]:.3f}</span>'
                )
            elif mode == "baseline":
                badges = rank_badge + f'<span class="score-badge badge-clip">Score {r["score"]:.3f}</span>'
            elif mode == "garment":
                garment_name = r.get("garment_label", "garment")
                icon = GARMENT_ICONS.get(garment_name, "🧩")
                badges = (
                    rank_badge
                    + f'<span class="score-badge badge-seg">Seg {r["score"]:.3f}</span> '
                    + f'<span class="score-badge badge-attr">{icon} {garment_name}</span>'
                )
            else:
                badges = rank_badge

            st.markdown(badges, unsafe_allow_html=True)
            gc = r.get("garment_colors", {})
            if gc:
                chips = " ".join(
                    f'<span class="chip">{g}:{c}</span>' for g, c in gc.items()
                )
                st.markdown(chips, unsafe_allow_html=True)


# ── Performance breakdown ──────────────────────────────────────────────────
def render_perf(timings: dict[str, float]):
    total = sum(timings.values()) + 1e-9
    cols = st.columns(len(timings))
    for i, (lbl, ms) in enumerate(timings.items()):
        with cols[i]:
            pct = int(ms / total * 100)
            st.markdown(
                f"**{lbl}**  \n"
                f'<div class="perf-bar" style="width:{max(pct,3)}%"></div>'
                f'<span style="font-size:0.78rem;color:var(--text-muted);font-weight:600;">{ms:.1f} ms</span>',
                unsafe_allow_html=True,
            )


# ── Export results as Light HTML ───────────────────────────────────────────
def build_export_html(results: list[dict], subset, mode: str, query: str) -> str:
    rows = ""
    for i, r in enumerate(results):
        try:
            img = get_image_from_index(subset, r["index"])
            buf = io.BytesIO()
            img.thumbnail((220, 220))
            img.save(buf, format="JPEG", quality=80)
            import base64
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:10px;border:1px solid #e2e8f0;">'
        except Exception:
            img_tag = "<div style='height:160px;background:#f1f5f9;border-radius:10px;'></div>"

        score_str = f"{r.get('final_score', r.get('score', 0)):.3f}"
        gc = r.get("garment_colors", {})
        colors = ", ".join(f"{g}:{c}" for g, c in gc.items()) if gc else "—"
        rows += f"""
        <div style="display:inline-block;width:190px;margin:10px;vertical-align:top;
                    background:#ffffff;border-radius:12px;padding:10px;border:1px solid #e2e8f0;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);">
          {img_tag}
          <div style="margin-top:8px;">
            <span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;background:#e0e7ff;color:#3730a3;">#{i+1}</span>
            <span style="font-size:12px;font-weight:700;color:#0f172a;margin-left:4px;">Score: {score_str}</span>
          </div>
          <p style="margin:4px 0 0;font-size:11px;color:#64748b;">{colors}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fashion Search Results</title>
  <style>
    body {{ background: #f8fafc; color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 28px; }}
    h2 {{ color: #4f46e5; margin: 0 0 6px; font-weight: 800; }}
    .sub {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h2>👔 Fashion Search Studio — Results</h2>
  <div class="sub">Mode: <b>{mode}</b> &nbsp;|&nbsp; Query: <i>{query}</i> &nbsp;|&nbsp; Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
  <div>{rows}</div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.8rem 0 0.5rem;">
            <span style="font-size:2.8rem;">👔</span>
            <h2 style="margin:0.3rem 0 0; font-size:1.3rem; font-weight:800; color:#0f172a;">Fashion Studio</h2>
            <p style="color:#64748b; font-size:0.78rem; font-weight:600; margin:0.2rem 0 0;">CLIP · SegFormer · FAISS</p>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Retrieval Parameters")
    top_k = st.slider("Results (top-k)", 1, 20, 6, key="top_k")
    candidate_pool = st.slider(
        "FAISS Candidate Pool", 10, 500, 100, key="pool",
        help="Stage 1 retrieval size before attribute re-ranking."
    )
    attribute_weight = st.slider(
        "Attribute Re-rank Weight (α)", 0.0, 1.0, 0.4, 0.05, key="attr_w",
        help="Weight assigned to color-garment match vs pure CLIP semantic similarity."
    )
    fuse_alpha = st.slider(
        "Image Weight in Multimodal Fusion", 0.0, 1.0, config.FUSE_ALPHA, 0.05, key="fuse_alpha",
        help="Weight of image embedding when fusing reference image + text."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Display Settings")
    cols_per_row = st.selectbox("Results per row", [3, 4, 5, 6], index=2)
    show_comparison = st.checkbox("Show CLIP baseline comparison", value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Search history
    if st.session_state.search_history:
        st.markdown("### 🕒 Recent Searches")
        for h in reversed(st.session_state.search_history[-5:]):
            st.markdown(
                f'<div class="hist-card">'
                f'<span style="font-size:0.7rem;color:var(--text-muted);font-weight:600;">{h["ts"]} · {h["mode"]}</span><br>'
                f'<span style="font-weight:600;color:var(--text-primary);">{h["query"][:36]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", key="clear_hist"):
            st.session_state.search_history = []
            st.rerun()
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size:0.73rem; color:#64748b; text-align:center; line-height:1.6; background:#f8fafc; padding:10px; border-radius:10px; border:1px solid #e2e8f0;">
        <b>SegFormer:</b> <code>mattmdjaga/segformer_b2_clothes</code><br>
        <b>Vision:</b> <code>CLIP ViT-B/32</code><br>
        <b>Dataset:</b> <code>Fashionpedia (Subsampled)</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN HEADER — Light Luxury Hero
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-banner">
      <div class="hero-inner">
        <div style="display:flex; align-items:center; gap:1.4rem;">
          <span style="font-size:3.2rem;">👔</span>
          <div>
            <h1 style="margin:0; background:linear-gradient(90deg, #4f46e5, #ec4899, #0284c7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; font-size:2.2rem !important; font-weight:800;">
              Fashion Search Studio
            </h1>
            <p style="margin:0.3rem 0 0.6rem; color:#475569; font-size:0.92rem; line-height:1.5;">
              Intelligent multimodal retrieval with SegFormer garment parsing, FAISS vector indexing, and attribute-aware re-ranking.
            </p>
            <div>
              <span class="pill-tag">✨ Multimodal Retrieval</span>
              <span class="pill-tag">🧩 SegFormer Segmentation</span>
              <span class="pill-tag">⚡ FAISS Indexing</span>
              <span class="pill-tag">🏷️ Attribute Re-ranking</span>
              <span class="pill-tag">🗺️ 2D Explorer</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load engines ───────────────────────────────────────────────────────────
engine = get_search_engine()
subset = get_subset()
garment_engine, garment_err = get_garment_engine()

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Text Search",
    "🖼️ Image Search",
    "🧩 Garment Segment Search",
    "🗺️ Embedding Explorer",
])


# ─────────────────────────────────────────────────────────────────────────
# TAB 1: Text Search
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🔍 Natural Language Query</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Describe the outfit or garment you are looking for in natural language. '
        'The query parser extracts color-garment bindings (e.g. <code>white shirt with red tie</code>) '
        'and re-ranks the top FAISS candidate images by verified attribute presence.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("💡 How Two-Stage Re-Ranking Works", expanded=False):
        st.markdown("""
**Retrieval Pipeline:**
1. **Stage 1 (Semantic Recall):** Your query is encoded into a 512-dim vector via **CLIP ViT-B/32**. FAISS performs inner-product similarity search over the indexed corpus to retrieve candidate images.
2. **Stage 2 (Attribute Extraction & Re-scoring):** The query parser extracts `{garment: color}` bindings. Candidates are re-scored using:
   $$\\text{Final Score} = (1 - \\alpha) \\cdot \\text{CLIP} + \\alpha \\cdot \\text{Attribute Match}$$
3. **Stage 3 (Precision Ranking):** Results are displayed sorted by `final_score`, overcoming compositional conflation (e.g. distinguishing *"red shirt, blue pants"* from *"blue shirt, red pants"*).
        """)

    # Quick example query buttons
    st.markdown("<p style='font-size:0.82rem;font-weight:700;color:#64748b;margin-bottom:6px;'>⚡ QUICK QUERY SUGGESTIONS:</p>", unsafe_allow_html=True)
    ex_c1, ex_c2, ex_c3, ex_c4 = st.columns(4)
    if ex_c1.button("👔 Red tie with white shirt", key="ex1", use_container_width=True):
        st.session_state["text_query"] = "a red tie with a white shirt"
        st.rerun()
    if ex_c2.button("👖 Blue jeans & black jacket", key="ex2", use_container_width=True):
        st.session_state["text_query"] = "casual blue jeans with a black jacket"
        st.rerun()
    if ex_c3.button("👗 Yellow summer dress", key="ex3", use_container_width=True):
        st.session_state["text_query"] = "bright yellow summer dress"
        st.rerun()
    if ex_c4.button("🧥 Red coat with black pants", key="ex4", use_container_width=True):
        st.session_state["text_query"] = "red coat with black pants"
        st.rerun()

    query_text = st.text_area(
        "Search Query",
        placeholder='e.g. "a red tie with a white shirt" or "casual blue jeans with black jacket"',
        height=75,
        key="text_query",
        label_visibility="collapsed",
    )
    run_text = st.button("🔍 Search Outfits", key="run_text")

    if run_text and query_text.strip():
        _add_history("Text", query_text.strip())
        timings = {}
        with st.spinner("Searching catalog…"):
            t0 = time.perf_counter()
            text_embed = engine.encode_query(query_text)
            timings["CLIP Encode"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            results, attrs = engine.search(
                query=query_text,
                top_k=top_k,
                candidate_pool=candidate_pool,
                attribute_weight=attribute_weight,
            )
            timings["FAISS + Re-rank"] = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieved", len(results))
        m2.metric("Latency", f"{total_ms:.0f} ms")
        m3.metric("Parsed Attributes", len(attrs))
        m4.metric("Candidate Pool", candidate_pool)

        if attrs:
            chips = " ".join(f'<span class="chip">✨ {g} → {c}</span>' for g, c in attrs.items())
            st.markdown(f"**Identified Outfit Attributes:** {chips}", unsafe_allow_html=True)

        with st.expander("⏱️ Latency Breakdown"):
            render_perf(timings)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if show_comparison:
            col_r, col_b = st.columns(2)
            with col_r:
                render_results_grid(results, subset, mode="rerank",
                                    cols_per_row=min(cols_per_row, 3), label="🏆 Re-ranked Results (Stage 2)")
            with col_b:
                baseline = engine.search_baseline(query=query_text, top_k=top_k)
                render_results_grid(baseline, subset, mode="baseline",
                                    cols_per_row=min(cols_per_row, 3), label="📊 CLIP-Only Baseline (Stage 1)")
        else:
            render_results_grid(results, subset, mode="rerank", cols_per_row=cols_per_row,
                                label="🏆 Re-ranked Results")

        html = build_export_html(results, subset, "Text Search", query_text)
        st.download_button(
            "⬇️ Export Results as HTML",
            data=html.encode(),
            file_name="fashion_results_text.html",
            mime="text/html",
            key="export_text",
        )


# ─────────────────────────────────────────────────────────────────────────
# TAB 2: Image Search
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🖼️ Reference Image Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Upload an outfit photo or fashion crop. You can optionally type text to refine or guide the query. '
        'Multimodal fusion blends image and text embeddings: '
        '<code>fused = α·image + (1-α)·text</code>.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_up, col_prev = st.columns([1, 1])
    with col_up:
        uploaded_img = st.file_uploader(
            "📸 Upload reference image", type=["jpg", "jpeg", "png", "webp"], key="img_upload",
        )
        img_text = st.text_input(
            "✍️ Optional refinement text",
            placeholder='e.g. "summer casual" or "dark evening style"',
            key="img_text",
        )
        run_img = st.button("🔍 Search with Image", key="run_img")

    with col_prev:
        if uploaded_img:
            ref_img = Image.open(uploaded_img).convert("RGB")
            st.image(ref_img, caption="Reference Query Image", use_container_width=True)

    if run_img and uploaded_img:
        ref_img = Image.open(uploaded_img).convert("RGB")
        _add_history("Image", img_text or "(reference image)")
        timings = {}
        with st.spinner("Encoding image & searching catalog…"):
            t0 = time.perf_counter()
            results, attrs = engine.search(
                query=img_text or None,
                image_input=ref_img,
                top_k=top_k,
                candidate_pool=candidate_pool,
                attribute_weight=attribute_weight,
            )
            timings["Total"] = (time.perf_counter() - t0) * 1000

        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieved", len(results))
        m2.metric("Latency", f"{timings['Total']:.0f} ms")
        m3.metric("Fusion Mode", "Multimodal (Image + Text)" if img_text else "Visual Only")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        render_results_grid(
            results, subset, mode="rerank", cols_per_row=cols_per_row, label="🏆 Search Results"
        )

        html = build_export_html(results, subset, "Image Search", img_text or "(image query)")
        st.download_button(
            "⬇️ Export Results as HTML",
            data=html.encode(),
            file_name="fashion_results_img.html",
            mime="text/html",
            key="export_img",
        )

    elif run_img and not uploaded_img:
        st.error("Please upload a reference image first.")


# ─────────────────────────────────────────────────────────────────────────
# TAB 3: Garment Segment Search
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🧩 Garment Segment Search</div>', unsafe_allow_html=True)

    if garment_engine is None:
        st.markdown(
            f"""
            <div class="disabled-hint">
            <b>⚠️ Garment FAISS index not found.</b><br><br>
            Build the per-garment index first:<br>
            <code>python -m indexer.build_garment_index</code><br><br>
            For faster approximate search (recommended for large corpora):<br>
            <code>python -m indexer.build_garment_index --use-ivf</code><br><br>
            SegFormer model: <code>{config.SEGFORMER_MODEL_NAME}</code><br>
            Output: <code>{config.GARMENT_FAISS_INDEX_PATH}</code><br><br>
            <i>Error: {garment_err}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box">'
            'Upload a full outfit image and choose a target garment piece (e.g. <b>dress, jacket, pants</b>). '
            'SegFormer segments the garment, isolates the bounding crop, and retrieves matching individual garments '
            'from the dedicated per-garment FAISS index.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_up5, col_cfg5 = st.columns([1, 1])

        with col_up5:
            uploaded_seg = st.file_uploader(
                "📸 Upload outfit image", type=["jpg", "jpeg", "png", "webp"], key="seg_upload",
            )
            if uploaded_seg:
                ref_seg_preview = Image.open(uploaded_seg).convert("RGB")
                st.image(ref_seg_preview, caption="Query Outfit Image", use_container_width=True)

        with col_cfg5:
            available = garment_engine.available_garments() if garment_engine else SEG_GARMENTS
            target_garment = st.selectbox(
                "Target garment to isolate",
                available,
                format_func=lambda g: f"{GARMENT_ICONS.get(g, '🏷️')} {g.capitalize()}",
                key="seg_garment",
            )
            show_overlay = st.checkbox("Show SegFormer segmentation overlay", value=True, key="show_overlay")
            show_confidence = st.checkbox("Show garment area breakdown chart", value=True, key="show_confidence")
            run_seg = st.button("🧩 Segment & Search Garment", key="run_seg")

        if uploaded_seg and show_overlay:
            ref_seg = Image.open(uploaded_seg).convert("RGB")
            with st.spinner("Running SegFormer neural segmentation…"):
                try:
                    from indexer.segformer_segmenter import get_default_segmenter
                    seg = get_default_segmenter()
                    overlay = seg.get_segmentation_overlay(ref_seg, annotate_labels=True)
                    seg_data = seg.get_all_segments_summary(ref_seg)
                    crops_raw = seg.segment_garments(ref_seg)

                    oc1, oc2 = st.columns(2)
                    with oc1:
                        st.image(overlay, caption="SegFormer Multi-Class Segmentation", use_container_width=True)
                    with oc2:
                        if crops_raw:
                            target_crop_prev = next(
                                (c for g, c in crops_raw if g == target_garment), crops_raw[0][1]
                            )
                            icon = GARMENT_ICONS.get(target_garment, "✨")
                            st.image(target_crop_prev, caption=f"{icon} Isolated Crop: {target_garment.capitalize()}", use_container_width=True)
                        else:
                            st.info("No distinct garments detected in this image.")

                    if show_confidence and seg_data:
                        try:
                            import plotly.graph_objects as go
                            fig = go.Figure(go.Bar(
                                x=[s["area_pct"] for s in seg_data],
                                y=[f"{GARMENT_ICONS.get(s['garment'], '🏷️')} {s['garment'].capitalize()}" for s in seg_data],
                                orientation="h",
                                marker=dict(color="#4f46e5"),
                                text=[f'{s["area_pct"]:.1f}%' for s in seg_data],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                title="Detected Garments (% of image area)",
                                xaxis_title="Coverage (%)",
                                paper_bgcolor="#ffffff",
                                plot_bgcolor="#f8fafc",
                                font=dict(color="#0f172a", family="Plus Jakarta Sans"),
                                margin=dict(l=10, r=20, t=40, b=10),
                                height=max(200, len(seg_data) * 40 + 70),
                            )
                            fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                            fig.update_yaxes(showgrid=False)
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            pass
                except Exception as e:
                    st.warning(f"Overlay generation note: {e}")

        if run_seg:
            if not uploaded_seg:
                st.error("Please upload an outfit image first.")
            else:
                ref_seg = Image.open(uploaded_seg).convert("RGB")
                _add_history("Garment Segment", f"{target_garment}")

                with st.spinner("Segmenting target garment & searching index…"):
                    t0 = time.perf_counter()
                    out = garment_engine.search(
                        image_input=ref_seg,
                        target_garment=target_garment,
                        top_k=top_k,
                        candidate_pool=candidate_pool,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000

                results5 = out["results"]
                q_crop = out["query_crop"]
                fallback = out["fallback"]
                found = out["found_garment"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Retrieved", len(results5))
                m2.metric("Latency", f"{elapsed:.0f} ms")
                m3.metric("Mode", "Fallback" if fallback else "Direct Segment")
                m4.metric("Found Garment", found or "—")

                if fallback:
                    st.warning(f"⚠️ '{target_garment}' was not detected in this image. Fell back to full-image search.")

                if q_crop:
                    with st.expander("🔍 Query Garment Crop Used"):
                        st.image(q_crop, width=220)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                render_results_grid(
                    results5, subset, mode="garment", cols_per_row=cols_per_row,
                    label=f"🧩 Matching '{target_garment.capitalize()}' Items",
                )

                html = build_export_html(results5, subset, "Garment Segment Search", f"{target_garment}")
                st.download_button(
                    "⬇️ Export Results as HTML",
                    data=html.encode(),
                    file_name="fashion_results_garment.html",
                    mime="text/html",
                    key="export_seg",
                )


# ─────────────────────────────────────────────────────────────────────────
# TAB 4: Embedding Explorer
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🗺️ 2D Embedding Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Project high-dimensional FAISS embeddings into 2D space via PCA or UMAP. '
        'Visualise how garments cluster by category and observe where your query lands.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_exp_l, col_exp_r = st.columns([1, 2])

    with col_exp_l:
        proj_method = st.selectbox("Projection Method", ["PCA", "UMAP"], key="proj_method",
                                   help="PCA is faster; UMAP produces tighter topological clusters.")
        n_points = st.slider("Corpus points to sample", 100, 5000, 1000, 100, key="n_points",
                             help="Random sample of indexed images plotted.")
        uploaded_exp = st.file_uploader(
            "📸 Upload query image (optional)", type=["jpg", "jpeg", "png", "webp"], key="exp_upload",
        )
        exp_query_text = st.text_input("✍️ Or enter text query", key="exp_text",
                                       placeholder='e.g. "blue denim jacket"')
        run_exp = st.button("🗺️ Explore Embeddings", key="run_exp")

    with col_exp_r:
        if run_exp:
            try:
                import plotly.graph_objects as go
                from sklearn.decomposition import PCA
            except ImportError:
                st.error("Install plotly and scikit-learn: `pip install plotly scikit-learn`")
                st.stop()

            with st.spinner(f"Computing {proj_method} projection…"):
                try:
                    faiss_idx = engine.index
                    total_vecs = faiss_idx.ntotal
                    np.random.seed(42)
                    sample_ids = np.random.choice(
                        min(total_vecs, len(engine.image_ids)), n_points, replace=False
                    ).tolist()

                    embs = np.zeros((len(sample_ids), faiss_idx.d), dtype="float32")
                    for i, sid in enumerate(sample_ids):
                        try:
                            faiss_idx.reconstruct(int(sid), embs[i])
                        except Exception:
                            pass

                    def _desc_label(desc: str) -> str:
                        for g in ALL_GARMENTS:
                            if g in desc.lower():
                                return g
                        return "other"

                    labels = [_desc_label(engine.descriptions[sid]) for sid in sample_ids]

                    if proj_method == "UMAP":
                        try:
                            from umap import UMAP
                            reducer = UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1)
                            coords_2d = reducer.fit_transform(embs)
                        except ImportError:
                            st.warning("umap-learn not installed; falling back to PCA.")
                            pca = PCA(n_components=2, random_state=42)
                            coords_2d = pca.fit_transform(embs)
                    else:
                        pca = PCA(n_components=2, random_state=42)
                        coords_2d = pca.fit_transform(embs)

                    query_coord = None
                    query_label = "query"
                    if uploaded_exp:
                        q_img = Image.open(uploaded_exp).convert("RGB")
                        q_emb = engine.encode_image_query(q_img)
                        if proj_method == "UMAP" and 'reducer' in dir():
                            try:
                                q_coord_2d = reducer.transform(q_emb.astype("float32"))
                            except Exception:
                                q_coord_2d = pca.transform(q_emb) if 'pca' in dir() else None
                        else:
                            q_coord_2d = pca.transform(q_emb.astype("float32"))
                        query_coord = q_coord_2d[0] if q_coord_2d is not None else None
                        query_label = "Uploaded Reference Image"
                    elif exp_query_text.strip():
                        q_emb = engine.encode_query(exp_query_text.strip())
                        if proj_method == "UMAP" and 'reducer' in dir():
                            try:
                                q_coord_2d = reducer.transform(q_emb.astype("float32"))
                            except Exception:
                                q_coord_2d = pca.transform(q_emb) if 'pca' in dir() else None
                        else:
                            q_coord_2d = pca.transform(q_emb.astype("float32"))
                        query_coord = q_coord_2d[0] if q_coord_2d is not None else None
                        query_label = f'"{exp_query_text.strip()}"'

                    unique_labels = sorted(set(labels))
                    palette = [
                        "#4f46e5", "#ec4899", "#059669", "#d97706", "#0284c7",
                        "#8b5cf6", "#f43f5e", "#10b981", "#eab308", "#06b6d4",
                        "#6366f1", "#f97316", "#14b8a6", "#84cc16", "#a855f7",
                    ]
                    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}

                    fig = go.Figure()

                    for lbl in unique_labels:
                        mask = [l == lbl for l in labels]
                        xs = coords_2d[mask, 0]
                        ys = coords_2d[mask, 1]
                        hover = [engine.descriptions[sid][:60] for sid, m in zip(sample_ids, mask) if m]
                        icon = GARMENT_ICONS.get(lbl, "🏷️")
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys,
                            mode="markers",
                            marker=dict(size=6, color=color_map[lbl], opacity=0.75,
                                        line=dict(width=0)),
                            name=f"{icon} {lbl.capitalize()}",
                            text=hover,
                            hovertemplate="%{text}<extra></extra>",
                        ))

                    if query_coord is not None:
                        fig.add_trace(go.Scatter(
                            x=[query_coord[0]], y=[query_coord[1]],
                            mode="markers",
                            marker=dict(size=18, color="#ec4899", symbol="star",
                                        line=dict(width=2, color="#ffffff")),
                            name=f"⭐ Query: {query_label}",
                            hovertemplate=f"Query: {query_label}<extra></extra>",
                        ))

                    dim_label = "PC" if proj_method == "PCA" else "UMAP"
                    fig.update_layout(
                        title=f"{proj_method} Projection ({n_points} Sampled Embeddings)",
                        xaxis_title=f"{dim_label} 1",
                        yaxis_title=f"{dim_label} 2",
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#f8fafc",
                        font=dict(color="#0f172a", family="Plus Jakarta Sans"),
                        legend=dict(
                            bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1,
                            font=dict(size=11),
                        ),
                        margin=dict(l=10, r=10, t=50, b=10),
                        height=520,
                    )
                    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
                    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
                    st.plotly_chart(fig, use_container_width=True)

                    if proj_method == "PCA" and 'pca' in dir():
                        var = pca.explained_variance_ratio_
                        st.caption(
                            f"PCA Variance Explained: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}% "
                            f"(total={sum(var)*100:.1f}%)"
                        )

                except Exception as e:
                    st.error(f"Embedding explorer failed: {e}")
                    st.exception(e)
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:center;
                            height:320px; border:2px dashed #cbd5e1;
                            background:#ffffff; border-radius:14px; flex-direction:column; gap:0.8rem; color:#64748b;">
                  <span style="font-size:3.2rem;">🗺️</span>
                  <p style="margin:0; font-size:0.95rem; font-weight:600; color:#334155;">
                    Click <b>Explore Embeddings</b> to visualise the 2D cluster projection.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="divider"></div>
    <div style="text-align:center; color:#64748b; font-size:0.8rem; padding-bottom:2rem; line-height:1.8;">
        👔 <b>Fashion Search Studio</b> &nbsp;·&nbsp;
        CLIP ViT-B/32 &nbsp;·&nbsp;
        SegFormer-B2 &nbsp;·&nbsp;
        FAISS IndexFlatIP &nbsp;·&nbsp;
        Fashionpedia &nbsp;·&nbsp;
        Two-Stage Attribute Re-ranking
    </div>
    """,
    unsafe_allow_html=True,
)
