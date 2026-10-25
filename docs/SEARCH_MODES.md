# 🔍 Search Modes Guide — Fashion Search Studio

A complete guide to all 6 search modes available in the Fashion Search Studio.

---

## Overview

| Tab | Mode | Inputs | Best For |
|---|---|---|---|
| 1 | 🔍 Text Search | Natural language query | Describing outfit attributes in words |
| 2 | 🖼️ Image Search | Reference image + optional text | Finding visually similar outfits |
| 3 | ✨ Multimodal + Overrides | Image + attribute overrides | "Same style, different colors" |
| 4 | 🔀 Full Triangulation | Text + image + overrides | Complex multi-signal queries (v3) |
| 5 | 🧩 Garment Segment Search | Image + target garment | Finding specific garment pieces |
| 6 | 🗺️ Embedding Explorer | Image or text (optional) | Visualizing embedding space |

---

## Tab 1: 🔍 Text Search

**Best for:** Queries where you can describe the outfit in natural language.

### How It Works

`
Your text query
      |
      v  CLIP ViT-B/32 encode_text()
  Text embedding (512-dim)
      |
      v  FAISS IndexFlatIP.search(embed, candidate_pool)
  Top-N candidates by cosine similarity (Stage 1)
      |
  Qwen 2.5-1.5B parses query --> {garment: color} attributes
      |
      v  Re-rank: final = (1-w)*clip_score + w*attr_match (Stage 2)
  Top-K results sorted by final_score
`

### Example Queries

- "a red tie with a white shirt in a formal setting"
- "casual blue jeans and a green sweater"
- "elegant black dress for an evening event"
- "yellow raincoat with brown boots"

### Parameters

| Parameter | Effect |
|---|---|
| **Attribute weight** | Higher = color matching matters more than visual similarity |
| **FAISS candidate pool** | Larger pool = better recall but slightly slower Stage 2 |

### Metrics Shown

- **Results** — actual count returned
- **Total latency** — end-to-end time including CLIP encode
- **Parsed attrs** — number of {garment: color} pairs extracted by LLM
- **Candidate pool** — Stage 1 recall width

### Sidebar: Baseline Comparison

Enable "Show baseline comparison" to see Stage 2 re-ranked results **alongside** pure CLIP-only results — useful for understanding how much the attribute re-ranking helps.

---

## Tab 2: 🖼️ Image Search

**Best for:** You have a reference outfit photo and want visually similar images.

### How It Works

`
Reference image (PIL)              Optional text
      |                                  |
      v  CLIP encode_image()             v  CLIP encode_text()
  Image embedding (512-dim)         Text embedding (512-dim)
      |                                  |
      v  fused = alpha*img + (1-alpha)*text  (if text provided)
  Fused embedding
      |
  Two-stage retrieval (FAISS + optional attr re-ranking)
`

### Fusion Formula

`
fused = normalize(alpha * E_image + (1 - alpha) * E_text)
`

- lpha controlled by "Image weight in fusion" slider (default: 0.7)
- Text-only query: alpha = 0 (effectively pure text search)
- Image-only: returns image embedding directly

### Tips

- Works best with clean, well-lit fashion photos with a single subject
- Adding text context (e.g., "casual summer outfit") helps bias results
- Higher alpha values emphasize visual similarity over text description

---

## Tab 3: ✨ Multimodal + Attribute Overrides

**Best for:** "I like this outfit's style, but I want different colors."

### How It Works

`
Reference image  +  {shirt: blue, pants: black}
      |                         |
      v  encode_image()         v  CLIP("a blue shirt"), CLIP("a black pants")
  img_embed              override_embeds (one per override)
      |
  accumulated = img_embed
  for each override:
      accumulated = Blend(accumulated, override_embed, alpha)
      # SLERP: interpolates along CLIP unit sphere
      # NLerp: linear blend + normalize
      |
  Two-stage retrieval with override_attrs as target
`

### Override Blend Alpha

Controls how strongly each override shifts the image embedding:
- **Low alpha (0.1-0.2):** Subtle color shift, strong style preservation
- **Medium alpha (0.3-0.5):** Balanced shift
- **High alpha (0.6-0.8):** Override dominates, less original style preserved

### SLERP vs NLerp

| Method | Description | When to Use |
|---|---|---|
| **NLerp** (default) | Linear blend + L2 normalize | Fast, good for small alphas |
| **SLERP** | Spherical Linear Interpolation | Smoother semantic trajectory, better for large alphas |

SLERP maintains the unit-sphere geometry of CLIP embeddings, which means the semantic direction changes smoothly rather than "shortcutting" through the interior of the embedding space.

### Override Slots

Up to 5 garment-color override pairs can be specified. Only valid combinations (garment in CATEGORY_TO_GARMENT.values() and color in KNOWN_COLORS) are applied.

---

## Tab 4: 🔀 Full Triangulation (v3 Feature)

**Best for:** Complex queries combining semantic description, visual reference, and specific color requirements.

### How It Works

`
"formal business outfit" + reference_image + {tie: red, pants: black}
            |                      |                    |
            v                      v                    v
       E_text (512)           E_image (512)       override_embeds
            |                      |
            v  weighted sum
  fused = normalize(img_alpha * E_image + txt_alpha * E_text)
            |
  for each override:
      fused = Blend(fused, override_embed, alpha_override)
            |
  Two-stage retrieval
`

### Example Use Cases

- *"Find a formal outfit similar to this photo, but with a red tie"*
  - Text: "formal business outfit", Image: photo of formal wear, Override: {tie: red}

- *"Casual weekend look like this photo, but in summer colors"*
  - Text: "casual summer weekend", Image: reference photo, Override: {shirt: white, pants: beige}

### Text vs Image Weight

The sliders control how the text and image signals are weighted before applying overrides:

`
total = text_alpha + image_alpha
contribution_img = image_alpha / total
contribution_txt = text_alpha / total
`

- Higher image weight: results look more like the reference photo
- Higher text weight: results match the semantic description more

---

## Tab 5: 🧩 Garment Segment Search

**Best for:** Finding images that contain a specific garment piece that looks like a crop from your reference photo.

### How It Works

`
Reference image
      |
      v  SegFormer-B2 (live segmentation)
  Garment segments + pixel masks
      |
  Find: target_garment == "top" (for example)
      |                           |
  Found: CLIP-encode crop    Not found: CLIP-encode full image (fallback)
      |
  (optional) Blend with "a {color} top" for color override
      |
      v  Search per-garment FAISS index
  Garment-level hits
      |
      v  De-duplicate by image_id, keep best garment-level score
  Top-K unique images
`

### Segmentation Overlay

When "Show segmentation overlay" is enabled, SegFormer runs on the uploaded image and displays:
- **Color-coded overlay:** Each detected garment segment shown in a distinct color
- **Garment label annotations:** Garment names burned into each region
- **Target crop preview:** The specific garment crop that will be used as query
- **Coverage chart:** Plotly bar chart showing each garment's pixel area percentage

### v3 Color Override

Select a color override to shift the garment crop embedding toward a different color before searching:

`
crop_embed = CLIP(garment_crop)
override_embed = CLIP("a {override_color} {target_garment}")
query_embed = Blend(crop_embed, override_embed, alpha)
`

This lets you ask: *"Find tops that look like this one, but in red"*

### Fallback Behavior

If the target garment is not detected in the query image:
- App shows a warning: *"'top' not detected — fell back to full-image CLIP search"*
- Full-image CLIP embedding is used instead of a crop embedding
- Results may be less precise but still returned

### Supported Garments

Garments available depend on what was indexed. Default supported types:
	op, pants, skirt, dress, elt, shoe, ag, scarf, hat

---

## Tab 6: 🗺️ Embedding Explorer

**Best for:** Understanding how the retrieval system "sees" the dataset — exploring the semantic structure of the embedding space.

### How It Works

`
FAISS index (30k vectors, 512-dim each)
      |
  Random sample of N_points
      |
      v  PCA or UMAP dimensionality reduction
  2D coordinates
      |
  Scatter plot colored by garment category
      |
  (Optional) Project query image or text to same 2D space
  and mark it as a star point
`

### Projection Methods

| Method | Speed | Cluster Quality | Notes |
|---|---|---|---|
| **PCA** | Very fast | Linear structure only | Always available |
| **UMAP** | Slower | Better non-linear clusters | Requires umap-learn package |

### Reading the Plot

- Each **dot** represents one indexed image at its position in CLIP embedding space
- **Color** indicates the dominant garment category derived from the image description
- **Clusters** of same-colored dots indicate visually/semantically similar outfits
- The **star (⭐)** shows where a query image or text lands relative to the corpus — images near the star are the most similar

### Use Cases for Embedding Explorer

- **Verify dataset coverage:** Are all garment categories well-represented?
- **Debug poor search results:** Is the query landing far from expected clusters?
- **Understand override effects:** Compare where image-only vs overridden embeddings land
- **Identify dataset biases:** Imbalanced clusters indicate non-IID sampling was effective

### PCA Variance Report

When using PCA, the app displays:
`
PCA variance explained: PC1=18.3%, PC2=12.1% (total=30.4%)
`
Low explained variance is normal for 512-dim CLIP embeddings — most structure is in higher dimensions.

---

## Sidebar Controls Summary

| Control | Default | Affects |
|---|---|---|
| Results (top-k) | 6 | All tabs |
| FAISS candidate pool | 100 | Stage 1 recall breadth (all except Garment) |
| Attribute weight | 0.4 | Stage 2 re-ranking strength (Tabs 1-4) |
| Override blend alpha | 0.35 | Override shift strength (Tabs 3, 4, 5) |
| Use SLERP interpolation | Off | Blend method for overrides (Tabs 3, 4, 5) |
| Image weight in fusion | 0.7 | Image vs text weight (Tabs 2, 4) |
| Results per row | 5 | Display grid layout |
| Show baseline comparison | On | Side-by-side CLIP-only results (Tabs 1, 2, 3, 4) |

---

## Score Badges

Results display color-coded score badges:

| Badge | Color | Meaning |
|---|---|---|
| #N rank badge | Purple | Result rank position |
| CLIP X.XXX | Purple | FAISS inner-product cosine similarity |
| Attr X.XX | Green | Garment color attribute match fraction |
| ★ X.XXX | Pink | Final combined score |
| Seg X.XXX | Yellow | Segment-level garment similarity (Tab 5) |
| ▲ X.XXX | Blue | Triangulation final score (Tab 4) |

---

## HTML Export

Every search result set can be exported as a standalone HTML file with embedded base64 thumbnail images and score annotations. Click the **"⬇️ Export results as HTML"** button after any search.

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details and [API_REFERENCE.md](API_REFERENCE.md) for function signatures.*
