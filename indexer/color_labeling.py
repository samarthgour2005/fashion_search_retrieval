"""
Per-garment dominant color extraction.

Fashionpedia's ontology has no color attribute, so color is derived directly
from the image: crop each annotated garment's bounding box, take the median
RGB (robust to shadows/edge noise), convert to HSV, and bucket into a coarse
color name. HSV is used instead of raw RGB-channel comparison because that
naive approach can't separate yellow from red (both have high R) and has no
way to represent white/black/gray at all.

This runs once per image at indexing time and is cached to disk — never
recomputed at query time.
"""

import os
import pickle
import colorsys

import numpy as np
from tqdm.auto import tqdm

import config


def hsv_to_color_name(h, s, v):
    """h in [0, 360), s and v in [0, 100]. Coarse heuristic buckets — enough to
    separate common garment colors (incl. white/black/gray), not a full color model."""
    if v < 15:
        return "black"
    if s < 12:
        return "white" if v > 80 else "gray"
    if h < 20 or h >= 340:
        return "red"
    if h < 45:
        return "brown" if v < 65 else "orange"
    if h < 65:
        return "yellow"
    if h < 170:
        return "green"
    if h < 260:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def get_dominant_color_name(image_rgb, bbox):
    """image_rgb: PIL Image (RGB). bbox: [x1, y1, x2, y2] (Fashionpedia format)."""
    w, h = image_rgb.size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_rgb.crop((x1, y1, x2, y2))
    arr = np.asarray(crop).reshape(-1, 3)
    if arr.size == 0:
        return None
    r, g, b = np.median(arr, axis=0) / 255.0
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    return hsv_to_color_name(hue * 360, sat * 100, val * 100)


def build_garment_category_ids(category_names):
    """category id -> simplified garment name, restricted to config.CATEGORY_TO_GARMENT."""
    return {
        category_names.index(raw): simple
        for raw, simple in config.CATEGORY_TO_GARMENT.items()
        if raw in category_names
    }


def extract_garment_colors(subset, category_names, cache_path=config.GARMENT_COLORS_CACHE_PATH):
    """Return a list of {garment: color} dicts, one per image in `subset`.

    Loads from `cache_path` if present; otherwise computes and saves there.
    """
    if os.path.exists(cache_path):
        print(f"Found cached garment colors at {cache_path} -- loading, skipping recomputation.")
        with open(cache_path, "rb") as f:
            garment_colors_per_image = pickle.load(f)
        print(f"Loaded colors for {len(garment_colors_per_image)} images.")
        return garment_colors_per_image

    garment_category_ids = build_garment_category_ids(category_names)

    garment_colors_per_image = []
    for i in tqdm(range(len(subset)), desc="Extracting garment colors"):
        example = subset[i]
        image_rgb = example["image"].convert("RGB")
        cats, bboxes = example["objects"]["category"], example["objects"]["bbox"]
        colors = {}
        for cat_id, bbox in zip(cats, bboxes):
            if cat_id not in garment_category_ids:
                continue
            garment_name = garment_category_ids[cat_id]
            color_name = get_dominant_color_name(image_rgb, bbox)
            if color_name:
                colors[garment_name] = color_name  # last detected instance wins if duplicated
        garment_colors_per_image.append(colors)

    with open(cache_path, "wb") as f:
        pickle.dump(garment_colors_per_image, f)
    print(f"Computed garment colors for {len(garment_colors_per_image)} images "
          f"and saved to {cache_path}.")
    return garment_colors_per_image
