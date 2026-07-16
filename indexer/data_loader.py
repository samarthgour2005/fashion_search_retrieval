"""
Loads Fashionpedia and draws a non-IID (class-skewed) image subset from it.

Also builds a short synthetic text description per image from its category
annotations, since Fashionpedia doesn't ship natural-language captions.
"""

import os
import numpy as np
from datasets import load_dataset, load_from_disk

import config


def _sample_non_iid_indices(bucket_sizes, buckets, total, alpha, rng, max_rounds=50):
    """Draw `total` image indices with a Dirichlet-skewed category distribution."""
    available = bucket_sizes.copy()
    selected = {c: 0 for c in range(len(bucket_sizes))}
    remaining = total

    for _ in range(max_rounds):
        if remaining <= 0:
            break
        active = np.where(available > 0)[0]
        if len(active) == 0:
            break
        proportions = rng.dirichlet(alpha * np.ones(len(active)))
        counts = (proportions * remaining).astype(int)
        for c, want in zip(active, counts):
            take = min(want, available[c])
            selected[c] += take
            available[c] -= take
            remaining -= take

    if remaining > 0:
        active = np.where(available > 0)[0]
        for c in active:
            if remaining <= 0:
                break
            take = min(remaining, available[c])
            selected[c] += take
            available[c] -= take
            remaining -= take

    final_indices = []
    for c, count in selected.items():
        if count > 0:
            final_indices.extend(buckets[c][:count].tolist())
    rng.shuffle(final_indices)
    return final_indices, selected


def build_non_iid_subset(total_images=config.TOTAL_IMAGES, alpha=config.ALPHA, seed=config.SEED):
    """Load Fashionpedia and return a (subset, category_names) pair sampled non-IID."""
    raw_ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    category_names = raw_ds.features["objects"]["category"].feature.names
    num_categories = len(category_names)

    rng = np.random.default_rng(seed)
    objects_column = raw_ds["objects"]  # columnar access -> doesn't decode images

    dominant_category = np.full(len(objects_column), -1, dtype=np.int64)
    for i, obj in enumerate(objects_column):
        cats, areas = obj["category"], obj["area"]
        if len(cats) == 0:
            continue
        dominant_category[i] = cats[int(np.argmax(areas))]

    buckets = {c: np.where(dominant_category == c)[0] for c in range(num_categories)}
    for idxs in buckets.values():
        rng.shuffle(idxs)
    bucket_sizes = np.array([len(buckets[c]) for c in range(num_categories)])

    selected_indices, _ = _sample_non_iid_indices(bucket_sizes, buckets, total_images, alpha, rng)
    subset = raw_ds.select(selected_indices)
    return subset, category_names, selected_indices


def build_description(example, category_names):
    """Synthesize a short text caption from an image's category annotations."""
    cats, areas = example["objects"]["category"], example["objects"]["area"]
    if len(cats) == 0:
        return "a photo of a fashion outfit"
    order = np.argsort(areas)[::-1]
    seen, ordered_names = set(), []
    for i in order:
        name = category_names[cats[i]]
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)
        if len(ordered_names) == 6:  # CLIP's text tower caps at 77 tokens
            break
    return "a photo of an outfit with " + ", ".join(ordered_names)


def load_or_build_subset():
    """Reuse a saved subset from disk if present (see config.SUBSET_PATH), else build fresh."""
    if os.path.exists(config.SUBSET_PATH):
        print(f"Loading cached subset from {config.SUBSET_PATH}")
        subset = load_from_disk(config.SUBSET_PATH)
        raw_ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
        category_names = raw_ds.features["objects"]["category"].feature.names
        return subset, category_names

    subset, category_names, _ = build_non_iid_subset()
    subset.save_to_disk(config.SUBSET_PATH)
    print(f"Saved subset ({len(subset)} images) to {config.SUBSET_PATH}")
    return subset, category_names
