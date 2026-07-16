"""
Encodes images + descriptions with CLIP, fuses image/text embeddings, and
builds a FAISS index over the fused vectors.
"""

import clip
import torch
import numpy as np
import faiss
from tqdm.auto import tqdm

import config


def load_clip():
    model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=config.DEVICE)
    model.eval()
    return model, preprocess


def encode_batch(model, preprocess, images, texts, fuse_alpha=config.FUSE_ALPHA):
    with torch.no_grad():
        image_input = torch.stack([preprocess(img) for img in images]).to(config.DEVICE)
        text_input = clip.tokenize(texts, truncate=True).to(config.DEVICE)

        image_embeds = model.encode_image(image_input)
        text_embeds = model.encode_text(text_input)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        fused = fuse_alpha * image_embeds + (1 - fuse_alpha) * text_embeds
        fused = fused / fused.norm(dim=-1, keepdim=True)

    return fused.cpu().numpy().astype("float32")


def build_fused_embeddings(model, preprocess, subset, descriptions, batch_size=config.BATCH_SIZE):
    all_fused = []
    n = len(subset)
    for start in tqdm(range(0, n, batch_size), desc="Encoding + fusing embeddings"):
        end = min(start + batch_size, n)
        batch_images = [subset[i]["image"].convert("RGB") for i in range(start, end)]
        batch_texts = descriptions[start:end]
        all_fused.append(encode_batch(model, preprocess, batch_images, batch_texts))
    return np.vstack(all_fused)


def build_faiss_index(fused_embeddings):
    embedding_dim = fused_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # inner product == cosine sim on normalized vectors

    use_gpu_index = hasattr(faiss, "StandardGpuResources") and config.DEVICE == "cuda"
    if use_gpu_index:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(fused_embeddings)
    return index, use_gpu_index


def save_faiss_index(index, use_gpu_index, path=config.FAISS_INDEX_PATH):
    cpu_index = faiss.index_gpu_to_cpu(index) if use_gpu_index else index
    faiss.write_index(cpu_index, path)


def load_faiss_index(path=config.FAISS_INDEX_PATH):
    return faiss.read_index(path)
