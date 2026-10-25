"""
SegFormer-based garment segmentation stage (v2/v3 extension).

Uses ``mattmdjaga/segformer_b2_clothes`` — a SegFormer-B2 model fine-tuned
on a fashion-specific clothes segmentation dataset.  Label space:
  background, hat, hair, sunglasses, upper-clothes, skirt, pants, dress,
  belt, left-shoe, right-shoe, face, left-leg, right-leg, left-arm,
  right-arm, bag, scarf

``segment_garments`` is the main public function.  It accepts a PIL RGB
image and returns a list of ``(garment_label, crop_pil)`` tuples, where
``garment_label`` is one of the simplified garment names in
``config.SEGFORMER_LABEL_TO_GARMENT`` and ``crop_pil`` is the tight
bounding-box crop of that segment — ready to be CLIP-encoded.

v3 additions
------------
- ``segment_garments_with_masks``: returns full pixel masks alongside crops,
  enabling mask-based visualisation and confidence scoring.
- ``get_segmentation_overlay``: now annotates each coloured region with its
  garment label text.
- Robustness: handles all-background segmentation maps (no garments found),
  degenerate crops (zero-size bounding boxes), and model output shape quirks.

Filtering:
- Segments whose area < ``config.SEG_MIN_AREA_FRAC * total_pixels`` are
  dropped (removes tiny artifacts / heavily occluded pieces).
- Segments whose label is not in ``config.SEGFORMER_LABEL_TO_GARMENT`` are
  silently skipped (hair, face, legs, etc. are not garments we index).

The segmenter is designed to be instantiated once and reused across many
images.  Loading is lazy by default when called through
``get_default_segmenter()``.
"""

from __future__ import annotations

import colorsys
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import config

_default_segmenter: "GarmentSegmenter | None" = None


class GarmentSegment(NamedTuple):
    """Structured result from ``segment_garments_with_masks``."""
    garment: str          # simplified garment name (e.g. "top", "pants")
    crop: Image.Image     # tight bounding-box crop (RGB)
    mask: np.ndarray      # boolean (H, W) mask in original image coordinates
    area_frac: float      # fraction of total pixels covered by this segment
    bbox: tuple           # (x1, y1, x2, y2) in pixel coordinates


class GarmentSegmenter:
    """Wraps SegFormer model + image processor for garment segmentation."""

    def __init__(self, model_name: str = config.SEGFORMER_MODEL_NAME):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        self.model.to(config.DEVICE)

        # Build id -> label name lookup from the HF model config
        self.id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def predict_segmentation_map(self, image_pil: Image.Image) -> np.ndarray:
        """Run SegFormer inference and return an (H, W) array of label ids."""
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # Upsample logits to original image resolution
        logits = outputs.logits  # (1, num_labels, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        return seg_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_segments(
        self,
        image_rgb: Image.Image,
        seg_map: np.ndarray,
        min_area_frac: float,
        label_to_garment: dict,
    ) -> list[GarmentSegment]:
        """Core segmentation logic shared by all public methods."""
        total_pixels = seg_map.size
        min_pixels = int(min_area_frac * total_pixels)

        raw_results: list[tuple[str, GarmentSegment]] = []  # (garment, segment)
        seen_garments: dict[str, GarmentSegment] = {}       # garment -> best segment

        unique_ids = np.unique(seg_map)
        for label_id in unique_ids:
            raw_label = self.id2label.get(int(label_id), "")
            garment_name = label_to_garment.get(raw_label)
            if garment_name is None:
                continue  # not a garment we index

            mask = seg_map == label_id
            area = int(mask.sum())
            if area < min_pixels:
                continue  # too small

            # Bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue  # degenerate mask

            y1 = int(np.argmax(rows))
            y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
            x1 = int(np.argmax(cols))
            x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))

            if x2 <= x1 or y2 <= y1:
                continue  # zero-size box

            crop = image_rgb.crop((x1, y1, x2 + 1, y2 + 1))
            area_frac = area / total_pixels
            segment = GarmentSegment(
                garment=garment_name,
                crop=crop,
                mask=mask,
                area_frac=area_frac,
                bbox=(x1, y1, x2, y2),
            )

            # Keep only the largest instance per garment type
            if garment_name not in seen_garments or area > seen_garments[garment_name].mask.sum():
                seen_garments[garment_name] = segment

        # Sort largest-first
        return sorted(seen_garments.values(), key=lambda s: s.area_frac, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_garments(
        self,
        image_pil: Image.Image,
        min_area_frac: float = config.SEG_MIN_AREA_FRAC,
        label_to_garment: dict = None,
    ) -> list[tuple[str, Image.Image]]:
        """Segment ``image_pil`` and return cropped garment regions.

        Parameters
        ----------
        image_pil:
            Input image in RGB mode.
        min_area_frac:
            Minimum fraction of total pixels for a segment to be returned.
        label_to_garment:
            Override mapping; defaults to ``config.SEGFORMER_LABEL_TO_GARMENT``.

        Returns
        -------
        List of ``(garment_name, crop_pil)`` tuples, sorted by crop area
        descending (largest garment first).
        """
        if label_to_garment is None:
            label_to_garment = config.SEGFORMER_LABEL_TO_GARMENT

        image_rgb = image_pil.convert("RGB")
        seg_map = self.predict_segmentation_map(image_rgb)
        segments = self._extract_segments(image_rgb, seg_map, min_area_frac, label_to_garment)
        return [(s.garment, s.crop) for s in segments]

    def segment_garments_with_masks(
        self,
        image_pil: Image.Image,
        min_area_frac: float = config.SEG_MIN_AREA_FRAC,
        label_to_garment: dict = None,
    ) -> list[GarmentSegment]:
        """Like ``segment_garments`` but returns full ``GarmentSegment`` structs.

        Each struct includes the pixel mask, bounding box, and area fraction
        alongside the crop and garment label — useful for:
        - Mask-based compositing / visualization
        - Reporting segment confidence (area_frac as proxy)
        - Computing overlap between query and retrieved garments

        Returns
        -------
        List of ``GarmentSegment`` named tuples, largest garment first.
        """
        if label_to_garment is None:
            label_to_garment = config.SEGFORMER_LABEL_TO_GARMENT

        image_rgb = image_pil.convert("RGB")
        seg_map = self.predict_segmentation_map(image_rgb)
        return self._extract_segments(image_rgb, seg_map, min_area_frac, label_to_garment)

    def get_garment_crop(
        self, image_pil: Image.Image, target_garment: str
    ) -> Image.Image | None:
        """Return the crop for a specific garment, or None if not found."""
        segments = self.segment_garments(image_pil)
        for garment, crop in segments:
            if garment == target_garment:
                return crop
        return None

    def get_segmentation_overlay(
        self,
        image_pil: Image.Image,
        alpha: float = 0.45,
        annotate_labels: bool = True,
    ) -> Image.Image:
        """Return a blended image with the segmentation map overlaid in color.

        v3: optionally burns garment name text into each detected region so the
        user can immediately see which color corresponds to which garment.

        Parameters
        ----------
        image_pil:
            Input image.
        alpha:
            Blend weight for the colour overlay (0 = original, 1 = full colour).
        annotate_labels:
            If True, draw garment name labels over their bounding boxes.

        Returns
        -------
        Blended RGB PIL image with optional text annotations.
        """
        image_rgb = image_pil.convert("RGB")
        seg_map = self.predict_segmentation_map(image_rgb)

        num_labels = len(self.id2label)
        # Generate distinct hue-based colors for each label id
        colors = []
        for i in range(num_labels):
            hue = i / max(num_labels, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        color_map = np.array(colors, dtype=np.uint8)
        overlay_arr = color_map[seg_map]  # (H, W, 3)
        overlay_pil = Image.fromarray(overlay_arr, mode="RGB")

        # Blend with original
        blended = Image.blend(image_rgb, overlay_pil, alpha=alpha)

        # Annotate garment labels
        if annotate_labels:
            label_to_garment = config.SEGFORMER_LABEL_TO_GARMENT
            draw = ImageDraw.Draw(blended)

            try:
                font = ImageFont.truetype("arial.ttf", size=max(12, min(blended.width // 30, 24)))
            except (OSError, IOError):
                font = ImageFont.load_default()

            unique_ids = np.unique(seg_map)
            for label_id in unique_ids:
                raw_label = self.id2label.get(int(label_id), "")
                garment_name = label_to_garment.get(raw_label)
                if garment_name is None:
                    continue

                mask = seg_map == label_id
                area = int(mask.sum())
                if area < int(config.SEG_MIN_AREA_FRAC * seg_map.size):
                    continue

                # Find centroid for label placement
                ys, xs = np.where(mask)
                cy, cx = int(ys.mean()), int(xs.mean())

                color_tuple = tuple(color_map[label_id].tolist())
                # Draw drop shadow for legibility
                draw.text((cx + 1, cy + 1), garment_name, fill=(0, 0, 0), font=font)
                draw.text((cx, cy), garment_name, fill=color_tuple, font=font)

        return blended

    def get_all_segments_summary(self, image_pil: Image.Image) -> list[dict]:
        """Return a list of segment summary dicts for UI display.

        Each dict has: garment, area_frac (%), bbox.
        Useful for displaying confidence bars in the Streamlit app.
        """
        segments = self.segment_garments_with_masks(image_pil)
        return [
            {
                "garment": s.garment,
                "area_pct": round(s.area_frac * 100, 1),
                "bbox": s.bbox,
            }
            for s in segments
        ]


def get_default_segmenter() -> GarmentSegmenter:
    """Return a lazily initialised module-level segmenter singleton."""
    global _default_segmenter
    if _default_segmenter is None:
        print(f"Loading SegFormer segmenter ({config.SEGFORMER_MODEL_NAME})…")
        _default_segmenter = GarmentSegmenter()
    return _default_segmenter
