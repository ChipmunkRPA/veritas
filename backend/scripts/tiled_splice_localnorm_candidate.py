"""Tile-local contrast normalization after whole-image recompression measurements.

This separate development hypothesis changes feature normalization, not the
checkpoint, scored JPEG grid, source coordinates, or customer configuration.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

from app.detectors.image import splice
from scripts.tiled_splice_candidate import Tile, boxes


def local_features(residuals: list[Image.Image], box: tuple[int, int, int, int]) -> torch.Tensor:
    channels = []
    for residual in residuals:
        selected = residual.crop(box)
        peak = max(maximum for _, maximum in selected.getextrema())
        normalized = ImageEnhance.Brightness(selected).enhance(255. / max(1, peak)).convert("L")
        channels.append(np.asarray(normalized.resize((splice.INPUT_SIZE, splice.INPUT_SIZE),
                                                    Image.Resampling.BILINEAR), dtype=np.uint8))
    values = torch.from_numpy(np.stack(channels).astype(np.float32) / 255.)
    return ((values - values.new_tensor(splice.MEAN)[:, None, None]) /
            (values.new_tensor(splice.STD)[:, None, None] + 1e-8))


@torch.inference_mode()
def predict(model, image: Image.Image) -> list[Tile]:
    regions = boxes(*image.size)
    if not regions:
        return []
    rgb = image.convert("RGB")
    residuals = []
    for quality in splice.QUALITIES:
        with io.BytesIO() as buffer:
            rgb.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                residuals.append(ImageChops.difference(rgb, encoded.convert("RGB")))
    values = torch.stack([local_features(residuals, box) for box in regions])
    maps = model(values).sigmoid()[:, 0].numpy()
    if maps.shape != (len(regions), splice.INPUT_SIZE, splice.INPUT_SIZE) or not np.isfinite(maps).all():
        raise ValueError("Invalid locally normalized tile response geometry")
    return [Tile(box, value.copy()) for box, value in zip(regions, maps, strict=True)]
