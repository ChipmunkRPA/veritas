"""Offline small-edit experiment using the existing commercial splice checkpoint.

Recompression is measured on the whole original image before cropping feature
planes. Tile boundaries therefore do not introduce fresh JPEG-grid evidence.
No detector registry or customer setting enables this experimental path.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

from app.detectors.image import splice

TILE_SHARE = .6
INNER_MARGIN = .06
MIN_SIDE = 320


@dataclass(frozen=True)
class Tile:
    box: tuple[int, int, int, int]  # Original pixel coordinates, end exclusive.
    response: np.ndarray

    @property
    def active_fraction(self) -> float:
        return float((self.response > splice.PIXEL_THRESHOLD).mean())


def boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if min(width, height) < MIN_SIDE:
        return []
    tw, th = math.ceil(width * TILE_SHARE), math.ceil(height * TILE_SHARE)
    return [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]


def feature_planes(image: Image.Image) -> tuple[Image.Image, ...]:
    rgb = image.convert("RGB")
    channels = []
    for quality in splice.QUALITIES:
        with io.BytesIO() as buffer:
            rgb.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                residual = ImageChops.difference(rgb, encoded.convert("RGB"))
        peak = max(maximum for _, maximum in residual.getextrema())
        channels.append(ImageEnhance.Brightness(residual).enhance(255. / max(1, peak)).convert("L"))
    return tuple(channels)


def features(planes: tuple[Image.Image, ...], box: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    channels = []
    for plane in planes:
        selected = plane.crop(box) if box is not None else plane
        channels.append(np.asarray(selected.resize((splice.INPUT_SIZE, splice.INPUT_SIZE), Image.Resampling.BILINEAR),
                                   dtype=np.uint8))
    value = torch.from_numpy(np.stack(channels).astype(np.float32) / 255.)
    return ((value - value.new_tensor(splice.MEAN)[:, None, None]) /
            (value.new_tensor(splice.STD)[:, None, None] + 1e-8))


@torch.inference_mode()
def predict(model, image: Image.Image) -> list[Tile]:
    regions = boxes(*image.size)
    if not regions:
        return []
    planes = feature_planes(image)
    batch = torch.stack([features(planes, box) for box in regions])
    maps = model(batch).sigmoid()[:, 0].numpy()
    if maps.shape != (len(regions), splice.INPUT_SIZE, splice.INPUT_SIZE) or not np.isfinite(maps).all():
        raise ValueError("Invalid tile response geometry")
    return [Tile(box, response.copy()) for box, response in zip(regions, maps, strict=True)]


def stitch(tiles: list[Tile], width: int, height: int, minimum_active_fraction: float) -> np.ndarray:
    """Reproject eligible tiles; suppress only their internal six-percent border."""
    if not math.isfinite(minimum_active_fraction) or not 0 <= minimum_active_fraction <= 1:
        raise ValueError("Invalid tile admission rule")
    selected = np.zeros((height, width), dtype=np.float32)
    for tile in tiles:
        x0, y0, x1, y1 = tile.box
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError("Tile lies outside the original image")
        if (tile.response.ndim != 2 or not np.isfinite(tile.response).all()
                or tile.response.min() < 0 or tile.response.max() > 1):
            raise ValueError("Tile contains invalid responses")
        if tile.active_fraction < minimum_active_fraction:
            continue
        projected = cv2.resize(tile.response, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        margin_x, margin_y = math.ceil((x1-x0)*INNER_MARGIN), math.ceil((y1-y0)*INNER_MARGIN)
        if x0:
            projected[:, :margin_x] = 0
        if x1 < width:
            projected[:, -margin_x:] = 0
        if y0:
            projected[:margin_y, :] = 0
        if y1 < height:
            projected[-margin_y:, :] = 0
        selected[y0:y1, x0:x1] = np.maximum(selected[y0:y1, x0:x1], projected)
    return selected
