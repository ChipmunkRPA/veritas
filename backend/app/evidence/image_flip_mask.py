"""Recompute recorded flip overlap from the two stored source-grid selections."""

from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np
from PIL import Image


def png_grid(body: bytes, size: tuple[int, int]) -> np.ndarray:
    with Image.open(io.BytesIO(body)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != size:
            raise ValueError("Image repeat-pass proof needs a grayscale PNG on the source grid")
        return np.array(image)


def verify_overlap(values: np.ndarray, reliable: np.ndarray, point: float, review: dict, body: bytes) -> None:
    height, width = values.shape
    if reliable.shape != values.shape or hashlib.sha256(body).hexdigest() != review["aligned_mask_png_sha256"]:
        raise ValueError("Image repeat-pass proof differs from its recorded grid or PNG digest")
    aligned = png_grid(body, (width, height))
    if not np.all((aligned == 0) | (aligned == 255)):
        raise ValueError("Image repeat-pass proof is not a binary mask")
    support = aligned == 255
    if hashlib.sha256(support.tobytes()).hexdigest() != review["aligned_mask_sha256"]:
        raise ValueError("Image repeat-pass mask pixels differ from their commitment")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(((values >= point) & reliable).astype(np.uint8), connectivity=8)
    for candidate in review["candidates"]:
        matches = [index for index, stat in enumerate(stats[1:], 1)
                   if stat.tolist() == [*candidate["bounds"], candidate["source_pixels"]]]
        if len(matches) != 1 or int(np.count_nonzero(support & (labels == matches[0]))) != candidate["shared_pixels"]:
            raise ValueError("The image repeat-pass overlap does not match the recorded source pixels")
