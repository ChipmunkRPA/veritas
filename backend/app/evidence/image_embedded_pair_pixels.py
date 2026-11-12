"""Render and independently replay source pixels for an exact image-area comparison."""
from __future__ import annotations

import hashlib
import io
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from app.evidence.image_copy_similarity_binding import box_bounds
from app.evidence.image_embedded_pair_binding import ASSETS, ROLES, SIZES


def source_image(data: bytes) -> Image.Image | None:
    with Image.open(io.BytesIO(data)) as raw:
        if (raw.format != "PNG" or raw.mode != "RGB" or raw.size not in SIZES or raw.n_frames != 1
                or int(raw.getexif().get(274, 1) or 1) != 1):
            return None
        raw.load()
        return raw.copy()


def overlay(crop: Image.Image, difference: np.ndarray) -> Image.Image:
    pixels = np.array(crop)
    original = pixels[difference].astype(np.uint16)
    orange = np.array([255, 135, 45], dtype=np.uint16)
    pixels[difference] = ((original + orange) // 2).astype(np.uint8)
    return Image.fromarray(pixels)


def plates(a: Image.Image, b: Image.Image, difference: np.ndarray) -> dict[str, Image.Image]:
    return {"area_a": a, "area_b": b, "overlay_a": overlay(a, difference), "overlay_b": overlay(b, difference),
            "difference_mask": Image.fromarray(difference.astype(np.uint8) * 255)}


def encode(image: Image.Image) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(image)).save(output, format="PNG")
    return output.getvalue()


def verify(source: Image.Image, pair: dict, read: Callable[[str], bytes]) -> None:
    """Recompute native mask, texture witnesses and every saved plate without detector inference."""
    bounds = {role: box_bounds(pair["regions"][role], *source.size) for role in ROLES}
    a, b = (source.crop(bounds[role]) for role in ROLES)
    original_a, original_b = np.asarray(a), np.asarray(b)
    difference = np.any(original_a != original_b, axis=2)
    y, x = np.nonzero(difference)
    if len(x) != pair["different_pixels"] or not len(x):
        raise ValueError("The measured pixels differ from the two original areas")
    _, _, components, _ = cv2.connectedComponentsWithStats(difference.astype(np.uint8), connectivity=8)
    if int(components[1:, 4].max()) < 48:
        raise ValueError("The native difference has insufficient connected support")
    for role in ROLES:
        left, top, _, _ = bounds[role]
        extent = left + int(x.min()), top + int(y.min()), left + int(x.max()) + 1, top + int(y.max()) + 1
        if extent != box_bounds(pair["difference_regions"][role], *source.size):
            raise ValueError("The recorded difference bounds differ from the original pixels")
    channels = original_a.astype(np.uint32)
    packed = (channels[:, :, 0] << 16) | (channels[:, :, 1] << 8) | channels[:, :, 2]
    for xx, yy in pair["witnesses"]:
        x0, y0 = xx - bounds["area_a"][0], yy - bounds["area_a"][1]
        witness = original_a[y0:y0 + 16, x0:x0 + 16]
        if (np.unique(packed[y0:y0 + 16, x0:x0 + 16]).size < 48
                or not np.array_equal(witness, original_b[y0:y0 + 16, x0:x0 + 16])):
            raise ValueError("An exact texture witness differs from the original pixels")
    expected = plates(a, b, difference)
    for role in ASSETS:
        digest = pair["artifacts"][role]["sha256"]
        data = read(digest)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("The exact comparison image differs from its recorded hash")
        with Image.open(io.BytesIO(data)) as saved:
            image = expected[role]
            if saved.format != "PNG" or saved.mode != image.mode or saved.size != image.size or saved.tobytes() != image.tobytes():
                raise ValueError(f"The {role} image differs from the original pixels")
