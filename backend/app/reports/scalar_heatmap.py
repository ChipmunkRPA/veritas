"""Verified scalar heat overlays for new reports (ARCHITECTURE D5; UX §8.3).

The caller binds an original, whole-image map to the frozen report. This module verifies bytes and geometry; it never resolves
live artifact IDs, reverses an RGB preview, or invents crop/frame transforms. Scalar values remain the recorded 8-bit samples.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import re
from functools import cache

from PIL import Image, ImageOps

from app.engine.media import MAX_PIXELS, MAX_SIDE
from app.storage.objects import InvalidObjectId, ObjectNotFound, ObjectStore, default_store

# frontend/src/lib/colormap.ts: the browser and vector legend use these eleven stops, not Matplotlib's inferno samples.
INFERNO_STOPS = ("000004", "160B39", "420A68", "6A176E", "932667", "BC3754", "DD513A", "F37819", "FCA50A", "F6D746", "FCFFA4")
SHOW_THRESHOLD = 0.5
MAX_OPACITY = 0.7
MAX_SCALAR_PIXELS = MAX_SIDE * MAX_SIDE  # native_forensics display cap; localisers and attribution maps are smaller
_SHA256 = re.compile(r"[a-f0-9]{64}")


def _size(value: object) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(type(v) is int and v > 0 for v in value)


def _range(value: object) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    if any(type(v) not in (int, float) for v in value):
        return False
    try:
        return all(math.isfinite(v) for v in value) and value[1] > value[0] and math.isfinite(value[1] - value[0])
    except OverflowError:
        return False


def _image(sha256: str, store: ObjectStore, *, scalar: bool, expected_grid: tuple[int, int] | None = None) -> Image.Image | None:
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        return None
    try:
        data = store.read_bytes(sha256)
    except (ObjectNotFound, InvalidObjectId):
        return None
    if hashlib.sha256(data).hexdigest() != sha256:
        return None
    # PIL decodes 2-/4-bit grayscale into L as well. Inspect IHDR so only the recorded 8-bit grayscale contract passes.
    if scalar and (len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR"
                   or data[24:26] != bytes((8, 0))):
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > (MAX_SCALAR_PIXELS if scalar else MAX_PIXELS):
                return None
            if expected_grid is not None and image.size != tuple(expected_grid):
                return None
            if getattr(image, "n_frames", 1) != 1:
                return None
            if scalar:
                if (image.format != "PNG" or image.mode != "L" or "transparency" in image.info
                        or image.getexif().get(274, 1) != 1):
                    return None
                image.load()
                if "transparency" in image.info:
                    return None
                return image.copy()  # byte values: no colour conversion, EXIF transpose, rescaling or smoothing
            # Matches StoredMedia._decode: orientation is applied before the recorded original dimensions are checked.
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


@cache
def _lut() -> tuple[tuple[int, int, int], ...]:
    rgb = [tuple(int(stop[i:i + 2], 16) for i in (0, 2, 4)) for stop in INFERNO_STOPS]
    result = []
    for value in range(256):
        position = value * (len(rgb) - 1) / 255
        index = min(len(rgb) - 2, math.floor(position))
        fraction = position - index
        # JS Math.round for these nonnegative channels, not Python's ties-to-even round.
        result.append(tuple(math.floor(rgb[index][c] + (rgb[index + 1][c] - rgb[index][c]) * fraction + 0.5) for c in range(3)))
    return tuple(result)


@cache
def _alpha_bytes(show: float) -> tuple[int, ...]:
    # heatPixels renders at opacity 1; the viewer applies .7 separately with canvas globalAlpha.
    return tuple(0 if value / 255 < show else
                 math.floor((0.35 + 0.65 * (value / 255 - show) / (1 - show)) * 255 + 0.5)
                 for value in range(256))


def _scalar_images(payload_sha256: str, grid: tuple[int, int], value_range: tuple[float, float],
                   reliability_sha256: str | None, require_reliability: bool,
                   store: ObjectStore) -> tuple[Image.Image, Image.Image | None] | None:
    if (not _size(grid) or not _range(value_range) or type(require_reliability) is not bool
            or (require_reliability and reliability_sha256 is None)):
        return None
    payload = _image(payload_sha256, store, scalar=True, expected_grid=grid)
    if payload is None or payload.size != tuple(grid):
        return None
    mask = _image(reliability_sha256, store, scalar=True, expected_grid=grid) if reliability_sha256 is not None else None
    if reliability_sha256 is not None and (mask is None or mask.size != payload.size):
        return None
    return payload, mask


def _composite(payload: Image.Image, mask: Image.Image | None, base: Image.Image, *, show: float, opacity: float) -> str:
    import numpy as np

    values = np.asarray(payload.resize(base.size, Image.Resampling.NEAREST))
    alpha = np.asarray(_alpha_bytes(show), dtype=np.float64)[values]
    if mask is not None:
        reliability = np.asarray(mask.resize(base.size, Image.Resampling.NEAREST), dtype=np.float64)
        alpha = np.floor(alpha * reliability / 255 + 0.5)
    alpha = (alpha / 255 * opacity)[..., None]
    heat = np.asarray(_lut(), dtype=np.float64)[values]
    blended = np.floor(np.asarray(base, dtype=np.float64) * (1 - alpha) + heat * alpha + 0.5).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(blended).save(out, "PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def scalar_overlay_plate(payload_sha256: str, source_sha256: str, box: tuple[int, int], *, grid: tuple[int, int],
                         value_range: tuple[float, float], source_size: tuple[int, int],
                         reliability_sha256: str | None = None, require_reliability: bool = False,
                         store: ObjectStore | None = None) -> str | None:
    """A verified whole-image scalar overlay, or None when any required evidence is unavailable or inconsistent.

    ``grid`` is the PNG width/height; ``source_size`` is the frozen original's EXIF-oriented width/height. The caller must verify
    media_ref/coordinate-space provenance before calling. Aspect-ratio checking allows at most one grid cell of integer
    rounding, as whole-image grid_shape and block maps can round their edges; it does not authorize crop/frame placement.
    ``value_range`` must be finite and increasing. It describes scalar units; like the viewer, the display threshold uses
    normalized stored bytes, not a calibrated-probability claim. An explicitly supplied mask always has to verify, even if
    ``require_reliability`` is False. An absent optional mask is rendered without a reliability adjustment.

    Source reduction uses Lanczos; map and mask use nearest-neighbor. LUT, full-opacity alpha rounding, and reliability byte
    rounding match heatPixels. Only then is the .7 lens opacity applied. Final RGB uses half-up integer rounding. Actual Canvas
    bitmap/compositing arithmetic is measured separately; exact final pixel parity or a universal error bound is not assumed.
    Output has no inherited image metadata. No fake high-precision map or fallback preview is produced.
    """
    if not _size(source_size) or not _size(box):
        return None
    store = store or default_store()
    decoded = _scalar_images(payload_sha256, grid, value_range, reliability_sha256, require_reliability, store)
    source = _image(source_sha256, store, scalar=False)
    if decoded is None or source is None or source.size != tuple(source_size):
        return None
    if abs(grid[0] * source_size[1] - grid[1] * source_size[0]) > max(source_size):
        return None
    source.thumbnail(tuple(box), Image.Resampling.LANCZOS)
    return _composite(*decoded, source, show=SHOW_THRESHOLD, opacity=MAX_OPACITY)


def scalar_map_plate(payload_sha256: str, box: tuple[int, int], *, grid: tuple[int, int], value_range: tuple[float, float],
                     reliability_sha256: str | None = None, require_reliability: bool = False,
                     store: ObjectStore | None = None) -> str | None:
    """The recorded scalar map on black: browser map-only show=0, opacity=1, with the same verified mask/rounding contract.

    Dimensions come from the map, so this plate makes no claim to place it on an original or frame. Only nearest-neighbor
    reduction is used; a low-resolution map is not represented as a higher-resolution measurement.
    """
    if not _size(box):
        return None
    decoded = _scalar_images(payload_sha256, grid, value_range, reliability_sha256, require_reliability, store or default_store())
    if decoded is None:
        return None
    payload, mask = decoded
    bounded = payload.copy()
    bounded.thumbnail(tuple(box), Image.Resampling.NEAREST)
    return _composite(payload, mask, Image.new("RGB", bounded.size, (0, 0, 0)), show=0.0, opacity=1.0)
