"""Classical forensics on native pixels for the v2 adapters (I-1, D13; IF §0.1, §4.1).

v0.3 measured ELA and noise on a ≤ 1024 px resample of the ≤ 4096 px analysis copy, which erases the JPEG grid and the
sensor noise, and the spectrum on a 512 × 512 resample. `forensics.py` keeps that code for the v1 pipeline and the B-5
baseline; the measurements below reuse its metrics and score mappings.

- **Inputs.** `native_pixels()` (after EXIF orientation) and the orientation tag from the original bytes. Nothing here
  reads the analysis copy (the D13 lint in `tests/test_input_discipline.py`).
- **ELA.** The pixels go back to the file's stored orientation, so the quality-90 re-save uses the file's 8 × 8 grid, and
  the error map is oriented like the pixels afterwards. The re-save runs in horizontal bands of `band_rows` with a 16-row
  margin on the 16 px MCU grid: every kept row equals the whole-image computation (tested).
- **Noise.** 3 px median residual and 16 px local variance per band, with the same margin (the filters reach 9 rows).
- **Spectrum.** Mean |FFT| over up to 3 × 3 non-overlapping native tiles of 512 px centred on the image (one tile of the
  shorter side below 512 px), Hann window, no resampling. Images below `MIN_SPECTRUM_PX` are not measured.
- **Scores** stay the v0.3 hand-set logistic mappings (`forensics.ela_score`, `noise_score`, `spectrum_score`): uncalibrated,
  capped to [0.40, 0.75]. Their rates on vb reals are in `docs/v2/qa/i1_forensic_fpr.md`.
- **Display maps** are area-resampled over the complete native image so the long side is ≤ 4096 px. Their coordinate
  space is the full native extent, including incomplete edge blocks. Only the display is reduced. At ≤ 1024 px with
  orientation 1, ELA and noise reproduce v0.3 exactly.
"""
from __future__ import annotations

import io
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from app.detectors.image import forensics
from app.engine.media import MAX_SIDE

BAND_ROWS = 1024  # a multiple of the 16 px MCU
MARGIN = 16
BLOCK = 16
TILE = 512
MAX_TILES = 3
MIN_SPECTRUM_PX = 64
ORIENTATION_TAG = 0x0112
DISPLAY_MAPPING_VERSION = "full-extent-area/1"

_METHODS = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1], 4: lambda a: a[::-1], 5: lambda a: a.swapaxes(0, 1),
            6: lambda a: np.rot90(a, -1), 7: lambda a: a.swapaxes(0, 1)[::-1, ::-1], 8: lambda a: np.rot90(a, 1)}


@dataclass
class NativeMap:
    metrics: dict[str, Any]
    display: np.ndarray
    coordinate_space: dict[str, Any]


def exif_orientation(data: bytes | memoryview) -> int:
    """The EXIF orientation `StoredMedia` applies (1–8); 1 when absent, invalid or unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            value = int(im.getexif().get(ORIENTATION_TAG, 1) or 1)
    except Exception:  # noqa: BLE001 - an unreadable header leaves the pixels as decoded
        return 1
    return value if value in _METHODS else 1


def orient(a: np.ndarray, orientation: int) -> np.ndarray:
    """The array as `ImageOps.exif_transpose` presents an image stored with this orientation."""
    method = _METHODS.get(orientation)
    return a if method is None else np.ascontiguousarray(method(a))


def unorient(a: np.ndarray, orientation: int) -> np.ndarray:
    """Inverse of `orient`: back to the stored orientation."""
    return orient(a, {6: 8, 8: 6}.get(orientation, orientation))


def native_box(box: tuple[int, int, int, int], shape: tuple[int, ...], max_side: int = MAX_SIDE) -> tuple[int, int, int, int]:
    """A face box from the faces stage (analysis-copy coordinates) in native pixels, clipped at the top-left edge."""
    height, width = shape[:2]
    scale = max(height, width) / max_side if max(height, width) > max_side else 1.0
    x, y, w, h = (int(round(v * scale)) for v in box)
    return max(0, x), max(0, y), w + min(0, x), h + min(0, y)


def _bands(height: int, band_rows: int) -> Iterator[tuple[int, int, int, int]]:
    step = max(BLOCK, (band_rows // BLOCK) * BLOCK)
    for core_lo in range(0, height, step):
        core_hi = min(height, core_lo + step)
        yield max(0, core_lo - MARGIN), min(height, core_hi + MARGIN), core_lo, core_hi


def _display(values: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = values.shape
    factor = max(1, math.ceil(max(height, width) / max_side))
    if factor > 1:
        # Keep the last row/column: trimming to a block multiple silently loses
        # measured edge pixels and makes full-source overlays geometrically wrong.
        values = cv2.resize(values, (math.ceil(width / factor), math.ceil(height / factor)), interpolation=cv2.INTER_AREA)
    return values, {"width": width, "height": height, "scale_to_original": 1.0}


def _face_ratio(values: np.ndarray, face_box: tuple[int, int, int, int] | None) -> float | None:
    """Face mean over surrounding-ring mean, as v0.3 computes it."""
    if face_box is None:
        return None
    x, y, w, h = face_box
    face = values[y:y + h, x:x + w]
    ring = forensics._ring_mask(values.shape, x, y, w, h)
    if face.size == 0 or ring.sum() == 0:
        return None
    return round(float(face.mean() / (values[ring].mean() + 1e-6)), 3)


def ela_error(stored: np.ndarray, *, quality: int = 90, band_rows: int = BAND_ROWS) -> np.ndarray:
    """Per-pixel mean absolute error of a quality re-save, in the stored orientation (float32, native size)."""
    height, width = stored.shape[:2]
    err = np.empty((height, width), dtype=np.float32)
    for lo, hi, core_lo, core_hi in _bands(height, band_rows):
        band = np.ascontiguousarray(stored[lo:hi])
        buf = io.BytesIO()
        Image.fromarray(band).save(buf, "JPEG", quality=quality)
        with Image.open(io.BytesIO(buf.getvalue())) as im:
            rec = np.asarray(im.convert("RGB"), dtype=np.int16)
        diff = np.abs(band.astype(np.int16) - rec).astype(np.float32)
        err[core_lo:core_hi] = diff.mean(axis=2)[core_lo - lo:core_hi - lo]
    return err


def ela(pixels: np.ndarray, *, orientation: int = 1, face_box: tuple[int, int, int, int] | None = None, quality: int = 90,
        band_rows: int = BAND_ROWS) -> NativeMap:
    """ELA at native resolution on the file's 8 × 8 grid; metrics as `forensics.ela_map`."""
    err = orient(ela_error(unorient(pixels, orientation), quality=quality, band_rows=band_rows), orientation)
    hh, ww = (err.shape[0] // BLOCK) * BLOCK, (err.shape[1] // BLOCK) * BLOCK
    metrics: dict[str, Any] = {"ela_mean": round(float(err.mean()), 3)}
    if hh and ww:
        blocks = err[:hh, :ww].reshape(hh // BLOCK, BLOCK, ww // BLOCK, BLOCK).mean(axis=(1, 3))
        metrics["ela_block_cv"] = round(float(blocks.std() / (blocks.mean() + 1e-6)), 3)
    else:  # below one block: no spatial variation to measure
        metrics["ela_block_cv"] = 0.0
    ratio = _face_ratio(err, face_box)
    if ratio is not None:
        metrics["ela_face_bg_ratio"] = ratio
    return NativeMap(metrics, *_display(err))


def noise(pixels: np.ndarray, *, face_box: tuple[int, int, int, int] | None = None, band_rows: int = BAND_ROWS) -> NativeMap:
    """Median-filter residual and 16 px local standard deviation at native resolution; metrics as `forensics.noise_map`."""
    height, width = pixels.shape[:2]
    lstd = np.empty((height, width), dtype=np.float32)
    n, total, squares, flat = 0, 0.0, 0.0, 0
    for lo, hi, core_lo, core_hi in _bands(height, band_rows):
        g = cv2.cvtColor(np.ascontiguousarray(pixels[lo:hi]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
        var = ndimage.uniform_filter(res**2, size=BLOCK) - ndimage.uniform_filter(res, size=BLOCK) ** 2
        core = slice(core_lo - lo, core_hi - lo)
        lstd[core_lo:core_hi] = np.sqrt(np.maximum(var[core], 0))
        kept = res[core].astype(np.float64)
        n, total, squares = n + kept.size, total + float(kept.sum()), squares + float((kept * kept).sum())
        flat += int((np.abs(res[core]) < 0.5).sum())
    mean = total / n
    metrics: dict[str, Any] = {"noise_std": round(math.sqrt(max(squares / n - mean * mean, 0.0)), 3),
                               "noise_local_cv": round(float(lstd.std() / (lstd.mean() + 1e-6)), 3),
                               "noise_flat_fraction": round(flat / n, 3)}
    ratio = _face_ratio(lstd, face_box)
    if ratio is not None:
        metrics["noise_face_bg_ratio"] = ratio
    return NativeMap(metrics, *_display(lstd))


def spectrum(pixels: np.ndarray, *, figure: bool = True) -> tuple[dict[str, Any], np.ndarray | None]:
    """Spectrum metrics (`forensics.spectrum_from_magnitude`) over native tiles; raises ValueError below MIN_SPECTRUM_PX."""
    height, width = pixels.shape[:2]
    side = min(TILE, height, width)
    if side < MIN_SPECTRUM_PX:
        raise ValueError(f"image smaller than {MIN_SPECTRUM_PX} px")
    nx, ny = min(MAX_TILES, width // side), min(MAX_TILES, height // side)
    x0, y0 = (width - nx * side) // 2, (height - ny * side) // 2
    window = np.outer(np.hanning(side), np.hanning(side))
    total = np.zeros((side, side), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            tile = pixels[y0 + j * side:y0 + (j + 1) * side, x0 + i * side:x0 + (i + 1) * side]
            g = cv2.cvtColor(np.ascontiguousarray(tile), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            total += np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * window)))
    metrics, art = forensics.spectrum_from_magnitude(np.log1p(total / (nx * ny)), figure=figure)
    return {**metrics, "spectrum_tiles": nx * ny, "spectrum_tile_px": side}, art
