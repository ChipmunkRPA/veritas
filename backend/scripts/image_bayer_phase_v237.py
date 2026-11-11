"""Source-coordinate Bayer-phase consistency for isolated image research."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

TILE_SIZE = 64
MIN_TILES_PER_FOLD = 8
MIN_SAMPLES_PER_PHASE = 64
MIN_PHASE_STRENGTH = 0.08
MIN_PHASE_COSINE = 0.80
MIN_PHASE_SUPPORT = 2 / 3

Status = Literal["scored", "insufficient_tiles", "phase_not_supported"]

_Y, _X = np.indices((TILE_SIZE - 2, TILE_SIZE - 2)) + 1
_PHASE = (_Y % 2) * 2 + _X % 2
# Avoid samples whose second difference crosses a JPEG block boundary.
_INTERIOR = (_Y % 8 != 0) & (_Y % 8 != 7) & (_X % 8 != 0) & (_X % 8 != 7)


@dataclass(frozen=True)
class PhaseResult:
    status: Status
    source_size: tuple[int, int]
    available_tiles: tuple[int, int]
    held_out_support: tuple[float, float] | None = None
    scores: np.ndarray | None = None


def phase_profile(green: np.ndarray) -> np.ndarray | None:
    """Measure horizontal and vertical residual magnitude by Bayer phase."""
    if green.shape != (TILE_SIZE, TILE_SIZE) or green.dtype != np.uint8:
        raise ValueError("One complete 64×64 uint8 green tile required")

    values = green.astype(np.int16)
    center = values[1:-1, 1:-1]
    left, right = values[1:-1, :-2], values[1:-1, 2:]
    top, bottom = values[:-2, 1:-1], values[2:, 1:-1]
    eligible = (
        _INTERIOR
        & (np.minimum.reduce((center, left, right, top, bottom)) >= 16)
        & (np.maximum.reduce((center, left, right, top, bottom)) <= 239)
        & (np.abs(left - right) <= 24)
        & (np.abs(top - bottom) <= 24)
    )
    phases = _PHASE[eligible]
    counts = np.bincount(phases, minlength=4)
    if np.any(counts < MIN_SAMPLES_PER_PHASE):
        return None

    errors = (np.abs(2 * center - left - right), np.abs(2 * center - top - bottom))
    magnitudes = np.stack(
        [np.bincount(phases, weights=error[eligible], minlength=4) / counts for error in errors]
    )
    log_magnitudes = np.log1p(magnitudes)
    return (log_magnitudes - log_magnitudes.mean(axis=1, keepdims=True)).ravel()


def _support(profiles: np.ndarray, reference: np.ndarray) -> float:
    lengths = np.linalg.norm(profiles, axis=1)
    cosines = (profiles @ reference) / np.maximum(lengths * np.linalg.norm(reference), 1e-12)
    strength = lengths / np.sqrt(profiles.shape[1])
    return float(np.mean((strength >= MIN_PHASE_STRENGTH) & (cosines >= MIN_PHASE_COSINE)))


def analyze_bayer_phase(pixels: np.ndarray) -> PhaseResult:
    """Score native full tiles against independent checkerboard-fold references.

    Cell (r, c) covers source pixels [64r, 64(r+1)) by [64c, 64(c+1)).
    Partial tiles are unmeasured. NaN marks an unavailable complete tile in a
    scored image. Scores measure log-residual mismatch, without an edit verdict.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Native uint8 RGB pixels required")

    height, width, _ = pixels.shape
    rows, columns = height // TILE_SIZE, width // TILE_SIZE
    signatures = np.full((rows, columns, 8), np.nan)
    for row, column in np.ndindex(rows, columns):
        y, x = row * TILE_SIZE, column * TILE_SIZE
        profile = phase_profile(pixels[y : y + TILE_SIZE, x : x + TILE_SIZE, 1])
        if profile is not None:
            signatures[row, column] = profile

    row, column = np.indices((rows, columns))
    valid = np.isfinite(signatures).all(axis=2)
    first = (row + column) % 2 == 0
    fold_a, fold_b = valid & first, valid & ~first
    a, b = signatures[fold_a], signatures[fold_b]
    size, counts = (width, height), (len(a), len(b))
    if min(counts) < MIN_TILES_PER_FOLD:
        return PhaseResult("insufficient_tiles", size, counts)

    ref_a, ref_b = np.median(a, axis=0), np.median(b, axis=0)
    strengths = np.sqrt(np.mean(np.stack((ref_a, ref_b)) ** 2, axis=1))
    support = (_support(a, ref_b), _support(b, ref_a))
    if strengths.min() < MIN_PHASE_STRENGTH or min(support) < MIN_PHASE_SUPPORT:
        return PhaseResult("phase_not_supported", size, counts, support)

    scores = np.full((rows, columns), np.nan)
    scores[fold_a] = np.sqrt(np.mean((a - ref_b) ** 2, axis=1))
    scores[fold_b] = np.sqrt(np.mean((b - ref_a) ** 2, axis=1))
    return PhaseResult("scored", size, counts, support, scores)
