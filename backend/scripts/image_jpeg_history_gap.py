# SPDX-License-Identifier: Apache-2.0
"""Measure whether a compact area lacks the compression pattern of its surroundings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.detectors.image import jpeg_dq
from app.provenance.metadata_rules import STD_LUMA, ZIGZAG, ijg_table

CELL = 16
SIZES = (2, 3, 4, 5, 6, 8)
QUALITIES = tuple(range(40, 80, 5))
FREQUENCIES = ZIGZAG[1:29]
MARGIN = 2
MIN_SAMPLES = 12
MIN_BACKGROUND = .32
MIN_GLOBAL_LEAD = .035
MIN_RING = .36


@dataclass(frozen=True)
class Gap:
    quality: int
    overall: float
    bounds: tuple[int, int, int, int] | None = None
    score: float = 0.
    inside: tuple[float, float] | None = None
    around: tuple[float, float] | None = None


def _directional_support(coefficients: jpeg_dq.Coefficients, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate excess compatibility separately for horizontal and vertical AC terms."""
    rows, cols = coefficients.height // CELL, coefficients.width // CELL
    freqs_y, freqs_x = np.divmod(FREQUENCIES, 8)
    current = coefficients.table[freqs_y, freqs_x]
    prior = ijg_table(STD_LUMA, quality)[FREQUENCIES]
    levels = coefficients.y[:2 * rows, :2 * cols, freqs_y, freqs_x]

    # A final quantization bin describes an interval of possible inputs. Allow
    # two units for pixel-domain roundoff. Account for the residues the current
    # quantization grid can represent before counting a compatible input.
    remainder = (levels * current) % prior
    distance = np.minimum(remainder, prior - remainder)
    stride = np.gcd(current, prior)
    allowance = current / 2. + MARGIN
    chance = np.minimum(1., (2 * np.floor(allowance / stride) + 1) / (prior // stride))
    eligible = (levels != 0) & (chance <= .6)
    evidence = np.where(eligible, (distance <= allowance) - chance, 0.)
    available = eligible * (1. - chance)

    observations, valid = [], []
    for direction in (freqs_x > freqs_y, freqs_y > freqs_x):
        per_block = evidence[..., direction].sum(axis=-1)
        per_block_weight = available[..., direction].sum(axis=-1)
        per_block_count = eligible[..., direction].sum(axis=-1)
        signal = per_block.reshape(rows, 2, cols, 2).sum(axis=(1, 3))
        weight = per_block_weight.reshape(rows, 2, cols, 2).sum(axis=(1, 3))
        count = per_block_count.reshape(rows, 2, cols, 2).sum(axis=(1, 3))
        observations.append(np.divide(signal, weight, out=np.zeros_like(signal), where=weight > 0))
        valid.append(count >= MIN_SAMPLES)
    return np.stack(observations), np.logical_and(*valid)


def _windows(integral: np.ndarray, width: int, height: int) -> np.ndarray:
    return integral[..., height:, width:] - integral[..., height:, :-width] - integral[..., :-height, width:] + integral[..., :-height, :-width]


def _compact_gap(quality: int, overall: float, field: np.ndarray, valid: np.ndarray) -> Gap:
    result = Gap(quality, overall)
    padded_field = np.pad(field * valid, ((0, 0), (1, 0), (1, 0)))
    padded_count = np.pad(valid, ((1, 0), (1, 0)))
    sums = padded_field.cumsum(axis=1).cumsum(axis=2)
    counts = padded_count.cumsum(axis=0).cumsum(axis=1)
    for height in SIZES:
        for width in SIZES:
            if max(width, height) > 3 * min(width, height):
                continue
            inner_count = _windows(counts, width, height)[1:-1, 1:-1]
            inner_sum = _windows(sums, width, height)[:, 1:-1, 1:-1]
            ring_count = _windows(counts, width + 2, height + 2) - inner_count
            ring_sum = _windows(sums, width + 2, height + 2) - inner_sum
            inside = inner_sum / np.maximum(inner_count, 1)
            around = ring_sum / np.maximum(ring_count, 1)
            eligible = ((4 * inner_count >= 3 * width * height)
                        & (4 * ring_count >= 3 * (2 * (width + height) + 4))
                        & (around.min(axis=0) >= MIN_RING))
            # Small proposals remain eligible, but more observed cells can
            # stabilize a candidate. Both frequency directions must agree.
            contrast = (around - inside).min(axis=0) * np.sqrt((width * height) / (width * height + 8.))
            ranked = np.where(eligible, contrast, 0.)
            y, x = np.unravel_index(ranked.argmax(), ranked.shape)
            score = float(ranked[y, x])
            if score > result.score:
                result = Gap(quality, overall, ((int(x) + 1) * CELL, (int(y) + 1) * CELL, width * CELL, height * CELL),
                             score, tuple(map(float, inside[:, y, x])), tuple(map(float, around[:, y, x])))
    return result


def measure(coefficients: jpeg_dq.Coefficients) -> Gap | None:
    """Require a widespread earlier-table pattern before searching its interior."""
    if not 256 <= min(coefficients.height, coefficients.width) <= max(coefficients.height, coefficients.width) <= 1024:
        return None
    candidates = []
    for quality in QUALITIES:
        field, valid = _directional_support(coefficients, quality)
        if valid.mean() >= .40:
            overall = float(np.median(field[:, valid], axis=1).min())
            candidates.append((overall, quality, field, valid))
    candidates.sort(reverse=True, key=lambda candidate: candidate[:2])
    if not candidates or candidates[0][0] < MIN_BACKGROUND:
        return None
    overall, quality, field, valid = candidates[0]
    # A quality five steps away can encode nearly the same quantization table.
    # Reject a broad family of unrelated settings with comparable support.
    if any(overall - score < MIN_GLOBAL_LEAD for score, other, _, _ in candidates if abs(other - quality) > 5):
        return None
    return _compact_gap(quality, overall, field, valid)
