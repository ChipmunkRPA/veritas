# SPDX-License-Identifier: Apache-2.0
"""Look for a local loss of a widespread earlier JPEG grid displaced by cropping."""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.provenance.metadata_rules import STD_LUMA, ijg_table

FREQUENCIES = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2))
FREQUENCY_Y, FREQUENCY_X = np.array(FREQUENCIES).T
HORIZONTAL = FREQUENCY_X > FREQUENCY_Y
PRIOR_QUALITIES = tuple(range(45, 81, 5))
SIZES = (2, 3, 4, 5, 6, 8, 10)
CELL = 16
MIN_BACKGROUND = .12
MIN_PHASE_LEAD = .035
MIN_CONTEXT = .14
MIN_COEFFICIENTS = 6
ARM_NAMES = ("periodicity", "with_edges", "region_scan", "scan_with_edges", "surrounded_scan")


@dataclass(frozen=True)
class Region:
    box: tuple[int, int, int, int]
    score: float
    inside: tuple[float, float]
    surrounding: tuple[float, float]
    cells: tuple[int, int]
    edge_contrast: tuple[float, float] | None


@dataclass(frozen=True)
class Measurement:
    phase: tuple[int, int]
    previous_quality: int
    background: float
    separated_from_next: float
    regions: dict[str, Region | None]


@lru_cache(maxsize=1)
def _bases():
    x = np.arange(8, dtype=np.float64) + .5
    dct = .5 * np.cos(np.pi * np.arange(8)[:, None] * x / 8)
    dct[0] /= np.sqrt(2)
    energy = []
    for phase in range(8):
        energy.append((dct[:, :8-phase] @ dct[:, phase:].T)**2 + (dct[:, 8-phase:] @ dct[:, :phase].T)**2)
    tables = np.array([ijg_table(STD_LUMA, quality)[8 * FREQUENCY_Y + FREQUENCY_X]
                       for quality in PRIOR_QUALITIES], dtype=np.float64)
    return dct, np.array(energy), tables


def read_luma(data):
    """Read the stored JPEG raster without applying its display orientation."""
    with Image.open(io.BytesIO(data)) as image:
        w, h = image.size
        if (image.format != "JPEG" or image.mode not in {"RGB", "L"} or image.bits != 8
                or min(w, h) < 320 or w * h > 1_048_576
                or (image.mode == "RGB" and tuple(layer[0] for layer in image.layer) != (1, 2, 3))):
            return None
        final = np.asarray(image.quantization[image.layer[0][3]], dtype=np.float64).reshape(8, 8)
        image.draft("L", image.size)
        return np.asarray(image.convert("L"), dtype=np.float32), final


def _phase_hypotheses(gray, final, *, check=None):
    basis, energies, periods = _bases()
    height, width = ((n - 7) // CELL * CELL for n in gray.shape)
    filters = [cv2.sepFilter2D(gray, cv2.CV_32F, basis[u], basis[v], anchor=(0, 0))
               for v, u in FREQUENCIES]
    # Independent final-coefficient rounding projected onto the shifted basis.
    variance = final**2 / 12
    for y in range(8):
        for x in range(8):
            if check:
                check()
            coeff = np.stack([f[y:y+height:8, x:x+width:8] for f in filters])
            sigma = np.sqrt(np.einsum("fi,ij,fj->f", energies[y, FREQUENCY_Y], variance, energies[x, FREQUENCY_X]) + .75**2)
            for quality, steps in zip(PRIOR_QUALITIES, periods, strict=True):
                active = steps >= 2.8 * sigma
                if min(np.count_nonzero(active & HORIZONTAL), np.count_nonzero(active & ~HORIZONTAL)) < 3:
                    continue
                units = coeff / steps.astype(np.float32)[:, None, None]
                use = (np.abs(units) >= .66) & active[:, None, None]
                periodic = np.where(use, np.cos(np.float32(2 * np.pi) * units), 0.)
                yield (x, y), quality, periodic, use


def _global_strength(periodic, use):
    h, w = periodic.shape[1:]
    quadrants = []
    for top, bottom in ((0, h // 2), (h // 2, h)):
        for left, right in ((0, w // 2), (w // 2, w)):
            counts = [int(use[mask, top:bottom, left:right].sum()) for mask in (HORIZONTAL, ~HORIZONTAL)]
            if min(counts) < 100:
                continue
            quadrants.append(min(periodic[mask, top:bottom, left:right].sum(dtype=np.float64) / count
                                 for mask, count in zip((HORIZONTAL, ~HORIZONTAL), counts, strict=True)))
    # One quadrant can contain the change or have comparatively little texture.
    return sorted(quadrants, reverse=True)[2] if len(quadrants) >= 3 else -1.


def _cell_totals(terms):
    rows, cols = terms.shape[1:]
    return np.stack([terms[band].sum(axis=0).reshape(rows // 2, 2, cols // 2, 2).sum(axis=(1, 3), dtype=np.float64)
                     for band in (HORIZONTAL, ~HORIZONTAL)])


def _grid_steps(gray, x, y, height, width):
    def steps(plane, offset, short, long, top):
        margins = np.abs(np.diff(plane, axis=1))
        values = np.clip(margins[:, 1:-1] - .5 * (margins[:, :-2] + margins[:, 2:]), -8., 8.)
        # First internal seam of each complete native cell, with all rows paired.
        selected = values[top:top+short, offset + 6:offset + long:16]
        return selected.reshape(short // CELL, CELL, long // CELL).mean(axis=1)

    if not x or not y:
        return None
    # A single cell includes 16 rows across one earlier block seam.
    return np.stack((steps(gray, x, height, width, y), steps(gray.T, y, width, height, x).T))


def _integral(values):
    padded = [(0, 0)] * (values.ndim - 2) + [(1, 0), (1, 0)]
    return np.pad(values, padded).cumsum(axis=-1).cumsum(axis=-2)


def _window(integral, h, w):
    return integral[..., h:, w:] - integral[..., :-h, w:] - integral[..., h:, :-w] + integral[..., :-h, :-w]


def _sides_supported(sums, counts, h, w, inside):
    def strips(total):
        horizontal = _window(total, 2, w)[..., :, 2:-2]
        vertical = _window(total, h, 2)[..., 2:-2, :]
        return (horizontal[..., :-(h+2), :], horizontal[..., h+2:, :],
                vertical[..., :, :-(w+2)], vertical[..., :, w+2:])

    sufficient = []
    for total, count, area in zip(strips(sums), strips(counts), (2*w, 2*w, 2*h, 2*h), strict=True):
        surface = total / np.maximum(1, count)
        sufficient.append((count >= .50 * area) & (surface.min(axis=0) >= MIN_CONTEXT)
                          & ((surface - inside).min(axis=0) > .035))
    return np.count_nonzero(np.stack(sufficient), axis=0) >= 3


def _regions(field, valid, edges, phase, *, check=None):
    phase_x, phase_y = phase
    sums, counts = _integral(field * valid), _integral(valid.astype(np.int32))
    edge_sums = _integral(edges * valid) if edges is not None else None
    chosen = dict.fromkeys(ARM_NAMES)
    for h in SIZES:
        for w in SIZES:
            if max(h, w) > 2.5 * min(h, w) or h * w > 100 or h + 4 >= counts.shape[0] or w + 4 >= counts.shape[1]:
                continue
            if check:
                check()
            ni = _window(counts, h, w)[2:-2, 2:-2]
            nr = _window(counts, h+4, w+4) - ni
            inner = _window(sums, h, w)[:, 2:-2, 2:-2]
            around = (_window(sums, h+4, w+4) - inner) / np.maximum(nr, 1)
            inside = inner / np.maximum(ni, 1)
            accepted = ((ni >= .67 * w * h) & (nr >= .60 * (4 * (h + w) + 16))
                        & (around.min(axis=0) >= MIN_CONTEXT))
            contrast = (around - inside).min(axis=0) * np.sqrt(h * w / (h * w + 8.))
            edge_delta = None
            if edge_sums is not None:
                ei = _window(edge_sums, h, w)[:, 2:-2, 2:-2]
                edge_delta = (_window(edge_sums, h+4, w+4) - ei) / np.maximum(nr, 1) - ei / np.maximum(ni, 1)
            scan = (around - inside).min(axis=0) * np.sqrt(ni * nr / np.maximum(1, ni + nr)) / 4
            arms = {"periodicity": (contrast, accepted), "region_scan": (scan, accepted)}
            arms["surrounded_scan"] = (scan, accepted & _sides_supported(sums, counts, h, w, inside))
            if edge_delta is not None:
                with_edges = accepted & (edge_delta.min(axis=0) > .015)
                arms.update(with_edges=(contrast, with_edges), scan_with_edges=(scan, with_edges))
            for arm, (response, mask) in arms.items():
                values = np.where(mask, response, 0.)
                y, x = np.unravel_index(np.argmax(values), values.shape)
                score = float(values[y, x])
                if score > 0 and (chosen[arm] is None or score > chosen[arm].score):
                    box = (phase_x + CELL * (int(x) + 2), phase_y + CELL * (int(y) + 2), w * CELL, h * CELL)
                    chosen[arm] = Region(box, score, tuple(map(float, inside[:, y, x])), tuple(map(float, around[:, y, x])),
                                         (int(ni[y, x]), int(nr[y, x])), None if edge_delta is None else tuple(map(float, edge_delta[:, y, x])))
    return chosen


def measure(data, *, check=None):
    source = read_luma(data)
    if source is None:
        return None
    gray, final = source
    scores = {}
    best = None
    for phase, quality, periodic, use in _phase_hypotheses(gray, final, check=check):
        strength = _global_strength(periodic, use)
        scores[phase] = max(scores.get(phase, -1.), strength)
        if best is None or strength > best[0]:
            best = strength, phase, quality, periodic, use
    if best is None or best[0] < MIN_BACKGROUND or best[1] == (0, 0):
        return None
    strength, phase, quality, periodic, use = best
    margin = strength - max((value for xy, value in scores.items() if xy != phase), default=-1.)
    if margin < MIN_PHASE_LEAD:
        return None
    totals, counts = _cell_totals(periodic), _cell_totals(use)
    field = totals / np.maximum(1, counts)
    valid = counts.min(axis=0) >= MIN_COEFFICIENTS
    h, w = (n * CELL for n in valid.shape)
    return Measurement(phase, quality, strength, margin, _regions(field, valid, _grid_steps(gray, *phase, h, w), phase, check=check))
