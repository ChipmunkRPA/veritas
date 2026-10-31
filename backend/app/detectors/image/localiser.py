"""`Localiser` output, resolution policy and the deterministic conditions mask (IF §3.2, §3.3; D-29).

- **Output.** `LocaliserOutput`: the anomaly map (float32 in [0, 1] on an aspect-correct grid registered to the native
  image), a reliability mask (a model reliability map when the localiser has one, else the conditions mask), the integrity
  score, connected-component regions in native px, the method, the weights sha256 and the resolution policy.
- **Resolution policy (IF §3.3).** Screening, every profile: one pass over the whole image resized to the model input
  (at most 2.4 MP). Deep: native-resolution tiles of the model input size with `overlap` px of overlap, blended with a
  raised-cosine window on the sides shared with a neighbour; regions come from the tiled map, and the integrity score is
  the larger of the tiled and screening scores.
- **Conditions mask (IF §3.2 (b), D-29).** Where the map is less reliable, measured on native pixels:
  - clipped: any channel ≥ 250, or every channel ≤ 5, dilated by 2 px;
  - flat: luma standard deviation below 1.5 grey levels over 9 × 9 px;
  - heavily compressed: when the estimated JPEG luma quality is below 70, 8 × 8 blocks whose luma stays within ±2 grey
    levels of the block mean (texture quantised away), dilated by one block;
  - upscaled: 32 × 32 px blocks that are not flat and whose Laplacian-to-gradient energy ratio is below `UPSCALE_RATIO`
    (interpolation removes the finest detail).
  A grid cell is unreliable when at least half of its native pixels are.
- **Regions.** 8-connected components of map ≥ 0.5 among reliable cells covering at least 0.1 % of the grid: bbox in native
  px, area share, mean and maximum map value, largest mean first, at most ten (R1 … R10).
- **Integrity score.** Mean of the top 1 % of the reliable map values; uncalibrated, 0 when no cell is reliable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

THRESHOLD = 0.5
MIN_REGION_FRACTION = 0.001
MAX_REGIONS = 10
TOP_FRACTION = 0.01
SCREENING_MAX_PIXELS = 2_400_000
GRID_MAX_SIDE = {"screening": 1024, "tiled": 2048}
CLIP_HIGH, CLIP_LOW, CLIP_DILATE_PX = 250, 5, 2
FLAT_WINDOW, FLAT_STD = 9, 1.5
COMPRESSED_QUALITY, COMPRESSED_RANGE = 70, 2.0
UPSCALE_BLOCK, UPSCALE_RATIO = 32, 0.35
CONDITIONS_LABEL = "Where the map is less reliable (saturated, flat, heavily compressed or upscaled areas)"


@dataclass(frozen=True)
class LocaliserOutput:
    anomaly_map: np.ndarray  # grid H × W, float32 in [0, 1]
    reliable: np.ndarray  # grid H × W bool, True = reliable
    reliability_source: str  # "conditions" (deterministic mask) or "model"
    integrity_score: float
    regions: list[dict[str, Any]]  # `ScoredRegion` dicts in native px
    region_stats: list[dict[str, float]]  # per region: mean, maximum, area share
    method: str
    weights_sha256: str
    resolution_policy: dict[str, Any]
    conditions: dict[str, float] = field(default_factory=dict)  # native-pixel shares per condition


# --- geometry --------------------------------------------------------------------------------------------------------

def grid_shape(width: int, height: int, max_side: int) -> tuple[int, int]:
    """(grid width, grid height): the native aspect ratio with the long side at most `max_side`."""
    scale = min(1.0, max_side / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    """Tile starts along one axis: stride `tile − overlap`, the last tile flush with the far edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    return starts + [length - tile]


def _ramp(n: int, overlap: int, taper_start: bool, taper_end: bool) -> np.ndarray:
    weights = np.ones(n, np.float64)
    if overlap <= 0:
        return weights
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)
    if taper_start:
        weights[:overlap] = ramp
    if taper_end:
        weights[n - overlap:] = np.minimum(weights[n - overlap:], ramp[::-1])
    return weights


def blend_tiles(tiles: list[tuple[int, int, np.ndarray]], width: int, height: int, overlap: int) -> np.ndarray:
    """Native-resolution map from `(x, y, map)` tiles, raised-cosine weighted on the sides shared with a neighbour."""
    total = np.zeros((height, width), np.float64)
    weight = np.zeros((height, width), np.float64)
    for x, y, values in tiles:
        th, tw = values.shape
        w = np.outer(_ramp(th, overlap, y > 0, y + th < height), _ramp(tw, overlap, x > 0, x + tw < width))
        total[y:y + th, x:x + tw] += values * w
        weight[y:y + th, x:x + tw] += w
    return (total / np.maximum(weight, 1e-12)).astype(np.float32)


def to_grid(values: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Area-resample a native-resolution array onto a (width, height) grid."""
    if values.shape[::-1] == grid:
        return values.astype(np.float32)
    return cv2.resize(values.astype(np.float32), grid, interpolation=cv2.INTER_AREA)


# --- conditions mask -------------------------------------------------------------------------------------------------

def _block_view(values: np.ndarray, block: int) -> np.ndarray:
    h, w = (values.shape[0] // block) * block, (values.shape[1] // block) * block
    return values[:h, :w].reshape(h // block, block, w // block, block)


def _expand_blocks(block_mask: np.ndarray, block: int, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, bool)
    expanded = np.kron(block_mask.astype(np.uint8), np.ones((block, block), np.uint8)).astype(bool)
    full[:expanded.shape[0], :expanded.shape[1]] = expanded
    return full


def upscale_ratios(luma: np.ndarray) -> np.ndarray:
    """Per `UPSCALE_BLOCK` block: mean squared 4-neighbour Laplacian over mean squared forward-difference gradient."""
    laplacian = cv2.Laplacian(luma, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT)
    gx, gy = np.zeros_like(luma), np.zeros_like(luma)
    gx[:, :-1] = np.diff(luma, axis=1)
    gy[:-1, :] = np.diff(luma, axis=0)
    energy = _block_view(laplacian * laplacian, UPSCALE_BLOCK).mean(axis=(1, 3))
    gradient = _block_view(gx * gx + gy * gy, UPSCALE_BLOCK).mean(axis=(1, 3))
    return energy / np.maximum(gradient, 1e-6)


def conditions_mask(pixels: np.ndarray, jpeg_quality: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Native-resolution unreliable mask (True = less reliable) and the share of pixels per condition."""
    rgb = np.ascontiguousarray(pixels)
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shape = luma.shape
    peak = rgb.max(axis=2)
    clipped = (peak >= CLIP_HIGH) | (peak <= CLIP_LOW)  # any channel clipped high, or every channel crushed
    if clipped.any():
        size = 2 * CLIP_DILATE_PX + 1
        clipped = cv2.dilate(clipped.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    mean = cv2.boxFilter(luma, cv2.CV_32F, (FLAT_WINDOW, FLAT_WINDOW), borderType=cv2.BORDER_REFLECT)
    square = cv2.boxFilter(luma * luma, cv2.CV_32F, (FLAT_WINDOW, FLAT_WINDOW), borderType=cv2.BORDER_REFLECT)
    flat = np.sqrt(np.maximum(square - mean * mean, 0.0)) < FLAT_STD

    compressed = np.zeros(shape, bool)
    blocks = _block_view(luma, 8)
    if jpeg_quality is not None and jpeg_quality < COMPRESSED_QUALITY and blocks.size:
        spread = np.abs(blocks - blocks.mean(axis=(1, 3), keepdims=True)).max(axis=(1, 3))
        smooth = cv2.dilate((spread <= COMPRESSED_RANGE).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        compressed = _expand_blocks(smooth, 8, shape)

    upscaled = np.zeros(shape, bool)
    if min(shape) >= UPSCALE_BLOCK:
        not_flat = _block_view(flat.astype(np.float32), UPSCALE_BLOCK).mean(axis=(1, 3)) < 0.5
        upscaled = _expand_blocks((upscale_ratios(luma) < UPSCALE_RATIO) & not_flat, UPSCALE_BLOCK, shape)

    unreliable = clipped | flat | compressed | upscaled
    n = float(luma.size)
    shares = {"clipped": clipped.sum() / n, "flat": flat.sum() / n, "compressed": compressed.sum() / n,
              "upscaled": upscaled.sum() / n, "unreliable": unreliable.sum() / n}
    return unreliable, {k: round(float(v), 4) for k, v in shares.items()}


def reliable_grid(unreliable: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """A grid cell is reliable when fewer than half of its native pixels are unreliable."""
    return to_grid(unreliable.astype(np.float32), grid) < 0.5


# --- regions and score -----------------------------------------------------------------------------------------------

def integrity_score(values: np.ndarray, reliable: np.ndarray) -> float:
    valid = values[reliable]
    if valid.size == 0:
        return 0.0
    k = max(1, int(np.ceil(valid.size * TOP_FRACTION)))
    return float(np.partition(valid.ravel(), valid.size - k)[valid.size - k:].mean())


def components(values: np.ndarray, reliable: np.ndarray, width: int, height: int
               ) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Connected components of map ≥ `THRESHOLD` among reliable cells: `ScoredRegion` dicts in native px, and their stats."""
    binary = ((values >= THRESHOLD) & reliable).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    grid_h, grid_w = values.shape
    sx, sy = width / grid_w, height / grid_h
    min_area = max(1, int(np.ceil(MIN_REGION_FRACTION * grid_w * grid_h)))
    found = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        inside = values[labels == i]
        found.append(({"kind": "bbox", "bbox": {"x": round(x * sx, 1), "y": round(y * sy, 1), "w": round(w * sx, 1),
                                                  "h": round(h * sy, 1)}},
                      {"mean": round(float(inside.mean()), 4), "max": round(float(inside.max()), 4),
                       "area_share": round(area / float(grid_w * grid_h), 4)}))
    found.sort(key=lambda r: (-r[1]["mean"], -r[1]["area_share"], r[0]["bbox"]["y"], r[0]["bbox"]["x"]))
    regions, region_stats = [], []
    for k, (region, stat) in enumerate(found[:MAX_REGIONS], 1):
        regions.append({"display_id": f"R{k}", "region": region, "score": stat["mean"],
                        "label": f"R{k}: mean {stat['mean']:.2f}, maximum {stat['max']:.2f}, "
                                 f"{100 * stat['area_share']:.1f} % of the image"})
        region_stats.append(stat)
    return regions, region_stats
