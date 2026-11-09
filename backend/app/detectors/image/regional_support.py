"""Recorded regional readout; never converts editing responses to AI confidence."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import jpeg_bitstream
from app.detectors.image import localiser as loc


@dataclass(frozen=True)
class RegionalReadout:
    floating: np.ndarray
    recorded: np.ndarray
    reliable: np.ndarray
    conditions: dict
    regions: list[dict]
    stats: list[dict]
    selected_grid_mask: np.ndarray

    def source_mask(self, width: int, height: int) -> np.ndarray:
        return np.asarray(Image.fromarray(self.selected_grid_mask.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST)) > 0


def readout(raw: np.ndarray, gate: bool, threshold: float, data: bytes, pixels: np.ndarray,
            *, per_map_cap: int = 2) -> RegionalReadout:
    """Same quantization, conditions, component order and display cap as evaluation."""
    height, width = pixels.shape[:2]
    native = np.asarray(raw)
    if native.ndim != 2 or native.dtype != np.float32 or not np.isfinite(native).all() or native.min() < 0 or native.max() > 1:
        raise ValueError("Finite float32 regional response required")
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
    floating = cv2.resize(native, grid, interpolation=cv2.INTER_LINEAR)
    recorded = np.rint(np.clip(floating.astype(np.float64), 0, 1) * 255) / 255.
    structure = jpeg_bitstream.read(data)
    excluded, conditions = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
    reliable = loc.reliable_grid(excluded, grid)
    regions, stats = loc.components(recorded, reliable, width, height, threshold=threshold)
    if not gate:
        regions, stats = [], []
    _, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        ((recorded >= threshold) & reliable).astype(np.uint8), connectivity=8)
    # Match the already ranked runtime regions back to their true memberships.
    # Bounding rectangles are only identifiers; they are never filled as masks.
    memberships = []
    gh, gw = recorded.shape
    for region in regions[:per_map_cap]:
        bbox = region["region"]["bbox"]
        matches = []
        for label, row in enumerate(component_stats[1:], 1):
            x, y, w, h, area = (int(v) for v in row)
            if area < max(1, int(np.ceil(loc.MIN_REGION_FRACTION * recorded.size))):
                continue
            left, top = round(x * (width / gw), 1), round(y * (height / gh), 1)
            right, bottom = round((x + w) * (width / gw), 1), round((y + h) * (height / gh), 1)
            if bbox == {"x": left, "y": top, "w": round(right - left, 1), "h": round(bottom - top, 1)}:
                matches.append(label)
        if len(matches) != 1:
            raise ValueError("Regional component membership is ambiguous")
        memberships.append(matches[0])
    selected = np.isin(labels, memberships)
    return RegionalReadout(floating, recorded, reliable, conditions, regions, stats, selected)


def shared_pixels(a: RegionalReadout, b: RegionalReadout) -> np.ndarray:
    """Actual intersection of independently retained masks; no new filtering/cap."""
    if a.selected_grid_mask.shape != b.selected_grid_mask.shape:
        raise ValueError("Shared evidence requires the same source-coordinate grid")
    return a.selected_grid_mask & b.selected_grid_mask
