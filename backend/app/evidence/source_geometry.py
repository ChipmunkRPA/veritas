"""Pixel bounds for rectangles recorded in source coordinates."""

from __future__ import annotations

import math
from typing import Any


def pixel_bounds(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(box, dict):
        raise ValueError("A source region is missing")
    values = [box.get(k) for k in ("x", "y", "w", "h")]
    if any(not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError("A source region has invalid coordinates")
    x, y, w, h = values
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 0.001 or y + h > height + 0.001:
        raise ValueError("A source region is outside the original image")
    return math.floor(x), math.floor(y), min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
