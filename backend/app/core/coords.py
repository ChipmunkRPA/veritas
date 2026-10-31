"""Exhibit coordinate spaces → original pixels (L-4, N-4; the viewer's `frontend/src/lib/coords.ts` `spaceFactors` / `scaleBox`).

An exhibit's `coordinate_space` names the space its geometry or map grid is expressed in, and that space spans the whole image.
The per-axis factor to original px is `source / space` (width and height separately), which equals `scale_to_original` only when
the aspect ratio is kept; without the source size the scalar `scale_to_original` is used. The report, the synthesis sentences and
the viewer apply this one rule; `tests/fixtures/coords_vectors.json` holds the shared vectors both test suites check.

Exhibits, detectors and the viewer work in EXIF-oriented pixels. The condition profile records the oriented size since N-4; a
result from before recorded the stored, pre-rotation size, which for EXIF Orientation 5–8 has the axes swapped. `oriented_source`
takes the transposed size when the space times its scale is that size and is not the recorded one (`fits`, the two-pixel
alignment bound of the overlay). A space that fits neither keeps the recorded size: a grid that spans the image without keeping
its aspect (a 64 × 64 map over a 3:2 photo) fits neither and needs the per-axis factors.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ALIGNMENT_PX = 2.0  # EA-04: a map is laid over the image only when its extent is known to this bound
SCALE_ROUNDING = 0.0000005  # half a unit in the sixth decimal of a recorded scale, per unit of extent


def fits(space: Mapping[str, Any], size: tuple[float, float]) -> bool:
    """Whether the space times its scale is this size on both axes, within the alignment bound."""
    scale = float(space.get("scale_to_original") or 1.0)
    return all(abs(float(n) * scale - float(target)) <= ALIGNMENT_PX + float(n) * SCALE_ROUNDING + 1e-9
               for n, target in zip((space["width"], space["height"]), size, strict=True))


def oriented_source(space: Mapping[str, Any] | None, source: tuple[float, float] | None) -> tuple[float, float] | None:
    """The source size as the exhibit's space sees it: transposed when the space fits the transposed size and not this one."""
    if not source or not source[0] or not source[1] or not space or not space.get("width") or not space.get("height"):
        return source
    transposed = (source[1], source[0])
    return transposed if not fits(space, source) and fits(space, transposed) else source


def space_factors(space: Mapping[str, Any] | None, source: tuple[float, float] | None) -> tuple[float, float]:
    """(sx, sy) from the exhibit space to original px; identity when the exhibit names no space."""
    if not space or not space.get("width") or not space.get("height"):
        return 1.0, 1.0
    if source and source[0] and source[1]:
        width, height = oriented_source(space, source)
        return float(width) / float(space["width"]), float(height) / float(space["height"])
    scale = float(space.get("scale_to_original") or 1.0)
    return scale, scale


def box_to_original(bbox: Mapping[str, Any], space: Mapping[str, Any] | None, source: tuple[float, float] | None) -> dict[str, float]:
    sx, sy = space_factors(space, source)
    return {"x": float(bbox["x"]) * sx, "y": float(bbox["y"]) * sy, "w": float(bbox["w"]) * sx, "h": float(bbox["h"]) * sy}


def area_share(bbox: Mapping[str, Any], space: Mapping[str, Any] | None, source: tuple[float, float] | None) -> float | None:
    """The box's share of the whole image (0–1): the same in every space that spans the image."""
    if space and space.get("width") and space.get("height"):
        return float(bbox["w"]) * float(bbox["h"]) / (float(space["width"]) * float(space["height"]))
    if source and source[0] and source[1]:
        return float(bbox["w"]) * float(bbox["h"]) / (float(source[0]) * float(source[1]))
    return None
