"""512 px tile pyramids for source images above 4096 px on the long side (ARCHITECTURE §4.3 `TileSet`, §7.4).

- **Levels.** `z = levels - 1` is the native (EXIF-oriented) image. Each lower level halves both sides, rounding up, with a
  box filter, until the whole image fits one tile at `z = 0`.
- **Tiles.** PNG without text chunks at a fixed compression level, so the bytes are deterministic. `x` counts columns and
  `y` rows from the top left; edge tiles are smaller than 512 px.
- **Manifest.** `veritas.tiles/1` maps `"z/x_y"` to tile artefact ids. It is an artefact itself, and
  `TileSet.manifest_artefact_id` points at it.
- **Budget.** The pyramid is a rendition of the original, so it is not charged to the exhibit storage budget (§4.3). The
  decoder's pixel limit (`media.MAX_PIXELS`) bounds it.
"""
from __future__ import annotations

import io
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.schemas.evidence import TileSet
from app.storage.artefacts import register_artefact
from app.storage.objects import ObjectStore

TILE_PX = 512
TILE_THRESHOLD_PX = 4096  # the viewer loads a single image up to this long side (§7.4)
MANIFEST_SCHEMA = "veritas.tiles/1"
_PNG_COMPRESS_LEVEL = 1  # fast; any fixed level is deterministic


def level_count(width: int, height: int, tile_px: int = TILE_PX) -> int:
    levels, side = 1, max(width, height)
    while side > tile_px:
        side = (side + 1) // 2
        levels += 1
    return levels


def tile_key(z: int, x: int, y: int) -> str:
    return f"{z}/{x}_{y}"


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    return buf.getvalue()


def pyramid(pixels: np.ndarray, tile_px: int = TILE_PX) -> Iterator[tuple[int, int, int, bytes]]:
    """Yield `(z, x, y, png)` for every tile, native level first."""
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("a tile pyramid needs an RGB uint8 image")
    image = Image.fromarray(pixels)
    levels = level_count(image.width, image.height, tile_px)
    for z in range(levels - 1, -1, -1):
        for y in range(math.ceil(image.height / tile_px)):
            for x in range(math.ceil(image.width / tile_px)):
                box = (x * tile_px, y * tile_px, min(image.width, (x + 1) * tile_px), min(image.height, (y + 1) * tile_px))
                yield z, x, y, _png(image.crop(box))
        if z:
            image = image.reduce(2)


def build_tileset(session: Session, store: ObjectStore, pixels: np.ndarray, *, analysis_id: str, created_by: str = "render",
                  tile_px: int = TILE_PX) -> dict[str, Any]:
    """Store and index the pyramid and its manifest (idempotent on the bytes); returns the `TileSet` JSON. The caller commits."""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    tiles: dict[str, str] = {}
    for z, x, y, data in pyramid(pixels, tile_px):
        row = register_artefact(session, store, data, media_type="image/png", role="tile", created_by=created_by,
                                analysis_id=analysis_id)
        tiles[tile_key(z, x, y)] = row.id
    levels = level_count(width, height, tile_px)
    manifest = {"schema": MANIFEST_SCHEMA, "tile_px": tile_px, "levels": levels, "width": width, "height": height,
                "format": "png", "tiles": tiles}
    row = register_artefact(session, store, canonical_json(manifest), media_type="application/json", role="tile",
                            created_by=created_by, analysis_id=analysis_id)
    return TileSet(tile_px=tile_px, levels=levels, width=width, height=height, manifest_artefact_id=row.id).model_dump(mode="json")
