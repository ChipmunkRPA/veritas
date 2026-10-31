"""Content the validation set does not hold (D-65; BUILD_SPEC N-18; browser QA P0-2).

Screenshots of text and interfaces, a flat grey image and a greyscale copy of a photograph were called AI at 92 to 98 %
"within the validated envelope". Two measurements of the pixels say that an image is of a kind the bench does not hold:

- **The flat-block share:** of the 8 × 8 blocks of the image as its file decodes (full size, no resampling, the grid from
  the top-left corner, partial blocks at the right and bottom edges left out), the share in which every pixel has one
  value in each of the three channels. Photographs and generated pictures have few such blocks; text pages, interfaces
  and flat graphics have many.
- **Greyscale:** a single-channel image, or one whose three channels are equal in every pixel.

The limit of the share is the larger of `FLOOR` and the `QUANTILE` of the share over the `fit` and `cal` images of the
bench, both classes (`scripts/v2/content_kinds.py` measures it and writes `content_kinds.json`, by bench manifest, beside
`validated_sizes.json`). The test split is not read to set it.

`measure` is the one function the bench script and the condition stage call, so a served image is measured as the bench's
images were.
"""
from __future__ import annotations

import functools
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BLOCK = 8
FLOOR = 0.25  # D-65: the limit is never below a quarter of the blocks
QUANTILE = 0.995
TABLE = Path(__file__).with_name("content_kinds.json")
SINGLE_CHANNEL_MODES = frozenset({"1", "L", "LA", "La", "I", "I;16", "I;16L", "I;16B", "F"})


def flat_block_share(rgb: np.ndarray) -> float | None:
    """The share of whole 8 × 8 blocks of an H × W × 3 array that are constant in all three channels; None for an image
    smaller than one block."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"an H x W x 3 array is needed, not {rgb.shape}")
    rows, cols = rgb.shape[0] // BLOCK, rgb.shape[1] // BLOCK
    if rows == 0 or cols == 0:
        return None
    blocks = rgb[:rows * BLOCK, :cols * BLOCK].reshape(rows, BLOCK, cols, BLOCK, 3)
    flat = (blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))).all(axis=2)
    return float(flat.mean())


def equal_channels(rgb: np.ndarray) -> bool:
    return bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))


def measure(source: bytes | Path) -> dict[str, Any]:
    """The two measurements of an image file, on the pixels of its first frame as the file decodes: no EXIF rotation (a
    rotation by a quarter turn moves the block grid by at most the partial edge blocks) and no alpha compositing."""
    opened = Image.open(io.BytesIO(source) if isinstance(source, bytes | bytearray | memoryview) else source)
    with opened as im:
        im.load()
        single = im.mode in SINGLE_CHANNEL_MODES
        rgb = np.asarray(im.convert("RGB"))
        return {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "mode": im.mode,
                "flat_block_share": flat_block_share(rgb), "greyscale": bool(single or equal_channels(rgb))}


def limit_from(shares: np.ndarray) -> float:
    """D-65 "The limit": the larger of the floor and the 99.5th percentile of the measured shares."""
    return max(FLOOR, float(np.quantile(shares, QUANTILE)))


@functools.cache
def limits() -> Mapping[str, Any]:
    """`content_kinds.json` by bench manifest sha256 (written by `scripts/v2/content_kinds.py`); empty before it exists."""
    if not TABLE.is_file():
        return {}
    return json.loads(TABLE.read_text(encoding="utf-8"))["by_manifest_sha256"]


def flat_limit(bench: Mapping[str, Any] | None) -> float | None:
    """The limit of the flat-block share for the bench an artefact was validated on; None when the bench has no entry
    (fixtures), as `abstain.size_floor` has no floor then."""
    entry = limits().get(str((bench or {}).get("manifest_sha256")))
    return None if not entry else float(entry["flat_blocks"]["limit"])
