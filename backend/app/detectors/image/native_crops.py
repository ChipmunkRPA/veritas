"""Native-resolution crop grid for learned whole-image detectors (ML N1, N2).

The centre and the four corners when the short side is at least the crop size, duplicates dropped; nothing is padded or
resampled. 8-bit RGB crops become normalised float32 NCHW batches.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

CROP_NAMES = ("centre", "top left", "top right", "bottom left", "bottom right")


def crop_origins(width: int, height: int, crop: int) -> list[tuple[int, int, str]]:
    """`(x, y, name)` of the N2 grid: centre, then the four corners; empty when the short side is below `crop`."""
    if min(width, height) < crop:
        return []
    points = [((width - crop) // 2, (height - crop) // 2), (0, 0), (width - crop, 0), (0, height - crop),
              (width - crop, height - crop)]
    seen: set[tuple[int, int]] = set()
    out = []
    for (x, y), name in zip(points, CROP_NAMES, strict=True):
        if (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y, name))
    return out


def to_batch(pixels: np.ndarray, origins: list[tuple[int, int, str]], crop: int, mean: Sequence[float],
             std: Sequence[float]) -> torch.Tensor:
    """8-bit RGB crops → float32 NCHW scaled to [0, 1] and normalised per channel."""
    crops = np.stack([pixels[y:y + crop, x:x + crop, :3] for x, y, _ in origins])
    batch = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    return (batch - torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)) / torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
