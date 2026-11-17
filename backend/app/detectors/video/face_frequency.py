"""Measure face and surrounding detail on one spatially aligned track."""

from __future__ import annotations

import numpy as np

LEVELS = 3
PLANES = np.array([[.299, .587, .114], [-.169, -.331, .5], [.5, -.419, -.081]], dtype=np.float32)


def frequency_detail(crops: np.ndarray) -> dict[str, list[float]]:
    """Reduce aligned RGB crops to median log energy and same-frame contrasts."""
    if crops.ndim != 4 or crops.shape[-1] != 3 or min(crops.shape[:3]) < 8:
        raise ValueError("Frequency detail needs a nonempty track of RGB crops")
    values = crops.astype(np.float32) @ PLANES.T
    absolute, relative = [], []
    for _ in range(LEVELS):
        height, width = values.shape[1:3]
        h, w = height // 2, width // 2
        a, b = values[:, :2*h:2, :2*w:2], values[:, :2*h:2, 1:2*w:2]
        c, d = values[:, 1:2*h:2, :2*w:2], values[:, 1:2*h:2, 1:2*w:2]
        bands = ((a - b + c - d) / 4, (a + b - c - d) / 4, (a - b - c + d) / 4)
        values = (a + b + c + d) / 4
        x, y = np.meshgrid((np.arange(w) + .5) / w, (np.arange(h) + .5) / h)
        distance = ((x - .5) / .35) ** 2 + ((y - .53) / .37) ** 2
        eligible = (x >= .10) & (x <= .90) & (y >= .14) & (y <= .88)
        masks = (eligible & (distance <= .61), eligible & (distance > .61) & (distance <= 1.16),
                 eligible & (distance > 1.16))
        if not all(mask.any() for mask in masks):
            raise ValueError("Frequency detail requires support in all spatial zones")
        for band in bands:
            power = band ** 2
            energy = [np.log(.25 + power[:, mask].mean(axis=1)) for mask in masks]
            absolute.extend(np.median(energy, axis=1).reshape(-1).tolist())
            relative.extend(np.median([energy[0] - energy[1], energy[1] - energy[2]], axis=1).reshape(-1).tolist())
    return {"absolute": absolute, "relative": relative}
