# SPDX-License-Identifier: Apache-2.0
"""Research-only fine background evidence for unavailable coarse scene tiles."""

from __future__ import annotations

import cv2
import numpy as np

CFG = {
    "width": 192, "height": 128, "rows": 8, "cols": 16, "texture_floor": 4.0,
    "min_fine_tiles": 16, "minimum_quadrants": 3, "min_tiles_per_quadrant": 3,
    "minimum_macro_sectors": 4, "minimum_median": .90, "minimum_quadrant_q25": .82,
    "luma_p90_max": 8.0, "chroma_p90_max": 6.0,
}


def fine_frame(rgb, face):
    width, height, rows, cols = (CFG[key] for key in ("width", "height", "rows", "cols"))
    yuv = cv2.cvtColor(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    blocks = yuv.reshape(rows, height//rows, cols, width//cols, 3).transpose(0, 2, 1, 3, 4).reshape(rows*cols, -1, 3)
    luma = blocks[:, :, 0]
    residual = luma - luma.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(residual, axis=1)
    energy = norm >= CFG["texture_floor"] * np.sqrt(luma.shape[1])
    spatial = np.ones(rows*cols, dtype=bool)
    if face:
        native_height, native_width = rgb.shape[:2]
        x, y, face_width, face_height = face.box()
        row, col = np.divmod(np.arange(rows*cols), cols)
        spatial &= ~((col*native_width/cols < x+1.1*face_width) & ((col+1)*native_width/cols > x-.1*face_width)
                     & (row*native_height/rows < y+1.1*face_height) & ((row+1)*native_height/rows > y-.1*face_height))
    return residual / np.maximum(norm[:, None], 1e-7), energy, spatial, blocks


def fine_edge(first, second):
    left, left_textured, left_spatial, left_pixels = first
    right, right_textured, right_spatial, right_pixels = second
    space = left_spatial & right_spatial
    mask = space & left_textured & right_textured
    row, col = np.divmod(np.arange(CFG["rows"]*CFG["cols"]), CFG["cols"])
    quadrants, sectors = (row//4)*2+col//8, (row//4)*4+col//4
    support = np.sum(left[mask]*right[mask], axis=1)
    median = float(np.median(support)) if len(support) else None
    qualified = []
    for quadrant in set(quadrants[mask]):
        scores = support[quadrants[mask] == quadrant]
        if len(scores) >= CFG["min_tiles_per_quadrant"] and float(np.quantile(scores, .25)) >= CFG["minimum_quadrant_q25"]:
            qualified.append(int(quadrant))
    sector_count = len(set(sectors[mask]))
    delta = np.mean(np.abs(left_pixels[space] - right_pixels[space]), axis=1)
    luma = float(np.quantile(delta[:, 0], .9)) if len(delta) else None
    chroma = float(np.quantile(np.max(delta[:, 1:], axis=1), .9)) if len(delta) else None
    count = int(np.count_nonzero(mask))
    passes = (count >= CFG["min_fine_tiles"] and len(qualified) >= CFG["minimum_quadrants"]
              and sector_count >= CFG["minimum_macro_sectors"] and median >= CFG["minimum_median"]
              and luma is not None and chroma is not None and luma <= CFG["luma_p90_max"] and chroma <= CFG["chroma_p90_max"])
    return {"count": count, "qualified_quadrants": qualified, "sectors": sector_count,
            "median": median, "luma_p90": luma, "chroma_p90": chroma, "passes": passes}
