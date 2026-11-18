# SPDX-License-Identifier: Apache-2.0
"""Verify picture correspondence for qualified brief soundtrack comparisons."""

import numpy as np

from app.evidence.video_reference_soundtrack_reencode import summarize

BASIS = "short_h264_soundtrack_picture_grid/1"
GRIDS = ((80, 60), (160, 90))
INSTANT_CELL_MAX = 13.0
MIN_PICTURES, MAX_PICTURES = 53, 72


def summarize_grids(bands):
    grids = []
    for band, (width, height) in zip(bands, GRIDS, strict=True):
        values = np.asarray(band)
        count = len(values)
        cells = values.reshape(count, height // 5, 5, width // 5, 5).mean(axis=(2, 4))
        grid = summarize(band)
        maximum = round(float(cells.max()), 6)
        grids.append({
            **grid,
            "width": width,
            "height": height,
            "instant_cell_max": maximum,
            "within_bounds": grid["within_bounds"] and maximum <= INSTANT_CELL_MAX,
        })
    return {"basis": BASIS, "grids": grids, "instant_cell_max_point": INSTANT_CELL_MAX,
            "within_bounds": all(grid["within_bounds"] for grid in grids)}
