"""Measure different prior JPEG grid phases on native RGB pixels."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

TILE = 8
HOP = 4
AC_THRESHOLD = 1.2
FREQUENCIES = ((0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3))
ARMS = ("dct_phase", "corroborated_phase")
PNG_ARM = "corroborated_phase"
PNG_REVIEW_POINT = 0.058593764901161194
DETECTOR_ID = "img.forensic.local_jpeg_grid"


def cells(values: np.ndarray) -> np.ndarray:
    """Means over 64 px windows, with 32 px spacing."""
    view = np.lib.stride_tricks.sliding_window_view(values, (TILE, TILE))[::HOP, ::HOP]
    return view.mean(axis=(-2, -1), dtype=np.float32)


def measured_grids(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rows, cols = (size // 8 - 2 for size in gray.shape)
    side_y, side_x = rows * 8, cols * 8
    dct_votes, texture = [], None
    for y in range(8):
        for x in range(8):
            blocks = gray[8+y:8+y+side_y, 8+x:8+x+side_x].reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3)
            if y == 0 and x == 0:
                texture = cells((blocks.std(axis=(-2, -1)) >= 5).astype(np.float32))
            transformed = dctn(blocks, axes=(-2, -1), norm="ortho", workers=1)
            yy, xx = zip(*FREQUENCIES, strict=True)
            evidence = (np.abs(transformed[:, :, yy, xx]) < AC_THRESHOLD).mean(axis=2)
            dct_votes.append(cells(evidence))

    def boundary(plane):
        delta = np.abs(np.diff(plane, axis=1))
        normalized = np.clip(delta[:, 1:-1] - (delta[:, :-2] + delta[:, 2:]) * .5, -12, 12)
        return normalized

    x_edges, y_edges = boundary(gray), boundary(gray.T).T
    xs, ys = [], []
    for phase in range(8):
        x = x_edges[8:8+side_y, 6+phase:6+phase+side_x:8]
        y = y_edges[6+phase:6+phase+side_y:8, 8:8+side_x]
        xs.append(cells(x.reshape(rows, 8, cols).mean(axis=1)))
        ys.append(cells(y.reshape(rows, cols, 8).mean(axis=2)))
    boundary_votes = (np.asarray(ys)[:, None] + np.asarray(xs)[None, :]).reshape(64, *dct_votes[0].shape)
    return np.asarray(dct_votes), boundary_votes, texture


def response(rgb: np.ndarray) -> dict:
    dct, boundaries, texture = measured_grids(rgb)
    measured = texture >= .40
    if np.count_nonzero(measured) < 9:
        return {"global_phase": None, "winner": None, "scores": {arm: 0. for arm in ARMS}}
    global_dct = dct[:, measured].mean(axis=1)
    global_boundary = boundaries[:, measured].mean(axis=1)
    native = int(global_dct.argmax())
    if float(global_dct[native] - np.median(global_dct)) < .015:
        return {"global_phase": None, "winner": None, "scores": {arm: 0. for arm in ARMS}}
    winner = dct.argmax(axis=0)
    support = np.take_along_axis(dct, winner[None], axis=0)[0] - dct[native]
    step = np.take_along_axis(boundaries, winner[None], axis=0)[0] - boundaries[native]
    locations = measured & (winner != native)
    arms = {"dct_phase": locations,
            "corroborated_phase": locations & (step >= .12) & (int(global_boundary.argmax()) == native)}
    ranked = {}
    for arm, allowed in arms.items():
        values = np.where(allowed, support, 0.)
        ranked[arm] = {"values": values, "score": float(values.max())}
    return {"global_phase": native, "winner": winner, "scores": {arm: r["score"] for arm, r in ranked.items()}, "maps": ranked}


def selected_regions(result, arm, point):
    if result["winner"] is None:
        return []
    phases = result["winner"]
    eligible = result["maps"][arm]["values"] > point
    regions = []
    for phase in sorted(np.unique(phases[eligible]).tolist()):
        active = eligible & (phases == phase)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), connectivity=4)
        for label, row in enumerate(stats[1:], 1):
            if row[4] < 2:
                continue
            regions.append((phase, np.nonzero(labels == label), tuple(int(value) for value in row)))
    return regions


def pixel_map(result, arm, point, shape):
    """Map each selected 32 px cell to its native pixel area and retain each phase's exact bounds."""
    values = np.zeros(shape, np.float32)
    boxes = []
    for _, (ys, xs), (left, top, width, height, _) in selected_regions(result, arm, point):
        cell_values = result["maps"][arm]["values"][ys, xs]
        for y, x, value in zip(ys, xs, cell_values, strict=True):
            py, px = 24 + int(y) * 32, 24 + int(x) * 32
            values[py:py+32, px:px+32] = value
        boxes.append((24 + left * 32, 24 + top * 32, width * 32, height * 32, float(cell_values.max())))
    return values, boxes
