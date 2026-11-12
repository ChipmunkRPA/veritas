# SPDX-License-Identifier: Apache-2.0
"""Look for small non-overlapping matches in encoded JPEG luminance blocks."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

from app.detectors.image.jpeg_dq import read_coefficients
from app.evidence.image_copy_similarity_binding import PAIR_LIMIT, V8, supported_size

BLOCK = 8
MIN_AC = 4
MIN_WITNESSES = 8
MIN_BOX_BLOCKS = 3
MAX_BOX_BLOCKS = 16
MAX_OFFSETS = 32


def _offsets(y):
    rows, cols, _ = y.shape
    flat = y.reshape(rows * cols, -1)
    textured = np.flatnonzero(np.count_nonzero(flat[:, 1:], axis=1) >= MIN_AC)
    candidates = np.ascontiguousarray(flat[textured]).view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()
    _, inverse, counts = np.unique(candidates, return_inverse=True, return_counts=True)
    copies = np.flatnonzero(counts[inverse] == 2)
    order = np.argsort(inverse[copies], kind="stable")
    before, after = textured[copies[order]].reshape(-1, 2).T
    ay, ax = np.divmod(before, cols)
    by, bx = np.divmod(after, cols)
    distances = np.column_stack((by - ay, bx - ax))
    offsets, votes = np.unique(distances, axis=0, return_counts=True)
    plausible = offsets[(votes >= MIN_WITNESSES) & (np.max(np.abs(offsets), axis=1) >= MIN_BOX_BLOCKS)]
    if len(plausible) > MAX_OFFSETS:
        return None
    return [(int(dy), int(dx), (ay[match], ax[match]))
            for dy, dx in plausible if (match := np.all(distances == [dy, dx], axis=1)).any()]


def _rectangles(y, dy, dx, witnesses):
    h, w, _ = y.shape
    x, end_x = max(0, -dx), min(w, w - dx)
    top, end_y = max(0, -dy), min(h, h - dy)
    first = y[top:end_y, x:end_x]
    second = y[top+dy:end_y+dy, x+dx:end_x+dx]
    equal = np.all(first == second, axis=2)
    _, labels, _, _ = cv2.connectedComponentsWithStats(equal.astype(np.uint8), connectivity=4)
    ay, ax = witnesses[0] - top, witnesses[1] - x
    cluster = labels[ay, ax]
    ids, counts = np.unique(cluster, return_counts=True)
    for group in ids[counts >= MIN_WITNESSES]:
        rows, cols = ay[cluster == group], ax[cluster == group]
        a, b, c, d = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
        width, height = c - a, d - b
        if (not MIN_BOX_BLOCKS <= min(width, height) <= max(width, height) <= MAX_BOX_BLOCKS
                or (abs(dx) < width and abs(dy) < height)
                or not equal[b:d, a:c].all()):
            continue
        quadrants = {(bool(col - a >= width / 2), bool(row - b >= height / 2)) for row, col in zip(rows, cols, strict=True)}
        if len(quadrants) != 4:
            continue
        first_box = (BLOCK * (x + a), BLOCK * (top + b), BLOCK * width, BLOCK * height)
        second_box = (first_box[0] + BLOCK * dx, first_box[1] + BLOCK * dy, *first_box[2:])
        yield len(rows), width * height, first_box, second_box


def analyse(data):
    """Return paired source boxes whose complete encoded luma blocks agree exactly."""
    try:
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            if (image.mode != "RGB" or JpegImagePlugin.get_sampling(image) != 2
                    or not supported_size(*image.size, version=V8) or image.getexif().get(274, 1) != 1
                    or max(image.size) > 1024):
                return "unsupported_profile", []
            width, height = image.size
        source = read_coefficients(data)
    except (OSError, ImportError, ValueError, SyntaxError, Image.DecompressionBombError):
        return "unavailable", []
    if source is None or (source.width, source.height) != (width, height):
        return "unsupported_profile", []
    blocks = source.y[:height // BLOCK, :width // BLOCK].reshape(height // BLOCK, width // BLOCK, BLOCK * BLOCK)
    offsets = _offsets(blocks)
    if offsets is None:
        return "unassessed_match_budget", []
    ranked = sorted((match for dy, dx, witnesses in offsets for match in _rectangles(blocks, dy, dx, witnesses)),
                    key=lambda row: (-row[0], -row[1], row[2:]))
    return ("supported_pair", [match[2:] for match in ranked[:PAIR_LIMIT]]) if ranked else ("no_pair", [])
