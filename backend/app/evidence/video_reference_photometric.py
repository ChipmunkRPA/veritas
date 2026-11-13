# SPDX-License-Identifier: Apache-2.0
"""Paired video comparison using a robust same-pixel photometric fit."""

import subprocess
from fractions import Fraction
from itertools import pairwise

import cv2
import numpy as np

from app.detectors.video.decode import DecodeError
from app.evidence import video_reference_local as local

BASIS = "paired_h264_local_area/1"
POINT = 1.5712549522005277
FLANK = 5
WIDTHS = (8, 12, 16, 20)
CELL = 8
SCOPE = ("Compares two upright 720 x 480 H.264 clips with the same verified picture clock, at 23–25 fps. "
         "The highlighted area estimates where a short visual difference persists after accounting for overall "
         "brightness and contrast. The difference may reflect ordinary editing and adds no AI confidence.")


def eligible(paths, tables, clock):
    if not 40 <= len(clock) <= local.MAX_FRAMES or any(
        table.rotation or (table.width, table.height) != (720, 480) for table in tables
    ):
        return False
    step = clock[1] - clock[0]
    if not Fraction(1, 25) <= step <= Fraction(1, 23) or any(b - a != step for a, b in pairwise(clock)):
        return False
    try:
        return all(local._carrier(path) == [("h264", "yuv420p")] for path in paths)
    except (DecodeError, OSError, ValueError, subprocess.SubprocessError):
        return False


def assess(signals, background_errors, rects):
    valid = [value for value in background_errors if value is not None]
    if len(valid) < local.MIN_FRACTION * len(signals):
        return "unverified_photometric_support", None
    if np.percentile(valid, 65) > 3 or np.percentile(valid, 25) > 2.5:
        return "insufficient_shared_picture_content", None
    values = np.asarray(signals)
    xs, ys, width, height = rects
    best = None
    for length in WIDTHS:
        for start in range(FLANK, len(values) - length - FLANK + 1):
            end = start + length
            span = values[start - FLANK:end + FLANK]
            if not np.isfinite(span).all():
                continue
            middle = np.median(span[FLANK:FLANK + length], axis=0)
            flanks = np.maximum(np.median(span[:FLANK], axis=0), np.median(span[-FLANK:], axis=0))
            response = middle - flanks
            index = int(np.argmax(response))
            score = float(response[index])
            if best is None or score > best["score"]:
                best = {"score": score, "frames": [start, end],
                        "tile": {"x": int(xs[index]), "y": int(ys[index]), "w": width, "h": height},
                        "samples": {"before": start - 1, "during": (start + end) // 2, "after": end}}
    if best is None:
        return "no_complete_flanks", None
    if best["score"] <= POINT:
        return "no_local_range", None
    return "local_difference", best


def residual(first, second):
    blurred = [cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 1.2)
               for frame in (first, second)]
    a, b = blurred
    coarse = [cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA).ravel() for frame in blurred]
    x, y = coarse[1], coarse[0]
    if min(np.quantile(frame, .8) - np.quantile(frame, .2) for frame in coarse) < 10:
        return None
    # Iterative least trimmed squares. Match spatial samples and exclude large deviations.
    scale, shift = 1., float(np.median(y - x))
    for _ in range(4):
        errors = np.abs(y - np.clip(x * scale + shift, 0, 255))
        keep = (errors <= np.quantile(errors, .72)) & (x > 8) & (x < 247)
        if keep.sum() < .45 * x.size:
            return None
        xx, yy = x[keep], y[keep]
        centered = xx - xx.mean()
        if float(np.mean(centered ** 2)) < 36:
            return None
        scale = float(np.clip(np.dot(centered, yy - yy.mean()) / np.dot(centered, centered), .65, 1.55))
        shift = float(np.clip(np.median(yy - scale * xx), -80, 80))
    return cv2.absdiff(a, np.clip(b * scale + shift, 0, 255))


def measurements(a, b, rects):
    difference = residual(a, b)
    height, width = a.shape[:2]
    cells = (width // CELL, height // CELL)
    if difference is None:
        return np.full(len(rects[0]), np.nan), None, np.full(cells[::-1], np.nan)
    xs, ys, w, h = rects
    summed = cv2.integral(difference, sdepth=cv2.CV_64F)
    tiles = (summed[ys + h, xs + w] - summed[ys, xs + w] - summed[ys + h, xs] + summed[ys, xs]) / (w * h)
    noise = float(np.percentile(tiles, 65))
    grid = cv2.resize(difference, cells, interpolation=cv2.INTER_AREA)
    return np.maximum(0, tiles - noise), noise, np.maximum(0, grid - np.percentile(grid, 65))


def expanded_region(grid, choice, *, constrained):
    first, stop = choice["frames"]
    flank, cell = FLANK, CELL
    quarter = (stop - first) // 4
    before, inside, after = [np.median(grid[lo:hi], axis=0) for lo, hi in (
        (first - flank, first), (first + quarter, stop - quarter), (stop, stop + flank),
    )]
    signal = np.maximum(0, inside - np.maximum(before, after))
    tile = choice["tile"]
    x, y, w, h = (tile[key] for key in ("x", "y", "w", "h"))
    left, top = x // cell, y // cell
    seed = signal[top:(y + h + cell - 1) // cell, left:(x + w + cell - 1) // cell]
    py, px = np.unravel_index(int(seed.argmax()), seed.shape)
    peak = float(seed[py, px])
    if peak < 1.75:
        return tile
    cap = .18 * signal.size
    fractions = (.07, .14, .28, .42) if constrained else (.07,)
    for fraction in fractions:
        support = (signal >= max(1.75, peak * fraction)).astype(np.uint8)
        if constrained:
            # A weak bridge can attach a distant change to the selected region.
            support = cv2.morphologyEx(support, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), anchor=(0, 0))
            support = np.roll(support, (1, 1), axis=(0, 1))
            support[:1, :] = 0
            support[:, :1] = 0
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        label = int(labels[top + py, left + px])
        if label == 0:
            continue
        bx, by, bw, bh = map(int, stats[label, :4])
        if 4 <= bw * bh <= cap:
            return {"x": cell * bx, "y": cell * by, "w": cell * bw, "h": cell * bh}
    return tile
