"""Bounded native picture comparisons for two independently encoded clips."""

from __future__ import annotations

import subprocess
from contextlib import closing
from fractions import Fraction
from itertools import pairwise, zip_longest

import cv2
import numpy as np

from app.detectors.video.decode import DecodeError, decode_frames
from app.evidence import video_reference_local as local

BASIS = "native_h264_extent_v1"
POINT = 1.5712549522005277
FLANK = 5
WIDTHS = (8, 12, 16, 20)
CELL = 8
SCOPE = ("Compares two upright 720 x 480 color H.264 videos at the same verified clock, between 23 and 25 fps. "
         "The marked area is an estimate of where a short difference persists relative to both surrounding ranges. "
         "Ordinary changes can produce this evidence. It carries zero weight in the AI estimate.")


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


def _residual(a, b):
    # Match the time selector's photometric transform before changing the spatial grid.
    a, b = (cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 1.2) for rgb in (a, b))
    a_q, b_q = (np.percentile(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA), (20, 50, 80)) for gray in (a, b))
    if min(a_q[2] - a_q[0], b_q[2] - b_q[0]) < 10:
        raise DecodeError("Video could not be decoded with measurable evidence pictures.")
    scale = float(np.clip((a_q[2] - a_q[0]) / (b_q[2] - b_q[0]), .75, 1.35))
    residual = cv2.absdiff(a, np.clip((b - b_q[1]) * scale + a_q[1], 0, 255).astype(np.float32))
    height, width = residual.shape
    cells = cv2.resize(residual, (width // CELL, height // CELL), interpolation=cv2.INTER_AREA)
    return np.maximum(0, cells - np.percentile(cells, 65))


def region(paths, tables, choice, decoded_hash):
    start, end = choice["frames"]
    windows = (range(start - FLANK, start),
               range(start + (end - start) // 4, end - (end - start) // 4),
               range(end, end + FLANK))
    indices = [index for window in windows for index in window]
    residuals = {}
    with (closing(decode_frames(paths[0], tables[0], indices, timeout_s=40)) as earlier,
          closing(decode_frames(paths[1], tables[1], indices, timeout_s=40)) as submitted):
        for index, a, b in zip_longest(indices, earlier, submitted):
            if (index is None or a is None or b is None or a.index != index or b.index != index
                    or not a.pts_verified or not b.pts_verified
                    or not decoded_hash(index, a.rgb, b.rgb)):
                raise DecodeError("Video could not be decoded with consistent picture evidence.")
            residuals[index] = _residual(a.rgb, b.rgb)
    before, center, after = (np.median([residuals[i] for i in window], axis=0) for window in windows)
    signal = np.maximum(0, center - np.maximum(before, after))
    tile = choice["tile"]
    x, y, w, h = (tile[key] for key in ("x", "y", "w", "h"))
    seed = signal[y // CELL:(y + h + CELL - 1) // CELL, x // CELL:(x + w + CELL - 1) // CELL]
    support = (signal >= max(1.75, .07 * float(seed.max()))).astype(np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    dy, dx = np.unravel_index(int(seed.argmax()), seed.shape)
    label = int(labels[y // CELL + dy, x // CELL + dx])
    if label == 0:
        return tile
    bx, by, bw, bh = [int(value) for value in stats[label, :4]]
    if not 4 <= bw * bh <= .18 * support.size:
        return tile
    return {"x": CELL * bx, "y": CELL * by, "w": CELL * bw, "h": CELL * bh}
