"""Bounded earlier-copy picture evidence for qualified lossless-to-H.264 pairs.

The measurements are frozen from the tested paired comparison. They locate
relative visual change for inspection and carry no authorship confidence.
"""

from __future__ import annotations

import json
import subprocess
from itertools import pairwise

import cv2
import numpy as np

from app.detectors.video.decode import DecodeError, _input, _tool

MIN_FRACTION = .90
POINT = 2.328647418186583
SHAPES = {(360, 288), (480, 360)}
MAX_FRAMES = 100
WIDTH_MS = (480, 640, 800, 960)
SCOPE = ("Compares a lossless FFV1 earlier copy with H.264 video at 360 x 288 and 25 fps, or 480 x 360 near 30 fps. "
         "Both files need upright pictures, identical relative timestamps and at most 100 pictures. This check measures "
         "short changes relative to both neighboring ranges. Ordinary edits may produce this evidence.")
STATUS = {
    "local_difference": "A local picture difference persisted across the indicated range.",
    "no_local_range": "This check found no supported local picture difference.",
    "insufficient_shared_picture_content": "The files did not have enough shared picture content for this check.",
    "unverified_photometric_support": "The pictures did not provide enough measurable brightness and contrast for this check.",
    "no_complete_flanks": "There were too few measured pictures before and after a possible local difference.",
}


def cadence(clock):
    steps = [b - a for a, b in pairwise(clock)]
    if not steps:
        return None
    dt = float(np.median([float(step) for step in steps]))
    return dt if 1 / 32 <= dt <= 1 / 24 and all(abs(float(step) - dt) <= .002 for step in steps) else None


def _carrier(path):
    result = subprocess.run([_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", _input(path)],
                            check=True, capture_output=True, text=True, timeout=3, stdin=subprocess.DEVNULL)
    return [(row.get("codec_name"), row.get("pix_fmt")) for row in json.loads(result.stdout).get("streams", [])]


def eligible(paths, tables, clock):
    dt = cadence(clock)
    if not dt or len(tables[0]) > MAX_FRAMES or any(
        table.rotation or (table.width, table.height) not in SHAPES for table in tables
    ):
        return None
    if (tables[0].width, tables[0].height) == (360, 288):
        if any(abs(float(b - a) - .04) > 1e-6 for a, b in pairwise(clock)):
            return None
    elif abs(float(clock[-1]) / (len(clock) - 1) - 1 / 30) > .001:
        return None
    try:
        source, submitted = (_carrier(path) for path in paths)
    except (DecodeError, OSError, ValueError, subprocess.SubprocessError):
        return None
    return dt if source == [("ffv1", "bgr0")] and submitted == [("h264", "yuv420p")] else None


def rectangles(height, width):
    side = min(height, width)
    w, h = round(.10 * side), round(.07 * side)
    ys, xs = np.meshgrid(np.arange(0, height - h + 1, max(1, h // 2)),
                         np.arange(0, width - w + 1, max(1, w // 2)), indexing="ij")
    return xs.ravel(), ys.ravel(), w, h


def measure(first, second, rects):
    a, b = (cv2.GaussianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 1.2)
            for pixels in (first, second))
    a_small, b_small = [cv2.resize(value, (32, 32), interpolation=cv2.INTER_AREA) for value in (a, b)]
    a_q = np.percentile(a_small, (20, 50, 80))
    b_q = np.percentile(b_small, (20, 50, 80))
    if min(a_q[2] - a_q[0], b_q[2] - b_q[0]) < 10:
        return np.full(len(rects[0]), np.nan), None
    scale = float(np.clip((a_q[2] - a_q[0]) / (b_q[2] - b_q[0]), .75, 1.35))
    residual = cv2.absdiff(a, np.clip((b - b_q[1]) * scale + a_q[1], 0, 255).astype(np.float32))
    xs, ys, w, h = rects
    total = cv2.integral(residual, sdepth=cv2.CV_64F)
    mean = (total[ys + h, xs + w] - total[ys, xs + w] - total[ys + h, xs] + total[ys, xs]) / (w * h)
    background = float(np.percentile(mean, 65))
    return np.maximum(0, mean - background), background


def assess(signals, background_errors, dt, rects):
    errors = [value for value in background_errors if value is not None]
    if len(errors) < MIN_FRACTION * len(signals):
        return "unverified_photometric_support", None
    if float(np.percentile(errors, 65)) > 3 or float(np.percentile(errors, 25)) > 2.5:
        return "insufficient_shared_picture_content", None
    xs, ys, w, h = rects
    signals = np.asarray(signals)
    flank = round(.24 / dt)
    best = None
    for width in sorted({round(ms / 1000 / dt) for ms in WIDTH_MS}):
        for start in range(flank, len(signals) - width - flank + 1):
            end = start + width
            if end >= len(signals):
                break
            block = signals[start - flank:end + flank]
            if not np.isfinite(block).all():
                continue
            before = np.median(block[:flank], axis=0)
            during = np.median(block[flank:flank + width], axis=0)
            after = np.median(block[flank + width:], axis=0)
            responses = during - np.maximum(before, after)
            index = int(np.argmax(responses))
            score = float(responses[index])
            if best is None or score > best["score"]:
                peak = start + int(np.argmax(signals[start:end, index]))
                best = {"score": score, "frames": [start, end], "tile": {"x": int(xs[index]), "y": int(ys[index]), "w": w, "h": h},
                        "samples": {"before": start - 1, "during": peak, "after": end}}
    if best is None:
        return "no_complete_flanks", None
    if best["score"] <= POINT:
        return "no_local_range", None
    return "local_difference", best
