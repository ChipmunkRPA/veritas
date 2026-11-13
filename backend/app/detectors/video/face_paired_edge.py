# SPDX-License-Identifier: Apache-2.0
"""Paired appearance transitions with nearby motion measured from native pictures."""

from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image.faces import iou
from app.detectors.video.face_appearance import distance
from app.evidence.face_paired_edge_binding import MARGIN, OFFSETS, POINT


def reduced_rgb(rgb):
    return cv2.cvtColor(cv2.resize(rgb, (384, 216), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def motion(edge, reduced, box):
    """Compare face and nearby motion to the median local movement on the same pixels."""
    height, width = reduced[edge].shape
    x, y, w, h = (box[key] * 0.3 for key in ("x", "y", "w", "h"))
    nx, ny = np.meshgrid(np.linspace(-0.8, 0.8, 96), np.linspace(-0.8, 0.8, 96))
    inner = (nx / 0.43) ** 2 + (ny / 0.47) ** 2 <= 1
    ring = ((nx / 0.77) ** 2 + (ny / 0.76) ** 2 <= 1) & ~inner
    masks = [inner, ring & (nx < -0.40), ring & (nx > 0.40), ring & (ny < -0.42), ring & (ny > 0.42)]
    # Preserve the experiment's arithmetic order before float32 interpolation coordinates are rounded.
    mx, my = (nx * w + (x + w / 2)).astype("float32"), (ny * h + (y + h / 2)).astype("float32")
    visible = (mx >= 0) & (mx <= width - 1) & (my >= 0) & (my <= height - 1)
    actual = [mask & visible for mask in masks]
    available = [
        i
        for i, (mask, shown) in enumerate(zip(masks, actual, strict=True))
        if np.count_nonzero(shown) >= 0.8 * np.count_nonzero(mask)
    ]
    neighbors = [
        edge + offset for offset in OFFSETS if edge + offset - 1 in reduced and edge + offset in reduced
    ]
    if 0 not in available or len(available) < 3 or len(neighbors) < 5:
        return None
    reprojected = {
        i: cv2.remap(reduced[i], mx, my, cv2.INTER_LINEAR).astype("float32")
        for i in {j + k for j in neighbors for k in (-1, 0)}
    }
    changes = {
        i: [float(np.abs(reprojected[i] - reprojected[i - 1])[actual[slot]].mean()) for slot in available]
        for i in neighbors
    }
    baseline = np.median([changes[i] for i in neighbors if i != edge], axis=0)
    central, *sectors = map(float, np.maximum(0.0, np.asarray(changes[edge]) - baseline))
    if central < 3.0 or sum(value >= max(6.0, 0.5 * central) for value in sectors) >= 2:
        return None
    return {
        "central_hex": central.hex(),
        "sector_ids": available[1:],
        "sectors_hex": [value.hex() for value in sectors],
    }


def estimate(coarse, values, reduced, check):
    first, last = coarse
    entries = range(max(5, first - MARGIN), min(115, first + MARGIN) + 1)
    exits = range(max(5, last - MARGIN), min(115, last + MARGIN) + 1)
    edges = {}
    for i in sorted(set(entries) | set(exits)):
        check("paired face timing")
        left, right = range(i - 3, i), range(i, i + 3)
        indices = [*left, *right]
        if not all(j in values for j in indices) or any(
            iou(values[a][1], values[b][1]) < 0.25 for a, b in zip(indices, indices[1:])
        ):
            continue
        before, after = [np.median([values[j][0] for j in side], axis=0) for side in (left, right)]
        separation = distance(before, after)
        if separation < 0.08:
            continue
        proof = motion(i, reduced, values[i - 1][2]["bbox"])
        if proof is None:
            continue
        floor = max(distance(values[j][0], before if j < i else after) for j in indices)
        proof.update(frame_indices=indices, separation_hex=separation.hex(), dispersion_hex=floor.hex())
        edges[i] = before, after, separation - floor, proof
    strongest = None
    for start in (i for i in entries if i in edges):
        before, first_middle, left_score, left = edges[start]
        for stop in (i for i in exits if i in edges and 5 <= i - start <= 16):
            if 2 * (min(stop, last) - max(start, first)) < min(stop - start, last - first):
                continue
            last_middle, after, right_score, right = edges[stop]
            gaps = (
                distance(before, first_middle),
                distance(after, last_middle),
                distance(before, after),
                distance(first_middle, last_middle),
            )
            score = min(left_score, right_score, min(gaps[:2]) - max(gaps[2:]))
            rank = score, -start, -stop
            if score > POINT and (strongest is None or rank > strongest[0]):
                strongest = rank, {"entry": left, "exit": right}, gaps
    if strongest is None:
        return None
    (_, start, stop), proof, gaps = strongest
    return (-start, -stop), proof, [value.hex() for value in gaps]
