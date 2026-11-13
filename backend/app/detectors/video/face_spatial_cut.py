# SPDX-License-Identifier: Apache-2.0
"""Measure paired off-face texture changes at an existing appearance range."""

from __future__ import annotations

import cv2
import numpy as np

WIDTH, HEIGHT, TILE = 320, 180, 12


def reduce_picture(rgb):
    return cv2.cvtColor(cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def face_box(rows):
    return np.median([[row["bbox"][key] for key in ("x", "y", "w", "h")] for row in rows], axis=0) / 4


def layout(box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    columns = range(TILE, WIDTH - TILE, TILE)
    rows = range(TILE, HEIGHT - TILE, TILE)
    return [
        (a, b, (a + TILE / 2 >= cx, b + TILE / 2 >= cy))
        for b in rows
        for a in columns
        if (a + TILE < x - 0.22 * w or a > x + 1.22 * w or b + TILE < y - 0.22 * h or b > y + 1.22 * h)
        and abs(a + TILE / 2 - cx) <= max(46.0, w * 1.45)
        and abs(b + TILE / 2 - cy) <= max(44.0, h * 1.45)
    ]


def local_change(reference, moving, a, b):
    template = reference[b : b + TILE, a : a + TILE]
    search = moving[b - 3 : b + TILE + 3, a - 3 : a + TILE + 3]
    if template.std() < 2.5:
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    return max(0.0, 1.0 - float(scores.max()))


def edge_features(i, pictures, tiles):
    baseline = [(i - 5, i - 4), (i - 4, i - 3), (i + 3, i + 4), (i + 4, i + 5)]
    output = {}
    for k, (x, y, sector) in enumerate(tiles):
        context = [local_change(pictures[a], pictures[b], x, y) for a, b in baseline]
        change = local_change(pictures[i - 1], pictures[i], x, y)
        if change is not None and all(v is not None for v in context):
            residual = float(change - np.median(context))
            if residual >= 0.20:
                output[k] = residual, sector
    return output


def paired_edges(pictures, box, interval, check=lambda _: None):
    a, b = interval
    tiles = layout(box)
    start = range(max(6, a - 12), min(114, a + 12) + 1)
    end = range(max(6, b - 12), min(114, b + 12) + 1)
    scores = {}
    for i in sorted(set(start) | set(end)):
        check("outside-face cut context")
        scores[i] = edge_features(i, pictures, tiles)
    strongest = None
    for left in start:
        for right in end:
            if not 5 <= right - left <= 24 or 2 * (min(right, b) - max(left, a)) < min(right - left, b - a):
                continue
            first, last = scores[left], scores[right]
            common = sorted(first.keys() & last.keys())
            if min(len(first), len(last)) < 5 or len(common) < 4 or len({first[k][1] for k in common}) < 2:
                continue
            persistent = []
            for index in common:
                x, y, _ = tiles[index]
                stable = local_change(pictures[left - 1], pictures[right], x, y)
                if stable is not None and stable <= 0.16:
                    persistent.append((min(first[index][0], last[index][0]), index))
            if len(persistent) < 5 or len({first[k][1] for _, k in persistent}) < 2:
                continue
            response = sorted((v for v, _ in persistent), reverse=True)[4]
            rank = response, len(persistent), -left, -right
            if strongest is None or rank > strongest[0]:
                strongest = (
                    rank,
                    {
                        "interval": [left, right],
                        "score": response,
                        "shared_tiles": len(persistent),
                        "entry_tiles": len(first),
                        "exit_tiles": len(last),
                        "sectors": len({first[k][1] for _, k in persistent}),
                    },
                )
    return strongest[1] if strongest else None, len(tiles)
