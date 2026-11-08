"""Independent source-block and temporal arithmetic for the context study."""
from __future__ import annotations

import numpy as np


def describe(rgb):
    # Integer integral sums, independently of the primary reshape/mean path.
    integral = np.pad(rgb.astype(np.uint64).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    rows = []
    for y in range(0, 480, 40):
        row = []
        for x in range(0, 640, 40):
            total = (integral[y + 40, x + 40].astype(np.int64) - integral[y, x + 40].astype(np.int64)
                     - integral[y + 40, x].astype(np.int64) + integral[y, x].astype(np.int64))
            row.append(total.astype(np.float64) / (1600 * 255))
        rows.append(row)
    return np.array(rows)


def measure(unit, descriptors, boxes):
    needed = {i for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]}
    if not needed <= descriptors.keys() or not needed <= boxes.keys():
        return None
    keep = []
    for y in range(12):
        for x in range(16):
            good = True
            for i in needed:
                a, b, w, h = boxes[i]
                left, top, right, bottom = max(0, a - w / 4), max(0, b - h / 4), min(640, a + 5 * w / 4), min(480, b + 5 * h / 4)
                if x * 40 < right and (x + 1) * 40 > left and y * 40 < bottom and (y + 1) * 40 > top:
                    good = False
                    break
            if good:
                keep.append((y, x))
    if len(keep) < 48:
        return None
    groups = []
    for key in ('before_centers', 'middle_centers', 'after_centers'):
        values = []
        for y, x in keep:
            for channel in range(3):
                values.append(sum(sum(float(descriptors[i + k][y, x, channel]) for k in (-1, 0, 1)) / 3
                                  for i in unit[key]) / len(unit[key]))
        groups.append(values)
    left, middle, right = groups
    def distance(a, b):
        return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)
    lm, rm, flank = distance(left, middle), distance(right, middle), distance(left, right)
    return {'raw': max(0., min(lm, rm) - .35 * flank), 'background_cells': len(keep)}


def intervals(locations):
    values = []
    for row in sorted(locations, key=lambda r: (r['start'], r['end'])):
        if values and row['start'] <= values[-1][1] + 550_000_000:
            values[-1][1] = max(values[-1][1], row['end'])
        else:
            values.append([row['start'], row['end']])
    return values


def iou(spans, truth):
    if truth is None:
        return None
    scores = [max(0, min(end, truth[1]) - max(start, truth[0])) /
              (max(end, truth[1]) - min(start, truth[0])) for start, end in spans]
    return max(scores, default=0.)
