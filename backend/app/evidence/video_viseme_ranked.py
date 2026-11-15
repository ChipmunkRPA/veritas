# SPDX-License-Identifier: MIT
"""Compare within-clip aperture ranks on the original picture and speech clocks."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

from app.evidence import video_viseme_timing as timing

STRIDE, CENTER = 320, 200
UNIT = 1_000_000
WINDOW = (1, 2, 3, 2, 1)
MIN_SPEECH = .10
POINT = .7909019
WEIGHTS = (
    .515588377, -.015710879, .029394551, -.07808157, .04632143, .018076469, .081967463,
    -.236155085, .047360523, -.067209957, .189931054, .066073465, .001340154,
    -.136775247, .005181847, -.040716522, .140648065, .031661857, -.094361837,
)


def acoustic(groups):
    nearby = [None] * len(groups)
    for i in range(2, len(groups) - 2):
        interval = groups[i - 2:i + 3]
        values = [sum(weight * row[j] for weight, row in zip(WINDOW, interval, strict=True)) for j in range(5)]
        mass = sum(values)
        if mass:
            nearby[i] = [v / mass for v in values] + [mass / (UNIT * sum(WINDOW))]
    return nearby


def features(nearby, index):
    start, stop = index - 3, index + 3
    if start < 2 or stop >= len(nearby) - 2:
        return None
    values = [nearby[i] for i in (start, index, stop)]
    if any(point is None for point in values) or values[1][-1] < MIN_SPEECH:
        return None
    return [1.] + [value for point in values for value in point]


def predict(groups, fitted=WEIGHTS):
    nearby = acoustic(groups)
    predictions = []
    for i in range(len(groups)):
        row = features(nearby, i)
        estimate = None if row is None else min(1.5, max(-.5, math.fsum(a * b for a, b in zip(row, fitted, strict=True))))
        predictions.append(None if estimate is None else round(UNIT * estimate))
    return predictions


def middle_rank(ordered, reading):
    return (bisect_left(ordered, reading) + bisect_right(ordered, reading)) / (2 * len(ordered))


def scores(row, prediction, *, lags=timing.LAGS):
    points = timing.observations(row)
    valid = sorted(p for p in prediction if p is not None)
    if points is None or len(valid) < 8 or valid[-1] == valid[0]:
        return []
    _, _, indices = points
    source_gaps = {frame["index"]: frame["gap"] for frame in row["frames"] if frame["gap"] is not None}
    raw = [source_gaps[index] for index in indices]
    sorted_gaps = sorted(raw)
    ranks = dict(zip(indices, (middle_rank(sorted_gaps, value) for value in raw), strict=True))
    result = []
    for i in range(2, len(prediction) - 2, 2):
        value = prediction[i]
        if value is None:
            continue
        sample = CENTER + i * STRIDE
        reads = [timing.nearest(points, sample - lag) for lag in lags]
        if any(item is None for item in reads):
            continue
        frames = [frame for _, frame in reads]
        target = middle_rank(valid, value)
        strengths = [round(-timing.SCORE_UNITS * ((ranks[j] - target) / timing.SCALE) ** 2) for j in frames]
        result.append((sample, "aperture", strengths, frames))
    return result
