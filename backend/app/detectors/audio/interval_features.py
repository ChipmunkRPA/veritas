# SPDX-License-Identifier: Apache-2.0
"""Measure complete short speech intervals and their adjacent context."""

from __future__ import annotations

import numpy as np

RATE = 16000
SAMPLES = 6 * RATE
FRAME = 160
STEP = 800
WIDTHS = (4000, 5600, 8000, 9600, 12000, 16000)
FLANK_GAP = 8  # 80 ms, in frames.
ARMS = ("pooled", "contrast", "combined")

WINDOWS = np.array([(start, start + width) for width in WIDTHS for start in range(0, SAMPLES - width + 1, STEP)])
LEFT, RIGHT = WINDOWS.T // FRAME
COUNT = RIGHT - LEFT
GROUPS = np.searchsorted(np.cumsum([1 + (SAMPLES - width) // STEP for width in WIDTHS]), np.arange(len(WINDOWS)), side="right")


def _measure(sequence, left, right):
    zero = np.zeros((1, sequence.shape[1]), dtype=np.float64)
    cumulative = np.concatenate((zero, np.cumsum(sequence, axis=0, dtype=np.float64)))
    squares = np.concatenate((zero, np.cumsum(sequence.astype(np.float64)**2, axis=0)))
    motion = np.diff(sequence.astype(np.float64), axis=0, prepend=sequence[:1])
    changes = np.concatenate((zero, np.cumsum(motion**2, axis=0)))
    size = (right - left)[:, None]
    averages = (cumulative[right] - cumulative[left]) / size
    spread = (squares[right] - squares[left]) / size - averages**2
    increments = (changes[right] - changes[left + 1]) / (size - 1)
    return averages, np.sqrt(np.maximum(spread, 0.)), np.sqrt(np.maximum(increments, 0.))


def features(sequence):
    if sequence.shape != (64, SAMPLES // FRAME) or not np.isfinite(sequence).all():
        raise ValueError("Expected the 64 time-aligned features of one six-second view")
    time = sequence.T
    core = _measure(time, LEFT, RIGHT)
    previous_right = np.clip(LEFT - FLANK_GAP, COUNT, len(time))
    next_left = np.clip(RIGHT + FLANK_GAP, 0, len(time) - COUNT)
    previous_ok = LEFT - FLANK_GAP >= COUNT
    next_ok = np.less_equal(RIGHT + FLANK_GAP + COUNT, len(time))
    previous = _measure(time, previous_right - COUNT, previous_right)
    following = _measure(time, next_left, next_left + COUNT)
    other = [(before * previous_ok[:, None] + after * next_ok[:, None]) /
             np.maximum(1, previous_ok.astype(int) + next_ok.astype(int))[:, None]
             for before, after in zip(previous, following, strict=True)]
    contrast = np.concatenate([a - b for a, b in zip(core, other, strict=True)], axis=1)
    pooled = np.concatenate(core, axis=1)
    return np.ascontiguousarray(np.concatenate((pooled, contrast), axis=1), dtype=np.float32)


def admitted(spans):
    clock = np.arange(SAMPLES // FRAME) * FRAME
    coverage = np.zeros(clock.shape, dtype=np.float64)
    for start, stop in spans:
        coverage += np.clip(np.minimum(clock + FRAME, stop * RATE) - np.maximum(clock, start * RATE), 0, FRAME)
    cumulative = np.r_[0., np.cumsum(np.minimum(coverage, FRAME))]
    return 2 * (cumulative[RIGHT] - cumulative[LEFT]) >= COUNT * FRAME - .00001


def ranks(grid, head):
    weights = np.asarray(head["weights"], dtype=np.float32)
    return grid @ weights


def readings(grid, head, spans):
    available = admitted(spans)
    scores = ranks(grid, head)
    results = []
    for width in range(len(WIDTHS)):
        choices = np.flatnonzero(available & np.equal(GROUPS, width))
        if len(choices):
            index = int(choices[int(np.argmax(scores[choices]))])
            results.append({"range": WINDOWS[index].tolist(), "score": round(float(scores[index]), 8)})
        else:
            results.append(None)
    return results
