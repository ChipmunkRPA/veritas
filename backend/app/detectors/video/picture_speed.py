"""Find two native spans with the same motion at a two-to-one playback ratio."""
from __future__ import annotations

import math

import numpy as np

from app.evidence.picture_sequence import MIN_SPAN, MOTION_PARAMETERS


def coarse_grid(values):
    """Average adjacent cells of the native 40 by 30 luminance grid."""
    return np.stack(values).reshape(-1, 15, 2, 20, 2).mean(axis=(2, 4)).reshape(-1, 300)


def _linked_movement(values, parameters):
    norm = (values * values).mean(axis=1)
    distances = norm[:, None] + norm[None, :] - 2 * values @ values.T / values.shape[1]
    close = distances <= parameters["maximum_picture_rms"] ** 2

    slow = values[2:] - values[:-2]
    fast = values[1:] - values[:-1]
    slow -= slow.mean(axis=1, keepdims=True)
    fast -= fast.mean(axis=1, keepdims=True)
    slow_rms = np.sqrt((slow * slow).mean(axis=1))[:, None]
    fast_rms = np.sqrt((fast * fast).mean(axis=1))[None, :]
    weaker = np.minimum(slow_rms, fast_rms)
    stronger = np.maximum(slow_rms, fast_rms)
    similarity = (slow @ fast.T / values.shape[1]) / np.maximum(slow_rms * fast_rms, 1e-12)
    moving = ((weaker >= parameters["minimum_motion_rms"])
              & (similarity >= parameters["minimum_motion_cosine"])
              & (weaker >= parameters["minimum_motion_rms_ratio"] * stronger))
    linked = close[:-2, :-1] & close[2:, 1:]
    return linked, moving & linked


def find(values, frame_rate, *, parameters=MOTION_PARAMETERS):
    linked, supported = _linked_movement(np.stack(values), parameters)
    minimum = parameters["minimum_supporting_edges"]
    min_edges = math.ceil(frame_rate * MIN_SPAN)
    max_step = parameters["maximum_support_edge_step"]
    best, match = None, None
    for parity in range(2):
        connections = linked[parity::2]
        support_axes = supported[parity::2]
        # A diagonal advances two native pictures in the slow span and one in the fast span.
        for lag in range(min_edges - len(connections), connections.shape[1] - min_edges + 1):
            points = np.flatnonzero(support_axes.diagonal(lag))
            if len(points) < minimum:
                continue
            links = connections.diagonal(lag)
            missing = np.cumsum(~links)
            breaks = (np.diff(points) > max_step) | (np.diff(missing[points]) > 0)
            slow_start, fast_start = 2 * max(0, -lag) + parity, max(0, lag)
            for chain in np.split(points, np.flatnonzero(breaks) + 1):
                for index in range(len(chain) - minimum + 1):
                    left = int(chain[index])
                    slow, fast = slow_start + 2 * left, fast_start + left
                    if slow < fast:
                        right_limit = (fast - slow_start - 1) // 2 - 1
                        spans = (slow, 2), (fast, 1)
                    else:
                        right_limit = slow - fast_start - 2
                        spans = (fast, 1), (slow, 2)
                    stop = int(np.searchsorted(chain, right_limit, side="right"))
                    if stop - index < minimum:
                        continue
                    edges = int(chain[stop - 1]) - left + 1
                    if edges < min_edges or 2 * (stop - index) < edges:
                        continue
                    rank = (edges, stop - index, -spans[0][0], -spans[1][0])
                    if best is None or rank > best:
                        best = rank
                        match = spans, edges + 1, [int(offset) - left for offset in chain[index:stop]]
    return match
