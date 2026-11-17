"""Find two native spans with the same motion at different playback speeds."""
from __future__ import annotations

import math

import numpy as np

from app.evidence.picture_sequence import MIN_SPAN, MOTION_PARAMETERS


def coarse_grid(values):
    """Average adjacent cells of the native 40 by 30 luminance grid."""
    return np.stack(values).reshape(-1, 15, 2, 20, 2).mean(axis=(2, 4)).reshape(-1, 300)


def _linked_movement(values, parameters, short_step, long_step):
    norm = (values * values).mean(axis=1)
    distances = norm[:, None] + norm[None, :] - 2 * values @ values.T / values.shape[1]
    close = distances <= parameters["maximum_picture_rms"] ** 2

    slow = values[long_step:] - values[:-long_step]
    fast = values[short_step:] - values[:-short_step]
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
    linked = close[:-long_step, :-short_step] & close[long_step:, short_step:]
    return linked, moving & linked


def _phase_matches(connections, support_axes, phases, strides, min_edges, minimum, max_step):
    slow_phase, fast_phase = phases
    long_step, short_step = strides
    for lag in range(min_edges - len(connections), connections.shape[1] - min_edges + 1):
        points = np.flatnonzero(support_axes.diagonal(lag))
        if len(points) < minimum:
            continue
        missing = np.cumsum(~connections.diagonal(lag))
        breaks = (np.diff(points) > max_step) | (np.diff(missing[points]) > 0)
        slow_start = long_step * max(0, -lag) + slow_phase
        fast_start = short_step * max(0, lag) + fast_phase
        for chain in np.split(points, np.flatnonzero(breaks) + 1):
            for index in range(len(chain) - minimum + 1):
                left = int(chain[index])
                slow, fast = slow_start + long_step * left, fast_start + short_step * left
                if slow < fast:
                    right_limit = (fast - slow_start - 1) // long_step - 1
                    spans = (slow, long_step), (fast, short_step)
                else:
                    right_limit = (slow - fast_start - 1) // short_step - 1
                    spans = (fast, short_step), (slow, long_step)
                stop = int(np.searchsorted(chain, right_limit, side="right"))
                if stop - index < minimum:
                    continue
                edges = int(chain[stop - 1]) - left + 1
                if edges >= min_edges and 2 * (stop - index) >= edges:
                    yield spans, edges + 1, [int(offset) - left for offset in chain[index:stop]]


def find(values, frame_rate, *, parameters=MOTION_PARAMETERS, steps=(1, 2)):
    short_step, long_step = steps
    linked, supported = _linked_movement(np.stack(values), parameters, short_step, long_step)
    minimum = parameters["minimum_supporting_edges"]
    min_edges = math.ceil(frame_rate * MIN_SPAN)
    max_step = parameters["maximum_support_edge_step"]
    matches = (
        match
        for slow_phase in range(long_step)
        for fast_phase in range(short_step)
        for match in _phase_matches(
            linked[slow_phase::long_step, fast_phase::short_step],
            supported[slow_phase::long_step, fast_phase::short_step],
            (slow_phase, fast_phase), (long_step, short_step), min_edges, minimum, max_step,
        )
    )
    return max(matches, key=lambda match: (match[1], len(match[2]), -match[0][0][0], -match[0][1][0]), default=None)


def fractional(values, frame_rate, *, parameters):
    """Prefer the longest verified frame range at a supported additional ratio."""
    matches = (find(values, frame_rate, parameters=parameters, steps=steps)
               for steps in parameters["playback_step_pairs"])
    return max((match for match in matches if match is not None),
               key=lambda match: ((match[1] - 1) * min(span[1] for span in match[0]), len(match[2])), default=None)
