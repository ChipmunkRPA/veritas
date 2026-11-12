"""Bounded repeated-picture comparison with matching signed motion."""
from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import numpy as np

from app.core.canonical import canonical_hash
from app.evidence import picture_sequence as binding


def profile(table, encoding):
    dimensions = table.width, table.height
    if (tuple(encoding or ()) not in binding.MOTION_ENCODINGS or table.rotation
            or dimensions != (table.coded_width, table.coded_height)):
        return None
    supported = binding.MOTION_PROFILES.get(dimensions)
    return supported if supported is not None and len(table) >= supported[1] else None


def descriptor(rgb):
    height, width = rgb.shape[:2]
    means = rgb.reshape(30, height // 30, 40, width // 40, 3).mean(axis=(1, 3), dtype=np.float64)
    return (means @ np.array([77., 150., 29.]) / 256).reshape(-1)


def _segments(indices, step):
    for span in np.split(indices, np.flatnonzero(np.diff(indices) > step) + 1):
        if span.size:
            yield span


def _pair(values, frame_rate):
    params = binding.MOTION_PARAMETERS
    values = np.stack(values)
    norm = (values * values).mean(axis=1)
    frame_sq = np.maximum(0, norm[:, None] + norm[None, :] - 2 * values @ values.T / values.shape[1])
    close = frame_sq <= params["maximum_picture_rms"] ** 2
    linked = close[:-1, :-1] & close[1:, 1:]
    motion = np.diff(values, axis=0)
    motion -= motion.mean(axis=1, keepdims=True)
    rms = np.sqrt((motion * motion).mean(axis=1))
    cos = (motion @ motion.T / values.shape[1]) / np.maximum(rms[:, None] * rms[None, :], 1e-12)
    weaker = np.minimum(rms[:, None], rms[None, :])
    stronger = np.maximum(rms[:, None], rms[None, :])
    moving = ((weaker >= params["minimum_motion_rms"]) & (cos >= params["minimum_motion_cosine"])
              & (weaker >= params["minimum_motion_rms_ratio"] * stronger))
    best, result = None, None
    minimum = params["minimum_supporting_edges"]
    for lag in range(minimum + 1, len(values) - minimum):
        supports = moving.diagonal(lag)
        for segment in _segments(np.flatnonzero(linked.diagonal(lag)), 1):
            for chain in _segments(segment[supports[segment]], params["maximum_support_edge_step"]):
                for a, left in enumerate(chain[:max(0, len(chain) - minimum + 1)]):
                    start = int(left)
                    b = int(np.searchsorted(chain, start + lag - 1, side="left")) - 1
                    end = int(chain[b]) + 1
                    if b - a + 1 < minimum or 2 * (b - a + 1) < end - start:
                        continue
                    rank = (end - start, -start, -start - lag)
                    if end - start >= frame_rate * binding.MIN_SPAN and (best is None or rank > best):
                        best, result = rank, (start, end, lag, [int(i) - start for i in chain[a:b + 1]])
    return result


def scan(observations, values, tb, encoding, exact_pair, *, source_profile):
    frame_rate, limit = source_profile
    result = {"schema": binding.MOTION_SCHEMA, "status": "unavailable", "reason": "source_clock", "frames_scanned": 0,
              "encoding": list(encoding), "parameters": binding.MOTION_PARAMETERS | {"frame_rate": frame_rate}, "pair": None}
    if len(observations) != limit:
        return result
    ticks, hashes = zip(*observations, strict=True)
    step = Fraction(1, frame_rate)
    if any(tick is None for tick in ticks) or any((b - a) * tb != step for a, b in pairwise(ticks)):
        return result
    if exact_pair is not None:
        return result | {"reason": "exact_sequence"}
    match = _pair(values, frame_rate)
    if match is None:
        return result | {"status": "no_match", "reason": None, "frames_scanned": len(ticks)}
    start, end, lag, supports = match
    count = end - start + 1
    spans = {}
    for name, i in zip(("first", "second"), (start, start + lag), strict=True):
        pts = list(ticks[i:i + count])
        digests = hashes[i:i + count]
        spans[name] = {"start_frame_index": i, "end_frame_index": i + count - 1, "pts": pts,
                       "start_stream_s": float(pts[0] * tb), "end_stream_s": float(pts[-1] * tb),
                       "endpoint_rgb_sha256": [digests[0], digests[-1]], "rgb_sequence_sha256": canonical_hash(digests)}
    return result | {"status": "matched", "reason": None, "frames_scanned": len(ticks),
                     "pair": spans | {"frame_count": count, "span_s": float((ticks[end] - ticks[start]) * tb),
                                       "supporting_edge_offsets": supports}}
