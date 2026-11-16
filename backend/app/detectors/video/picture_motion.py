"""Bounded repeated-picture comparison with matching signed motion."""
from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.video import picture_speed
from app.evidence import picture_sequence as binding


def profile(table, encoding, *, schema=binding.MOTION_SCHEMA):
    dimensions = table.width, table.height
    if (not binding.motion_encoding_allowed(schema, encoding) or table.rotation
            or dimensions != (table.coded_width, table.coded_height)):
        return None
    supported = binding.motion_profile(dimensions, schema)
    if supported is None or len(table) < supported[1]:
        return None
    if schema == binding.MOTION_EXTENDED_SCHEMA:
        rate, limit = supported
        tb, step = table.tb, Fraction(1, rate)
        if any(i < limit for i in table.interpolated) or any(
                (b - a) * tb != step for a, b in pairwise(table.pts[:limit])):
            return None
    return supported


def descriptor(rgb):
    height, width = rgb.shape[:2]
    means = rgb.reshape(30, height // 30, 40, width // 40, 3).mean(axis=(1, 3), dtype=np.float64)
    return (means @ np.array([77., 150., 29.]) / 256).reshape(-1)


def _segments(indices, step):
    for span in np.split(indices, np.flatnonzero(np.diff(indices) > step) + 1):
        if span.size:
            yield span


def _pair(values, frame_rate, *, reverse=False):
    params = binding.MOTION_PARAMETERS
    values = np.stack(values)
    norm = (values * values).mean(axis=1)
    frame_sq = np.maximum(0, norm[:, None] + norm[None, :] - 2 * values @ values.T / values.shape[1])
    close = frame_sq <= params["maximum_picture_rms"] ** 2
    motion = np.diff(values, axis=0)
    motion -= motion.mean(axis=1, keepdims=True)
    rms = np.sqrt((motion * motion).mean(axis=1))
    cos = (motion @ motion.T / values.shape[1]) / np.maximum(rms[:, None] * rms[None, :], 1e-12)
    weaker = np.minimum(rms[:, None], rms[None, :])
    stronger = np.maximum(rms[:, None], rms[None, :])
    if reverse:
        close = close[:, ::-1]
        cos = -cos[:, ::-1]
        weaker, stronger = weaker[:, ::-1], stronger[:, ::-1]
    linked = close[:-1, :-1] & close[1:, 1:]
    moving = ((weaker >= params["minimum_motion_rms"]) & (cos >= params["minimum_motion_cosine"])
              & (weaker >= params["minimum_motion_rms_ratio"] * stronger))
    best, result = None, None
    minimum = params["minimum_supporting_edges"]
    frames = len(values)
    lags = range(1-frames, frames-1) if reverse else range(minimum + 1, frames - minimum)
    for lag in lags:
        supports = moving.diagonal(lag)
        comparable = linked.diagonal(lag)
        first_index = max(0, -lag)
        if reverse:
            # In reverse coordinates both indices rise. Convert the second
            # back before either range can run into the other.
            available = max(0, (frames-2-lag)//2 - first_index)
            supports, comparable = supports[:available], comparable[:available]
        for segment in _segments(np.flatnonzero(comparable), 1):
            for chain in _segments(segment[supports[segment]], params["maximum_support_edge_step"]):
                for a, left in enumerate(chain[:max(0, len(chain) - minimum + 1)]):
                    start = int(left)
                    b = (len(chain) if reverse else int(np.searchsorted(chain, start + lag - 1, side="left"))) - 1
                    end = int(chain[b]) + 1
                    if b - a + 1 < minimum or 2 * (b - a + 1) < end - start:
                        continue
                    source_start, source_end = start + first_index, end + first_index
                    other = frames - 1 - source_end - lag if reverse else source_start + lag
                    rank = (end - start, -source_start, -other)
                    if end - start >= frame_rate * binding.MIN_SPAN and (best is None or rank > best):
                        best, result = rank, (source_start, source_end, other, [int(i) - start for i in chain[a:b + 1]])
    return result


def scan(observations, values, tb, encoding, exact_pair, *, source_profile, schema=binding.MOTION_SCHEMA):
    frame_rate, limit = source_profile
    result = {"schema": schema, "status": "unavailable", "reason": "source_clock", "frames_scanned": 0,
              "encoding": list(encoding), "parameters": binding.motion_parameters(schema, frame_rate), "pair": None}
    if len(observations) != limit:
        return result
    ticks, hashes = zip(*observations, strict=True)
    step = Fraction(1, frame_rate)
    if any(tick is None for tick in ticks) or any((b - a) * tb != step for a, b in pairwise(ticks)):
        return result
    if exact_pair is not None:
        return result | {"reason": "exact_sequence"}
    reverse = schema == binding.MOTION_REVERSE_SCHEMA
    retimed = schema == binding.MOTION_RETIMED_SCHEMA
    if retimed:
        match = picture_speed.find(values, frame_rate)
        if match is not None:
            occurrences, count, supports = match
    else:
        match = _pair(values, frame_rate, reverse=reverse)
        if match is not None:
            start, end, other, supports = match
            count = end - start + 1
            occurrences = (start, 1), (other, 1)
    if match is None:
        return result | {"status": "no_match", "reason": None, "frames_scanned": len(ticks)}
    spans = {}
    for name, (start, stride) in zip(("first", "second"), occurrences, strict=True):
        end = start + stride * (count - 1)
        pts = list(ticks[start:end + 1:stride])
        digests = hashes[start:end + 1:stride]
        spans[name] = {"start_frame_index": start, "end_frame_index": end, "pts": pts,
                       "start_stream_s": float(pts[0] * tb), "end_stream_s": float(pts[-1] * tb),
                       "endpoint_rgb_sha256": [digests[0], digests[-1]], "rgb_sequence_sha256": canonical_hash(digests)}
    pair = spans | {"frame_count": count, "span_s": float(Fraction(count - 1, frame_rate)), "supporting_edge_offsets": supports}
    if reverse:
        pair["order"] = "reverse"
    if retimed:
        pair["frame_steps"] = dict(zip(("first", "second"), (stride for _, stride in occurrences), strict=True))
    return result | {"status": "matched", "reason": None, "frames_scanned": len(ticks), "pair": pair}
