"""Bounded repeated-picture comparison with matching signed motion."""
from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.video import picture_speed
from app.evidence import picture_sequence as binding


def profile(table, encoding, *, schema=binding.MOTION_SCHEMA):
    dimensions = table.width, table.height
    if (not binding.motion_encoding_allowed(schema, encoding) or table.rotation
            or dimensions != (table.coded_width, table.coded_height)):
        return None
    if schema == binding.MOTION_REFLECTED_HD_SCHEMA:
        return binding.hd_profile(dimensions, table.pts, table.tb)
    supported = binding.motion_profile(dimensions, schema)
    if supported is None or len(table) < supported[1]:
        return None
    if schema in {binding.MOTION_EXTENDED_SCHEMA, *binding.MOTION_CROP_SCHEMAS}:
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


def hd_descriptor(rgb):
    grid = Image.fromarray(rgb).resize((40, 30), Image.Resampling.BOX)
    return (np.asarray(grid) @ np.array([77., 150., 29.]) / 256).reshape(-1)


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


def _reflected_pair(values, frame_rate, parameters):
    values = np.stack(values)
    columns, rows = parameters["luma_grid"]
    reflected = values.reshape(-1, rows, columns)[:, :, ::-1].copy().reshape(values.shape)
    norm = (values * values).mean(axis=1)
    summed_norm = norm[:, None] + norm[None, :]
    pixels = values.shape[1]
    direct_distance = summed_norm - 2 * (values @ values.T) / pixels
    reflected_distance = summed_norm - 2 * (values @ reflected.T) / pixels
    close = ((reflected_distance <= parameters["maximum_picture_rms"] ** 2)
             & (direct_distance >= parameters["minimum_unflipped_picture_rms"] ** 2))

    motion = np.diff(values, axis=0)
    motion -= motion.mean(axis=1, keepdims=True)
    reflected_motion = motion.reshape(-1, rows, columns)[:, :, ::-1].copy().reshape(motion.shape)
    rms = np.sqrt((motion * motion).mean(axis=1))
    stronger = np.maximum(rms[:, None], rms[None, :])
    weaker = np.minimum(rms[:, None], rms[None, :])
    cosine = (motion @ reflected_motion.T / pixels) / np.maximum(rms[:, None] * rms[None, :], 1e-12)
    supported = ((weaker >= parameters["minimum_motion_rms"])
                 & (cosine >= parameters["minimum_motion_cosine"])
                 & (weaker >= parameters["minimum_motion_rms_ratio"] * stronger))
    linked = close[:-1, :-1] & close[1:, 1:]

    minimum_edges = parameters["minimum_supporting_edges"]
    minimum_span = frame_rate * binding.MIN_SPAN
    best, result = None, None
    for lag in range(int(minimum_span) + 1, len(values) - int(minimum_span)):
        matches = linked.diagonal(lag)
        changes = supported.diagonal(lag)
        for segment in _segments(np.flatnonzero(matches), 1):
            for chain in _segments(segment[changes[segment]], parameters["maximum_support_edge_step"]):
                for first, start in enumerate(chain[:max(0, len(chain) - minimum_edges + 1)]):
                    # Both endpoints of the first range must precede the second range.
                    stop = int(np.searchsorted(chain, start + lag - 1, side="left"))
                    end = int(chain[stop - 1]) + 1
                    width = end - start
                    if (stop - first < minimum_edges or width < minimum_span
                            or stop - first < parameters["minimum_support_fraction"] * width):
                        continue
                    rank = width, -int(start), -int(start + lag)
                    if best is None or rank > best:
                        best = rank
                        result = int(start), end, int(start + lag), [int(i - start) for i in chain[first:stop]]
    return result


def scan(observations, values, tb, encoding, exact_pair, *, source_profile, schema=binding.MOTION_SCHEMA):
    frame_rate, limit = source_profile
    result = {"schema": schema, "status": "unavailable", "reason": "source_clock", "frames_scanned": 0,
              "encoding": list(encoding), "parameters": binding.motion_parameters(schema, frame_rate), "pair": None}
    if len(observations) != limit:
        return result
    ticks, hashes = zip(*observations, strict=True)
    step = Fraction(1, frame_rate)
    hd = schema == binding.MOTION_REFLECTED_HD_SCHEMA
    if hd:
        clock_ok = binding.hd_clock_matches(ticks, tb, frame_rate)
    else:
        clock_ok = all(tick is not None for tick in ticks) and all((b - a) * tb == step for a, b in pairwise(ticks))
    if not clock_ok:
        return result
    reflected = schema in binding.MOTION_REFLECTED_SCHEMAS
    cropped = schema in binding.MOTION_CROP_SCHEMAS
    if exact_pair is not None and not (reflected or cropped):
        return result | {"reason": "exact_sequence"}
    reverse = schema in {binding.MOTION_REVERSE_SCHEMA, binding.MOTION_CROP_REVERSE_SCHEMA}
    retimed = schema in binding.MOTION_RETIMED_SCHEMAS
    if cropped:
        match, alignment = _crop_match(values, frame_rate, schema)
        if match is not None:
            occurrences = tuple((start, 1) for start, _ in match.ranges)
            count = match.ranges[0][1] - match.ranges[0][0] + 1
            supports = list(match.support)
    elif retimed:
        if schema != binding.MOTION_RETIMED_SCHEMA:
            values = picture_speed.coarse_grid(values)
        compare = picture_speed.fractional if schema == binding.MOTION_RETIMED_FRACTIONAL_SCHEMA else picture_speed.find
        match = compare(values, frame_rate, parameters=result["parameters"])
        if match is not None:
            occurrences, count, supports = match
    else:
        match = (_reflected_pair(values, frame_rate, result["parameters"]) if reflected
                 else _pair(values, frame_rate, reverse=reverse))
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
    first = spans["first"]["pts"]
    span = (first[-1] - first[0]) * tb if hd else (count - 1) * step
    pair = spans | {"frame_count": count, "span_s": float(span), "supporting_edge_offsets": supports}
    if reverse:
        pair["order"] = "reverse"
    if retimed:
        pair["frame_steps"] = dict(zip(("first", "second"), (stride for _, stride in occurrences), strict=True))
    if reflected:
        pair["spatial_transform"] = binding.HORIZONTAL_REFLECTION
    if cropped:
        pair["framing"] = {"wider_range": match.wider_range, "crop_box": list(match.crop_box)}
        if alignment:
            pair["framing"]["alignment"] = alignment
    return result | {"status": "matched", "reason": None, "frames_scanned": len(ticks), "pair": pair}


def _crop_match(values, frame_rate, schema):
    if schema == binding.MOTION_CROP_SCHEMA:
        from app.detectors.video.picture_crop import find

        return find(values, frame_rate), None
    if schema == binding.MOTION_CROP_REVERSE_SCHEMA:
        from app.detectors.video.picture_crop_reverse import find

        return find(values)
    from app.detectors.video.picture_crop_aligned import find as aligned

    matches = [(match, method) for method, match in aligned(values).items() if match]
    return max(matches, key=lambda row: (row[0].ranges[0][1] - row[0].ranges[0][0], len(row[0].support)),
               default=(None, None))
