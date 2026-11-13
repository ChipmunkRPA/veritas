# SPDX-License-Identifier: Apache-2.0
"""Source and arithmetic commitments for the optional paired face timing estimate."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from app.evidence.face_interval_binding import APPEARANCE_METHOD, validate_face_input

METHOD = "face-appearance-paired-edge/1"
ENCODING = ("hevc", "yuv420p", "hvc1")
FRAME_COUNT, RATE = 120, 30
SIZE = (1280, 720)
MARGIN, MAX_READS = 12, 52
COARSE_FRAMES = 16
POINT = 0.03176017255556951
OFFSETS = (-5, -4, -3, 0, 3, 4, 5)
ROLES = ("before", "during", "after")
_SHA = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule(first, last):
    return list(range(max(0, first - MARGIN - 6), min(FRAME_COUNT, last + MARGIN + 6)))


def motion_inputs(edge, planned):
    return sorted(
        {
            i
            for offset in OFFSETS
            if {edge + offset - 1, edge + offset} <= planned
            for i in (edge + offset - 1, edge + offset)
        }
    )


def supported(table, encoding, carrier):
    return (
        tuple(encoding) == ENCODING
        and carrier == "mp4"
        and (table.width, table.height) == SIZE
        and (table.coded_width, table.coded_height) == SIZE
        and len(table) == FRAME_COUNT
        and not (table.rotation or table.truncated or table.interpolated)
        and all((pts - table.pts[0]) * table.tb == Fraction(i, RATE) for i, pts in enumerate(table.pts))
    )


def box_overlap(a, b):
    width = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    height = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    common = width * height
    return common / (a["w"] * a["h"] + b["w"] * b["h"] - common)


def _exact(text, low, high):
    require(isinstance(text, str) and len(text) <= 24, "Invalid paired face measurement")
    try:
        value = float.fromhex(text)
    except OverflowError as error:
        raise ValueError("Paired face measurement is outside its range") from error
    require(low <= value <= high, "Paired face measurement is outside its range")
    return value


def _frame_rows(rows, source, planned, first_pts, base, *, face=False):
    keys = {"frame_index", "pts", "decoded_rgb_sha256"}
    if face:
        keys |= {
            "bbox",
            "landmarks",
            "alignment_matrix",
            "alignment_matrix_hex",
            "alignment_side",
            "aligned_size",
            "aligned_rgb_sha256",
        }
    by_index = {}
    for row in rows:
        i, pts = row["frame_index"], row["pts"]
        require(
            set(row) == keys
            and type(i) is int
            and type(pts) is int
            and i in planned
            and (not by_index or i > next(reversed(by_index)))
            and (pts - first_pts) * base == Fraction(i, RATE)
            and _SHA.fullmatch(row["decoded_rgb_sha256"]),
            "Paired face input differs from its source clock or hash",
        )
        if face:
            validate_face_input(row, source, appearance=True)
            require(
                min(row["bbox"]["w"], row["bbox"]["h"]) >= 48, "Paired face input is below its measured size"
            )
        by_index[i] = row
    return by_index


def validate_record(record, coarse, source, *, table_pts=None, interpolated=()):
    base = Fraction(source["time_base"])
    require(
        set(record)
        == {
            "method",
            "time_base",
            "track_id",
            "start_pts",
            "end_pts",
            "start_stream_s",
            "end_stream_s",
            "source_first_pts",
            "source_encoding",
            "source_carrier",
            "start_frame_index",
            "end_frame_index",
            "anchors",
            "scheduled_frame_indices",
            "frame_inputs",
            "scene_frame_inputs",
            "edges",
            "return_hex",
        }
        and record["method"] == METHOD
        and record["time_base"] == source["time_base"]
        and record["source_encoding"] == list(ENCODING)
        and record["source_carrier"] == "mp4"
        and (source["width"], source["height"]) == SIZE
        and source["frame_count"] == FRAME_COUNT
        and not interpolated
        and coarse["method"] == APPEARANCE_METHOD
        and record["track_id"] == coarse["track_id"],
        "Invalid paired face delivery or scope",
    )
    first_pts = record["source_first_pts"]
    require(
        type(first_pts) is int
        and (
            table_pts is None
            or (
                len(table_pts) == FRAME_COUNT
                and all((pts - first_pts) * base == Fraction(i, RATE) for i, pts in enumerate(table_pts))
            )
        ),
        "Paired face timing requires an exact 30 fps native clock",
    )
    coarse_frames = [(coarse[key] - first_pts) * base * RATE for key in ("start_pts", "end_pts")]
    require(
        all(0 <= i < FRAME_COUNT and i.denominator == 1 for i in coarse_frames)
        and coarse_frames[1] - coarse_frames[0] == COARSE_FRAMES,
        "Paired face timing requires a short coarse range",
    )
    first, last = map(int, coarse_frames)
    start, stop = (record[key] for key in ("start_frame_index", "end_frame_index"))
    require(
        all(type(i) is int and 5 <= i <= 115 for i in (start, stop))
        and 5 <= stop - start <= 16
        and abs(start - first) <= MARGIN
        and abs(stop - last) <= MARGIN
        and 2 * (min(stop, last) - max(start, first)) >= min(stop - start, last - first)
        and (start, stop) != (first, last),
        "Paired face range differs from its local search",
    )
    for key, i in (("start", start), ("end", stop)):
        pts, seconds = record[f"{key}_pts"], record[f"{key}_stream_s"]
        require(
            type(pts) is int
            and (pts - first_pts) * base == Fraction(i, RATE)
            and type(seconds) in (int, float)
            and math.isclose(seconds, float(pts * base), abs_tol=0.0000011, rel_tol=0),
            "Paired face boundary differs from its native timestamp",
        )
    planned = schedule(first, last)
    require(
        record["scheduled_frame_indices"] == planned, "Paired face timing does not match its native schedule"
    )
    faces = _frame_rows(record["frame_inputs"], source, planned, first_pts, base, face=True)
    scenes = _frame_rows(record["scene_frame_inputs"], source, planned, first_pts, base)
    require(
        set(scenes) == set(motion_inputs(start, set(planned))) | set(motion_inputs(stop, set(planned)))
        and all(
            faces[i]["decoded_rgb_sha256"] == scenes[i]["decoded_rgb_sha256"]
            for i in faces.keys() & scenes.keys()
        ),
        "Paired face motion omits native inputs",
    )
    require(set(record["edges"]) == {"entry", "exit"}, "Paired face timing needs two measured boundaries")
    distances = []
    for edge, key in ((start, "entry"), (stop, "exit")):
        proof = record["edges"][key]
        indices = list(range(edge - 3, edge + 3))
        require(
            set(proof)
            == {
                "frame_indices",
                "separation_hex",
                "dispersion_hex",
                "central_hex",
                "sector_ids",
                "sectors_hex",
            }
            and proof["frame_indices"] == indices
            and set(indices) <= faces.keys()
            and all(
                box_overlap(faces[a]["bbox"], faces[b]["bbox"]) >= 0.25 for a, b in zip(indices, indices[1:])
            ),
            "Paired face timing lacks consistent nearby face crops",
        )
        distances.append(
            _exact(proof["separation_hex"], 0.08, 2.000000000001)
            - _exact(proof["dispersion_hex"], 0.0, 2.000000000001)
        )
        central = _exact(proof["central_hex"], 3.0, 255.0)
        sectors = [_exact(value, 0.0, 255.0) for value in proof["sectors_hex"]]
        ids = proof["sector_ids"]
        require(
            2 <= len(sectors) == len(ids) <= 4
            and all(type(i) is int and 1 <= i <= 4 for i in ids)
            and ids == sorted(set(ids))
            and sum(value >= max(6.0, central * 0.5) for value in sectors) < 2,
            "Near-head motion does not support an isolated boundary",
        )
    gaps = [_exact(value, 0.0, 2.000000000001) for value in record["return_hex"]]
    require(
        len(gaps) == 4 and min(*distances, min(gaps[:2]) - max(gaps[2:])) > POINT,
        "Paired face timing does not meet its fixed support point",
    )
    middle = min((i for i in faces if start <= i < stop), key=lambda i: (abs(2 * i - start - stop), i))
    require(
        record["anchors"] == dict(zip(ROLES, (start - 1, middle, stop), strict=True))
        and all(type(i) is int for i in record["anchors"].values()),
        "Paired face timing lacks its three native roles",
    )
