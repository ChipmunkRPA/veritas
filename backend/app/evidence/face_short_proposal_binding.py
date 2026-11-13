# SPDX-License-Identifier: Apache-2.0
"""Portable commitments for a brief native face proposal with a pixel shortlist."""

from __future__ import annotations

from fractions import Fraction

from app.evidence import face_paired_edge_binding as paired

METHOD = "face-short-return/9"
POINT = 0.10
ANCHOR_SAMPLES = tuple(range(4, 120, 10))
EDGE_SCAN = range(6, 115)
DURATIONS = (5, 6, 7)
MAX_EDGES = 16
MAX_FACE_READS = 84
PLAN = set(range(paired.FRAME_COUNT))


def edge_frames(edge):
    return list(range(edge - 3, edge + 3))


def face_schedule(edges):
    return sorted(set(ANCHOR_SAMPLES) | {i for edge in edges for i in edge_frames(edge)})


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    require = paired.require
    base = Fraction(source["time_base"])
    first_pts = record["source_first_pts"]
    require(
        set(record)
        == {
            "method",
            "time_base",
            "start_pts",
            "end_pts",
            "start_stream_s",
            "end_stream_s",
            "start_frame_index",
            "end_frame_index",
            "source_first_pts",
            "source_encoding",
            "source_carrier",
            "response",
            "response_hex",
            "return_hex",
            "edges",
            "anchors",
            "scheduled_frame_indices",
            "shortlist_edges",
            "shortlist_anchor_indices",
            "frame_inputs",
            "scene_frame_inputs",
        }
        and record["method"] == METHOD
        and record["time_base"] == source["time_base"]
        and type(first_pts) is int
        and record["source_encoding"] == list(paired.ENCODING)
        and record["source_carrier"] == "mp4"
        and (source["width"], source["height"]) == paired.SIZE
        and source["frame_count"] == paired.FRAME_COUNT
        and not (rotation or truncated or interpolated)
        and (
            table_pts is None
            or (
                len(table_pts) == paired.FRAME_COUNT
                and all(
                    (pts - first_pts) * base == Fraction(i, paired.RATE) for i, pts in enumerate(table_pts)
                )
            )
        ),
        "Brief face proposal requires its measured HEVC source and exact native clock",
    )
    start, stop = (record[key] for key in ("start_frame_index", "end_frame_index"))
    edges = record["shortlist_edges"]
    require(
        isinstance(edges, list)
        and 2 <= len(edges) <= MAX_EDGES
        and len(edges) == len(set(edges))
        and all(type(i) is int and i in EDGE_SCAN for i in edges)
        and start in edges
        and stop in edges
        and type(start) is int
        and type(stop) is int
        and stop - start in DURATIONS,
        "Brief face proposal has an invalid edge shortlist",
    )
    planned = face_schedule(edges)
    require(
        record["scheduled_frame_indices"] == planned and len(planned) <= MAX_FACE_READS,
        "Brief face proposal exceeds its face sample budget",
    )
    faces = paired.frame_rows(record["frame_inputs"], source, planned, first_pts, base, face=True)
    scenes = paired.frame_rows(record["scene_frame_inputs"], source, PLAN, first_pts, base)
    require(
        set(scenes) == PLAN
        and all(scenes[i]["decoded_rgb_sha256"] == row["decoded_rgb_sha256"] for i, row in faces.items()),
        "Brief face proposal omits native pixel inputs",
    )
    observed = [i for i in ANCHOR_SAMPLES if i in faces]
    require(
        record["shortlist_anchor_indices"] == observed and len(observed) >= 4,
        "Brief face proposal lacks original shortlist anchors",
    )
    for key, i in (("start", start), ("end", stop)):
        pts = scenes[i]["pts"]
        require(
            type(record[f"{key}_pts"]) is int
            and record[f"{key}_pts"] == pts
            and abs(record[f"{key}_stream_s"] - float(pts * base)) <= 0.0000011,
            "Brief face proposal differs from its exact source interval",
        )
    score = paired.edge_response(record, faces, start, stop)
    exact = paired.exact(record["response_hex"], POINT, 2.000000000001)
    require(
        score > POINT and exact == score and abs(record["response"] - exact) <= 0.0000011,
        "Brief face proposal does not meet its fixed support point",
    )
    paired.validate_roles(record, faces, start, stop)
