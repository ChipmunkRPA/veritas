# SPDX-License-Identifier: Apache-2.0
"""Portable commitments for a brief native face proposal with a pixel shortlist."""

from __future__ import annotations

from fractions import Fraction

from app.evidence import face_paired_edge_binding as paired
from app.evidence import face_primary_track_binding
from app.evidence import scene_return_binding as webm

METHOD = "face-short-return/9"
WEBM_METHOD = "face-short-return/12"
WEBM_ENCODING = webm.WEBM_ENCODING
AVC_METHOD = "face-short-return/13"
SECOND_AVC_METHOD = "face-short-return/14"
MOVING_AVC_METHOD = "face-short-return/15"
PRIMARY_TRACK_AVC_METHOD = "face-short-return/16"
POSITION_TRACK_AVC_METHOD = "face-short-return/17"
PANE_TRACK_AVC_METHOD = "face-short-return/18"
SECOND_PANE_TRACK_AVC_METHOD = "face-short-return/19"
OPPOSITE_PANE_TRACK_AVC_METHOD = "face-short-return/20"
OVERLAPPING_PANE_TRACK_AVC_METHOD = "face-short-return/21"
BOUNDARY_AVC_METHOD = "face-short-return/22"
VERY_BRIEF_AVC_METHOD = "face-short-return/28"
AVC_ENCODING = ("h264", "yuv420p", "avc1")
AVC_FRAME_STEPS = (Fraction(1, 30), Fraction(1001, 30_000))
SOURCE_PROFILES = {METHOD: (paired.ENCODING, "mp4"), WEBM_METHOD: (WEBM_ENCODING, "webm"),
                   AVC_METHOD: (AVC_ENCODING, "mp4"), SECOND_AVC_METHOD: (AVC_ENCODING, "mp4"),
                   MOVING_AVC_METHOD: (AVC_ENCODING, "mp4"), PRIMARY_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"),
                   POSITION_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"), PANE_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"),
                   SECOND_PANE_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"), OPPOSITE_PANE_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"),
                   OVERLAPPING_PANE_TRACK_AVC_METHOD: (AVC_ENCODING, "mp4"), BOUNDARY_AVC_METHOD: (AVC_ENCODING, "mp4"),
                   VERY_BRIEF_AVC_METHOD: (AVC_ENCODING, "mp4")}
METHODS = frozenset(SOURCE_PROFILES)
OPPOSITE_PANE_METHODS = frozenset({OPPOSITE_PANE_TRACK_AVC_METHOD, OVERLAPPING_PANE_TRACK_AVC_METHOD})
PANE_METHODS = OPPOSITE_PANE_METHODS | {PANE_TRACK_AVC_METHOD, SECOND_PANE_TRACK_AVC_METHOD}
SECOND_METHODS = OPPOSITE_PANE_METHODS | {SECOND_AVC_METHOD, SECOND_PANE_TRACK_AVC_METHOD}
TRACK_METHODS = PANE_METHODS | {PRIMARY_TRACK_AVC_METHOD, POSITION_TRACK_AVC_METHOD}
MOVING_METHODS = TRACK_METHODS | {MOVING_AVC_METHOD}
POINT = 0.10
ANCHOR_SAMPLES = tuple(range(4, 120, 10))
EDGE_SCAN = range(6, 115)
DURATIONS = (5, 6, 7)
VERY_BRIEF_DURATIONS = (3, 4)
VERY_BRIEF_TICKS = tuple(range(0, 120000, 1000))
SECOND_AVC_DURATIONS = (5, 6, 7, 8, 9)
BOUNDARY_EDGES = frozenset((3, 4, 5, 115, 116, 117))
BOUNDARY_PAIRS = frozenset((start, start + length) for start in range(3, 118) for length in SECOND_AVC_DURATIONS
                           if start + length <= 117 and (start in BOUNDARY_EDGES or start + length in BOUNDARY_EDGES))
BOUNDARY_CANDIDATES = frozenset(edge for pair in BOUNDARY_PAIRS for edge in pair)
SECOND_AVC_GAP = Fraction(3, paired.RATE)
OPPOSITE_PANE_MIN_OVERLAP = Fraction(3, paired.RATE)
MAX_EDGES = 16
MAX_FACE_READS = 84
PLAN = set(range(paired.FRAME_COUNT))


def webm_source(table, encoding, carrier):
    return (tuple(encoding) == WEBM_ENCODING and carrier == "webm" and table.pts
            and (table.width, table.height) == (table.coded_width, table.coded_height) == paired.SIZE
            and not (table.rotation or table.truncated or table.interpolated)
            and webm.valid_webm_clock(table.time_base, table.pts, table.pts[0]))


def avc_source(table, encoding, carrier):
    return (tuple(encoding) == AVC_ENCODING and carrier == "mp4"
            and (table.width, table.height) == (table.coded_width, table.coded_height) == paired.SIZE
            and len(table) == paired.FRAME_COUNT and not (table.rotation or table.truncated or table.interpolated)
            and _avc_tick_step(table.tb, table.pts, table.pts[0]) is not None)


def very_brief_source(table, encoding, carrier):
    return (table.time_base == "1/30000" and table.pts == VERY_BRIEF_TICKS
            and avc_source(table, encoding, carrier))


def _avc_tick_step(base, pts, first):
    if len(pts) != paired.FRAME_COUNT or any(type(tick) is not int for tick in pts) or type(first) is not int:
        return None
    ticks = pts[1] - first
    if ticks * base in AVC_FRAME_STEPS and all(tick == first + index * ticks for index, tick in enumerate(pts)):
        return ticks
    return None


def exact_clock(base, pts, first):
    return len(pts) == paired.FRAME_COUNT and all(
        (tick - first) * base == Fraction(i, paired.RATE) for i, tick in enumerate(pts))


def edge_frames(edge):
    return list(range(edge - 3, edge + 3))


def face_schedule(edges):
    return sorted(set(ANCHOR_SAMPLES) | {i for edge in edges for i in edge_frames(edge)})


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    require = paired.require
    base = Fraction(source["time_base"])
    first_pts = record["source_first_pts"]
    is_webm = record["method"] == WEBM_METHOD
    encoding, carrier = SOURCE_PROFILES.get(record["method"], ((), None))
    relative_tick = None
    if is_webm:
        clock = base == webm.WEBM_TIME_BASE and (table_pts is None or webm.valid_webm_clock(base, table_pts, first_pts))
        relative_tick = webm.webm_tick
    elif encoding == AVC_ENCODING:
        pts = table_pts if table_pts is not None else [row["pts"] for row in record["scene_frame_inputs"]]
        step = _avc_tick_step(base, pts, first_pts)
        clock = step is not None

        def relative_tick(index):
            return index * step
    else:
        clock = table_pts is None or exact_clock(base, table_pts, first_pts)
    extra = {"face_detection_boxes"} if record["method"] in TRACK_METHODS else set()
    is_pane = record["method"] in PANE_METHODS
    very_brief = record["method"] == VERY_BRIEF_AVC_METHOD
    if very_brief:
        extra.add("shot_boundary_frames")
        pts = table_pts if table_pts is not None else [row["pts"] for row in record["scene_frame_inputs"]]
        clock = clock and record["time_base"] == "1/30000" and tuple(pts) == VERY_BRIEF_TICKS
    if is_pane:
        extra.update(("pane", "off_face_motion_tiles"))
    require(
        set(record)
        == extra | {
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
        and record["method"] in METHODS
        and record["time_base"] == source["time_base"]
        and type(first_pts) is int
        and record["source_encoding"] == list(encoding)
        and record["source_carrier"] == carrier
        and (source["width"], source["height"]) == paired.SIZE
        and source["frame_count"] == paired.FRAME_COUNT
        and not (rotation or truncated or interpolated)
        and clock,
        "Brief face proposal requires its measured carrier and exact native clock",
    )
    start, stop = (record[key] for key in ("start_frame_index", "end_frame_index"))
    edges = record["shortlist_edges"]
    near_boundary = record["method"] == BOUNDARY_AVC_METHOD
    allowed_edges = BOUNDARY_CANDIDATES if near_boundary else EDGE_SCAN
    durations = SECOND_AVC_DURATIONS if record["method"] in MOVING_METHODS | SECOND_METHODS else DURATIONS
    if very_brief:
        durations = VERY_BRIEF_DURATIONS
    require(
        isinstance(edges, list)
        and 2 <= len(edges) <= MAX_EDGES
        and len(edges) == len(set(edges))
        and all(type(i) is int and i in allowed_edges for i in edges)
        and start in edges
        and stop in edges
        and type(start) is int
        and type(stop) is int
        and ((start, stop) in BOUNDARY_PAIRS if near_boundary else stop - start in durations),
        "Brief face proposal has an invalid edge shortlist",
    )
    if very_brief:
        cuts = record["shot_boundary_frames"]
        require(isinstance(cuts, list) and all(type(i) is int and 0 < i < paired.FRAME_COUNT for i in cuts)
                and cuts == sorted(set(cuts)) and not any(abs(a - b) <= 1 for a in (start, stop) for b in cuts),
                "Very brief face range has conflicting source shot context")
    planned = face_schedule(edges)
    require(
        record["scheduled_frame_indices"] == planned and len(planned) <= MAX_FACE_READS,
        "Brief face proposal exceeds its face sample budget",
    )
    faces = paired.frame_rows(record["frame_inputs"], source, planned, first_pts, base, face=True, relative_tick=relative_tick)
    scenes = paired.frame_rows(record["scene_frame_inputs"], source, PLAN, first_pts, base, relative_tick=relative_tick)
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
    if record["method"] in TRACK_METHODS:
        pane = record.get("pane")
        if is_pane:
            tiles = record["off_face_motion_tiles"]
            require(pane in {"left", "right"} and isinstance(tiles, list) and len(tiles) == 2
                    and all(type(count) is int and 0 <= count < face_primary_track_binding.PANE_CHANGE_TILES
                            for count in tiles), "The selected range does not meet its pane motion guard")
        face_primary_track_binding.validate(record, source, planned, faces, ANCHOR_SAMPLES,
                                            position_only=record["method"] == POSITION_TRACK_AVC_METHOD, pane=pane)
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
