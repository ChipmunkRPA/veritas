# SPDX-License-Identifier: Apache-2.0
"""Portable bindings for a single face-appearance transition observation."""

from __future__ import annotations

import math
import re
from fractions import Fraction

KEY = "face_single_boundary"
COVERAGE = "face_single_boundary_coverage"
COMMITMENT = "face_single_boundary_sha256"
COVERAGE_COMMITMENT = "face_single_boundary_coverage_sha256"
COMMITMENTS = (COMMITMENT, COVERAGE_COMMITMENT)
METHOD = "single-face-boundary/1"
SPARSE_METHOD = "single-face-boundary/2"
NATIVE30_METHOD = "single-face-boundary/3"
NATIVE2997_METHOD = "single-face-boundary/4"
METHODS = {METHOD, SPARSE_METHOD, NATIVE30_METHOD, NATIVE2997_METHOD}
COARSE_CONTEXT, FINE_CONTEXT, MOTION_CONTEXT = "existing", "fine_tiles", "off_face_motion"
ALTERNATE_CONTEXTS = (MOTION_CONTEXT, FINE_CONTEXT)
PACKET_ID = "VFB01"
ROLES = ("before", "transition", "after")
WIDTH, HEIGHT = 640, 360
FRAME_COUNT, RATE = 100, 25
NATIVE30_FRAME_COUNT = 120
NATIVE30_TIME_BASE, NATIVE30_TICKS = "1/30000", 1000
NATIVE_CLOCK_TICKS = {NATIVE30_METHOD: NATIVE30_TICKS, NATIVE2997_METHOD: 1001}
COARSE_GRID, NATIVE30_COARSE_GRID = range(22, 79, 2), range(28, 93, 2)
POINT = 0.165750624
MIN_SCENE = .60
MINIMUM_FACE_IOU = .35
TITLE = "Face appearance transition"
OBSERVATION = "The face looked consistently different on the two sides of this source frame while the surrounding picture stayed similar."
SPARSE_OBSERVATIONS = {
    FINE_CONTEXT: "The face looked consistently different across this frame. A finer grid of textured areas outside the face supported continuity near the transition.",
    MOTION_CONTEXT: "The face looked consistently different across this frame. Near the transition, detail below the face stayed similar and the picture change was consistent with movement in nearby frames.",
}
INTERPRETATION = "Review the last frame before the transition, the first frame of the new appearance, and a later supporting frame. The cause and the affected side remain undetermined."
LIMITATION = "This check contributes zero weight to AI confidence. It looks for one change in a short continuous shot and can miss other edits."
UNAVAILABLE = {
    "provenance": "Verified source credentials determined the analysis path.",
    "profile": "This check covers four-second, 640 × 360 H.264 MP4 with a verified 25 fps clock.",
    "unsupported_video": "The original video could not be prepared for this check.",
    "models_unavailable": "The face appearance check was unavailable on this instance.",
    "time_budget": "The check did not finish within its local time limit.",
    "native_measurement_unavailable": "The native pictures could not be verified for this check.",
    "shot_change": "The clip contains a detected shot change. This check requires one continuous shot.",
    "insufficient_face_context": "The face views and surrounding picture did not support a stable comparison.",
    "insufficient_native_boundary": "The nearby source frames did not establish a precise transition.",
    "conflicting_boundary_context": "The background checks pointed to different source frames, so this check did not display a transition.",
}
_SHA = re.compile(r"[a-f0-9]{64}")
READINGS = ("scene", "persistent", "edge_supported", "jump_excess", "fine_margin", "fine_scene")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, lo, hi):
    return type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi


def _exact(pinned, displayed):
    _require(isinstance(pinned, str) and len(pinned) <= 32, "Invalid exact face transition value")
    try:
        value = float.fromhex(pinned)
    except (ValueError, OverflowError) as error:
        raise ValueError("Invalid exact face transition value") from error
    _require(value.hex() == pinned and math.isfinite(value) and type(displayed) in (float, int)
             and math.isclose(value, displayed, abs_tol=.000000500001, rel_tol=0),
             "Face transition display differs from its exact recorded value")
    return value


def eyes(row):
    originals, display = row["eyes_hex"], row["eyes"]
    _require(len(originals) == len(display) == 2 and all(len(point) == 2 for point in (*originals, *display)),
             "Face transition needs two exact eye locations")
    return tuple(tuple(_exact(raw, printed) for raw, printed in zip(pinned, shown, strict=True))
                 for pinned, shown in zip(originals, display, strict=True))


def reading(record, name):
    return _exact(record["readings_hex"][name], record[name])


def native30_clock(pts, method=NATIVE30_METHOD):
    ticks = NATIVE_CLOCK_TICKS[method]
    return len(pts) == NATIVE30_FRAME_COUNT and all(value == index * ticks for index, value in enumerate(pts))


def native30_source(table, encoding, carrier, method=NATIVE30_METHOD):
    return (table is not None and tuple(encoding) == ("h264", "yuv420p", "avc1") and carrier == "mp4"
            and (table.width, table.height) == (table.coded_width, table.coded_height) == (WIDTH, HEIGHT)
            and not (table.rotation or table.truncated or table.interpolated)
            and table.time_base == NATIVE30_TIME_BASE and native30_clock(table.pts, method))


def validate_clock(source, table_pts=None, *, interpolated=(), rotation=0, truncated=False, method=None):
    base = Fraction(source["time_base"])
    native30 = method in NATIVE_CLOCK_TICKS
    count = NATIVE30_FRAME_COUNT if native30 else FRAME_COUNT
    if native30:
        cadence = "29.97" if method == NATIVE2997_METHOD else "30"
        _require(source["time_base"] == NATIVE30_TIME_BASE and (table_pts is None or native30_clock(table_pts, method)),
                 f"Face transition differs from the verified {cadence} fps source clock")
    _require(base > 0 and source["width"] == WIDTH and source["height"] == HEIGHT
             and source["frame_count"] == count and not (rotation or truncated or interpolated)
             and (native30 or table_pts is None or len(table_pts) == count
                  and all((pts - table_pts[0]) * base * RATE == index for index, pts in enumerate(table_pts))),
             "Face transition differs from the verified short video clock")
    return base


def measurements(record):
    final = record.get("scene_context", {}).get("boundary")
    label = {FINE_CONTEXT: "Similarity across finer background tiles",
             MOTION_CONTEXT: "Picture similarity below the face"}.get(final, "Surrounding picture similarity at the transition")
    return [{"label": "Difference between sustained face appearances", "value": record["persistent"], "unit": "relative separation"},
            {"label": label, "value": record["fine_scene"], "unit": "correlation"}]


def observation(record):
    context = record.get("scene_context")
    return OBSERVATION if context is None else SPARSE_OBSERVATIONS[context["arm"]]


def _box(box):
    return (set(box) == {"x", "y", "w", "h"} and all(type(value) is int for value in box.values())
            and min(box["w"], box["h"]) >= 48 and 0 <= box["x"] <= WIDTH - box["w"]
            and 0 <= box["y"] <= HEIGHT - box["h"])


def coverage_text(coverage):
    status = coverage["status"]
    if status == "localized":
        return "One source frame marks a sustained face appearance transition."
    if status == "no_match":
        return "No qualifying face appearance transition was found in the portion checked."
    if coverage.get("method") == NATIVE30_METHOD and coverage["reason"] == "profile":
        return "This check covers four-second, 640 × 360 H.264 MP4 with a verified 25 fps or qualified 30 fps clock."
    if coverage.get("method") == NATIVE2997_METHOD and coverage["reason"] == "profile":
        return "This check requires a short 640 × 360 H.264 MP4 and a supported fixed source clock."
    return UNAVAILABLE[coverage["reason"]]


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    method = record.get("method")
    ticks = NATIVE_CLOCK_TICKS.get(method)
    native30 = ticks is not None
    sparse = record.get("method") == SPARSE_METHOD or native30 and "scene_context" in record
    base = validate_clock(source, table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated, method=method)
    fields = {"method", "coarse", "scene", "persistent", "edge_supported", "jump_excess", "boundary",
              "pts", "time_base", "time_s", "fine_margin", "fine_scene", "source_frames",
              "affects_overall_confidence", "readings_hex"}
    _require(set(record) == (fields | {"scene_context"} if sparse else fields)
             and record["method"] in METHODS and record["affects_overall_confidence"] is False
             and record["time_base"] == source["time_base"] and set(record["readings_hex"]) == set(READINGS),
             "Invalid face transition measurement")
    raw = {name: reading(record, name) for name in READINGS}
    coarse, boundary, pts = (record[key] for key in ("coarse", "boundary", "pts"))
    _require(type(coarse) is int and coarse in (NATIVE30_COARSE_GRID if native30 else COARSE_GRID) and type(boundary) is int
             and abs(boundary - coarse) <= 2 and type(pts) is int
             and (not native30 or pts == boundary * ticks)
             and type(record["time_s"]) in (float, int)
             and math.isclose(record["time_s"], float(pts * base), abs_tol=1e-6, rel_tol=0)
             and all(_number(raw[key], MIN_SCENE, 1.000001) for key in ("scene", "fine_scene"))
             and POINT < raw["persistent"] <= 2
             and _number(raw["jump_excess"], -2, 2) and _number(raw["edge_supported"], -2, 2)
             and raw["edge_supported"] == min(raw["jump_excess"], raw["persistent"])
             and 0 < raw["fine_margin"] <= 2,
             "Face transition is outside the frozen measurement range")
    if sparse:
        context = record["scene_context"]
        _require(set(context) == {"arm", "coarse", "boundary", "coarse_boxes"}
                 and context["arm"] in ALTERNATE_CONTEXTS
                 and all(context[key] in (COARSE_CONTEXT, context["arm"]) for key in ("coarse", "boundary"))
                 and context["arm"] in (context["coarse"], context["boundary"])
                 and len(context["coarse_boxes"]) == 2 and all(_box(box) for box in context["coarse_boxes"]),
                 "Invalid sparse scene context for face transition")
    rows = record["source_frames"]
    _require(set(rows) == set(ROLES), "Face transition must have three original frame records")
    for role, index in zip(ROLES, (boundary - 1, boundary, coarse + 11), strict=True):
        row = rows[role]
        _require(set(row) == {"index", "pts", "box", "eyes", "eyes_hex", "decoded_rgb_sha256", "aligned_rgb_sha256"}
                 and type(row["index"]) is int and row["index"] == index and type(row["pts"]) is int
                 and (row["pts"] - pts == (index - boundary) * ticks if native30 else
                      (row["pts"] - pts) * base * RATE == index - boundary)
                 and (table_pts is None or row["pts"] == table_pts[index])
                 and all(isinstance(row[key], str) and _SHA.fullmatch(row[key])
                         for key in ("decoded_rgb_sha256", "aligned_rgb_sha256")),
                 "Face transition frame does not match its exact source position")
        box, points = row["box"], eyes(row)
        _require(_box(box)
                 and all(_number(point[0], 0, WIDTH) and _number(point[1], 0, HEIGHT) for point in points),
                 "A face transition crop lies outside its native frame")


def _validate_packet(packet, record, source):
    _require(packet.get("id") == PACKET_ID and packet.get("semantics") == record["method"]
             and (record["method"] == METHOD or packet.get("derivation", {}).get("selection") == record["method"])
             and packet.get("claim_scope") == "sampled_window" and packet.get("evidence_type") == "anomaly"
             and packet.get("affects_overall_confidence") is False and packet.get("title") == TITLE
             and packet.get("observation") == observation(record) and packet.get("interpretation") == INTERPRETATION
             and packet.get("limitations") == [LIMITATION] and packet.get("measurements") == measurements(record)
             and packet.get("signal_refs") == [] and packet.get("finding_ids") == [] and "window" not in packet,
             "Face transition evidence must present a single confidence-neutral boundary")
    _require(len(packet["frames"]) == len(ROLES), "Face transition evidence has incomplete display frames")
    base, origin = Fraction(source["time_base"]), source.get("timeline_origin_s")
    for role, frame in zip(ROLES, packet["frames"], strict=True):
        frozen = record["source_frames"][role]
        anchor = frame["anchor"]
        stream = float(frozen["pts"] * base)
        refs = frame["artifact_refs"]
        _require(frame["id"] == f"{PACKET_ID}.{role}" and frame["interval_role"] == role
                 and frame["score"] is None and frame["is_peak"] is False
                 and frame["region_meaning"] == "detected_face_context" and frame["context_region"] == frozen["box"]
                 and anchor["frame_index"] == frozen["index"] and anchor["pts"] == frozen["pts"]
                 and anchor["decoded_rgb_sha256"] == frozen["decoded_rgb_sha256"]
                 and anchor["original_sha256"] == source["sha256"] and anchor["time_base"] == source["time_base"]
                 and anchor["stream_selector"] == "v:0" and (anchor["width"], anchor["height"]) == (WIDTH, HEIGHT)
                 and abs(anchor["stream_time_s"] - stream) <= 1e-6
                 and (anchor["playback_time_s"] is None if origin is None else
                      abs(anchor["playback_time_s"] - (stream - origin)) <= 1e-6)
                 and set(refs) == {"original", "context", "crop"}
                 and set(refs.values()) <= packet["artifacts"].keys(),
                 "Displayed face transition view differs from its native frame")


def validate(document, canonical_hash, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    source = document["source"]
    coverage = (document.get("coverage") or {}).get(COVERAGE)
    found = [packet for packet in document["packets"]
             if KEY in packet or packet.get("semantics") in METHODS or packet["id"] == PACKET_ID]
    record = found[0].get(KEY) if found else None
    _require(len(found) <= 1 and (not found or isinstance(record, dict))
             and (canonical_hash(record) if record is not None else None) == source.get(COMMITMENT)
             and (canonical_hash(coverage) if coverage is not None else None) == source.get(COVERAGE_COMMITMENT),
             "Face transition differs from its published commitments")
    if coverage is None:
        _require(not found, "Face transition evidence requires coverage")
        return
    status = coverage["status"]
    counts = coverage["frames_scanned"], coverage["measured_face_frames"]
    method = coverage.get("method")
    sparse, native30 = method == SPARSE_METHOD, method in NATIVE_CLOCK_TICKS
    frame_count, minimum_measured = (NATIVE30_FRAME_COUNT, 46) if native30 else (FRAME_COUNT, 38)
    base_fields = {"status", "reason", "frames_scanned", "measured_face_frames"}
    _require(set(coverage) == (base_fields | {"method"} if sparse or native30 else base_fields)
             and status in {"localized", "no_match", "unavailable"} and (status == "localized") == bool(found)
             and (coverage["reason"] in UNAVAILABLE if status == "unavailable" else coverage["reason"] is None)
             and (coverage["reason"] != "conflicting_boundary_context" or sparse or native30)
             and all(type(count) is int for count in counts) and 0 <= counts[0] <= frame_count
             and 0 <= counts[1] <= (counts[0] + 1) // 2
             and (status == "unavailable" or counts[0] == frame_count and counts[1] >= minimum_measured),
             "Invalid face transition coverage")
    if status != "unavailable" or counts[0] == frame_count or native30 and counts[0] > 0:
        validate_clock(source, table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated, method=method)
    if record is not None:
        _require((not native30 and record.get("method") not in NATIVE_CLOCK_TICKS) or
                 (native30 and record.get("method") == method), "Native cadence face transition requires matching coverage")
        _require(record.get("method") != SPARSE_METHOD or sparse, "Sparse face transition requires matching coverage")
        validate_record(record, source, table_pts=table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated)
        _validate_packet(found[0], record, source)
