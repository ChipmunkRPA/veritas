"""Source and range checks for a supplemental short face-change observation."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from app.evidence.face_interval_binding import face_context_box, validate_face_input
from app.evidence.video_packet_versions import (
    DENSE_DURATION_FACE_VERSION,
    DENSE_SHORT_FACE_VERSION,
    FINE_SCENE_FACE_VERSION,
    GUARDED_720P30_FACE_VERSION,
    GUARDED_CODEC_FACE_VERSION,
    SECOND_HD_FACE_VERSION,
    SECOND_SHORT_FACE_VERSION,
    SHORT_FACE_VERSION,
)

METHOD = "face-short-return/1"
DENSE_METHOD = "face-short-return/2"
DENSE_DURATION_METHOD = "face-short-return/3"
FINE_SCENE_METHOD = "face-short-return/4"
GUARDED_CODEC_METHOD = "face-short-return/5"
GUARDED_720P30_METHOD = "face-short-return/6"
SECOND_METHOD = "face-short-return/7"
SECOND_HD_METHOD = "face-short-return/8"
SECOND_PRIOR_METHODS = frozenset({METHOD, DENSE_METHOD, FINE_SCENE_METHOD})
SECOND_METHODS = frozenset({SECOND_METHOD, SECOND_HD_METHOD})
SECOND_MIN_GAP = 2
GUARDED_METHODS = frozenset({GUARDED_CODEC_METHOD, GUARDED_720P30_METHOD}) | SECOND_METHODS
SECOND_BY_VERSION = {SECOND_SHORT_FACE_VERSION: SECOND_METHOD, SECOND_HD_FACE_VERSION: SECOND_HD_METHOD}
PRIMARY_BY_SECOND = {SECOND_METHOD: SECOND_PRIOR_METHODS, SECOND_HD_METHOD: frozenset({GUARDED_720P30_METHOD})}
GUARDED_720P30_DIMENSIONS = (1280, 720)
GUARDED_720P30_STEP = Fraction(1, 30)
FINE_SCENE_FRAME_COUNT = 120
GUARDED_CODEC_FRAME_COUNT = 120
GUARDED_SCENE_QUARTILE_POINT = .60
COARSE_TILE_MIN = 8
FINE_TILE_MIN = 16
FINE_QUADRANT_MIN = 3
FINE_SECTOR_MIN = 4
FINE_MEDIAN_MIN = .90
FINE_LUMA_MAX = 8.0
FINE_CHROMA_MAX = 6.0
DENSE_ENCODING = ("h264", "yuv420p")
GUARDED_CODEC_CARRIERS = {("vp9", "yuv420p"): "webm", ("hevc", "yuv420p"): "mp4"}
DENSE_DIMENSIONS = (640, 360)
DENSE_FRAME_PROFILES = {80: DENSE_DURATION_METHOD, 120: DENSE_METHOD, 150: DENSE_DURATION_METHOD}
METHOD_VERSIONS = {METHOD: SHORT_FACE_VERSION, DENSE_METHOD: DENSE_SHORT_FACE_VERSION,
                   DENSE_DURATION_METHOD: DENSE_DURATION_FACE_VERSION, FINE_SCENE_METHOD: FINE_SCENE_FACE_VERSION,
                   GUARDED_CODEC_METHOD: GUARDED_CODEC_FACE_VERSION, GUARDED_720P30_METHOD: GUARDED_720P30_FACE_VERSION,
                   SECOND_METHOD: SECOND_SHORT_FACE_VERSION, SECOND_HD_METHOD: SECOND_HD_FACE_VERSION}
POINT = 0.10
SCENE_POINT = 0.55
ROLES = ("before", "during", "after")
PROFILE_STEPS = {
    (1920, 1080): (Fraction(1, 25), Fraction(1, 30)),
    (1280, 720): (Fraction(1, 25),),
    (640, 360): (Fraction(1, 25),),
}
_SHA = re.compile(r"[a-f0-9]{64}")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _exact(value, text, lower, upper):
    exact = float.fromhex(text)
    _require(type(value) in (int, float) and lower <= exact <= upper and math.isclose(exact, value, abs_tol=1e-6, rel_tol=0),
             "Short face-change measurement differs from its exact value")
    return exact


def _time(value, expected):
    return ((value is None) if expected is None else
            type(value) in (int, float) and math.isclose(value, expected, rel_tol=0, abs_tol=1.1e-6))


def observation(record):
    return (
        f"The face appearance changed briefly from {record['start_stream_s']:.3f} to "
        f"{record['end_stream_s']:.3f} s and then returned toward its surrounding appearance."
    )


def validate_distinct(record, incumbent):
    if incumbent is None:
        return
    left, right = record["start_pts"], record["end_pts"]
    a, b = incumbent["start_pts"], incumbent["end_pts"]
    _require(incumbent["time_base"] == record["time_base"]
             and 2 * max(0, min(right, b) - max(left, a)) < max(right, b) - min(left, a),
             "Supplemental face range duplicates the incumbent range")


def separated(left, right, before, after, base):
    gap = Fraction(SECOND_MIN_GAP, 25)
    return (before - right) * base >= gap or (left - after) * base >= gap


def validate_second(record, primary, *incumbents):
    _require(isinstance(primary, dict) and primary.get("method") in PRIMARY_BY_SECOND.get(record.get("method"), ()),
             "A second face range requires an earlier qualified brief face range")
    left, right = record["start_pts"], record["end_pts"]
    base = Fraction(record["time_base"])
    for other in (primary, *incumbents):
        if other is not None:
            _require(record["time_base"] == other["time_base"]
                     and separated(left, right, other["start_pts"], other["end_pts"], base),
                     "The second face range overlaps or is too close to an earlier range")


def validate_set(document):
    packets = [packet for packet in document["packets"]
               if "face_return" in packet or packet.get("semantics") in METHOD_VERSIONS]
    selected = [packet.get("semantics") for packet in packets]
    if second := SECOND_BY_VERSION.get(document["version"]):
        _require(len(packets) == 2 and selected[0] in PRIMARY_BY_SECOND[second] and selected[1] == second
                 and packets[0].get("signal_refs") == packets[1].get("signal_refs"),
                 "The evidence version requires two qualified ranges from the same review")
        prior = [packet.get(key) for packet in document["packets"] for key in ("face_sequence", "face_boundary")
                 if packet.get(key)]
        validate_second(packets[1]["face_return"], packets[0]["face_return"], *prior)
        return
    expected = [method for method, version in METHOD_VERSIONS.items() if version == document["version"]]
    _require(selected == expected, "The evidence version disagrees with its supplemental face range")


def _native_inputs(items, source, table_pts, interpolated):
    by_index = {}
    last_pts = None
    for item in items:
        i, pts = item["frame_index"], item["pts"]
        _require(type(i) is int and type(pts) is int and 0 <= i < source["frame_count"] and i not in interpolated
                 and (not by_index or i > next(reversed(by_index))) and (last_pts is None or pts > last_pts)
                 and _SHA.fullmatch(str(item["decoded_rgb_sha256"]))
                 and (table_pts is None or table_pts[i] == pts), "Short face-change input lacks an exact source timestamp")
        by_index[i] = item
        last_pts = pts
    return by_index


def _validate_scene_boxes(faces, scene, dimensions):
    for item in scene.values():
        box = item["face_box"]
        if box is not None:
            _require(isinstance(box, list) and len(box) == 4 and all(type(value) is int for value in box),
                     "Fine scene face exclusion needs native integer coordinates")
            x, y, w, h = box
            _require(0 <= x < x + w <= dimensions[0] and 0 <= y < y + h <= dimensions[1],
                     "Fine scene face exclusion exceeds the native picture")
        if observed := faces.get(item["frame_index"]):
            bbox = observed["bbox"]
            _require(box == [bbox[key] for key in ("x", "y", "w", "h")],
                     "Fine scene exclusion disagrees with the measured face box")


def _validate_fine_scene(record):
    edges = record["fine_scene_edges"]
    _require(isinstance(edges, list) and len(edges) == 2 and any(edge is not None for edge in edges),
             "Fine scene support must supplement at least one sampled boundary")
    for edge, correlation in zip(edges, record["scene_correlations_hex"], strict=True):
        if edge is None:
            continue
        _require(isinstance(edge, dict) and set(edge) == {
            "coarse_tiles", "fine_tiles", "qualified_quadrants", "sectors", "luma_p90_hex", "chroma_p90_hex",
        }, "Fine scene support has incomplete boundary measurements")
        quadrants = edge["qualified_quadrants"]
        _require(type(edge["coarse_tiles"]) is int and 0 <= edge["coarse_tiles"] < COARSE_TILE_MIN
                 and type(edge["fine_tiles"]) is int and FINE_TILE_MIN <= edge["fine_tiles"] <= 128
                 and type(edge["sectors"]) is int and FINE_SECTOR_MIN <= edge["sectors"] <= 8
                 and isinstance(quadrants, list) and all(type(q) is int and 0 <= q < 4 for q in quadrants)
                 and quadrants == sorted(set(quadrants)) and len(quadrants) >= FINE_QUADRANT_MIN
                 and FINE_MEDIAN_MIN <= float.fromhex(correlation) <= 1.00001
                 and 0 <= float.fromhex(edge["luma_p90_hex"]) <= FINE_LUMA_MAX
                 and 0 <= float.fromhex(edge["chroma_p90_hex"]) <= FINE_CHROMA_MAX,
                 "Fine scene support does not meet the measured background requirements")


def _validate_guarded_scene(record):
    values = record["scene_lower_quartiles"]
    exact = record["scene_lower_quartiles_hex"]
    _require(isinstance(values, list) and isinstance(exact, list) and len(values) == len(exact) == 2,
             "Guarded face evidence needs two background quartiles")
    for value, hex_value, correlation in zip(values, exact, record["scene_correlations_hex"], strict=True):
        _exact(value, hex_value, GUARDED_SCENE_QUARTILE_POINT, min(1.00001, float.fromhex(correlation) + 1e-6))


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    method = record["method"]
    _require(method in METHOD_VERSIONS and record["time_base"] == source["time_base"],
             "Short face-change method or source clock differs")
    dimensions = (source["width"], source["height"])
    hd = method in (GUARDED_720P30_METHOD, SECOND_HD_METHOD)
    allowed_steps = (GUARDED_720P30_STEP,) if hd else PROFILE_STEPS.get(dimensions)
    _require(allowed_steps is not None and not (rotation or truncated or interpolated),
             "Short face-change evidence requires a supported upright source and complete native timestamps")
    base = Fraction(source["time_base"])
    a, b = record["start_pts"], record["end_pts"]
    _require(type(a) is int and type(b) is int and 0 < (b - a) * base < Fraction(1, 2)
             and abs(record["start_stream_s"] - float(a * base)) <= 1e-6
             and abs(record["end_stream_s"] - float(b * base)) <= 1e-6,
             "Short face-change interval is inconsistent with its exact half-open time range")
    _require(_exact(record["response"], record["response_hex"], 0, 2) > POINT,
             "Short face-change response is below its review point")
    scene_values, scene_hex = record["scene_correlations"], record["scene_correlations_hex"]
    _require(len(scene_values) == len(scene_hex) == 2, "Short face-change evidence needs two scene comparisons")
    for value, exact in zip(scene_values, scene_hex, strict=True):
        _exact(value, exact, SCENE_POINT, 1.00001)
    groups = record["support_indices"]
    _require(set(groups) == set(ROLES), "Short face-change evidence needs before, during and after support")
    before, during, after = (groups[role] for role in ROLES)
    faces = _native_inputs(record["frame_inputs"], source, table_pts, interpolated)
    scheduled = record["scheduled_frame_indices"]
    frame_count = source["frame_count"]
    maximum_samples = 75 if method == DENSE_DURATION_METHOD else 64
    _require(50 <= frame_count <= 240 and 11 <= len(scheduled) <= maximum_samples
             and all(type(i) is int and 0 <= i < frame_count for i in scheduled)
             and scheduled == sorted(set(scheduled)) and scheduled[0] == 0
             and not set(scheduled).intersection(interpolated), "Short face-change evidence has an invalid sample grid")
    if method != METHOD:
        _require(dimensions == (GUARDED_720P30_DIMENSIONS if hd else DENSE_DIMENSIONS)
                 and scheduled == list(range(0, frame_count, 2)),
                 "Dense short face-change evidence requires its measured native sample grid")
        encoding = tuple(record.get("source_encoding", ()))
        if hd:
            _require(frame_count == GUARDED_CODEC_FRAME_COUNT and encoding == DENSE_ENCODING and record.get("source_carrier") == "mp4",
                     "Guarded 720p30 face evidence requires 120 H.264 pictures in MP4")
        elif method == GUARDED_CODEC_METHOD:
            carrier = GUARDED_CODEC_CARRIERS.get(encoding)
            _require(frame_count == GUARDED_CODEC_FRAME_COUNT and carrier is not None and record.get("source_carrier") == carrier,
                     "Guarded short face-change evidence requires its measured VP9/WebM or HEVC/MP4 profile")
        elif method == SECOND_METHOD:
            _require(frame_count == GUARDED_CODEC_FRAME_COUNT and encoding == DENSE_ENCODING and record.get("source_carrier") == "mp4",
                     "A second face range requires 120 H.264 pictures in MP4")
        else:
            matches_profile = (frame_count == FINE_SCENE_FRAME_COUNT if method == FINE_SCENE_METHOD
                               else DENSE_FRAME_PROFILES.get(frame_count) == method)
            _require(matches_profile and encoding == DENSE_ENCODING,
                     "Dense short face-change evidence requires its measured H.264 delivery")
    entry = scheduled.index(record["start_frame_index"])
    exit_ = scheduled.index(record["end_frame_index"])
    core_size = exit_ - entry
    _require(entry >= 4 and exit_ + 4 <= len(scheduled) and core_size in (3, 4, 6)
             and 3 <= len(before) <= 4 and (4 if core_size == 6 else 3) <= len(during) <= core_size
             and 3 <= len(after) <= 4 and list(faces) == [*before, *during, *after]
             and set(before) <= set(scheduled[entry - 4:entry]) and set(during) <= set(scheduled[entry:exit_])
             and set(after) <= set(scheduled[exit_:exit_ + 4]), "Short face-change support groups are incomplete or unordered")
    if table_pts is not None:
        times = [(pts - table_pts[0]) * base for pts in table_pts]
        step = times[1] - times[0]
        _require(step in allowed_steps and 2 <= times[-1] + step <= 8
                 and all(t == i * step for i, t in enumerate(times)),
                 "Short face-change evidence requires a bounded native frame clock")
        if method == METHOD:
            targets = (Fraction(i, 8) for i in range(int(times[-1] * 8) + 1))
            exact_schedule = list(dict.fromkeys(min(range(len(times)), key=lambda i: (abs(times[i] - target), i)) for target in targets))
            _require(scheduled == exact_schedule, "Short face-change evidence differs from the source sample grid")
    for item in faces.values():
        validate_face_input(item, source, appearance=True)
    _require(faces[before[-1]]["pts"] < a <= faces[during[0]]["pts"] <= faces[during[-1]]["pts"] < b <= faces[after[0]]["pts"]
             and record["anchors"] == dict(zip(ROLES, (before[-1], during[len(during) // 2], after[0]), strict=True)),
             "Short face-change frame roles disagree with its interval")
    scene = _native_inputs(record["scene_frame_inputs"], source, table_pts, interpolated)
    indices = list(scene)
    _require(indices == [scheduled[entry - 1], scheduled[entry], scheduled[exit_ - 1], scheduled[exit_]]
             and (scene[indices[1]]["pts"], scene[indices[3]]["pts"]) == (a, b),
             "Short face-change scene samples disagree with its boundary positions")
    if hd or allowed_steps == (Fraction(1, 25),):
        # A verifier without the original source still checks the clock of every committed picture.
        first = before[0]
        first_pts = faces[first]["pts"]
        _require(source["frame_count"] <= 200 and all(
            (item["pts"] - first_pts) * base == (i - first) * allowed_steps[0]
            for i, item in (*faces.items(), *scene.items())),
            "Short face-change evidence differs from its exact source clock")
    for i in faces.keys() & scene.keys():
        _require(all(faces[i][key] == scene[i][key] for key in ("pts", "decoded_rgb_sha256")),
                 "Short face and scene measurements disagree on a source picture")
    if method in GUARDED_METHODS or method == FINE_SCENE_METHOD:
        _validate_scene_boxes(faces, scene, dimensions)
    else:
        _require(all("face_box" not in item for item in scene.values()),
                 "Scene exclusion boxes require their dedicated evidence version")
    if method == FINE_SCENE_METHOD:
        _validate_fine_scene(record)
    else:
        _require("fine_scene_edges" not in record, "Fine scene support requires its dedicated evidence version")
    if method in GUARDED_METHODS:
        _validate_guarded_scene(record)
    else:
        _require(not {"source_carrier", "scene_lower_quartiles", "scene_lower_quartiles_hex"} & record.keys(),
                 "Guarded codec support requires its dedicated evidence version")


def validate_packet(packet, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    record = packet.get("face_return")
    semantics = packet.get("semantics")
    if record is None and semantics not in METHOD_VERSIONS:
        return
    _require(isinstance(record, dict) and semantics in METHOD_VERSIONS and record.get("method") == semantics
             and packet.get("affects_overall_confidence") is False
             and packet.get("claim_scope") == "sampled_window" and len(packet["signal_refs"]) == 1,
             "Supplemental face evidence must remain a confidence-neutral sampled window")
    validate_record(record, source, table_pts=table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated)
    _require(len(packet["frames"]) == 3, "Supplemental face evidence needs three original frame views")
    inputs = {row["frame_index"]: row for row in record["frame_inputs"]}
    window = packet["window"]
    base = Fraction(source["time_base"])
    origin = source["timeline_origin_s"]
    _require(window["frame_indices"] == sorted(inputs.keys() | {row["frame_index"] for row in record["scene_frame_inputs"]})
             and all(window[key] == record[key] for key in ("time_base", "start_pts", "end_pts", "start_stream_s", "end_stream_s"))
             and all(_time(window[f"{edge}_playback_s"], None if origin is None else float(record[f"{edge}_pts"] * base) - origin)
                     for edge in ("start", "end")),
             "Supplemental face evidence does not match its displayed time range")
    for frame, role in zip(packet["frames"], ROLES, strict=True):
        i = record["anchors"][role]
        item = inputs[i]
        anchor = frame["anchor"]
        stream_s = float(item["pts"] * base)
        _require(frame["interval_role"] == role and frame["score"] is None and frame["is_peak"] is False
                 and anchor["frame_index"] == i and all(anchor[key] == item[key] for key in ("pts", "decoded_rgb_sha256"))
                 and anchor["time_base"] == record["time_base"] and _time(anchor["stream_time_s"], stream_s)
                 and _time(anchor["playback_time_s"], None if origin is None else stream_s - origin)
                 and frame["context_region"] == face_context_box(item["bbox"], source)
                 and {"original", "context", "crop"} <= set(frame["artifact_refs"]),
                 "Displayed short face-change image differs from its native sample")
