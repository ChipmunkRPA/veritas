"""Source and range checks for a supplemental short face-change observation."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from app.evidence.face_interval_binding import face_context_box, validate_face_input
from app.evidence.video_packet_versions import SHORT_FACE_VERSION

METHOD = "face-short-return/1"
POINT = 0.10
SCENE_POINT = 0.55
ROLES = ("before", "during", "after")
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


def validate_set(document):
    count = sum("face_return" in packet or packet.get("semantics") == METHOD for packet in document["packets"])
    _require(count == int(document["version"] == SHORT_FACE_VERSION),
             "The evidence version disagrees with its supplemental face range")


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


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    _require(record["method"] == METHOD and record["time_base"] == source["time_base"],
             "Short face-change method or source clock differs")
    _require((source["width"], source["height"]) == (1920, 1080) and not (rotation or truncated or interpolated),
             "Short face-change evidence requires an upright 1080p source and complete native timestamps")
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
    _require(50 <= source["frame_count"] <= 240 and 11 <= len(scheduled) <= 64
             and all(type(i) is int and 0 <= i < source["frame_count"] for i in scheduled)
             and scheduled == sorted(set(scheduled)) and scheduled[0] == 0
             and not set(scheduled).intersection(interpolated), "Short face-change evidence has an invalid sample grid")
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
        _require(step in (Fraction(1, 25), Fraction(1, 30)) and 2 <= times[-1] + step <= 8
                 and all(t == i * step for i, t in enumerate(times)),
                 "Short face-change evidence requires a bounded native frame clock")
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
    for i in faces.keys() & scene.keys():
        _require(all(faces[i][key] == scene[i][key] for key in ("pts", "decoded_rgb_sha256")),
                 "Short face and scene measurements disagree on a source picture")


def validate_packet(packet, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    record = packet.get("face_return")
    if record is None and packet.get("semantics") != METHOD:
        return
    _require(isinstance(record, dict) and packet.get("semantics") == METHOD and packet.get("affects_overall_confidence") is False
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
