"""Portable source bindings for a short held mouth region."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from itertools import pairwise

from app.core.canonical import canonical_hash
from app.evidence import mouth_hold_sound
from app.evidence.source_geometry import pixel_bounds

SCHEMA = "chipmunk.video-mouth-local-hold/1"
HD_SCHEMA = "chipmunk.video-mouth-local-hold/2"
PACKET_ID = "VPH02"
FRAME_COUNT = 50
STEP = Fraction(1, 25)
PROFILE = {(360, 288), (480, 360)}
HD_COUNT = 120
HD_STEP = Fraction(1, 30)
HD_PROFILE = {(1280, 720)}
HD_WINDOW = 60
HD_STARTS = (0, 30, 60)
HD_FLANK = 5
HD_SPAN = (8, 24)
SCOPES = {SCHEMA: (FRAME_COUNT, STEP, PROFILE), HD_SCHEMA: (HD_COUNT, HD_STEP, HD_PROFILE)}
ENCODING = ("h264", "yuv420p")
CONTEXT = ("A mouth region remained nearly unchanged while other face areas changed in brightness. Ordinary editing and compression "
           "can create this pattern. This observation carries zero weight in the AI estimate.")
STATUSES = {"matched", "no_match", "no_stable_face", "unavailable"}
ROLES = ("before", "during", "after")
SHA = re.compile(r"[0-9a-f]{64}")


def eligible(table, encoding, *, schema=SCHEMA):
    if schema not in SCOPES:
        return False
    count, step, dimensions = SCOPES[schema]
    return (tuple(encoding or ()) == ENCODING and (table.width, table.height) in dimensions and len(table) >= count
            and (schema == HD_SCHEMA or len(table) == count) and not (table.rotation or table.truncated)
            and all(i >= count for i in table.interpolated)
            and all((b - a) * table.tb == step for a, b in pairwise(table.pts[:count])))


def region(box):
    return dict(zip(("x", "y", "w", "h"), box, strict=True))


def observation(review, tb):
    match = review["match"]
    start, end = match["frames"]
    return (f"A textured mouth region stayed nearly unchanged from {float(review['pts'][start] * tb):.3f} to "
            f"{float(review['pts'][end] * tb):.3f} s on the source clock. The end time marks the first following picture. "
            f"At least two surrounding face regions changed in brightness on {match['moving_context_edges']} of {match['total_edges']} consecutive transitions.")


def measurements(match):
    return [{"label": "Consecutive pictures in the held region", "value": match["frames"][1] - match["frames"][0], "unit": "frames"},
            {"label": "Lowest share of near-unchanged pixels between pictures", "value": round(match["minimum_patch_near_equal_fraction"] * 100, 2), "unit": "%"},
            {"label": "Transitions with change in two surrounding face regions", "value": match["moving_context_edges"], "unit": "transitions"}]


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _close(value, expected):
    return type(value) in (int, float) and math.isfinite(value) and abs(value - float(expected)) <= .000002


def _playback(value, stream_time, origin):
    return value is None if origin is None else _close(value, float(stream_time) - origin)


def validate(cover, document):
    packets = [packet for packet in document["packets"] if packet.get("id") == PACKET_ID
               or packet.get("semantics") in SCOPES or "mouth_local_hold" in packet]
    review = cover.get("mouth_local_hold")
    if review is None:
        _require(not packets, "Mouth hold has no recorded source inspection")
        return
    source = document["source"]
    tb = Fraction(source["time_base"])
    origin = source["timeline_origin_s"]
    status, pts, schema = review["status"], review["pts"], review["schema"]
    _require(schema in SCOPES, "Mouth hold uses an unsupported source profile")
    count, step, dimensions = SCOPES[schema]
    hd = schema == HD_SCHEMA
    _require("sequence_review" in cover and review["frames_scanned"] == count <= cover["frames_scanned"] <= source["frame_count"]
             and (hd or source["frame_count"] == count) and tuple(review["encoding"]) == ENCODING
             and (source["width"], source["height"]) in dimensions and len(pts) == count and all(type(tick) is int for tick in pts)
             and all((b - a) * tb == step for a, b in pairwise(pts)) and len(review["frame_hashes"]) == count
             and all(SHA.fullmatch(item) for item in review["frame_hashes"]) and canonical_hash(review["frame_hashes"]) == review["input_sha256"]
             and status in STATUSES and review["affects_overall_confidence"] is False,
             "Mouth hold has an unsupported source or clock")
    if hd:
        assessed = review["assessed_window_starts"]
        _require(review["carrier"] == "mp4" and assessed == sorted(set(assessed)) and all(type(i) is int and i in HD_STARTS for i in assessed)
                 and bool(assessed) == (status in {"matched", "no_match"}), "HD mouth hold has inconsistent measured windows")
    else:
        _require("assessed_window_starts" not in review, "Historical mouth hold includes undeclared windows")
    match = review["match"]
    if status != "matched":
        _require(match is None and not packets and review["packet_id"] is None and "sound_context" not in review,
                 "Mouth hold evidence differs from its status")
        return
    _require(len(packets) == 1 and review["packet_id"] == PACKET_ID, "Mouth hold omits its recorded evidence")
    packet, = packets
    _require(packet["id"] == PACKET_ID and packet["semantics"] == schema and packet["claim_scope"] == "sampled_window"
             and packet["evidence_type"] == "anomaly" and packet["mouth_local_hold"] == match and packet["affects_overall_confidence"] is False
             and not (packet["signal_refs"] or packet["finding_ids"] or packet["exhibit_refs"])
             and packet["interpretation"] == CONTEXT, "Mouth hold evidence misstates the recorded observation")
    start, end = match["frames"]
    index = [start - 1, (start + end) // 2, end]
    bounds = region(match["box"])
    offset = match["window_start"] if hd else 0
    _require((type(offset) is int and offset in assessed) if hd else "window_start" not in match,
             "Mouth hold differs from its measured review window")
    window, flank = (HD_WINDOW, HD_FLANK) if hd else (count, 4)
    minimum, maximum = HD_SPAN if hd else (7, 20)
    _require(all(type(i) is int for i in (start, end, *match["box"])) and offset + flank <= start < end <= offset + window - flank
             and minimum <= end - start <= maximum and match["anchors"] == index and len(match["context_boxes"]) in range(2, 6)
             and match["total_edges"] == end - start - 1 and 2 * match["total_edges"] <= 3 * match["moving_context_edges"] <= 3 * match["total_edges"],
             "Mouth hold range or motion support is invalid")
    _require(0 <= match["maximum_patch_mean_difference"] <= .42 and .94 <= match["minimum_patch_near_equal_fraction"] <= 1
             and 2.4 <= match["minimum_flank_peak"] <= 255, "Mouth hold pixel measurements are inconsistent")
    for box in [match["box"], *match["context_boxes"]]:
        _require(all(type(value) is int for value in box), "Mouth hold requires exact native pixel coordinates")
        pixel_bounds(region(box), source["width"], source["height"])
    spoken = review.get("sound_context")
    mouth_hold_sound.validate(review, packet, document.get("audio", {}).get("source", {}), source["time_base"])
    text = observation(review, tb) + (" " + mouth_hold_sound.observation(spoken) if spoken else "")
    numbers = measurements(match) + (mouth_hold_sound.measurements(spoken) if spoken else [])
    _require(packet["observation"] == text and packet["measurements"] == numbers,
             "Mouth hold wording differs from its measured source range")
    window = packet["window"]
    _require(window["frame_indices"] == list(range(start - 1, end + 1)) and window["time_base"] == source["time_base"]
             and window["start_pts"] == pts[start] and window["end_pts"] == pts[end]
             and _close(window["start_stream_s"], pts[start] * tb) and _close(window["end_stream_s"], pts[end] * tb)
             and _playback(window["start_playback_s"], pts[start] * tb, origin) and _playback(window["end_playback_s"], pts[end] * tb, origin),
             "Mouth hold display window differs from the committed clock")
    frames, assets = packet["frames"], packet["artifacts"]
    _require(len(frames) == 3, "Mouth hold omits before, during or after pictures")
    for i, role, frame in zip(index, ROLES, frames, strict=True):
        anchor, refs = frame["anchor"], frame["artifact_refs"]
        _require(frame["interval_role"] == role and frame["context_region"] == bounds and frame["score"] is None and not frame["is_peak"]
                 and frame["region_meaning"] == "measured_mouth_region"
                 and (anchor["frame_index"], anchor["pts"], anchor["decoded_rgb_sha256"]) == (i, pts[i], review["frame_hashes"][i])
                 and anchor["original_sha256"] == source["sha256"] and anchor["time_base"] == source["time_base"]
                 and all(anchor[key] == source[key] for key in ("width", "height", "stream_selector", "coordinate_space"))
                 and _playback(anchor["playback_time_s"], pts[i] * tb, origin)
                 and _close(anchor["stream_time_s"], pts[i] * tb) and set(refs) == {"original", "crop", "context"}
                 and refs["original"] == refs["context"], "Mouth hold image differs from its recorded source anchor")
        original, crop = assets[refs["original"]], assets[refs["crop"]]
        _require(original["media_type"] == crop["media_type"] == "image/png" and
                 (original["width"], original["height"], crop["width"], crop["height"])
                 == (source["width"], source["height"], bounds["w"], bounds["h"]), "Mouth hold crop differs from its source box")


def validate_clock(review, table_pts, interpolated):
    if review is not None:
        count = review["frames_scanned"]
        _require(review["pts"] == list(table_pts[:count]) and all(i >= count for i in interpolated),
                 "Mouth hold differs from the committed native clock")
