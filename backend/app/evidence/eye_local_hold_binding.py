"""Source clock and evidence bindings for the 30 fps eye-region observation."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from itertools import pairwise

from app.core.canonical import canonical_hash
from app.evidence.source_geometry import pixel_bounds

KEY = "eye_local_hold"
SCHEMA = "chipmunk.video-eye-local-hold/1"
PACKET_ID = "VPH03"
FRAME_COUNT = 60
DIMENSIONS = (480, 360)
STEP = Fraction(1, 30)
ENCODING = ("h264", "yuv420p")
CARRIER = "mp4"
ROLES = ("before", "during", "after")
COMMON = {"schema", "status", "frames_scanned", "packet_id", "affects_overall_confidence", "match"}
SOURCE_FIELDS = {"encoding", "carrier", "pts", "frame_hashes", "input_sha256"}
MEASURED = {"matched", "no_match", "no_stable_face"}
SHA = re.compile(r"[0-9a-f]{64}")
TEXTURE_POINT = .115
NEARBY_POINT = .100
MIN_EDGE_CHANGE = .105
MAX_RELATIVE_CHANGE = .40
MIN_CONTEXT_FRACTION = Fraction(2, 3)
LENGTHS = range(10, 23)
CONTEXT = ("One eye region changed less than nearby areas within this interval. Local editing, compression, and ordinary motion "
           "can produce this pattern. The observation carries zero weight in the AI assessment.")


def _exact_clock(pts, time_base):
    return (len(pts) == FRAME_COUNT and all(type(value) is int for value in pts)
            and all((end - start) * time_base == STEP for start, end in pairwise(pts)))


def eligible(table, encoding, carrier):
    return (tuple(encoding or ()) == ENCODING and carrier == CARRIER and (table.width, table.height) == DIMENSIONS
            and not (table.rotation or table.truncated or table.interpolated) and _exact_clock(table.pts, table.tb))


def record(status, *, pts=(), hashes=(), match=None):
    result = {"schema": SCHEMA, "status": status, "frames_scanned": len(pts), "match": match,
              "affects_overall_confidence": False, "packet_id": PACKET_ID if match is not None else None}
    if status in MEASURED:
        result.update(encoding=list(ENCODING), carrier=CARRIER, pts=list(pts),
                      frame_hashes=list(hashes), input_sha256=canonical_hash(hashes))
    return result


def region(box):
    return dict(zip(("x", "y", "w", "h"), box, strict=True))


def observation(review, time_base):
    start, stop = review["match"]["frames"]
    first, following = (float(review["pts"][i] * time_base) for i in (start, stop))
    return (f"An eye region showed lower texture change than nearby areas from {first:.3f} to {following:.3f} s "
            "on the source clock. The end marks the first picture after the measured interval.")


def measurements(match):
    return [{"label": "Consecutive pictures measured in the eye region", "value": match["frames"][1] - match["frames"][0], "unit": "frames"},
            {"label": "Share of transitions with nearby face motion", "value": round(100 * match["moving_context_fraction"], 2), "unit": "%"},
            {"label": "Nearby minus eye texture change", "value": round(match["nearby_gap"], 4), "unit": "normalized difference"}]


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _near(value, expected):
    return type(value) in (int, float) and math.isfinite(value) and abs(value - float(expected)) <= .000002


def _playback(value, stream_time, origin):
    return value is None if origin is None else _near(value, float(stream_time) - origin)


def _bounded(value, low, high):
    return type(value) in (int, float) and low <= value <= high


def _box(value):
    return (isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value)
            and min(value[2:]) >= 8 and pixel_bounds(region(value), *DIMENSIONS) is not None)


def _match_packet(packet, review, source, tb):
    match = review["match"]
    start, stop = match["frames"]
    anchors = [start - 1, (start + stop) // 2, stop]
    _require(type(start) is int and type(stop) is int and 5 <= start < stop <= FRAME_COUNT - 5 and stop - start in LENGTHS
             and match["anchors"] == anchors and type(match["eye"]) is int and match["eye"] in (0, 1)
             and _box(match["box"]) and 3 <= len(match["context_boxes"]) <= 4 and all(map(_box, match["context_boxes"])),
             "Eye region has invalid source geometry or frame bounds")
    change = match["internal_texture_change"]
    flank = match["minimum_flank_texture_change"]
    nearby = match["nearby_medians"]
    _require(_bounded(change, 0, TEXTURE_POINT) and _bounded(match["midpoint_texture_change"], 0, TEXTURE_POINT)
             and _bounded(flank, MIN_EDGE_CHANGE, 2.000002) and change <= MAX_RELATIVE_CHANGE * flank + .000002
             and _bounded(match["moving_context_fraction"], MIN_CONTEXT_FRACTION, 1)
             and _bounded(match["box_median"], 0, 2) and len(nearby) == 4 and all(_bounded(value, 0, 2) for value in nearby)
             and _bounded(match["nearby_gap"], NEARBY_POINT, 2)
             and _near(match["nearby_gap"], min(max(nearby[:2]), max(nearby[2:])) - match["box_median"]),
             "Eye region has inconsistent texture or nearby motion measurements")
    _require(packet["id"] == PACKET_ID and packet["semantics"] == SCHEMA and packet[KEY] == match
             and packet["claim_scope"] == "sampled_window" and packet["evidence_type"] == "anomaly"
             and packet["affects_overall_confidence"] is False and not (packet["finding_ids"] or packet["signal_refs"] or packet["exhibit_refs"])
             and packet["observation"] == observation(review, tb) and packet["interpretation"] == CONTEXT
             and packet["measurements"] == measurements(match), "Eye region differs from the recorded picture observation")

    pts, hashes, origin = review["pts"], review["frame_hashes"], source["timeline_origin_s"]
    window = packet["window"]
    _require(window["frame_indices"] == list(range(start - 1, stop + 1)) and window["time_base"] == source["time_base"],
             "Eye-region range omits a measured or neighboring frame")
    for name, index in (("start", start), ("end", stop)):
        stream = pts[index] * tb
        _require(window[f"{name}_pts"] == pts[index] and _near(window[f"{name}_stream_s"], stream)
                 and _playback(window[f"{name}_playback_s"], stream, origin), "Eye-region range differs from the native clock")

    _require(len(packet["frames"]) == len(ROLES), "Eye-region observation requires three original source pictures")
    expected_assets = set()
    for frame, index, role in zip(packet["frames"], anchors, ROLES, strict=True):
        anchor = frame["anchor"]
        original, crop = f"{role}_original", f"{role}_crop"
        expected_assets.update((original, crop))
        stream = pts[index] * tb
        _require(frame["id"] == f"{PACKET_ID}.{role}" and frame["interval_role"] == role
                 and frame["context_region"] == region(match["box"]) and frame["region_meaning"] == "measured_eye_region"
                 and frame["score"] is None and frame["is_peak"] is False
                 and frame["artifact_refs"] == {"original": original, "context": original, "crop": crop}
                 and anchor["frame_index"] == index and anchor["pts"] == pts[index] and anchor["decoded_rgb_sha256"] == hashes[index]
                 and anchor["original_sha256"] == source["sha256"] and anchor["time_base"] == source["time_base"]
                 and anchor["stream_selector"] == source["stream_selector"] and anchor["coordinate_space"] == source["coordinate_space"]
                 and (anchor["width"], anchor["height"]) == DIMENSIONS and _near(anchor["stream_time_s"], stream)
                 and _playback(anchor["playback_time_s"], stream, origin), "Eye-region frame or crop differs from its recorded source")
        for name, size in ((original, DIMENSIONS), (crop, tuple(match["box"][2:]))):
            artifact = packet["artifacts"][name]
            _require(artifact["media_type"] == "image/png" and (artifact["width"], artifact["height"]) == size,
                     "Eye-region image has an invalid size or media type")
    _require(packet["artifacts"].keys() == expected_assets, "Eye-region observation has undeclared assets")


def validate(cover, document):
    review = cover.get(KEY)
    packets = [p for p in document["packets"] if p.get("id") == PACKET_ID or p.get("semantics") == SCHEMA or KEY in p]
    if review is None:
        _require(not packets, "Eye-region evidence has no recorded source inspection")
        return
    status = review["status"]
    _require(review["schema"] == SCHEMA and review["affects_overall_confidence"] is False,
             "Invalid eye-region source inspection policy")
    if status not in MEASURED:
        _require(status in {"outside_scope", "unavailable"} and review.keys() == COMMON and review["frames_scanned"] == 0
                 and not packets and review["packet_id"] is None and review["match"] is None,
                 "An unmeasured eye-region inspection claims evidence")
        return
    source = document["source"]
    tb = Fraction(source["time_base"])
    hashes = review["frame_hashes"]
    _require(review.keys() == COMMON | SOURCE_FIELDS and review["frames_scanned"] == cover["frames_scanned"] == source["frame_count"] == FRAME_COUNT
             and (source["width"], source["height"]) == DIMENSIONS and _exact_clock(review["pts"], tb)
             and review["carrier"] == CARRIER and tuple(review["encoding"]) == ENCODING and len(hashes) == FRAME_COUNT
             and all(isinstance(value, str) and SHA.fullmatch(value) for value in hashes)
             and canonical_hash(hashes) == review["input_sha256"], "Eye-region inspection differs from the verified source profile")
    if status != "matched":
        _require(not packets and review["packet_id"] is None and review["match"] is None,
                 "Eye-region evidence differs from the recorded no-match status")
    else:
        _require(len(packets) == 1 and review["packet_id"] == PACKET_ID, "Eye-region observation omits its recorded evidence")
        _match_packet(packets[0], review, source, tb)


def validate_clock(review, table_pts, interpolated):
    if review and review["status"] in MEASURED:
        _require(review["pts"] == list(table_pts) and not interpolated,
                 "Eye-region timestamps differ from the verified source clock")
