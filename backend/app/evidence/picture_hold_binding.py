"""Portable verification for exact repeated pictures on a source video clock."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence.video_packet_versions import PICTURE_HOLD_VERSIONS

METHOD = "exact-native-picture-hold/1"
PACKET_ID = "VPH01"
MAX_FRAMES = 120
MAX_RGB_BYTES = 384 * 1024**2
MAX_FRAME_BYTES = 32 * 1024**2
EXTENDED_SCOPE = "extended-native-prefix/1"
EXTENDED_MAX_FRAMES = 900
EXTENDED_MAX_RGB_BYTES = 1536 * 1024**2
EXTENDED_MAX_SECONDS = 30
MIN_SPAN = Fraction(1, 2)
MAX_STEP = Fraction(1, 8)

HISTORICAL = "This result predates the repeated-picture check."
PROVENANCE = "Trusted signed Content Credentials established AI involvement before the repeated-picture check."
INCOMPLETE = "The opening frames could not all be decoded within the repeated-picture check's limits."
TOO_SHORT = "The opening frames did not provide half a second of consecutive, verifiable picture timestamps."
NO_MATCH = "No exact picture hold met the rule in the checked opening frames."
EXTENDED_TOO_SHORT = "The inspected frames did not provide half a second of consecutive, verifiable picture timestamps."
EXTENDED_NO_MATCH = "No exact picture hold met the rule in the inspected frame sequence."
MATCH = "Consecutive source frames carry the same decoded picture."
CONTEXT = "Still cards, a camera hold, screen recording, or editing can produce identical frames. This check carries no weight in the AI estimate."


def frame_limit(count: int, width: int, height: int) -> int:
    size = width * height * 3
    return min(count, MAX_FRAMES, MAX_RGB_BYTES // size) if 0 < size <= MAX_FRAME_BYTES else 0


def _extended_frame_cap(count: int, width: int, height: int) -> int:
    size = width * height * 3
    return min(count, EXTENDED_MAX_FRAMES, EXTENDED_MAX_RGB_BYTES // size) if 0 < size <= MAX_FRAME_BYTES else 0


def extended_frame_limit(pts: Sequence[int], tb: Fraction, width: int, height: int) -> int:
    cap = _extended_frame_cap(len(pts), width, height)
    if not cap:
        return 0
    cutoff = pts[0] + int(EXTENDED_MAX_SECONDS / tb)
    return next((i for i in range(cap) if pts[i] > cutoff), cap)


def coverage(status: str, reason: str, time_base: str, *, limit: int = 0, scanned: int = 0,
             last_pts: int | None = None, continuity_breaks: int | None = None) -> dict:
    result = {"status": status, "reason": reason, "frame_limit": limit, "frames_scanned": scanned,
              "last_frame_index": scanned - 1 if scanned else None, "last_pts": last_pts, "time_base": time_base,
              "last_stream_time_s": None if last_pts is None else float(last_pts * Fraction(time_base)),
              "packet_id": PACKET_ID if status == "matched" else None}
    if continuity_breaks is not None:
        result.update(scan_scope=EXTENDED_SCOPE, continuity_breaks=continuity_breaks)
    return result


def observation(match: dict) -> str:
    return (f"{match['frame_count']} consecutive decoded frames have identical pixels. "
            f"The first and last timestamps are {round(match['span_s'], 6):.3f} seconds apart.")


def measurements(match: dict) -> list[dict]:
    return [{"label": "Identical consecutive frames", "value": match["frame_count"], "unit": "frames"},
            {"label": "First-to-last source timestamp span", "value": match["span_s"], "unit": "seconds"}]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value) -> bool:
    return type(value) is int and abs(value) < 2**63


def _near(value, expected) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and abs(value - float(expected)) <= 0.000002


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _match(packet: dict, cover: dict, source: dict, tb: Fraction) -> None:
    match = packet["picture_hold"]
    start, end, pts = match["start_frame_index"], match["end_frame_index"], match["pts"]
    _require(_integer(start) and _integer(end) and 0 <= start < end < cover["frames_scanned"]
             and isinstance(pts, list) and len(pts) >= 3 and _integer(match["frame_count"])
             and len(pts) == end - start + 1 == match["frame_count"]
             and all(_integer(p) for p in pts), "Picture hold has invalid consecutive frame support")
    span = (pts[-1] - pts[0]) * tb
    _require(span >= MIN_SPAN and all(0 < (b - a) * tb <= MAX_STEP for a, b in zip(pts, pts[1:])),
             "Picture hold bridges a missing or discontinuous timestamp")
    _require(match["start_pts"] == pts[0] and match["end_pts"] == pts[-1] and match["time_base"] == source["time_base"]
             and _near(match["start_stream_s"], pts[0] * tb) and _near(match["end_stream_s"], pts[-1] * tb)
             and _near(match["span_s"], span) and _digest(match["decoded_rgb_sha256"]),
             "Picture hold has an invalid source clock or image commitment")
    origin = source.get("timeline_origin_s")
    for role, point in (("start", pts[0]), ("end", pts[-1])):
        playback = match[f"{role}_playback_s"]
        _require(playback is None if origin is None else _near(playback, float(point * tb) - origin),
                 "Picture hold playback time differs from the source clock")
    _require(packet.get("semantics") == METHOD and packet.get("claim_scope") == "consecutive_frames"
             and packet.get("evidence_type") == "anomaly" and packet.get("signal_refs") == []
             and packet.get("finding_ids") == [] and packet.get("exhibit_refs") == [],
             "Picture hold is misrepresented as statistical evidence")
    _require(packet.get("observation") == observation(match) and packet.get("measurements") == measurements(match)
             and packet.get("interpretation") == CONTEXT and not packet.get("window"),
             "Picture hold description differs from its measured interval")
    frames, assets = packet["frames"], packet["artifacts"]
    _require(len(frames) == 2 and set(assets) == {"held_picture", "contact_sheet"}, "Picture hold omits its two endpoints")
    original = assets["held_picture"]
    _require(original.get("media_type") == "image/png" and (original.get("width"), original.get("height")) == (source["width"], source["height"]),
             "Picture hold original is not the native source size")
    for frame, index, point in zip(frames, (start, end), (pts[0], pts[-1]), strict=True):
        anchor = frame["anchor"]
        _require(anchor["frame_index"] == index and anchor["pts"] == point and anchor["time_base"] == source["time_base"]
                 and anchor["original_sha256"] == source["sha256"] and anchor["decoded_rgb_sha256"] == match["decoded_rgb_sha256"]
                 and (anchor["width"], anchor["height"]) == (source["width"], source["height"])
                 and anchor["coordinate_space"] == source["coordinate_space"] and anchor["stream_selector"] == source["stream_selector"]
                 and _near(anchor["stream_time_s"], point * tb) and frame.get("context_region") is None
                 and frame.get("score") is None and frame.get("is_peak") is False
                 and frame["artifact_refs"] == {"original": "held_picture", "context": "held_picture"},
                 "Picture hold endpoint differs from its source image or timestamp")
        playback = anchor["playback_time_s"]
        _require(playback is None if origin is None else _near(playback, float(point * tb) - origin),
                 "Picture hold endpoint playback time differs from the source clock")


def validate(document: dict) -> None:
    """Validate signed evidence structure without media, models or application services."""
    packets = [p for p in document["packets"] if p.get("id") == PACKET_ID or "picture_hold" in p
               or p.get("claim_scope") == "consecutive_frames" or p.get("semantics") == METHOD]
    source = document["source"]
    if document["version"] not in PICTURE_HOLD_VERSIONS:
        _require("picture_hold" not in document and "picture_hold_sha256" not in source and not packets,
                 "Historical video evidence includes an undeclared picture hold")
        return
    cover = document["picture_hold"]
    tb = Fraction(source["time_base"])
    status, limit, scanned, last_pts = (cover[k] for k in ("status", "frame_limit", "frames_scanned", "last_pts"))
    commitment = source["picture_hold_sha256"]
    _require(tb > 0 and cover["time_base"] == source["time_base"] and _integer(limit) and _integer(scanned)
             and 0 <= scanned <= limit and cover["last_frame_index"] == (scanned - 1 if scanned else None),
             "Picture hold check has invalid prefix coverage")
    _require(cover["last_stream_time_s"] is None if last_pts is None else
             scanned > 0 and _integer(last_pts) and _near(cover["last_stream_time_s"], last_pts * tb),
             "Picture hold prefix time differs from its timestamp")
    if commitment is None:
        _require(cover == coverage("unavailable", HISTORICAL, source["time_base"]) and not packets,
                 "Picture hold check claims coverage without a recorded inspection")
        return
    native_limit = frame_limit(source["frame_count"], source["width"], source["height"])
    _require(_digest(commitment), "Picture hold check omits its recorded commitment")
    expanded = "scan_scope" in cover
    if expanded:
        breaks = cover.get("continuity_breaks")
        _require(cover["scan_scope"] == EXTENDED_SCOPE and native_limit < limit <= _extended_frame_cap(
            source["frame_count"], source["width"], source["height"]) and scanned == limit
            and _integer(breaks) and 0 <= breaks < scanned,
            "Picture hold check differs from its extended native budget")
    else:
        _require(limit == native_limit and "continuity_breaks" not in cover,
                 "Picture hold check differs from its recorded native budget")
    if status == "matched":
        _require(scanned == limit and cover["reason"] == MATCH and cover["packet_id"] == PACKET_ID and len(packets) == 1
                 and packets[0].get("id") == PACKET_ID, "Picture hold check omits its recorded match")
        _match(packets[0], cover, source, tb)
    else:
        no_match, too_short = (EXTENDED_NO_MATCH, EXTENDED_TOO_SHORT) if expanded else (NO_MATCH, TOO_SHORT)
        _require(not packets and cover["packet_id"] is None and
                 ((status == "no_match" and cover["reason"] == no_match and scanned == limit and scanned >= 3)
                  or (status == "unavailable" and cover["reason"] == too_short and scanned == limit)
                  or (not expanded and status == "unavailable" and cover["reason"] == INCOMPLETE and scanned < limit)
                  or (not expanded and status == "unavailable" and cover["reason"] == PROVENANCE and scanned == 0)),
                 "Picture hold coverage has an invalid outcome")
    match = packets[0]["picture_hold"] if packets else None
    _require(canonical_hash({"coverage": cover, "match": match}) == commitment,
             "Picture hold coverage differs from the recorded native inspection")


def validate_record(document: dict, record: dict | None, *, table_pts: Sequence[int], interpolated: Sequence[int],
                    canonical_hash: Callable[[dict], str]) -> None:
    """Compare the package to the native check frozen in the render stage and result."""
    validate(document)
    if document["version"] not in PICTURE_HOLD_VERSIONS:
        return
    commitment = document["source"]["picture_hold_sha256"]
    if commitment is None:
        _require(record is None, "Picture hold inspection is missing its published commitment")
        return
    _require(isinstance(record, dict) and canonical_hash(record) == commitment, "Picture hold differs from the recorded native inspection")
    validate_source_clock(document, table_pts, interpolated)


def validate_source_clock(document: dict, table_pts: Sequence[int], interpolated: Sequence[int]) -> None:
    """Check that covered source indices, clocks and declared gaps match the original frame table."""
    cover = document["picture_hold"]
    if cover.get("scan_scope") == EXTENDED_SCOPE:
        source = document["source"]
        tb = Fraction(source["time_base"])
        expected = extended_frame_limit(table_pts, tb, source["width"], source["height"])
        _require(expected == cover["frame_limit"], "Picture hold extension differs from its source-clock budget")
        inferred = set(interpolated)
        minimum_breaks = sum(i in inferred or i - 1 in inferred or not 0 < (right - left) * tb <= MAX_STEP
                             for i, (left, right) in enumerate(zip(table_pts[:expected - 1], table_pts[1:expected]), 1))
        _require(cover["continuity_breaks"] >= minimum_breaks, "Picture hold extension omits source-clock breaks")
    match = next((p["picture_hold"] for p in document["packets"] if p["id"] == PACKET_ID), None)
    last = cover["last_frame_index"]
    if cover["last_pts"] is not None:
        _require(last not in interpolated and cover["last_pts"] == table_pts[last], "Picture hold prefix has an inferred source timestamp")
    if match is not None:
        start, end = match["start_frame_index"], match["end_frame_index"]
        _require(match["pts"] == list(table_pts[start:end + 1]) and not set(interpolated).intersection(range(start, end + 1)),
                 "Picture hold differs from the committed source frame table")
