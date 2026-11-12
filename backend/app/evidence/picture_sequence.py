"""Exact repeated moving-picture sequences on the native presentation clock."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction

from app.core.canonical import canonical_hash

SCHEMA = "chipmunk.video-exact-sequence/1"
SCAN_SCOPE = "native-sequence-prefix/1"
PACKET_ID = "VPR01"
MIN_UNIQUE = 8
MIN_SPAN = Fraction(1, 2)
MAX_STEP = Fraction(1, 8)
CONTEXT = "Loops, transitions and ordinary editing can repeat a moving sequence. This check carries zero weight in the AI estimate."
_SHA = re.compile(r"[a-f0-9]{64}")


def _steps(ticks, tb):
    result = [None]
    maximum = int(MAX_STEP / tb)
    for a, b in zip(ticks, ticks[1:]):
        result.append(b - a if a is not None and b is not None and 0 < b - a <= maximum else None)
    return result


def _two_spans(ticks, steps, minimum):
    start, found = 0, 0
    for end, pts in enumerate(ticks):
        if pts is None or steps[end] is None:
            start = end
        if end - start + 1 >= MIN_UNIQUE and pts - ticks[start] >= minimum:
            found += 1
            if found == 2:
                return True
            start = end + 1
    return False


def _pair(ticks, digests, steps, minimum):
    if sum(code is not None and count >= 2 for code, count in Counter(digests).items()) < MIN_UNIQUE:
        return None
    best = None
    n = len(ticks)
    for lag in range(MIN_UNIQUE, n - MIN_UNIQUE + 1):
        start, counts = 0, {}
        for end in range(n - lag):
            code = digests[end]
            other = end + lag
            if code is None or code != digests[other]:
                start = end + 1
                counts.clear()
                continue
            if steps[end] is None or steps[end] != steps[other]:
                start = end
                counts.clear()
            counts[code] = counts.get(code, 0) + 1
            if end - start + 1 > lag:
                before = digests[start]
                counts[before] -= 1
                if counts[before] == 0:
                    del counts[before]
                start += 1
            if len(counts) >= MIN_UNIQUE:
                elapsed = ticks[end] - ticks[start]
                rank = elapsed, -start, -(start + lag)
                if elapsed >= minimum and (best is None or rank > best[0]):
                    best = rank, end, len(counts)
    return best


def scan(observations, tb: Fraction):
    """Compare equal pixels and frame spacing in two disjoint continuous occurrences."""
    ticks = [row[0] for row in observations]
    digests = [row[1] if row[0] is not None else None for row in observations]
    minimum = math.ceil(MIN_SPAN / tb)
    steps = _steps(ticks, tb)
    if not _two_spans(ticks, steps, minimum):
        return {"schema": SCHEMA, "status": "unavailable", "pair": None}
    best = _pair(ticks, digests, steps, minimum)
    if best is None:
        return {"schema": SCHEMA, "status": "no_match", "pair": None}
    (elapsed, neg_start, neg_other), end, unique = best
    start, other = -neg_start, -neg_other
    width = end - start + 1
    spans = []
    for i in (start, other):
        pts = ticks[i:i + width]
        spans.append({"start_frame_index": i, "end_frame_index": i + width - 1, "pts": pts,
                      "start_stream_s": float(pts[0] * tb), "end_stream_s": float(pts[-1] * tb)})
    pair = {"first": spans[0], "second": spans[1], "frame_count": width, "unique_pictures": unique,
            "span_s": float(elapsed * tb), "rgb_sequence_sha256": canonical_hash(digests[start:end + 1]),
            "endpoint_rgb_sha256": [digests[start], digests[end]]}
    return {"schema": SCHEMA, "status": "matched", "pair": pair}


def observation(pair):
    return f"Two disjoint sequences contain the same {pair['frame_count']} decoded pictures in the same order and at the same frame spacing."


def endpoints(pair):
    for occurrence in ("first", "second"):
        span = pair[occurrence]
        for offset, role in enumerate(("start", "end")):
            yield occurrence, role, span[f"{role}_frame_index"], span["pts"][0 if offset == 0 else -1], pair["endpoint_rgb_sha256"][offset]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(number, expected):
    return type(number) in (int, float) and math.isfinite(number) and abs(number - float(expected)) <= .000002


def validate(cover, document):
    record = cover.get("sequence_review")
    packets = [p for p in document["packets"] if p.get("id") == PACKET_ID or "picture_sequence" in p
               or p.get("claim_scope") == "paired_frame_sequences"]
    if record is None:
        _require(not packets, "Repeated sequence lacks a recorded scan")
        return
    _require(record["schema"] == SCHEMA and cover.get("scan_scope") == SCAN_SCOPE
             and cover["frames_scanned"] == cover["frame_limit"], "Repeated sequence lacks completed coverage")
    if record["status"] != "matched":
        _require(record["status"] in {"unavailable", "no_match"} and record["pair"] is None and not packets,
                 "Repeated sequence evidence disagrees with its status")
        return
    pair = record["pair"]
    _require(len(packets) == 1 and packets[0]["id"] == PACKET_ID and packets[0].get("picture_sequence") == pair,
             "Repeated sequence omits the recorded pair")
    source = document["source"]
    tb = Fraction(source["time_base"])
    count, unique = pair["frame_count"], pair["unique_pictures"]
    _require(type(count) is int and type(unique) is int and MIN_UNIQUE <= unique <= count <= cover["frames_scanned"] // 2,
             "Repeated sequence does not contain enough distinct moving pictures")
    spans = (pair["first"], pair["second"])
    _require(spans[0]["end_frame_index"] < spans[1]["start_frame_index"], "Repeated sequences overlap")
    intervals = []
    for span in spans:
        start, end, pts = span["start_frame_index"], span["end_frame_index"], span["pts"]
        _require(type(start) is int and type(end) is int and 0 <= start < end < cover["frames_scanned"]
                 and end - start + 1 == len(pts) == count and all(type(p) is int for p in pts), "Repeated sequence has an invalid source interval")
        steps = _steps(pts, tb)[1:]
        elapsed = (pts[-1] - pts[0]) * tb
        _require(None not in steps and elapsed >= MIN_SPAN and _close(pair["span_s"], elapsed)
                 and _close(span["start_stream_s"], pts[0] * tb) and _close(span["end_stream_s"], pts[-1] * tb),
                 "Repeated sequence has incomplete clock support")
        intervals.append(steps)
    _require(intervals[0] == intervals[1], "Repeated sequence changes the native frame spacing")
    digests = pair["endpoint_rgb_sha256"]
    _require(len(digests) == 2 and all(_SHA.fullmatch(d) for d in [*digests, pair["rgb_sequence_sha256"]]),
             "Repeated sequence has invalid image commitments")
    packet = packets[0]
    _require(packet.get("claim_scope") == "paired_frame_sequences" and packet.get("semantics") == SCHEMA
             and packet.get("evidence_type") == "anomaly" and packet.get("affects_overall_confidence") is False
             and packet.get("signal_refs") == packet.get("finding_ids") == packet.get("exhibit_refs") == []
             and packet.get("observation") == observation(pair) and packet.get("interpretation") == CONTEXT,
             "Repeated sequence is misrepresented as statistical evidence")
    _require(set(packet["artifacts"]) == {"start_picture", "end_picture", "contact_sheet"}, "Repeated sequence lacks endpoint images")
    for frame, (_, role, i, pts, digest) in zip(packet["frames"], endpoints(pair), strict=True):
        anchor = frame["anchor"]
        ref = f"{role}_picture"
        original = packet["artifacts"][ref]
        origin = source.get("timeline_origin_s")
        stream_s = float(pts * tb)
        _require((anchor["frame_index"], anchor["pts"], anchor["decoded_rgb_sha256"]) == (i, pts, digest)
                 and anchor["time_base"] == source["time_base"] and anchor["original_sha256"] == source["sha256"]
                 and all(anchor[key] == source[key] for key in ("width", "height", "stream_selector", "coordinate_space"))
                 and original["media_type"] == "image/png" and (original["width"], original["height"]) == (source["width"], source["height"])
                 and _close(anchor["stream_time_s"], stream_s)
                 and (anchor["playback_time_s"] is None if origin is None else _close(anchor["playback_time_s"], stream_s - origin))
                 and frame.get("context_region") is None and frame.get("score") is None and not frame.get("is_peak")
                 and frame["artifact_refs"] == {"original": ref, "context": ref}, "Repeated sequence endpoint differs from its image or clock")


def validate_clock(record, table_pts: Sequence[int], interpolated: Sequence[int]):
    if not record or record["pair"] is None:
        return
    inferred = set(interpolated)
    for key in ("first", "second"):
        span = record["pair"][key]
        start, end = span["start_frame_index"], span["end_frame_index"]
        _require(span["pts"] == list(table_pts[start:end + 1]) and not inferred.intersection(range(start, end + 1)),
                 "Repeated sequence differs from the committed source frame table")
