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
REVERSE_SCHEMA = "chipmunk.video-reverse-sequence/1"
REVERSE_KEY = "reverse_sequence_review"
REVERSE_PACKET_ID = "VPR03"
MIN_UNIQUE = 8
MIN_SPAN = Fraction(1, 2)
MAX_STEP = Fraction(1, 8)
CONTEXT = "Loops, transitions and ordinary editing can repeat a moving sequence. This check carries zero weight in the AI estimate."
REVERSE_CONTEXT = "Reverse playback can be part of ordinary editing. This check carries zero weight in the AI estimate."
MOTION_SCHEMA = "chipmunk.video-motion-sequence/1"
MOTION_EXTENDED_SCHEMA = "chipmunk.video-motion-sequence/2"
MOTION_REVERSE_SCHEMA = "chipmunk.video-reverse-motion-sequence/1"
MOTION_RETIMED_SCHEMA = "chipmunk.video-retimed-motion-sequence/1"
MOTION_REFLECTED_SCHEMA = "chipmunk.video-reflected-motion-sequence/1"
MOTION_SCHEMAS = frozenset({MOTION_SCHEMA, MOTION_EXTENDED_SCHEMA})
MOTION_PACKET_ID = "VPR02"
MOTION_REVERSE_KEY = "reverse_motion_sequence_review"
MOTION_REVERSE_PACKET_ID = "VPR04"
MOTION_RETIMED_KEY = "retimed_motion_sequence_review"
MOTION_RETIMED_PACKET_ID = "VPR05"
MOTION_RETIMED_CONTEXT = "Playback speed changes are common in ordinary editing. This check carries zero weight in the AI estimate."
MOTION_REFLECTED_KEY = "reflected_motion_sequence_review"
MOTION_REFLECTED_PACKET_ID = "VPR06"
HORIZONTAL_REFLECTION = "horizontal_reflection"
MOTION_REFLECTED_CONTEXT = ("Ordinary editing can reuse footage in a mirrored orientation. The comparison leaves the order of creation "
                            "undetermined and carries zero weight in the AI estimate.")
MOTION_REVIEWS = (
    ("motion_sequence_review", MOTION_PACKET_ID, MOTION_SCHEMAS),
    (MOTION_REVERSE_KEY, MOTION_REVERSE_PACKET_ID, frozenset({MOTION_REVERSE_SCHEMA})),
    (MOTION_RETIMED_KEY, MOTION_RETIMED_PACKET_ID, frozenset({MOTION_RETIMED_SCHEMA})),
    (MOTION_REFLECTED_KEY, MOTION_REFLECTED_PACKET_ID, frozenset({MOTION_REFLECTED_SCHEMA})),
)
MOTION_PROFILES = {(320, 180): (20, 180), (640, 360): (25, 225)}
MOTION_EXTENDED_PROFILES = {(640, 360): (25, 450)}
MOTION_360P_PROFILES = {(640, 360): (25, 225)}
MOTION_ENCODINGS = {("h264", "yuv420p"), ("vp9", "yuv420p")}
MOTION_PARAMETERS = {
    "luma_grid": [40, 30], "maximum_picture_rms": 1.6, "minimum_motion_rms": .4,
    "minimum_motion_cosine": .65, "minimum_motion_rms_ratio": .55, "minimum_supporting_edges": 8,
    "maximum_support_edge_step": 3, "minimum_support_fraction": .5, "minimum_span_seconds": .5,
}
_SHA = re.compile(r"[a-f0-9]{64}")


def motion_profile(dimensions, schema):
    if schema in {MOTION_SCHEMA, MOTION_REVERSE_SCHEMA}:
        return MOTION_PROFILES.get(dimensions)
    if schema == MOTION_EXTENDED_SCHEMA:
        return MOTION_EXTENDED_PROFILES.get(dimensions)
    if schema in {MOTION_RETIMED_SCHEMA, MOTION_REFLECTED_SCHEMA}:
        return MOTION_360P_PROFILES.get(dimensions)
    return None


def motion_encoding_allowed(schema, encoding):
    return tuple(encoding or ()) in (MOTION_ENCODINGS if schema != MOTION_RETIMED_SCHEMA else {("h264", "yuv420p")})


def motion_parameters(schema, rate):
    parameters = MOTION_PARAMETERS | {"frame_rate": rate}
    if schema == MOTION_RETIMED_SCHEMA:
        parameters["native_frame_steps"] = [1, 2]
    if schema == MOTION_REFLECTED_SCHEMA:
        parameters.update(maximum_picture_rms=2.4, minimum_unflipped_picture_rms=4.8,
                          spatial_transform=HORIZONTAL_REFLECTION)
    return parameters


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


def _reverse_pair(ticks, digests, steps, minimum):
    if sum(code is not None and count >= 2 for code, count in Counter(digests).items()) < MIN_UNIQUE:
        return None
    best = None
    n = len(digests)
    # Corresponding frame indices add up to the same value across a reversed pair.
    for index_sum in range(2 * MIN_UNIQUE - 1, 2 * n - 2 * MIN_UNIQUE):
        distinct = set()
        for left in range((index_sum - 1) // 2, max(-1, index_sum - n), -1):
            right = index_sum - left
            code = digests[left]
            if code is None or code != digests[right]:
                distinct.clear()
                continue
            if not distinct or steps[left + 1] is None or steps[left + 1] != steps[right]:
                first_end, second_start = left, right
                distinct.clear()
            distinct.add(code)
            if len(distinct) >= MIN_UNIQUE:
                elapsed = ticks[first_end] - ticks[left]
                rank = elapsed, -left, -second_start
                if elapsed >= minimum and (best is None or rank > best[0]):
                    best = rank, first_end, len(distinct)
    return best


def scan(observations, tb: Fraction, *, reverse=False):
    """Compare equal pixels and corresponding frame spacing in two disjoint source ranges."""
    schema = REVERSE_SCHEMA if reverse else SCHEMA
    ticks = [row[0] for row in observations]
    digests = [row[1] if row[0] is not None else None for row in observations]
    minimum = math.ceil(MIN_SPAN / tb)
    steps = _steps(ticks, tb)
    if not _two_spans(ticks, steps, minimum):
        return {"schema": schema, "status": "unavailable", "pair": None}
    find_pair = _reverse_pair if reverse else _pair
    best = find_pair(ticks, digests, steps, minimum)
    if best is None:
        return {"schema": schema, "status": "no_match", "pair": None}
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
    if reverse:
        pair["order"] = "reverse"
    return {"schema": schema, "status": "matched", "pair": pair}


def observation(pair):
    if pair.get("spatial_transform") == HORIZONTAL_REFLECTION:
        return (f"Two separate ranges contain {pair['frame_count']} visually similar pictures after a left-to-right flip. "
                f"{len(pair['supporting_edge_offsets'])} matching picture changes support the comparison.")
    if steps := pair.get("frame_steps"):
        faster, slower = ("first", "second") if steps["first"] == 1 else ("second", "first")
        return (f"The {faster} range shows {pair['frame_count']} matching pictures at twice the speed of the {slower} range. "
                f"{len(pair['supporting_edge_offsets'])} matching changes across the picture support the comparison.")
    if "supporting_edge_offsets" in pair:
        if pair.get("order") == "reverse":
            return (f"The second source range contains {pair['frame_count']} visually similar pictures in reverse order. "
                    f"{len(pair['supporting_edge_offsets'])} transitions support the reversed motion.")
        return (f"Two separate source ranges contain {pair['frame_count']} visually similar pictures with the same timing. "
                f"{len(pair['supporting_edge_offsets'])} transitions show matching changes across the picture.")
    if pair.get("order") == "reverse":
        return (f"The second source range plays the same {pair['frame_count']} decoded pictures backward, "
                "with the corresponding frame spacing preserved.")
    return f"Two disjoint sequences contain the same {pair['frame_count']} decoded pictures in the same order and at the same frame spacing."


def interpretation(pair):
    if pair.get("spatial_transform") == HORIZONTAL_REFLECTION:
        return MOTION_REFLECTED_CONTEXT
    if "frame_steps" in pair:
        return MOTION_RETIMED_CONTEXT
    return REVERSE_CONTEXT if pair.get("order") == "reverse" else CONTEXT


def endpoints(pair):
    for occurrence in ("first", "second"):
        span = pair[occurrence]
        for offset, role in enumerate(("start", "end")):
            motion = "supporting_edge_offsets" in pair
            digests = span["endpoint_rgb_sha256"] if motion else pair["endpoint_rgb_sha256"]
            digest_offset = 1 - offset if not motion and occurrence == "second" and pair.get("order") == "reverse" else offset
            yield occurrence, role, span[f"{role}_frame_index"], span["pts"][0 if offset == 0 else -1], digests[digest_offset]


def picture_ref(pair, occurrence, role):
    if "supporting_edge_offsets" in pair:
        return f"{occurrence}_{role}_picture"
    if occurrence == "second" and pair.get("order") == "reverse":
        role = "end" if role == "start" else "start"
    return f"{role}_picture"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(number, expected):
    return type(number) in (int, float) and math.isfinite(number) and abs(number - float(expected)) <= .000002


def validate(cover, document):
    packet_ids = {PACKET_ID, REVERSE_PACKET_ID, *(packet_id for _, packet_id, _ in MOTION_REVIEWS)}
    schemas = {SCHEMA, REVERSE_SCHEMA}.union(*(schemas for _, _, schemas in MOTION_REVIEWS))
    packets = [p for p in document["packets"] if p.get("id") in packet_ids or "picture_sequence" in p
               or p.get("claim_scope") == "paired_frame_sequences"
               or p.get("semantics") in schemas]
    _require(all(p.get("id") in packet_ids for p in packets), "Unexpected repeated sequence packet")
    exact = cover.get("sequence_review")
    reverse = cover.get(REVERSE_KEY)
    if exact is not None:
        _require(cover.get("scan_scope") == SCAN_SCOPE and cover["frames_scanned"] == cover["frame_limit"],
                 "Repeated sequence lacks completed coverage")
    _require(reverse is None or exact is not None, "Reversed sequence lacks a completed native scan")
    _validate_pair(exact, document["source"], cover["frames_scanned"] if exact is not None else 0,
                   [p for p in packets if p["id"] == PACKET_ID], schemas={SCHEMA})
    _validate_pair(reverse, document["source"], cover["frames_scanned"] if reverse is not None else 0,
                   [p for p in packets if p["id"] == REVERSE_PACKET_ID], schemas={REVERSE_SCHEMA})
    for key, packet_id, schemas in MOTION_REVIEWS:
        motion = cover.get(key)
        if key == MOTION_RETIMED_KEY and motion is not None:
            _require(all(record and record["pair"] is None for record in (
                exact, reverse, cover.get("motion_sequence_review"), cover.get(MOTION_REVERSE_KEY),
            )), "Retimed sequence requires completed earlier repeat comparisons without a match")
        source = document["source"]
        rate = _validate_motion_scope(motion, source, reverse if key == MOTION_REVERSE_KEY else exact,
                                      cover["frames_scanned"] if motion is not None else 0, schemas=schemas)
        _validate_pair(motion, source, motion["frames_scanned"] if motion is not None else 0,
                       [p for p in packets if p["id"] == packet_id], schemas=schemas, motion_rate=rate)


def _validate_motion_scope(record, source, exact, scanned, *, schemas):
    if record is None:
        return None
    schema = record["schema"]
    profile = motion_profile((source["width"], source["height"]), schema)
    _require(schema in schemas and profile is not None, "Motion sequence has unsupported source dimensions or method")
    rate, limit = profile
    _require(exact is not None and scanned >= limit and record["parameters"] == motion_parameters(schema, rate)
             and motion_encoding_allowed(schema, record["encoding"]), "Motion sequence has unsupported source scope or parameters")
    exact_takes_priority = exact["status"] == "matched" and schema != MOTION_REFLECTED_SCHEMA
    if record["status"] == "unavailable":
        _require(schema != MOTION_EXTENDED_SCHEMA and record["frames_scanned"] == 0 and (record["reason"] == "source_clock"
                 or (record["reason"] == "exact_sequence" and exact_takes_priority)),
                 "Motion sequence has an invalid skipped outcome")
    else:
        _require(record["frames_scanned"] == limit and record["reason"] is None and not exact_takes_priority,
                 "Motion sequence has invalid measured coverage")
    return rate


def _validate_pair(record, source, scanned, packets, *, schemas, motion_rate=None):
    if record is None:
        _require(not packets, "Repeated sequence lacks a recorded scan")
        return
    motion = motion_rate is not None
    schema = record["schema"]
    _require(schema in schemas,
             "Repeated sequence lacks completed coverage")
    if record["status"] != "matched":
        _require(record["status"] in {"unavailable", "no_match"} and record["pair"] is None and not packets,
                 "Repeated sequence evidence disagrees with its status")
        return
    pair = record["pair"]
    reverse = schema in {REVERSE_SCHEMA, MOTION_REVERSE_SCHEMA}
    retimed = schema == MOTION_RETIMED_SCHEMA
    reflected = schema == MOTION_REFLECTED_SCHEMA
    _require(len(packets) == 1 and packets[0].get("picture_sequence") == pair,
             "Repeated sequence omits the recorded pair")
    _require(("supporting_edge_offsets" in pair) == motion and pair.get("order") == ("reverse" if reverse else None),
             "Repeated sequence has the wrong matching method")
    _require(("frame_steps" in pair) == retimed, "Repeated sequence has unexpected frame strides")
    _require(("spatial_transform" not in pair and not reflected)
             or (reflected and pair.get("spatial_transform") == HORIZONTAL_REFLECTION),
             "Repeated sequence has the wrong spatial transform")
    strides = pair.get("frame_steps", {"first": 1, "second": 1})
    if retimed:
        _require(set(strides) == {"first", "second"} and all(type(step) is int for step in strides.values())
                 and sorted(strides.values()) == [1, 2], "Retimed sequence requires a two-to-one frame stride")
    tb = Fraction(source["time_base"])
    count = pair["frame_count"]
    _require(type(count) is int and MIN_UNIQUE <= count <= scanned // 2,
             "Repeated sequence does not contain enough distinct moving pictures")
    if motion:
        supports = pair["supporting_edge_offsets"]
        _require(all(type(value) is int for value in supports) and len(supports) >= 8 and len(supports) * 2 >= count - 1
                 and supports[0] == 0 and supports[-1] == count - 2 and all(0 < b - a <= 3 for a, b in zip(supports, supports[1:])),
                 "Motion sequence lacks enough repeated adjacent movement")
    else:
        unique = pair["unique_pictures"]
        _require(type(unique) is int and MIN_UNIQUE <= unique <= count, "Repeated sequence lacks distinct pictures")
    spans = (pair["first"], pair["second"])
    if schema == MOTION_EXTENDED_SCHEMA:
        _, opening_limit = MOTION_PROFILES[(source["width"], source["height"])]
        _require(spans[1]["end_frame_index"] >= opening_limit, "Extended motion sequence lies wholly inside the opening scope")
    _require(spans[0]["end_frame_index"] < spans[1]["start_frame_index"], "Repeated sequences overlap")
    intervals = []
    for side, span in zip(("first", "second"), spans, strict=True):
        stride = strides[side]
        start, end, pts = span["start_frame_index"], span["end_frame_index"], span["pts"]
        _require(type(start) is int and type(end) is int and 0 <= start < end < scanned
                 and end - start == stride * (count - 1) and len(pts) == count
                 and all(type(p) is int for p in pts), "Repeated sequence has an invalid source interval")
        steps = _steps(pts, tb)[1:]
        elapsed = (pts[-1] - pts[0]) * tb
        _require(None not in steps and elapsed / stride >= MIN_SPAN and _close(pair["span_s"], elapsed / stride)
                 and _close(span["start_stream_s"], pts[0] * tb) and _close(span["end_stream_s"], pts[-1] * tb),
                 "Repeated sequence has incomplete clock support")
        intervals.append([Fraction(step, stride) for step in steps] if retimed else steps)
    _require(intervals[0] == (intervals[1][::-1] if reverse else intervals[1]),
             "Repeated sequence changes the native frame spacing")
    if motion:
        _require(all(step * tb == Fraction(1, motion_rate) for step in intervals[0]), "Motion sequence uses unsupported frame spacing")
    for span in spans if motion else (pair,):
        digests = span["endpoint_rgb_sha256"]
        _require(len(digests) == 2 and all(_SHA.fullmatch(d) for d in [*digests, span["rgb_sequence_sha256"]]),
                 "Repeated sequence has invalid image commitments")
    packet = packets[0]
    _require(packet.get("claim_scope") == "paired_frame_sequences" and packet.get("semantics") == schema
             and packet.get("evidence_type") == "anomaly" and packet.get("affects_overall_confidence") is False
             and packet.get("signal_refs") == packet.get("finding_ids") == packet.get("exhibit_refs") == []
             and packet.get("observation") == observation(pair)
             and packet.get("interpretation") == interpretation(pair),
             "Repeated sequence is misrepresented as statistical evidence")
    refs = {picture_ref(pair, occurrence, role) for occurrence, role, *_ in endpoints(pair)}
    _require(set(packet["artifacts"]) == refs | {"contact_sheet"}, "Repeated sequence lacks endpoint images")
    for frame, (occurrence, role, i, pts, digest) in zip(packet["frames"], endpoints(pair), strict=True):
        anchor = frame["anchor"]
        ref = picture_ref(pair, occurrence, role)
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


def validate_clock(record, table_pts: Sequence[int], interpolated: Sequence[int], *, tb: Fraction):
    fixed_clock = {MOTION_EXTENDED_SCHEMA, MOTION_REVERSE_SCHEMA, MOTION_RETIMED_SCHEMA, MOTION_REFLECTED_SCHEMA}
    if record and record["schema"] in fixed_clock and record["status"] != "unavailable":
        limit = record["frames_scanned"]
        pts = table_pts[:limit]
        step = Fraction(1, record["parameters"]["frame_rate"])
        _require(len(pts) == limit and not set(interpolated).intersection(range(limit))
                 and all((b - a) * tb == step for a, b in zip(pts, pts[1:])),
                 "Motion sequence lacks a complete verified native clock")
    if not record or record["pair"] is None:
        return
    inferred = set(interpolated)
    for key in ("first", "second"):
        span = record["pair"][key]
        start, end = span["start_frame_index"], span["end_frame_index"]
        stride = record["pair"].get("frame_steps", {}).get(key, 1)
        _require(span["pts"] == list(table_pts[start:end + 1:stride]) and not inferred.intersection(range(start, end + 1)),
                 "Repeated sequence differs from the committed source frame table")
