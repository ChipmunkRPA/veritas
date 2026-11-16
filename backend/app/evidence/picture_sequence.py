"""Exact repeated moving-picture sequences on the native presentation clock."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence.source_geometry import pixel_bounds

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
MOTION_REFLECTED_HD_SCHEMA = "chipmunk.video-reflected-motion-sequence/2"
MOTION_CROP_SCHEMA = "chipmunk.video-cropped-motion-sequence/1"
MOTION_CROP_ALIGNED_SCHEMA = "chipmunk.video-cropped-motion-sequence/2"
MOTION_CROP_SCHEMAS = frozenset({MOTION_CROP_SCHEMA, MOTION_CROP_ALIGNED_SCHEMA})
MOTION_REFLECTED_SCHEMAS = frozenset({MOTION_REFLECTED_SCHEMA, MOTION_REFLECTED_HD_SCHEMA})
MOTION_SCHEMAS = frozenset({MOTION_SCHEMA, MOTION_EXTENDED_SCHEMA})
MOTION_PACKET_ID = "VPR02"
MOTION_REVERSE_KEY = "reverse_motion_sequence_review"
MOTION_REVERSE_PACKET_ID = "VPR04"
MOTION_RETIMED_KEY = "retimed_motion_sequence_review"
MOTION_RETIMED_PACKET_ID = "VPR05"
MOTION_RETIMED_CONTEXT = "Playback speed changes are common in ordinary editing. This check carries zero weight in the AI estimate."
MOTION_REFLECTED_KEY = "reflected_motion_sequence_review"
MOTION_REFLECTED_PACKET_ID = "VPR06"
MOTION_CROP_KEY = "cropped_motion_sequence_review"
MOTION_CROP_PACKET_ID = "VPR07"
HORIZONTAL_REFLECTION = "horizontal_reflection"
MOTION_REFLECTED_CONTEXT = ("Ordinary editing can reuse footage in a mirrored orientation. The comparison leaves the order of creation "
                            "undetermined and carries zero weight in the AI estimate.")
MOTION_CROP_CONTEXT = ("Ordinary editing can reuse the same footage with a tighter crop. The comparison carries zero weight "
                       "in the AI estimate.")
MOTION_REVIEWS = (
    ("motion_sequence_review", MOTION_PACKET_ID, MOTION_SCHEMAS),
    (MOTION_REVERSE_KEY, MOTION_REVERSE_PACKET_ID, frozenset({MOTION_REVERSE_SCHEMA})),
    (MOTION_RETIMED_KEY, MOTION_RETIMED_PACKET_ID, frozenset({MOTION_RETIMED_SCHEMA})),
    (MOTION_REFLECTED_KEY, MOTION_REFLECTED_PACKET_ID, MOTION_REFLECTED_SCHEMAS),
    (MOTION_CROP_KEY, MOTION_CROP_PACKET_ID, MOTION_CROP_SCHEMAS),
)
MOTION_PROFILES = {(320, 180): (20, 180), (640, 360): (25, 225)}
MOTION_EXTENDED_PROFILES = {(640, 360): (25, 450)}
MOTION_360P_PROFILES = {(640, 360): (25, 225)}
MOTION_CROP_PROFILES = {(640, 360): (25, 225), (1280, 720): (25, 200)}
CROP_WORK_WIDTH, CROP_WORK_HEIGHT = 192, 108
CROP_PERCENTAGES = (70, 75, 80, 85, 90)
CROP_ALIGNED_UNIT = 32
MOTION_ENCODINGS = {("h264", "yuv420p"), ("vp9", "yuv420p")}
HD_ENCODINGS = MOTION_ENCODINGS | {("hevc", "yuv420p")}
HD_DIMENSIONS = frozenset({(1280, 720), (1920, 1080), (720, 1280), (1080, 1920)})
HD_RATES = tuple(Fraction(rate) for rate in ("24", "24000/1001", "25", "30", "30000/1001"))
HD_PREFIX_S = 8
HD_MINIMUM_S = 2
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
    if schema in MOTION_CROP_SCHEMAS:
        return MOTION_CROP_PROFILES.get(dimensions)
    return None


def motion_encoding_allowed(schema, encoding):
    if schema == MOTION_REFLECTED_HD_SCHEMA:
        return tuple(encoding or ()) in HD_ENCODINGS
    return tuple(encoding or ()) in (MOTION_ENCODINGS if schema not in {MOTION_RETIMED_SCHEMA, *MOTION_CROP_SCHEMAS}
                                     else {("h264", "yuv420p")})


def motion_parameters(schema, rate):
    parameters = MOTION_PARAMETERS | {"frame_rate": rate}
    if schema == MOTION_RETIMED_SCHEMA:
        parameters["native_frame_steps"] = [1, 2]
    if schema in MOTION_REFLECTED_SCHEMAS:
        parameters.update(maximum_picture_rms=2.4, minimum_unflipped_picture_rms=4.8,
                          spatial_transform=HORIZONTAL_REFLECTION)
    if schema == MOTION_REFLECTED_HD_SCHEMA:
        parameters.update(frame_rate=str(rate), minimum_motion_rms=float(Fraction(10) / rate),
                          descriptor="box_rgb8_luminance")
    if schema in MOTION_CROP_SCHEMAS:
        parameters.update(luma_grid=[32, 18], maximum_picture_rms=3., minimum_direct_rms=7.,
                          minimum_motion_rms=.37, minimum_motion_cosine=.82, minimum_motion_rms_ratio=.60,
                          minimum_support_fraction=.60, crop_percentages=list(CROP_PERCENTAGES),
                          working_grid=[CROP_WORK_WIDTH, CROP_WORK_HEIGHT])
    if schema == MOTION_CROP_ALIGNED_SCHEMA:
        parameters.update(maximum_picture_rms=3.25, crop_units_per_working_pixel=CROP_ALIGNED_UNIT,
                          landmark_frame_step=6, maximum_landmark_features=100, maximum_crop_proposals=16)
    return parameters


def crop_region(pair, occurrence, width, height):
    crop = pair.get("framing")
    if crop is None or crop["wider_range"] != occurrence:
        return None
    unit = CROP_ALIGNED_UNIT if "alignment" in crop else 1
    left, top, right, bottom = crop["crop_box"]
    work_width, work_height = CROP_WORK_WIDTH * unit, CROP_WORK_HEIGHT * unit
    x, y = left * width // work_width, top * height // work_height
    x2, y2 = -(-right * width // work_width), -(-bottom * height // work_height)
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def valid_crop_box(box, *, alignment=None):
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        return False
    left, top, right, bottom = box
    if alignment == "landmarks":
        width, height = CROP_WORK_WIDTH * CROP_ALIGNED_UNIT, CROP_WORK_HEIGHT * CROP_ALIGNED_UNIT
        return (0 <= left < right <= width and 0 <= top < bottom <= height
                and 69 * width - 100 <= 100 * (right - left)
                and 1000 * (right - left) <= 925 * width + 1000
                and abs(CROP_WORK_HEIGHT * (right - left) - CROP_WORK_WIDTH * (bottom - top))
                <= CROP_WORK_HEIGHT + CROP_WORK_WIDTH)
    if alignment == "centered":
        for percent in CROP_PERCENTAGES:
            w, h = CROP_WORK_WIDTH * CROP_ALIGNED_UNIT, CROP_WORK_HEIGHT * CROP_ALIGNED_UNIT
            outside = (100 - percent) / 200
            if box == [round(w * outside), round(h * outside), round(w * (1 - outside)), round(h * (1 - outside))]:
                return True
        return False
    if alignment is not None:
        return False
    for scale in CROP_PERCENTAGES:
        width, height = round(CROP_WORK_WIDTH * scale / 100), round(CROP_WORK_HEIGHT * scale / 100)
        spare_x, spare_y = CROP_WORK_WIDTH - width, CROP_WORK_HEIGHT - height
        if (right - left, bottom - top) == (width, height):
            return left in (0, spare_x // 2, spare_x) and top in (0, spare_y // 2, spare_y)
    return False


def hd_frame_limit(dimensions, count, rate):
    limit = min(count, math.ceil(HD_PREFIX_S * rate))
    return limit if dimensions in HD_DIMENSIONS and rate in HD_RATES and limit >= math.ceil(HD_MINIMUM_S * rate) else 0


def hd_clock_matches(pts, tb, rate):
    """Check cadence while leaving measured, possibly rounded, timestamp ticks intact."""
    if not pts or any(type(point) is not int for point in pts) or not 0 < tb <= Fraction(1, 1000):
        return False
    step = 1 / (tb * rate)
    first = pts[0]
    # At most one tick of relative error from rounding the first and current timestamp.
    tolerance = step.denominator - 1
    return all(abs((point - first) * step.denominator - index * step.numerator) <= tolerance
               for index, point in enumerate(pts))


def hd_profile(dimensions, pts, tb):
    for rate in HD_RATES:
        limit = hd_frame_limit(dimensions, len(pts), rate)
        if limit and hd_clock_matches(pts[:limit], tb, rate):
            return rate, limit
    return None


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
    if framing := pair.get("framing"):
        return (f"Cropping the {framing['wider_range']} range to the displayed area aligns {pair['frame_count']} pictures "
                f"with the other range. {len(pair['supporting_edge_offsets'])} matching picture changes support the comparison.")
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
    if "framing" in pair:
        return MOTION_CROP_CONTEXT
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
    dimensions = source["width"], source["height"]
    _require(schema in schemas, "Motion sequence has an unsupported method")
    if schema == MOTION_REFLECTED_HD_SCHEMA:
        rate_string = record["parameters"]["frame_rate"]
        _require(rate_string in {str(rate) for rate in HD_RATES}, "HD motion sequence has an unsupported clock")
        rate = Fraction(rate_string)
        limit = hd_frame_limit(dimensions, source["frame_count"], rate)
        expected = canonical_hash(motion_parameters(schema, rate))
        _require(limit > 0 and canonical_hash(record["parameters"]) == expected
                 and motion_encoding_allowed(schema, record["encoding"]), "HD motion sequence has unsupported source scope or parameters")
        _require((record["status"] == "unavailable" and record["frames_scanned"] == 0 and record["reason"] == "source_clock")
                 or (record["status"] != "unavailable" and record["frames_scanned"] == limit and record["reason"] is None),
                 "HD motion sequence has invalid scan coverage")
        return rate
    profile = motion_profile(dimensions, schema)
    _require(profile is not None, "Motion sequence has unsupported source dimensions")
    rate, limit = profile
    _require(exact is not None and scanned >= limit and record["parameters"] == motion_parameters(schema, rate)
             and motion_encoding_allowed(schema, record["encoding"]), "Motion sequence has unsupported source scope or parameters")
    exact_takes_priority = exact["status"] == "matched" and schema not in {MOTION_REFLECTED_SCHEMA, *MOTION_CROP_SCHEMAS}
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
    reflected = schema in MOTION_REFLECTED_SCHEMAS
    cropped = schema in MOTION_CROP_SCHEMAS
    hd = schema == MOTION_REFLECTED_HD_SCHEMA
    _require(len(packets) == 1 and packets[0].get("picture_sequence") == pair,
             "Repeated sequence omits the recorded pair")
    _require(("supporting_edge_offsets" in pair) == motion and pair.get("order") == ("reverse" if reverse else None),
             "Repeated sequence has the wrong matching method")
    _require(("frame_steps" in pair) == retimed, "Repeated sequence has unexpected frame strides")
    _require(("spatial_transform" not in pair and not reflected)
             or (reflected and pair.get("spatial_transform") == HORIZONTAL_REFLECTION),
             "Repeated sequence has the wrong spatial transform")
    _require(("framing" in pair) == cropped, "Repeated sequence has unexpected crop geometry")
    if cropped:
        framing = pair["framing"]
        aligned = schema == MOTION_CROP_ALIGNED_SCHEMA
        fields = {"wider_range", "crop_box"} | ({"alignment"} if aligned else set())
        _require(set(framing) == fields and framing["wider_range"] in {"first", "second"}
                 and (not aligned or framing["alignment"] in {"centered", "landmarks"})
                 and valid_crop_box(framing["crop_box"], alignment=framing.get("alignment")),
                 "Repeated sequence has an unsupported crop")
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
        if cropped:
            _require(5 * len(supports) >= 3 * (count - 1), "Cropped sequence lacks repeated adjacent movement")
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
        _require(None not in steps and elapsed / stride >= MIN_SPAN
                 and _close(span["start_stream_s"], pts[0] * tb) and _close(span["end_stream_s"], pts[-1] * tb),
                 "Repeated sequence has incomplete clock support")
        if hd:
            _require(hd_clock_matches(pts, tb, motion_rate) and (side != "first" or _close(pair["span_s"], elapsed)),
                     "HD motion sequence has unsupported frame spacing")
        else:
            _require(_close(pair["span_s"], elapsed / stride), "Repeated sequence has an incorrect source span")
        intervals.append([Fraction(step, stride) for step in steps] if retimed else steps)
    if not hd:
        _require(intervals[0] == (intervals[1][::-1] if reverse else intervals[1]),
                 "Repeated sequence changes the native frame spacing")
    if motion and not hd:
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
    dimensions = source["width"], source["height"]
    refs.update(f"{picture_ref(pair, occurrence, role)}_crop" for occurrence, role, *_ in endpoints(pair)
                if crop_region(pair, occurrence, *dimensions) is not None)
    _require(set(packet["artifacts"]) == refs | {"contact_sheet"}, "Repeated sequence lacks endpoint images")
    for frame, (occurrence, role, i, pts, digest) in zip(packet["frames"], endpoints(pair), strict=True):
        anchor = frame["anchor"]
        ref = picture_ref(pair, occurrence, role)
        original = packet["artifacts"][ref]
        region = crop_region(pair, occurrence, *dimensions)
        expected_refs = {"original": ref, "context": ref}
        if region:
            crop_ref = f"{ref}_crop"
            expected_refs["crop"] = crop_ref
            asset = packet["artifacts"][crop_ref]
            x, y, right, bottom = pixel_bounds(region, source["width"], source["height"])
            _require(asset["media_type"] == "image/png" and (asset["width"], asset["height"]) == (right - x, bottom - y)
                     and frame.get("region_meaning") == "matched_frame_crop", "Cropped sequence has an invalid source crop")
        origin = source.get("timeline_origin_s")
        stream_s = float(pts * tb)
        _require((anchor["frame_index"], anchor["pts"], anchor["decoded_rgb_sha256"]) == (i, pts, digest)
                 and anchor["time_base"] == source["time_base"] and anchor["original_sha256"] == source["sha256"]
                 and all(anchor[key] == source[key] for key in ("width", "height", "stream_selector", "coordinate_space"))
                 and original["media_type"] == "image/png" and (original["width"], original["height"]) == (source["width"], source["height"])
                 and _close(anchor["stream_time_s"], stream_s)
                 and (anchor["playback_time_s"] is None if origin is None else _close(anchor["playback_time_s"], stream_s - origin))
                 and frame.get("context_region") == region and frame.get("score") is None and not frame.get("is_peak")
                 and frame["artifact_refs"] == expected_refs, "Repeated sequence endpoint differs from its image or clock")


def validate_clock(record, table_pts: Sequence[int], interpolated: Sequence[int], *, tb: Fraction):
    fixed_clock = {MOTION_EXTENDED_SCHEMA, MOTION_REVERSE_SCHEMA, MOTION_RETIMED_SCHEMA, MOTION_REFLECTED_SCHEMA, *MOTION_CROP_SCHEMAS}
    if record and record["schema"] in fixed_clock and record["status"] != "unavailable":
        limit = record["frames_scanned"]
        pts = table_pts[:limit]
        step = Fraction(1, record["parameters"]["frame_rate"])
        _require(len(pts) == limit and not set(interpolated).intersection(range(limit))
                 and all((b - a) * tb == step for a, b in zip(pts, pts[1:])),
                 "Motion sequence lacks a complete verified native clock")
    if record and record["schema"] == MOTION_REFLECTED_HD_SCHEMA and record["status"] != "unavailable":
        limit = record["frames_scanned"]
        rate = Fraction(record["parameters"]["frame_rate"])
        pts = table_pts[:limit]
        _require(len(pts) == limit and not set(interpolated).intersection(range(limit)) and hd_clock_matches(pts, tb, rate),
                 "HD motion sequence lacks a complete verified native clock")
    if not record or record["pair"] is None:
        return
    inferred = set(interpolated)
    for key in ("first", "second"):
        span = record["pair"][key]
        start, end = span["start_frame_index"], span["end_frame_index"]
        stride = record["pair"].get("frame_steps", {}).get(key, 1)
        _require(span["pts"] == list(table_pts[start:end + 1:stride]) and not inferred.intersection(range(start, end + 1)),
                 "Repeated sequence differs from the committed source frame table")
