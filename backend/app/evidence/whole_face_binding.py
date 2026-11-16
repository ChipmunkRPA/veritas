"""Frozen sampling and paired responses for whole-clip face-swap review."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from fractions import Fraction
from math import isfinite
from statistics import median

FRAME_COUNT = 6
MIN_SCORED_FRAMES = 4
MIN_FACE_PX = 48
METHOD = "paired-whole-face-samples/1"
SAME_FRAME_METHOD = "same-frame-face-samples/1"
METHODS = frozenset({METHOD, SAME_FRAME_METHOD})
SAME_FRAME_ENCODING = ("h264", "yuv420p", "avc1")
SAME_FRAME_DIMENSIONS = (640, 360)
SAME_FRAME_PICTURES = 100
SAME_FRAME_STEP = Fraction(1, 25)
# Each arm is centered and scaled by its median and 90th percentile on the
# same ten development originals. The paired point was frozen on those ten.
REFERENCE_SCALE = (0.024325260892510414, 0.43718233704566956)
COMPLEMENT_SCALE = (-3.6829073429107666, 0.11348772048950195)
REVIEW_POINT = 1.2940809706865464
# Pooled ordinary calibration in docs/video-face-quantization-calibration-2026-11-16.md.
SAME_FRAME_NORMALIZERS = ((0.03397737070918083, 0.6322437501512468),
                          (-3.1351442337036133, 7.178168892860413))
SAME_FRAME_POINT = 0.4046108311641258


def _fixed_clock(pts, time_base) -> bool:
    return len(pts) == SAME_FRAME_PICTURES and all(
        (tick - pts[0]) * time_base == index * SAME_FRAME_STEP for index, tick in enumerate(pts))


def same_frame_source(table, encoding, carrier) -> bool:
    return (encoding == SAME_FRAME_ENCODING and carrier == "mp4"
            and (table.width, table.height) == SAME_FRAME_DIMENSIONS
            and not (table.rotation or table.truncated or table.interpolated)
            and _fixed_clock(table.pts, table.tb))


def review_point(schema: str) -> float:
    return SAME_FRAME_POINT if schema == SAME_FRAME_METHOD else REVIEW_POINT


def sample_indices(pts: Sequence[int]) -> tuple[int, ...]:
    """Choose six interior times by exact integer distance, with earlier ties."""
    if len(pts) < FRAME_COUNT or any(a >= b for a, b in zip(pts, pts[1:])):
        return ()
    denominator = FRAME_COUNT + 1
    start, span = pts[0], pts[-1] - pts[0]
    selected = []
    for step in range(1, denominator):
        target = start * denominator + span * step
        right = bisect_left(pts, target, key=lambda point: point * denominator)
        index = min((right - 1, right), key=lambda i: (abs(pts[i] * denominator - target), i))
        selected.append(index)
    return tuple(selected) if len(set(selected)) == FRAME_COUNT else ()


def paired_score(reference: Sequence[float | None], complement: Sequence[float | None]) -> float | None:
    """Return the greater normalized median when both checks have support."""
    if len(reference) != FRAME_COUNT or len(complement) != FRAME_COUNT:
        raise ValueError("Face review needs the six planned positions per check")
    scores = []
    for arm, (center, upper) in zip((reference, complement), (REFERENCE_SCALE, COMPLEMENT_SCALE), strict=True):
        assessed = [value for value in arm if value is not None]
        if any(not isfinite(value) for value in assessed):
            raise ValueError("Face review responses must be finite")
        if len(assessed) < MIN_SCORED_FRAMES:
            return None
        scores.append((median(assessed) - center) / (upper - center))
    return max(scores)


def same_frame_score(reference: Sequence[float | None], complement: Sequence[float | None]) -> float | None:
    """Combine the two readings at each jointly measured native position."""
    (first_center, first_scale), (second_center, second_scale) = SAME_FRAME_NORMALIZERS
    values = [((first - first_center) / first_scale + (second - second_center) / second_scale) / 2
              for first, second in zip(reference, complement, strict=True)
              if first is not None and second is not None]
    return median(values) if len(values) >= MIN_SCORED_FRAMES else None


def validate(record: dict, source: dict, *, pts: Sequence[int] | None = None) -> tuple[int, ...]:
    """Verify recorded source positions, per-arm support and the paired result."""
    if (record.get("schema") not in METHODS or record.get("time_base") != source["time_base"]
            or type(record.get("flagged")) is not bool or not isinstance(record.get("score_hex"), str)):
        raise ValueError("Paired face evidence has an invalid schema or source clock")
    samples = record["samples"]
    if not isinstance(samples, list) or len(samples) != FRAME_COUNT:
        raise ValueError("Paired face evidence must bind all six planned source positions")
    indices = tuple(row["frame_index"] for row in samples)
    timestamps = tuple(row["pts"] for row in samples)
    if (any(type(i) is not int or not 0 <= i < source["frame_count"] for i in indices)
            or any(type(p) is not int for p in timestamps)
            or any(a >= b for items in (indices, timestamps) for a, b in zip(items, items[1:]))):
        raise ValueError("Paired face positions must be distinct and in source order")
    if pts is not None and (indices != sample_indices(pts) or timestamps != tuple(pts[i] for i in indices)):
        raise ValueError("Paired face evidence differs from the committed source frame table")
    responses = []
    for row in samples:
        box, raw = row["box"], row["responses_hex"]
        if not isinstance(raw, list) or len(raw) != 2 or any(v is not None and not isinstance(v, str) for v in raw):
            raise ValueError("Paired face evidence has malformed responses")
        try:
            first, second = (float.fromhex(value) if value is not None else None for value in raw)
        except (ValueError, OverflowError) as error:
            raise ValueError("Paired face evidence has an invalid numerical response") from error
        if box is not None:
            if not isinstance(box, dict) or set(box) != {"x", "y", "w", "h"} or any(type(v) is not int for v in box.values()):
                raise ValueError("Invalid native face box")
            x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
            if (min(x, y) < 0 or min(w, h) < MIN_FACE_PX or x + w > source["width"] or y + h > source["height"]
                    or first is None or not 0.0 <= first <= 1.0):
                raise ValueError("Paired face evidence falls outside its native face input")
        elif first is not None or second is not None:
            raise ValueError("A face response has no recorded face input")
        responses.append((first, second))
    arms = tuple(zip(*responses, strict=True))
    score = paired_score(*arms)
    if record["schema"] == SAME_FRAME_METHOD:
        time_base = Fraction(source["time_base"])
        if ((source["width"], source["height"]) != SAME_FRAME_DIMENSIONS or source["frame_count"] != SAME_FRAME_PICTURES
                or any((stamp - timestamps[0]) * time_base != (i - indices[0]) * SAME_FRAME_STEP
                       for i, stamp in zip(indices, timestamps, strict=True))
                or (pts is not None and not _fixed_clock(pts, time_base))
                or score is None or score > REVIEW_POINT or not record["flagged"]):
            raise ValueError("Same-frame face evidence is outside its fallback profile")
        score = same_frame_score(*arms)
    if score is None or score.hex() != record["score_hex"] or (score > review_point(record["schema"])) != record["flagged"]:
        raise ValueError("The paired face outcome does not match its recorded responses")
    return indices


def representatives(record: dict) -> list[dict]:
    measured = [row for row in record["samples"] if all(value is not None for value in row["responses_hex"])]
    return [measured[0], measured[len(measured) // 2], measured[-1]]


def validate_packet(packet: dict, source: dict, *, pts: Sequence[int] | None = None) -> None:
    record = packet.get("whole_face_review")
    if record is None and packet.get("semantics") not in METHODS:
        return
    if (not isinstance(record, dict) or packet.get("semantics") != record.get("schema") or packet.get("claim_scope") != "sampled_window"
            or packet.get("affects_overall_confidence") is not False or len(packet.get("signal_refs", ())) != 1):
        raise ValueError("Paired face evidence has an invalid scope or check reference")
    indices = validate(record, source, pts=pts)
    window = packet["window"]
    if (window["frame_indices"] != list(indices) or window["time_base"] != source["time_base"]
            or window["start_pts"] != record["samples"][0]["pts"] or window["end_pts"] != record["samples"][-1]["pts"]):
        raise ValueError("Paired face window differs from the recorded positions")
    time_base = Fraction(source["time_base"])
    origin = source["timeline_origin_s"]

    def check_time(value, expected):
        # The packet is stored as canonical JSON, which rounds decimal seconds.
        if expected is None:
            return value is None
        return type(value) in (int, float) and abs(value - expected) <= 2e-6

    for endpoint in ("start", "end"):
        stream_s = float(window[f"{endpoint}_pts"] * time_base)
        if (not check_time(window[f"{endpoint}_stream_s"], stream_s)
                or not check_time(window[f"{endpoint}_playback_s"], None if origin is None else stream_s - origin)):
            raise ValueError("Paired face window time differs from the source clock")
    frames = packet["frames"]
    if len(frames) != 3:
        raise ValueError("Paired face review needs three source examples")
    for frame, expected in zip(frames, representatives(record), strict=True):
        anchor = frame["anchor"]
        stream_s = float(expected["pts"] * time_base)
        if (anchor["frame_index"] != expected["frame_index"] or anchor["pts"] != expected["pts"]
                or anchor["time_base"] != source["time_base"]
                or not check_time(anchor["stream_time_s"], stream_s)
                or not check_time(anchor["playback_time_s"], None if origin is None else stream_s - origin)
                or frame.get("context_region") != expected["box"] or frame.get("score") is not None or frame.get("is_peak")
                or not {"original", "crop"} <= frame["artifact_refs"].keys()):
            raise ValueError("Displayed face evidence differs from the paired input context")
