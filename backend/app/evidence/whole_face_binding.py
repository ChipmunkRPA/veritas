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
# Each arm is centered and scaled by its median and 90th percentile on the
# same ten development originals. The paired point was frozen on those ten.
REFERENCE_SCALE = (0.024325260892510414, 0.43718233704566956)
COMPLEMENT_SCALE = (-3.6829073429107666, 0.11348772048950195)
REVIEW_POINT = 1.2940809706865464


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


def validate(record: dict, source: dict, *, pts: Sequence[int] | None = None) -> tuple[int, ...]:
    """Verify recorded source positions, per-arm support and the paired result."""
    if (record.get("schema") != METHOD or record.get("time_base") != source["time_base"]
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
    score = paired_score(*zip(*responses, strict=True))
    if score is None or score.hex() != record["score_hex"] or (score > REVIEW_POINT) != record["flagged"]:
        raise ValueError("The paired face outcome does not match its recorded responses")
    return indices


def representatives(record: dict) -> list[dict]:
    measured = [row for row in record["samples"] if all(value is not None for value in row["responses_hex"])]
    return [measured[0], measured[len(measured) // 2], measured[-1]]


def validate_packet(packet: dict, source: dict, *, pts: Sequence[int] | None = None) -> None:
    record = packet.get("whole_face_review")
    if record is None and packet.get("semantics") != METHOD:
        return
    if (not isinstance(record, dict) or packet.get("semantics") != METHOD or packet.get("claim_scope") != "sampled_window"
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
