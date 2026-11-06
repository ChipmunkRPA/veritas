"""Validate the public interval evidence without loading detector dependencies."""
from __future__ import annotations

from fractions import Fraction
from statistics import mean

METHOD = "face-track-response/1"
MAX_GAP_S = 0.8
MAX_INTERVAL_S = 4.0
CONTEXT_COUNT = 2
MIN_CORE_COUNT = 3


def validate_packet(packet: dict, source: dict, *, table_pts: tuple[int, ...] | None = None, interpolated: tuple[int, ...] = ()) -> None:
    validate_sequence_packet(packet, source, table_pts=table_pts, interpolated=interpolated)
    interval = packet.get("face_interval")
    if interval is None and packet.get("semantics") != METHOD:
        return
    if (not isinstance(interval, dict) or packet.get("semantics") != METHOD
            or interval.get("method") != METHOD or interval.get("time_base") != source["time_base"]):
        raise ValueError("Face interval has an invalid method or source clock")
    before, during, after = (interval["samples"][role] for role in ("before", "during", "after"))
    if len(before) != CONTEXT_COUNT or len(after) != CONTEXT_COUNT or len(during) < MIN_CORE_COUNT:
        raise ValueError("Face interval lacks measured context or support")
    support = [*before, *during, *after]
    indices = [s["frame_index"] for s in support]
    pts = [s["pts"] for s in support]
    base = Fraction(source["time_base"])
    if (indices != sorted(set(indices)) or any(not 0 <= i < source["frame_count"] for i in indices)
            or pts != sorted(set(pts)) or any(float((b-a)*base) > MAX_GAP_S + 1e-6 for a, b in zip(pts, pts[1:]))
            or any(not 0 <= s["score"] <= 1 for s in support)
            or (table_pts is not None and any(table_pts[i] != p or i in interpolated for i, p in zip(indices, pts, strict=True)))):
        raise ValueError("Face interval has inconsistent frame support")
    a, b = during[0]["pts"], during[-1]["pts"]
    if ((interval["start_pts"], interval["end_pts"]) != (a, b)
            or abs(interval["start_stream_s"] - float(a * base)) > 1e-6
            or abs(interval["end_stream_s"] - float(b * base)) > 1e-6):
        raise ValueError("Face interval does not match the first and last supported samples")
    lift = mean(s["score"] for s in during) - max(mean(s["score"] for s in before), mean(s["score"] for s in after))
    if float((b-a)*base) > MAX_INTERVAL_S + 1e-6 or not 0 < interval["lift"] <= 1 or abs(lift - interval["lift"]) > 1e-6:
        raise ValueError("Face interval has an invalid response difference")
    peak = max(during, key=lambda s: s["score"])
    expected = [("before", before[-1]), ("during", peak), ("after", after[0])]
    if len(packet["frames"]) != len(expected):
        raise ValueError("Face interval must show before, during and after frames")
    for frame, (role, sample) in zip(packet["frames"], expected, strict=True):
        anchor = frame["anchor"]
        if (frame.get("interval_role") != role or anchor["frame_index"] != sample["frame_index"]
                or anchor["pts"] != sample["pts"] or abs(frame["score"] - sample["score"]) > 1e-6
                or not frame.get("context_region") or frame.get("is_peak") != (role == "during")):
            raise ValueError("Face interval image differs from its measured role")


SEQUENCE_METHOD = "face-sequence-change/1"
SEQUENCE_REVIEW_POINT = 0.007326911389827726


def validate_sequence_record(record: dict, source: dict, *, table_pts=None, interpolated=(), require_flag=True) -> None:
    """Check window inputs and arithmetic without importing inference dependencies."""
    import math
    import re

    if record.get("method") != SEQUENCE_METHOD or record.get("time_base") != source["time_base"]:
        raise ValueError("Face sequence has an invalid method or source clock")
    groups = record["centers"]
    before, during, after = (groups[role] for role in ("before", "during", "after"))
    centers = [*before, *during, *after]
    if (len(before) != 2 or len(after) != 2 or len(during) < 3 or len(centers) > 64
            or any(type(i) is not int for i in centers) or centers != sorted(set(centers))):
        raise ValueError("Face sequence lacks ordered measured context")
    inputs = record["frame_inputs"]
    indices = [item["frame_index"] for item in inputs]
    expected = sorted({i+k for i in centers for k in (-1, 0, 1)})
    if (indices != expected or any(type(i) is not int or not 0 <= i < source["frame_count"] for i in indices)
            or set(indices).intersection(interpolated)):
        raise ValueError("Face sequence does not bind all consecutive input frames")
    by_index = {item["frame_index"]: item for item in inputs}
    pts = [item["pts"] for item in inputs]
    base = Fraction(source["time_base"])
    if (any(type(p) is not int for p in pts) or pts != sorted(set(pts))
            or (table_pts is not None and any(table_pts[i] != p for i, p in zip(indices, pts, strict=True)))):
        raise ValueError("Face sequence input timestamps differ from the source")
    for item in inputs:
        box = item["bbox"]
        matrix = item["alignment_matrix"]
        exact_matrix = [[float.fromhex(value) for value in row] for row in item["alignment_matrix_hex"]]
        landmarks = item["landmarks"]
        if (not all(re.fullmatch(r"[a-f0-9]{64}", str(item.get(key, ""))) for key in ("decoded_rgb_sha256", "aligned_rgb_sha256"))
                or item.get("aligned_size") != 112 or type(item.get("alignment_side")) is not int or item["alignment_side"] <= 0
                or len(matrix) != 2 or any(len(row) != 3 for row in matrix)
                or len(exact_matrix) != 2 or any(len(row) != 3 for row in exact_matrix)
                or any(not math.isfinite(exact) or abs(exact-approx) > .000001
                       for raw, approximate in zip(exact_matrix, matrix, strict=True)
                       for exact, approx in zip(raw, approximate, strict=True))
                or len(landmarks) != 5 or any(len(row) != 2 for row in landmarks)
                or any(not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
                       for row in [*matrix, *landmarks] for value in row)
                or any(not isinstance(box[k], int | float) or isinstance(box[k], bool) or not math.isfinite(box[k]) for k in ("x", "y", "w", "h"))
                or min(box["x"], box["y"]) < 0 or min(box["w"], box["h"]) <= 0
                or box["x"]+box["w"] > source["width"] or box["y"]+box["h"] > source["height"]):
            raise ValueError("Face sequence has invalid source crop commitments")
    stamps = [by_index[i]["pts"] for i in centers]
    if (any(not 0 < float((b-a)*base) <= .55+1e-6 for a, b in zip(stamps, stamps[1:]))
            or any(float((by_index[i+1]["pts"]-by_index[i-1]["pts"])*base) > .13+1e-6 for i in centers)):
        raise ValueError("Face sequence has unsupported source spacing")
    a, b = by_index[during[0]]["pts"], by_index[during[-1]]["pts"]
    if (not .52-1e-6 <= float((b-a)*base) <= 2.6+1e-6
            or (record["start_pts"], record["end_pts"]) != (a, b)
            or abs(record["start_stream_s"]-float(a*base)) > 1e-6
            or abs(record["end_stream_s"]-float(b*base)) > 1e-6):
        raise ValueError("Face sequence range differs from its measured core")
    measures = [record[key] for key in ("response", "inner_change", "outer_change", "flank_change")]
    if any(not isinstance(x, int | float) or isinstance(x, bool) or not math.isfinite(x) for x in measures):
        raise ValueError("Face sequence has a nonfinite response")
    response, inner, outer, flank = measures
    if ((require_flag and response < SEQUENCE_REVIEW_POINT) or not all(-1e-5 <= value <= 2.00001 for value in (inner, outer, flank))
            or abs(response-(inner-.60*outer-.35*flank)) > .0000011):
        raise ValueError("Face sequence review response has inconsistent arithmetic")


def validate_sequence_packet(packet: dict, source: dict, *, table_pts=None, interpolated=()) -> None:
    import math

    record = packet.get("face_sequence")
    if record is None and packet.get("semantics") != SEQUENCE_METHOD:
        return
    if (not isinstance(record, dict) or packet.get("semantics") != SEQUENCE_METHOD
            or packet.get("claim_scope") != "sampled_window" or packet.get("evidence_type") != "anomaly"):
        raise ValueError("Face sequence misrepresents its measurement scope")
    validate_sequence_record(record, source, table_pts=table_pts, interpolated=interpolated)
    groups = record["centers"]
    by_index = {item["frame_index"]: item for item in record["frame_inputs"]}
    expected = [("before", groups["before"][-1]), ("during", groups["during"][len(groups["during"])//2]),
                ("after", groups["after"][0])]
    if len(packet["frames"]) != 3 or packet["window"]["frame_indices"] != sorted(by_index):
        raise ValueError("Face sequence omits source coverage or representative frames")
    for frame, (role, index) in zip(packet["frames"], expected, strict=True):
        item = by_index[index]
        box = item["bbox"]
        x, y = math.floor(box["x"]), math.floor(box["y"])
        expected_box = {"x": x, "y": y, "w": min(source["width"], math.ceil(box["x"]+box["w"]))-x,
                        "h": min(source["height"], math.ceil(box["y"]+box["h"]))-y}
        if (frame.get("interval_role") != role or frame.get("score") is not None or frame.get("is_peak") is not False
                or frame["anchor"]["frame_index"] != index or frame["anchor"]["pts"] != item["pts"]
                or frame["anchor"]["decoded_rgb_sha256"] != item["decoded_rgb_sha256"] or frame.get("context_region") != expected_box):
            raise ValueError("Face sequence image differs from its measured role")
