"""Validate the public refinement record without the detector runtime."""
from __future__ import annotations

import math
from fractions import Fraction

from app.evidence.face_interval_binding import APPEARANCE_METHOD, face_context_box, validate_face_input
from app.evidence.video_packet_versions import BOUNDARY_VERSION

METHOD = "face-appearance-boundary/1"
MAX_FRAMES = 30
ROLES = ("before", "during", "after")
EDGE_LIMIT = Fraction(3, 8)
MAX_GAP = Fraction(1, 8)
SCHEDULE_MARGIN = Fraction(5, 8)
_INPUT_KEYS = frozenset({"frame_index", "pts", "decoded_rgb_sha256", "bbox", "landmarks", "alignment_matrix",
                         "alignment_matrix_hex", "alignment_side", "aligned_size", "aligned_rgb_sha256"})
_RECORD_KEYS = frozenset({"method", "time_base", "track_id", "start_pts", "end_pts", "start_stream_s", "end_stream_s",
                          "anchors", "scheduled_frame_indices", "frame_inputs", "edges"})


def _overlap(a: dict, b: dict) -> float:
    w = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    intersection = max(w, 0) * max(h, 0)
    return intersection / (a["w"] * a["h"] + b["w"] * b["h"] - intersection)


def _same_time(actual: float, expected: Fraction | float) -> bool:
    return type(actual) in (float, int) and math.isfinite(actual) and abs(actual - float(expected)) <= .0000011


def validate_record(record: dict, coarse: dict, source: dict, *, table_pts=None, interpolated=()) -> None:
    base = Fraction(source["time_base"])
    if (set(record) != _RECORD_KEYS or record["method"] != METHOD or record["time_base"] != source["time_base"]
            or coarse["method"] != APPEARANCE_METHOD or record["track_id"] != coarse["track_id"]
            or not 0 < (coarse["end_pts"] - coarse["start_pts"]) * base <= 1):
        raise ValueError("Refined boundaries require a matching short face appearance measurement")

    schedule = record["scheduled_frame_indices"]
    if (not isinstance(schedule, list) or not 4 <= len(schedule) <= MAX_FRAMES
            or any(type(i) is not int or not 0 <= i < source["frame_count"] for i in schedule)
            or schedule != sorted(set(schedule))):
        raise ValueError("Refined boundary sample schedule is invalid")
    inputs = record["frame_inputs"]
    if not isinstance(inputs, list) or not 4 <= len(inputs) <= len(schedule):
        raise ValueError("Refined boundary input coverage is invalid")
    indices = [item["frame_index"] for item in inputs]
    pts = [item["pts"] for item in inputs]
    start, end = coarse["start_pts"] * base, coarse["end_pts"] * base
    if (any(type(i) is not int for i in indices) or indices != sorted(set(indices)) or not set(indices).issubset(schedule)
            or set(indices).intersection(interpolated) or any(type(p) is not int for p in pts)
            or pts != sorted(set(pts)) or pts[0] * base < start - SCHEDULE_MARGIN or pts[-1] * base > end + SCHEDULE_MARGIN
            or (table_pts is not None and any(table_pts[i] != p for i, p in zip(indices, pts, strict=True)))):
        raise ValueError("Refined boundary inputs differ from the source clock")
    by_index = dict(zip(indices, inputs, strict=True))
    for item in inputs:
        if set(item) != _INPUT_KEYS or set(item["bbox"]) != {"x", "y", "w", "h"}:
            raise ValueError("Refined boundary input has unexpected fields")
        validate_face_input(item, source, appearance=True)

    edges = record["edges"]
    if set(edges) != {"entry", "exit"}:
        raise ValueError("Refined boundary needs two transitions")
    for label, coarse_time in (("entry", start), ("exit", end)):
        edge = edges[label]
        group = edge["frame_indices"]
        values = edge["signed_distance_hex"]
        if (set(edge) != {"frame_indices", "signed_distance_hex"} or len(group) != 4 or len(values) != 4
                or any(type(i) is not int or i not in by_index for i in group)
                or group != schedule[schedule.index(group[0]):schedule.index(group[0]) + 4]):
            raise ValueError("Refined boundary omits adjacent measured frames around a transition")
        try:
            signed = [float.fromhex(value) for value in values]
        except OverflowError as error:
            raise ValueError("Refined boundary response is outside the supported range") from error
        support = [by_index[i] for i in group]
        if (any(not -2.000000000001 <= value <= 2.000000000001 for value in signed)
                or not max(signed[:2]) < 0 < min(signed[2:])
                or abs(support[2]["pts"] * base - coarse_time) > EDGE_LIMIT
                or any((b["pts"] - a["pts"]) * base > MAX_GAP or _overlap(a["bbox"], b["bbox"]) < .35 - 1e-6
                       for a, b in zip(support, support[1:]))):
            raise ValueError("Refined boundary transitions lack measured change or consistent face context")

    entry, exit_group = (edges[key]["frame_indices"] for key in ("entry", "exit"))
    a, b = by_index[entry[2]]["pts"], by_index[exit_group[2]]["pts"]
    if ((b - a) * base < EDGE_LIMIT or any(type(record[key]) is not int for key in ("start_pts", "end_pts"))
            or (record["start_pts"], record["end_pts"]) != (a, b)
            or not _same_time(record["start_stream_s"], a * base)
            or not _same_time(record["end_stream_s"], b * base)):
        raise ValueError("Refined range differs from its measured transitions")
    during = min((i for i in indices if entry[2] <= i < exit_group[2]),
                 key=lambda i: (abs(2 * by_index[i]["pts"] - a - b), i))
    if (record["anchors"] != dict(zip(ROLES, (entry[1], during, exit_group[2]), strict=True))
            or any(type(i) is not int for i in record["anchors"].values())):
        raise ValueError("Refined range omits its representative frames")


def validate_packet(packet: dict, source: dict, version: str, *, table_pts=None, interpolated=()) -> None:
    if not {"face_boundary", "face_boundary_frames"}.intersection(packet):
        return
    record, frames = packet.get("face_boundary"), packet.get("face_boundary_frames")
    if (version != BOUNDARY_VERSION or not isinstance(record, dict) or not isinstance(frames, list)
            or len(frames) != len(ROLES) or packet.get("affects_overall_confidence") is not False
            or packet.get("semantics") != APPEARANCE_METHOD or len(packet["signal_refs"]) != 1):
        raise ValueError("Refined boundaries need a corresponding confidence-neutral face check")
    validate_record(record, packet["face_sequence"], source, table_pts=table_pts, interpolated=interpolated)
    by_index = {row["frame_index"]: row for row in record["frame_inputs"]}
    origin = source["timeline_origin_s"]
    for position, (frame, role) in enumerate(zip(frames, ROLES, strict=True), 1):
        index = record["anchors"][role]
        item, anchor = by_index[index], frame["anchor"]
        instant = item["pts"] * Fraction(source["time_base"])
        refs = {key: f"boundary_frame_{position}_{key}" for key in ("original", "context", "crop")}
        if (frame["id"] != f"{packet['id']}.B{position}" or frame.get("interval_role") != role
                or frame.get("score") is not None or frame.get("is_peak") is not False
                or frame.get("context_region") != face_context_box(item["bbox"], source)
                or frame.get("region_meaning") != "detected_face_context"
                or frame["artifact_refs"] != refs or any(packet["artifacts"][ref]["media_type"] != "image/png" for ref in refs.values())
                or anchor["frame_index"] != index or anchor["pts"] != item["pts"]
                or anchor["decoded_rgb_sha256"] != item["decoded_rgb_sha256"]
                or anchor["original_sha256"] != source["sha256"] or anchor["stream_selector"] != "v:0"
                or anchor["time_base"] != source["time_base"] or anchor["coordinate_space"] != source["coordinate_space"]
                or not _same_time(anchor["stream_time_s"], instant)
                or (origin is None and anchor["playback_time_s"] is not None)
                or (origin is not None and not _same_time(anchor["playback_time_s"], float(instant) - origin))):
            raise ValueError("Refined boundary image differs from its measured source position")
