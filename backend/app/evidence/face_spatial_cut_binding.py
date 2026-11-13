# SPDX-License-Identifier: Apache-2.0
"""Portable evidence checks for a possible picture cut around a coarse face change."""

from __future__ import annotations

import math
from fractions import Fraction

from app.evidence import face_paired_edge_binding as paired
from app.evidence.face_interval_binding import APPEARANCE_METHOD
from app.evidence.video_packet_versions import SPATIAL_CUT_VERSION

METHOD = "face-picture-cut-context/1"
POINT = 0.30
ROLES = ("before", "during", "after")


def schedule(a, b):
    roles = [max(0, a - 4), (a + b) // 2, min(119, b + 4)]
    return sorted(set(range(max(0, a - 17), min(120, b + 18))) | set(roles)), roles


def span(first_pts, record, base):
    values = [(record[key] - first_pts) * base * paired.RATE for key in ("start_pts", "end_pts")]
    paired.require(
        all(value.denominator == 1 and 0 <= value < 120 for value in values),
        "Picture cut differs from its native clock",
    )
    return tuple(map(int, values))


def validate_record(record, coarse, source, *, table_pts=None, interpolated=()):
    require = paired.require
    base = Fraction(source["time_base"])
    first = record["source_first_pts"]
    require(
        set(record)
        == {
            "method",
            "time_base",
            "source_first_pts",
            "source_encoding",
            "source_carrier",
            "start_pts",
            "end_pts",
            "start_stream_s",
            "end_stream_s",
            "start_frame_index",
            "end_frame_index",
            "coarse_start_pts",
            "coarse_end_pts",
            "score_hex",
            "shared_tiles",
            "entry_tiles",
            "exit_tiles",
            "sectors",
            "tiles",
            "face_inputs",
            "frame_inputs",
            "anchors",
        }
        and record["method"] == METHOD
        and record["time_base"] == coarse["time_base"] == source["time_base"]
        and record["source_encoding"] == list(paired.ENCODING)
        and record["source_carrier"] == "mp4"
        and (source["width"], source["height"]) == paired.SIZE
        and source["frame_count"] == paired.FRAME_COUNT
        and type(first) is int
        and not interpolated
        and coarse["method"] == APPEARANCE_METHOD
        and (
            table_pts is None
            or (
                len(table_pts) == 120
                and all((p - first) * base == Fraction(i, 30) for i, p in enumerate(table_pts))
            )
        )
        and (record["coarse_start_pts"], record["coarse_end_pts"])
        == (coarse["start_pts"], coarse["end_pts"]),
        "Picture-cut context differs from its original appearance range or delivery profile",
    )
    a, b = span(first, coarse, base)
    left, right = span(first, record, base)
    require(
        5 <= b - a <= 42
        and 5 <= right - left <= 24
        and 6 <= left < right <= 114
        and abs(left - a) <= 12
        and abs(right - b) <= 12
        and 2 * (min(right, b) - max(left, a)) >= min(right - left, b - a)
        and (record["start_frame_index"], record["end_frame_index"]) == (left, right),
        "Picture-cut boundaries are outside their local search",
    )
    for key in ("start", "end"):
        require(
            type(record[f"{key}_pts"]) is int
            and math.isclose(
                record[f"{key}_stream_s"], float(record[f"{key}_pts"] * base), rel_tol=0, abs_tol=0.0000011
            ),
            "Picture-cut context differs from its source time",
        )
    planned, roles = schedule(a, b)
    frames = paired.frame_rows(record["frame_inputs"], source, planned, first, base)
    require(list(frames) == planned, "Picture-cut context omits a native frame input")
    face_indices = []
    for row in record["face_inputs"]:
        i, box = row["frame_index"], row["bbox"]
        require(
            set(row) == {"frame_index", "bbox"}
            and i in roles
            and set(box) == {"x", "y", "w", "h"}
            and all(type(v) is int for v in box.values())
            and 0 <= box["x"] <= 1280 - box["w"]
            and 0 <= box["y"] <= 720 - box["h"]
            and min(box["w"], box["h"]) >= 48,
            "Picture-cut context has an invalid native face exclusion",
        )
        face_indices.append(i)
    require(
        2 <= len(face_indices) <= 3 and face_indices == sorted(set(face_indices)),
        "Picture-cut context has insufficient face exclusion support",
    )
    tiles, shared, entry, exit_ = (
        record[key] for key in ("tiles", "shared_tiles", "entry_tiles", "exit_tiles")
    )
    require(
        all(type(v) is int for v in (tiles, shared, entry, exit_, record["sectors"]))
        and 5 <= shared <= min(entry, exit_) <= max(entry, exit_) <= tiles <= 350
        and 2 <= record["sectors"] <= 4
        and paired.exact(record["score_hex"], POINT, 1.0) >= POINT,
        "Picture-cut context has insufficient surrounding texture support",
    )
    require(
        record["anchors"] == dict(zip(ROLES, (left - 1, (left + right) // 2, right), strict=True)),
        "Picture-cut context has wrong representative frames",
    )


def observation(record):
    return (
        f"Picture detail outside the face changed at both ends of {record['start_stream_s']:.3f}–{record['end_stream_s']:.3f} s. "
        "A cut between shots is one plausible explanation for this face change."
    )


def validate_set(document):
    packets = document["packets"]
    count = sum("face_spatial_cut" in packet for packet in packets)
    if count != int(document["version"] == SPATIAL_CUT_VERSION) or (
        count and any(packet.get("face_return") for packet in packets)
    ):
        raise ValueError("The evidence version disagrees with its picture-cut context")


def validate_packet(packet, source, version, *, table_pts=None, interpolated=()):
    if not {"face_spatial_cut", "face_spatial_cut_frames"} & packet.keys():
        return
    record, frames = packet["face_spatial_cut"], packet["face_spatial_cut_frames"]
    require = paired.require
    require(
        version == SPATIAL_CUT_VERSION
        and packet.get("affects_overall_confidence") is False
        and packet.get("semantics") == APPEARANCE_METHOD
        and len(frames) == 3
        and len(packet["signal_refs"]) == 1,
        "Picture-cut context needs its own confidence-neutral evidence version",
    )
    validate_record(record, packet["face_sequence"], source, table_pts=table_pts, interpolated=interpolated)
    by_index = {row["frame_index"]: row for row in record["frame_inputs"]}
    base = Fraction(source["time_base"])
    origin = source["timeline_origin_s"]
    for pos, (frame, role) in enumerate(zip(frames, ROLES, strict=True), 1):
        index = record["anchors"][role]
        item = by_index[index]
        anchor = frame["anchor"]
        stamp = float(item["pts"] * base)
        refs = {key: f"cut_frame_{pos}_{key}" for key in ("original", "context")}
        require(
            frame["id"] == f"{packet['id']}.C{pos}"
            and frame["interval_role"] == role
            and frame["score"] is None
            and frame["is_peak"] is False
            and frame["context_region"] is None
            and frame["region_meaning"] == "whole_frame_context"
            and frame["artifact_refs"] == refs
            and all(packet["artifacts"][ref]["media_type"] == "image/png" for ref in refs.values())
            and anchor["frame_index"] == index
            and all(anchor[key] == item[key] for key in ("pts", "decoded_rgb_sha256"))
            and anchor["original_sha256"] == source["sha256"]
            and anchor["stream_selector"] == "v:0"
            and anchor["time_base"] == source["time_base"]
            and anchor["coordinate_space"] == source["coordinate_space"]
            and math.isclose(anchor["stream_time_s"], stamp, rel_tol=0, abs_tol=0.0000011)
            and (
                anchor["playback_time_s"] is None
                if origin is None
                else math.isclose(anchor["playback_time_s"], stamp - origin, rel_tol=0, abs_tol=0.0000011)
            ),
            "Picture-cut context image differs from its source picture",
        )
