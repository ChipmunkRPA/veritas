# SPDX-License-Identifier: Apache-2.0
"""Absolute source binding for overlapping instances of the bounded face proposal."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from app.core.canonical import canonical_hash
from app.evidence import face_paired_edge_binding as paired
from app.evidence import face_short_proposal_binding as proposal

METHOD = "face-short-return/10"
KEY = "brief_face_windows"
COMMITMENT = "brief_face_windows_sha256"
CHECKS_COMMITMENT = "brief_face_window_measurements_sha256"
WINDOW_STARTS = (0, 60, 120)
COUNT = 240
WINDOW = 120
REASONS = {
    "time_budget": "The check reached its time limit.",
    "native_measurement_unavailable": "The native pictures could not be measured.",
    "incomplete_native_decode": "Some native pictures could not be read or timestamp-verified.",
    "insufficient_same_shot_context": "The source pictures did not show enough comparable views of one face.",
}


def supported(table, encoding, carrier):
    return (
        tuple(encoding) == paired.ENCODING
        and carrier == "mp4"
        and (table.width, table.height) == (table.coded_width, table.coded_height) == paired.SIZE
        and len(table) == COUNT
        and not (table.rotation or table.truncated or table.interpolated)
        and all((tick - table.pts[0]) * table.tb == Fraction(i, 30) for i, tick in enumerate(table.pts))
    )


def local_record(record):
    offset = record["window_start_frame_index"]
    step = Fraction(1, 30) / Fraction(record["time_base"])
    paired.require(step.denominator == 1, "The window cannot use fractional source ticks")
    local = {k: v for k, v in record.items() if k != "window_start_frame_index"}
    local.update(method=proposal.METHOD, source_first_pts=record["source_first_pts"] + offset * int(step))
    for key in ("start_frame_index", "end_frame_index"):
        local[key] = record[key] - offset
    for key in ("scheduled_frame_indices", "shortlist_edges", "shortlist_anchor_indices"):
        local[key] = [i - offset for i in record[key]]
    local["anchors"] = {role: i - offset for role, i in record["anchors"].items()}
    local["edges"] = {
        role: {**edge, "frame_indices": [i - offset for i in edge["frame_indices"]]}
        for role, edge in record["edges"].items()
    }
    for key in ("frame_inputs", "scene_frame_inputs"):
        local[key] = [{**row, "frame_index": row["frame_index"] - offset} for row in record[key]]
    return local


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    offset = record["window_start_frame_index"]
    first = record["source_first_pts"]
    base = Fraction(source["time_base"])
    paired.require(
        record["method"] == METHOD
        and type(offset) is int
        and offset in WINDOW_STARTS
        and type(first) is int
        and source["frame_count"] == COUNT
        and not (rotation or truncated or interpolated)
        and (
            table_pts is None
            or len(table_pts) == COUNT
            and all((p - first) * base == Fraction(i, 30) for i, p in enumerate(table_pts))
        ),
        "Brief face window differs from its complete native source clock",
    )
    proposal.validate_record(
        local_record(record),
        {**source, "frame_count": WINDOW},
        table_pts=None if table_pts is None else table_pts[offset : offset + WINDOW],
    )


def observations(windows):
    result = []
    for row in windows:
        a, b = (row[k] / 30 for k in ("start_frame_index", "end_frame_index_exclusive"))
        scope = f"Window {a:.3f}–{b:.3f} s from the start of the picture stream"
        status = row["status"]
        if status == "unassessed":
            result.append(f"{scope} was unassessed. {REASONS[row['reason']]}")
        else:
            extent = (
                "supplied a brief face-change range"
                if status == "localized"
                else "supplied no additional brief face-change range"
            )
            result.append(f"{scope} {extent}. {row['measured_face_frames']} face samples were measured.")
    return result


def coverage(windows):
    return checked(
        {"windows": windows, "observations": observations(windows), "affects_overall_confidence": False},
        COUNT,
    )


def checked(record, frame_count):
    paired.require(
        isinstance(record, dict)
        and set(record) == {"windows", "observations", "affects_overall_confidence"}
        and record["affects_overall_confidence"] is False
        and frame_count == COUNT
        and isinstance(record["windows"], list)
        and len(record["windows"]) == len(WINDOW_STARTS),
        "Brief face window coverage requires the complete supported source plan",
    )
    windows = record["windows"]
    for start, row in zip(WINDOW_STARTS, windows, strict=True):
        paired.require(
            set(row)
            == {
                "start_frame_index",
                "end_frame_index_exclusive",
                "status",
                "reason",
                "pixel_frames",
                "scheduled_frames",
                "measured_face_frames",
            }
            and type(row["start_frame_index"]) is int
            and row["start_frame_index"] == start
            and type(row["end_frame_index_exclusive"]) is int
            and row["end_frame_index_exclusive"] == start + WINDOW
            and all(
                type(row[key]) is int for key in ("pixel_frames", "scheduled_frames", "measured_face_frames")
            )
            and 0 <= row["pixel_frames"] <= WINDOW
            and 0 <= row["measured_face_frames"] <= row["scheduled_frames"] <= proposal.MAX_FACE_READS
            and (
                row["status"] == "unassessed"
                and row["reason"] in REASONS
                or row["status"] in {"localized", "no_distinct_range"}
                and row["reason"] is None
                and row["pixel_frames"] == WINDOW
                and row["scheduled_frames"] >= 12
                and row["measured_face_frames"] >= 4
                and (row["status"] != "localized" or row["measured_face_frames"] >= 11)
            ),
            "Brief face window has invalid measured coverage",
        )
    paired.require(
        record["observations"] == observations(windows), "Brief face window coverage wording differs"
    )
    return record


def from_result(result, frame_count=COUNT):
    record = (result.get("manifest", {}).get("sampling", {})).get(KEY)
    return checked(record, frame_count) if record is not None else None


def validate_packets(document, packets):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        paired.require(not packets, "Brief face window range has no measured coverage")
        return None
    checked(record, document["source"]["frame_count"])
    windows = [row for row in record["windows"] if row["status"] == "localized"]
    paired.require(len(windows) == len(packets), "Brief face window coverage differs from displayed ranges")
    for row, packet in zip(windows, packets, strict=True):
        selected = packet["face_return"]
        paired.require(
            row["start_frame_index"] == selected["window_start_frame_index"]
            and row["pixel_frames"] == len(selected["scene_frame_inputs"])
            and row["scheduled_frames"] == len(selected["scheduled_frame_indices"])
            and row["measured_face_frames"] == len(selected["frame_inputs"]),
            "Brief face window coverage differs from its recorded inputs",
        )
    for first, second in combinations(packets, 2):
        a, b = first["face_return"], second["face_return"]
        x, y, u, v = a["start_pts"], a["end_pts"], b["start_pts"], b["end_pts"]
        paired.require(
            a["source_first_pts"] == b["source_first_pts"]
            and 2 * max(0, min(y, v) - max(x, u)) < max(y, v) - min(x, u),
            "Brief face window ranges overlap or use different source clocks",
        )
    return record


def validate_commitment(document, packets, expected):
    record = validate_packets(document, packets)
    paired.require(
        expected == (canonical_hash(record) if record is not None else None),
        "Brief face window coverage differs from the signed result",
    )
