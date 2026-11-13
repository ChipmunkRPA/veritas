# SPDX-License-Identifier: Apache-2.0
"""Native timing commitments for the AVC MP4 and VP9 WebM face window review."""

from __future__ import annotations

from fractions import Fraction

from app.evidence import face_paired_edge_binding as paired
from app.evidence import face_short_proposal_binding as proposal
from app.evidence import face_short_window_binding as windows

METHOD = "face-short-return/11"
ENCODINGS = {("h264", "yuv420p", "avc1"): "mp4", ("vp9", "yuv420p", "[0][0][0][0]"): "webm"}
META = frozenset(("source_encoding", "source_carrier", "source_first_pts", "source_time_base"))


def tick(index, base, carrier):
    if carrier == "webm":
        paired.require(base == Fraction(1, 1000), "WebM face review requires whole-millisecond ticks")
        return (100 * index + 1) // 3
    ticks = Fraction(index, 30) / base
    paired.require(ticks.denominator == 1, "MP4 face review requires a complete native 30 fps clock")
    return int(ticks)


def supported(table, encoding, carrier):
    try:
        return (
            ENCODINGS.get(tuple(encoding)) == carrier
            and carrier is not None
            and (table.width, table.height) == (table.coded_width, table.coded_height) == paired.SIZE
            and len(table) == windows.COUNT
            and not (table.rotation or table.truncated or table.interpolated)
            and all(p - table.pts[0] == tick(i, table.tb, carrier) for i, p in enumerate(table.pts))
        )
    except ValueError:
        return False


def source_metadata(table, encoding):
    return {
        "source_encoding": list(encoding),
        "source_carrier": ENCODINGS[tuple(encoding)],
        "source_first_pts": table.pts[0],
        "source_time_base": table.time_base,
    }


def validate_metadata(record, source, *, table_pts=None):
    first, carrier, base = (
        record["source_first_pts"],
        record["source_carrier"],
        Fraction(record["source_time_base"]),
    )
    paired.require(
        isinstance(record["source_encoding"], list)
        and ENCODINGS.get(tuple(record["source_encoding"])) == carrier
        and carrier in {"mp4", "webm"}
        and type(first) is int
        and base > 0
        and record["source_time_base"] == source["time_base"]
        and source["frame_count"] == windows.COUNT
        and (source["width"], source["height"]) == paired.SIZE
        and tick(1, base, carrier) > 0
        and (
            table_pts is None
            or len(table_pts) == windows.COUNT
            and all(p - first == tick(i, base, carrier) for i, p in enumerate(table_pts))
        ),
        "Overlapping face windows differ from the measured native carrier or cadence",
    )
    return first, carrier, base


def local_record(record):
    # The prior validator works on a 120-picture exact clock. Here only its internal
    # arithmetic sees synthetic ticks. The supplied source PTS were checked separately.
    offset = record["window_start_frame_index"]
    timed = {
        **record,
        "time_base": "1/30000",
        "source_first_pts": 0,
        "source_encoding": list(paired.ENCODING),
        "source_carrier": "mp4",
    }
    for key in ("start", "end"):
        i = record[f"{key}_frame_index"]
        timed[f"{key}_pts"] = i * 1000
        timed[f"{key}_stream_s"] = i / 30
    for key in ("frame_inputs", "scene_frame_inputs"):
        timed[key] = [{**row, "pts": row["frame_index"] * 1000} for row in record[key]]
    translated = windows.local_record(timed)
    paired.require(translated["source_first_pts"] == offset * 1000, "Invalid internal window offset")
    return translated


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    offset = record["window_start_frame_index"]
    paired.require(
        record["method"] == METHOD
        and type(offset) is int
        and offset in windows.WINDOW_STARTS
        and not (rotation or truncated or interpolated),
        "Unsupported overlapping face window source",
    )
    first, carrier, base = validate_metadata(
        {**record, "source_time_base": record["time_base"]}, source, table_pts=table_pts
    )
    for key in ("start", "end"):
        i = record[f"{key}_frame_index"]
        native = first + tick(i, base, carrier)
        paired.require(
            type(i) is int
            and type(record[f"{key}_pts"]) is int
            and record[f"{key}_pts"] == native
            and abs(record[f"{key}_stream_s"] - float(native * base)) <= 0.0000011,
            "Brief face change does not match its actual source timestamps",
        )
    for key in ("frame_inputs", "scene_frame_inputs"):
        for row in record[key]:
            i = row["frame_index"]
            paired.require(
                type(i) is int and type(row["pts"]) is int and row["pts"] == first + tick(i, base, carrier),
                "Brief face input does not match its actual source timestamp",
            )
    proposal.validate_record(
        local_record(record), {**source, "frame_count": windows.WINDOW, "time_base": "1/30000"}
    )
