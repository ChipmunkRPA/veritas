"""Portable validation for one optional temporary upper-band drop."""

from __future__ import annotations

import math

RATE, CELL, FLANK = 16_000, 1280, 5
LENGTHS = (7, 9, 11, 13, 15)
POINT_DB = 16.
EXPANDED_DURATIONS = (8, 12)
EXPANDED_SCHEMA = "chipmunk.acoustic-context/4"
EXPANDED_VERSION = "mixed-acoustics-4"
LEGACY_SCHEMA = "chipmunk.acoustic-context/3"
LEGACY_VERSION = "mixed-acoustics-3"
BANDWIDTH_SCHEMAS = {LEGACY_SCHEMA, EXPANDED_SCHEMA}
BANDWIDTH_METHOD = {
    "rate_hz": RATE, "cell_samples": CELL, "flank_cells": FLANK, "range_cells": list(LENGTHS),
    "bands_hz": [[500, 1900], [1900, 2700], [4100, 5500], [5500, 7400]], "point_db": POINT_DB,
    "minimum_upper_flank_dbfs": -80., "maximum_upper_flank_difference_db": 12.,
    "maximum_middle_change_db": 7., "flank_percentile": 50, "interval_percentile": 65,
}


def native_input(samples, source, version):
    formats = {("wav", "pcm_s16le")}
    durations = (6,)
    if version == EXPANDED_VERSION:
        formats.add(("flac", "flac"))
        durations = (6, *EXPANDED_DURATIONS)
        if samples == 6 * RATE and source.get("source_container") == "wav":
            return False  # This input retains the earlier evidence version.
        if source.get("source_bit_depth") != 16 or source.get("timeline_time_base") != "1/16000":
            return False
    return ((source.get("source_container"), source.get("source_codec")) in formats
            and any(samples == seconds * RATE and source.get("source_duration_s") == seconds for seconds in durations)
            and source.get("source_channels") == 1 and source.get("source_sample_rate") == RATE
            and len(source.get("audio_stream_indices", [0])) == 1
            and source.get("timeline") == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}])


def maximum_ranges(samples):
    return sum(max(0, samples // CELL - 2 * FLANK - cells + 1) for cells in LENGTHS)


def validate(detail, *, start, end, samples):
    if (not FLANK * CELL <= start < end <= (samples // CELL - FLANK) * CELL
            or start % CELL or end % CELL or (end - start) // CELL not in LENGTHS):
        raise ValueError("Temporary upper-frequency evidence has invalid source extent")
    if detail["range"] != [start, end]:
        raise ValueError("Temporary upper-frequency evidence has mismatched source extent")
    left, interior, right = detail["relative_band_db"]
    contrast = [min(a, c) - b for a, b, c in zip(left, interior, right, strict=True)]
    measured = detail["drop_db"]
    if (len(contrast) != 3 or not all(math.isfinite(db) and abs(db) <= 121 for side in (left, interior, right) for db in side)
            or not math.isfinite(measured) or measured <= POINT_DB or abs(min(contrast[1:]) - measured) > .000002
            or abs(contrast[0]) > 7.000002 or max(abs(a - b) for a, b in zip(left[1:], right[1:], strict=True)) > 12.000002):
        raise ValueError("Temporary upper-frequency evidence differs from its band measurements")
    return measured
