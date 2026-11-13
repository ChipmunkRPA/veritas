"""Portable validation for one optional temporary upper-band drop."""

from __future__ import annotations

import math

RATE, CELL, FLANK = 16_000, 1280, 5
LENGTHS = (7, 9, 11, 13, 15)
POINT_DB = 16.
BANDWIDTH_METHOD = {
    "rate_hz": RATE, "cell_samples": CELL, "flank_cells": FLANK, "range_cells": list(LENGTHS),
    "bands_hz": [[500, 1900], [1900, 2700], [4100, 5500], [5500, 7400]], "point_db": POINT_DB,
    "minimum_upper_flank_dbfs": -80., "maximum_upper_flank_difference_db": 12.,
    "maximum_middle_change_db": 7., "flank_percentile": 50, "interval_percentile": 65,
}


def validate(detail, *, start, end, samples):
    if (samples != 6 * RATE or not FLANK * CELL <= start < end <= samples - FLANK * CELL
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
