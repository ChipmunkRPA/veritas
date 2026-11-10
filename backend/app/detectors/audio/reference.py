"""Known speech-reference bindings shared by customer presentation and fusion."""

from __future__ import annotations

import math

from app.detectors.audio import calibration, combined

COMBINED_SHA256 = "e7de202c42b80fc81d8ab7faa0ce3538c6af5a4c28a9830e4b9dfbcaa6b63641"


def combined_tendency(screen):
    """Use only a valid, scoped saved result from this exact calibrated method."""
    p = screen.get("probability_ai")
    ranges = screen.get("checked_source_ranges")
    bound = (
        isinstance(ranges, list)
        and len(ranges) == 1
        and isinstance(ranges[0], dict)
        and ranges[0].get("start_sample") == 0
        and ranges[0].get("end_sample") == 64000
        and ranges[0].get("time_base") == "1/16000"
        and type(ranges[0].get("source_start_pts")) is int
        and type(ranges[0].get("source_end_pts")) is int
        and ranges[0]["source_end_pts"] - ranges[0]["source_start_pts"] == 64000
    )
    valid = (
        screen.get("calibration_sha256") == COMBINED_SHA256
        and screen.get("status") == "ok"
        and screen.get("scope_supported") is True
        and screen.get("checked_windows")
        == [{"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}]
        and bound
        and isinstance(p, (int, float))
        and not isinstance(p, bool)
        and math.isfinite(p)
        and 0 <= p <= 1
    )
    if not valid:
        return None, "inconclusive"
    direction = (
        "ai"
        if screen.get("decision") == "ai" and p > 0.5
        else "original"
        if screen.get("decision") == "original" and p < 0.5
        else "inconclusive"
    )
    return p, direction


def note(pin):
    return (
        combined.NOTE if pin == COMBINED_SHA256 else calibration.NOTE if pin == calibration.SHA256 else None
    )
