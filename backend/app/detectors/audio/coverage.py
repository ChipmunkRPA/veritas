"""Coverage of successfully scored voice windows, distinct from decoded audio duration."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def interval_union(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching non-negative finite intervals without counting overlap twice."""
    valid = sorted((a, b) for a, b in spans if math.isfinite(a) and math.isfinite(b) and 0 <= a < b)
    merged: list[tuple[float, float]] = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def scored_voice_spans(entries: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Union of timestamped probability windows from completed voice detectors only."""
    spans: list[tuple[float, float]] = []
    for entry in entries:
        if entry.get("status") != "ok" or entry.get("family") != "voice":
            continue
        for unit in entry.get("unit_scores") or ():
            if not str(unit.get("unit_ref", "")).startswith("window:") or unit.get("raw_kind") != "probability":
                continue
            quality = unit.get("quality") or {}
            try:
                start, end = float(quality["start_s"]), float(quality["end_s"])
                raw = float(unit["raw"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(raw):
                spans.append((start, end))
    return interval_union(spans)


def coverage_seconds(spans: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in interval_union(spans))


def content_rows(runs: Iterable[Sequence[Any]], scored: Iterable[tuple[float, float]]) -> list[list[Any]]:
    """Split content runs at scored-window boundaries; content eligibility alone is not coverage."""
    spans = interval_union(scored)
    rows: list[list[Any]] = []
    for a, b, label in runs:
        start, end = float(a), float(b)
        boundaries = sorted({start, end, *(t for span in spans for t in span if start < t < end)})
        for left, right in zip(boundaries, boundaries[1:], strict=False):
            covered = any(a <= left and right <= b for a, b in spans)
            rows.append([round(left, 3), round(right, 3), label, "scored" if covered else "not scored"])
    return rows
