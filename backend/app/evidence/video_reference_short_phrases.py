"""Compare short sound regions between two copies on a common sample clock."""

from __future__ import annotations

import numpy as np

from app.evidence import video_reference_sound_difference as original

BASIS = "short_phrase_correlation/1"
WINDOW, HOP = 960, 320
THREE_BANDS = original.speech.BANDS
WEIGHT = np.hanning(WINDOW)
WIDTHS = (8, 10, 12, 14, 16, 18, 20, 22)
FLANK, GUARD, SUPPORT = 28, 2, 3
CORRELATION_FLOOR = .90


def locate(earlier, submitted):
    """Require clear synchronized audio before and after a compact difference."""
    size = len(earlier)
    if size != len(submitted) or not original.SHORT_MIN_SAMPLES <= size <= original.SAMPLES:
        return None
    bands = [original._band_statistics(earlier, submitted, band, weight=WEIGHT, hop=HOP, floor=.09) for band in THREE_BANDS]
    cells = 1 + (size - WINDOW) // HOP
    strongest = None
    for width in WIDTHS:
        for start in range(SUPPORT + GUARD, cells - width - SUPPORT - GUARD + 1):
            stop = start + width
            limits = np.array(((max(0, start - GUARD - FLANK), start - GUARD), (start, stop),
                               (stop + GUARD, min(cells, stop + GUARD + FLANK))))
            readings = []
            for frequency, stats in zip(THREE_BANDS, bands, strict=True):
                if not (row := original._band_reading(stats, limits, min_support=SUPPORT)):
                    continue
                a, b, c = row["correlation"]
                if min(a, c) >= CORRELATION_FLOOR and b <= .30 and row["local_relative_gain"] <= .44:
                    readings.append({"band_hz": list(frequency), **row})
            if len(readings) < 2:
                continue
            strength = float(np.median([1. - row["local_relative_gain"] for row in readings]))
            # Prefer a supported extent over a small peak in one part of the same word.
            rank = (round(strength * np.sqrt(width), 8), -start, width)
            if strongest is None or rank > strongest[0]:
                offset = (WINDOW - HOP) // 2
                strongest = (rank, {"samples": [start * HOP + offset, stop * HOP + offset],
                                    "strength": round(strength, 6), "bands": readings, "measurement_basis": BASIS})
    return strongest[1] if strongest else None
