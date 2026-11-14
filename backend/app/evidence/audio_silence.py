"""Locate fully muted intervals in decoded PCM with a verified source clock."""
from __future__ import annotations

import numpy as np

from app.evidence.audio_silence_binding import (
    FLANK,
    MAXIMUM_EVENTS,
    MIN_FLANK_DBFS,
    MINIMUM,
    SCHEMA,
    continuous_spans,
)


def measure(pcm, timeline):
    """Search each continuous source span. Keep the longest qualifying runs."""
    pcm = np.asarray(pcm)
    checked = 0
    candidates = []
    for a, b in continuous_spans(timeline):
        checked += b - a
        zero = pcm[a:b] == 0
        edges = np.flatnonzero(np.diff(np.r_[False, zero, False])) + a
        starts, stops = edges[::2], edges[1::2]
        eligible = (stops - starts >= MINIMUM) & (starts >= a + FLANK) & (stops <= b - FLANK)
        for start, stop in zip(starts[eligible], stops[eligible], strict=True):
            flanks = (pcm[start - FLANK:start], pcm[stop:stop + FLANK])
            levels = [round(20 * np.log10(max(1e-6, np.sqrt(np.mean(x**2, dtype=np.float64)))), 6) for x in flanks]
            if min(levels) >= MIN_FLANK_DBFS:
                candidates.append({"start_sample": int(start), "end_sample": int(stop),
                                   "flank_rms_dbfs": [float(value) for value in levels]})
    candidates.sort(key=lambda row: (row["start_sample"] - row["end_sample"], row["start_sample"]))
    return {"schema": SCHEMA, "checked_samples": checked,
            "qualifying_intervals": len(candidates), "selected": candidates[:MAXIMUM_EVENTS],
            "affects_overall_confidence": False}
