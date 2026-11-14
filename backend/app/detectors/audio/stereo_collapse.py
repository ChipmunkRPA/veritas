"""Find audible, sample-identical audio between two distinct stereo flanks."""

from __future__ import annotations

import math

import numpy as np

from app.evidence import stereo_collapse_binding as binding


def _variation_dbfs(samples, *, divisor):
    # Centering excludes a constant DC level from audible signal support.
    power = float(np.var(samples, dtype=np.float64)) / (divisor * divisor)
    return round(10 * math.log10(power), 3) if power else -140.0


def measure(stereo, timeline, ranges, *, minimum_difference=binding.MIN_DIFFERENCE, minimum_signal=binding.MIN_SIGNAL):
    spans = binding.spans(timeline, len(stereo))
    diff = stereo[:, 0].astype(np.int32) - stereo[:, 1]
    flank = binding.FLANK
    candidates = []
    for a, b in spans:
        edges = np.flatnonzero(np.diff(np.r_[False, diff[a:b] == 0, False])) + a
        starts, ends = edges[::2], edges[1::2]
        lengths = ends - starts
        eligible = ((lengths >= binding.MINIMUM) & (lengths <= binding.MAXIMUM)
                    & (starts >= a + flank) & (ends <= b - flank))
        for start, end in zip(starts[eligible].tolist(), ends[eligible].tolist(), strict=True):
            levels = [_variation_dbfs(values, divisor=65536)
                      for values in (diff[start-flank:start], diff[end:end+flank])]
            if min(levels) < minimum_difference:
                continue
            blocks = range(start, end - flank, flank)
            signal = min(_variation_dbfs(stereo[p:p+flank, 0], divisor=32768)
                         for p in (*blocks, end - flank))
            if signal >= minimum_signal:
                candidates.append({"start_sample": start, "end_sample": end, "flank_difference_dbfs": levels,
                                   "minimum_signal_dbfs": signal})
    candidates.sort(key=lambda row: (row["start_sample"] - row["end_sample"], row["start_sample"]))
    selected = sorted(candidates[:binding.MAX_EVENTS], key=lambda row: row["start_sample"])
    for row in selected:
        row["source_timestamp_ranges"] = ranges(row["start_sample"], row["end_sample"])
    method = {**binding.METHOD, "minimum_flank_difference_dbfs": minimum_difference,
              "minimum_signal_dbfs": minimum_signal}
    return {"schema": binding.SCHEMA, "method": method, "affects_overall_confidence": False,
            "checked_samples": sum(b - a for a, b in spans), "qualifying_intervals": len(candidates),
            "selected": selected}
