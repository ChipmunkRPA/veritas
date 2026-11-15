# SPDX-License-Identifier: MIT
"""Reproducible speech and picture timing on a verified audio sample clock."""

from __future__ import annotations

import bisect

RATE = 16000
MINIMUM_CHANGE, MAXIMUM_CHANGE = 1760, 5440

PHONES = {
    "closure": ("b", "bʲ", "bʰ", "bː", "p", "pʰ", "pʲ", "pː", "m", "mʲ"),
    "open": ("a", "aː", "æ", "æː", "ɑ", "ɑː", "ʌ", "ɐ"),
    "rounded": ("o", "oː", "ɔ", "ɔː", "ɒ", "u", "uː", "ʊ", "w"),
    "spread": ("i", "iː", "ɪ", "e", "eː", "ɛ", "j"),
}
LAGS = tuple(range(-3840, 3841, 640))
WIDTHS = (6400, 8960, 11520)
STEP = 640
EDGE = 640
MAX_NEAREST = 400
SCALE = .30
SCORE_UNITS = 10_000_000


def observations(row):
    points = [(frame["sample"], frame["gap"], frame["index"]) for frame in row["frames"] if frame["gap"] is not None]
    if len(points) < 20:
        return None
    levels = sorted(level for _, level, _ in points)
    a, b = levels[len(levels)//10], levels[-1-len(levels)//10]
    if b-a < .025:
        return None
    return [sample for sample, _, _ in points], [min(1.5, max(-.5, (level-a)/(b-a))) for _, level, _ in points], [
        index for _, _, index in points]


def nearest(points, sample):
    times, gaps, indices = points
    i = bisect.bisect_left(times, sample)
    candidates = [j for j in (i-1, i) if 0 <= j < len(times)]
    winner = min(candidates, key=lambda j: (abs(times[j]-sample), j))
    if abs(times[winner]-sample) > MAX_NEAREST:
        return None
    return gaps[winner], indices[winner]


def compatible(row, sounds, references, *, sample_at=nearest):
    points = observations(row)
    if points is None:
        return []
    events = []
    for label, sample, probability in sounds:
        if label not in references:
            continue
        values = [sample_at(points, sample-lag) for lag in LAGS]
        if not all(value is not None for value in values):
            continue
        center = references[label]["median"]
        # Quantize before summing so the replay stays identical on Python
        # versions that use different floating-point summation algorithms.
        strength = [round(-SCORE_UNITS * probability * (min(1.5, abs(level-center)) / SCALE)**2)
                    for level, _ in values]
        events.append((sample, label, strength, [index for _, index in values]))
    return events


def totals(events, *, lags=LAGS):
    return [sum(event[2][i] for event in events) for i in range(len(lags))]


def hypotheses(events, samples, *, lags=LAGS):
    if len(events) < 4:
        return {"status": "insufficient_events", "eligible_events": len(events)}
    constant = max(totals(events, lags=lags))
    alternatives = []
    for width in WIDTHS:
        for start in range(EDGE, samples-width-EDGE+1, STEP):
            stop = start + width
            # Use every event in exactly one part of both competing hypotheses.
            before = [event for event in events if event[0] < start]
            local = [event for event in events if start <= event[0] < stop]
            after = [event for event in events if event[0] >= stop]
            if not before or len(local) < 2 or not after:
                continue
            left, middle, right = (totals(group, lags=lags) for group in (before, local, after))
            flank_order = sorted(range(len(lags)), key=lambda i: (-(left[i] + right[i]), abs(lags[i]), lags[i]))
            outer = flank_order[0]
            # The contrast should be against the strongest common flank lag.
            for inner, lag in enumerate(lags):
                shift = lag - lags[outer]
                if not MINIMUM_CHANGE <= abs(shift) <= MAXIMUM_CHANGE:
                    continue
                if min(left[outer] - left[inner], right[outer] - right[inner], middle[inner] - middle[outer]) <= 0:
                    continue
                gain = round((left[outer]+middle[inner]+right[outer]-constant) / (SCORE_UNITS * len(events)), 7)
                if gain <= 0:
                    continue
                alternatives.append({"score": gain, "span": [start, stop], "baseline_lag": lags[outer],
                                     "change": shift, "frames": sorted({event[3][inner] for event in local}),
                                     "before_frame": before[-1][3][outer], "after_frame": after[0][3][outer],
                                     "local_events": [event[0] for event in local]})
    if not alternatives:
        return {"status": "no_supported_change", "eligible_events": len(events)}
    alternatives.sort(key=lambda item: (-item["score"], item["span"], abs(item["change"])))
    winner = alternatives[0]
    for item in alternatives[1:]:
        if item["score"] < winner["score"] - .015:
            break
        a, b = winner["span"]
        c, d = item["span"]
        if winner["change"] * item["change"] < 0 or 2 * (min(b, d) - max(a, c)) < max(b, d) - min(a, c):
            return {"status": "ambiguous_change", "eligible_events": len(events)}
    return {"status": "measured", "eligible_events": len(events), **winner}


def picture_evidence(frames, result, samples):
    lag = result["baseline_lag"] + result["change"]
    start, stop = (min(samples, max(0, sample-lag)) for sample in result["span"])
    available = [frame for frame in frames if frame.get("mouth_box")]
    earlier = [frame for frame in available if start - 5120 <= frame["sample"] < start]
    within = [frame for frame in available if start <= frame["sample"] < stop]
    later = [frame for frame in available if stop <= frame["sample"] <= stop + 5120]
    center = (start + stop) // 2
    bracket = {"before": earlier[-1] if earlier else None,
               "during": min(within, key=lambda frame: (abs(frame["sample"]-center), frame["index"])) if within else None,
               "after": later[0] if later else None}
    return {"span": result["span"], "picture_span": [start, stop], "change": result["change"],
            "baseline_lag": result["baseline_lag"], "picture_brackets": bracket,
            "timing_anchor_frames": {"left_audio_context": frames[result["before_frame"]],
                                     "local_audio_events": [frames[i] for i in result["frames"]],
                                     "right_audio_context": frames[result["after_frame"]]}}
