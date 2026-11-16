# SPDX-License-Identifier: MIT
"""Compare several visible lip shapes to short source-clock speech events."""

from __future__ import annotations

from bisect import bisect_left
from math import exp, log

from app.evidence import video_viseme_timing as timing

PHONES = timing.PHONES | {"lower_lip": ("f", "fʲ", "v", "vʲ")}
LABELS = tuple(PHONES)
FIT_LAG = 320  # Visible articulation often starts slightly before the sound.
MAX_LOG_SUPPORT = 2.
LAGS = timing.LAGS


def _deciles(values):
    ordered = sorted(values)
    bounds = []
    for percentile in (.1, .9):
        position = percentile * (len(ordered) - 1)
        index = int(position)
        fraction = position - index
        low, high = ordered[index:index + 2]
        difference = high - low
        bounds.append(low + difference * fraction if fraction < .5 else high - difference * (1 - fraction))
    return bounds


def _clip(value, low, high):
    return min(high, max(low, value))


def _speed(values, indices):
    return [0., *(current - previous if i - j == 1 else 0.
                  for current, previous, i, j in zip(values[1:], values[:-1], indices[1:], indices[:-1], strict=True))]


def visual_features(frames, *, shape):
    rows = [(frame["index"], frame["sample"], frame["shape"]) for frame in frames if "shape" in frame]
    if len(rows) < 20:
        return None
    index, times, measurements = zip(*rows, strict=True)
    scaled = [tuple(value / timing.SCORE_UNITS for value in row) for row in measurements]
    gap = [sorted(row[:3])[1] for row in scaled]
    base, top = _deciles(gap)
    if top - base < .025:
        return None
    opening = [_clip((value - base) / (top - base), -.5, 1.5) for value in gap]
    # Derivatives are admitted only on contiguous picture samples.
    features = [gap, opening, _speed(opening, index)]
    if shape:
        width = [row[3] for row in scaled]
        lo, hi = _deciles(width)
        relative_width = [_clip((value - lo) / max(.02, hi - lo), -.5, 1.5) for value in width]
        center_to_width = [center / w for _, center, _, w in scaled]
        lip_curve = [(center - (left + right) * .5) / (g + .03)
                     for (left, center, right, _), g in zip(scaled, gap, strict=True)]
        asymmetry = [abs(left - right) for left, _, right, _ in scaled]
        features.extend((width, relative_width, _speed(relative_width, index), center_to_width, lip_curve, asymmetry))
    return times, index, list(zip(*features, strict=True))


def _index(times, sample):
    i = bisect_left(times, sample)
    best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: (abs(times[j] - sample), j))
    return best if abs(times[best] - sample) <= timing.MAX_NEAREST else None


def _log_support(features, classifier):
    columns = list(zip(*classifier["coefficients"], strict=True))
    result = []
    for row in features:
        normalized = [(value - midpoint) / scale
                      for value, midpoint, scale in zip(row, classifier["midpoint"], classifier["scale"], strict=True)]
        logits = [sum(a * b for a, b in zip(normalized, column, strict=True)) + bias
                  for column, bias in zip(columns, classifier["bias"], strict=True)]
        peak = max(logits)
        shifted = [logit - peak for logit in logits]
        chance = log(sum(exp(logit) for logit in shifted) / len(shifted))
        result.append([logit - chance for logit in shifted])
    return result


def compatible(frames, sounds, classifier):
    visual = visual_features(frames, shape=classifier["shape"])
    if visual is None:
        return []
    times, indices, features = visual
    labels = classifier["labels"]
    support = _log_support(features, classifier)
    answers = [max(range(len(scores)), key=scores.__getitem__) for scores in support]
    label_index = {label: index for index, label in enumerate(labels)}
    events = []
    for label, sample, weight in sounds:
        if label not in label_index:
            continue
        picture = [_index(times, sample - lag) for lag in LAGS]
        if any(i is None for i in picture):
            continue
        scores = [round(_clip(support[i][label_index[label]], -MAX_LOG_SUPPORT, MAX_LOG_SUPPORT)
                        * weight * timing.SCORE_UNITS) for i in picture]
        events.append((sample, label, scores, [indices[i] for i in picture], [labels[answers[i]] for i in picture]))
    return events


def _candidates(events, samples):
    for width in timing.WIDTHS:
        for start in range(timing.STEP, samples - width - timing.STEP + 1, timing.STEP):
            stop = start + width
            left = [event for event in events if event[0] < start]
            middle = [event for event in events if start <= event[0] < stop]
            right = [event for event in events if event[0] >= stop]
            if min(len(left), len(middle), len(right)) < 2:
                continue
            a, b, c = ([v / (len(part) * timing.SCORE_UNITS) for v in timing.totals(part)] for part in (left, middle, right))
            lane = max(range(len(LAGS)), key=lambda i: (min(a[i], c[i]), a[i] + c[i], -abs(LAGS[i]), -LAGS[i]))
            flank = min(a[lane], c[lane])
            local = max(b[max(0, lane - 1):lane + 2])
            if flank <= 0 or local >= 0:
                continue
            observations = [
                {"sample": at, "heard": heard, "best_visual_class": seen[lane], "frame": frames[lane], "score": values[lane]}
                for at, heard, values, frames, seen in middle if values[lane] < 0
            ]
            yield {"score": round(flank - local, 7), "span": [start, stop], "baseline_lag": LAGS[lane], "change": 0,
                   "before_frame": left[-1][3][lane], "after_frame": right[0][3][lane],
                   "frames": sorted({event[3][lane] for event in middle}), "disagreements": observations}


def locate(frames, sounds, samples, classifier):
    events = compatible(frames, sounds, classifier)
    if len(events) < 6:
        return {"status": "insufficient_events", "eligible_events": len(events)}
    proposals = list(_candidates(events, samples))
    if not proposals:
        return {"status": "no_supported_change", "eligible_events": len(events)}
    best = min(proposals, key=lambda row: (-row["score"], row["span"]))
    a, b = best["span"]
    for item in proposals:
        c, d = item["span"]
        if item["score"] >= best["score"] - .015 and 5 * (min(b, d) - max(a, c)) < 2 * (max(b, d) - min(a, c)):
            return {"status": "ambiguous_change", "eligible_events": len(events)}
    return {"status": "measured", "eligible_events": len(events), **best}
