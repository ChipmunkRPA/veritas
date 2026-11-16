# SPDX-License-Identifier: MIT
"""Compare several visible lip shapes to short source-clock speech events."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np

from app.evidence import video_viseme_timing as timing

PHONES = timing.PHONES | {"lower_lip": ("f", "fʲ", "v", "vʲ")}
LABELS = tuple(PHONES)
FIT_LAG = 320  # Visible articulation often starts slightly before the sound.
MAX_LOG_SUPPORT = 2.
LAGS = timing.LAGS


def visual_features(frames, *, shape):
    rows = [(frame["index"], frame["sample"], frame["shape"]) for frame in frames if "shape" in frame]
    if len(rows) < 20:
        return None
    index, times, measurements = zip(*rows, strict=True)
    left, center, right, width = np.asarray(measurements, dtype=np.float64).T / timing.SCORE_UNITS
    gap = np.median([left, center, right], axis=0)
    base, top = np.percentile(gap, [10, 90])
    if top - base < .025:
        return None
    opening = np.clip((gap - base) / (top - base), -.5, 1.5)
    # Derivatives are admitted only on contiguous picture samples.
    adjacent = np.array([0, *np.diff(index)]) == 1
    speed = np.where(adjacent, np.r_[0, np.diff(opening)], 0)
    features = [gap, opening, speed]
    if shape:
        lo, hi = np.percentile(width, [10, 90])
        relative_width = np.clip((width - lo) / max(.02, hi - lo), -.5, 1.5)
        spread_speed = np.where(adjacent, np.r_[0, np.diff(relative_width)], 0)
        features.extend((width, relative_width, spread_speed, center / width,
                         (center - (left + right) * .5) / (gap + .03), abs(left - right)))
    return times, index, np.stack(features, axis=1)


def _index(times, sample):
    i = bisect_left(times, sample)
    best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: (abs(times[j] - sample), j))
    return best if abs(times[best] - sample) <= timing.MAX_NEAREST else None


def fit(train, *, shape):
    """Fit balanced visual classes on observed events from unchanged recordings."""
    profiles = {label: [] for label in LABELS}
    speakers = {label: set() for label in LABELS}
    for row in train:
        visual = visual_features(row["frames"], shape=shape)
        if visual is None:
            continue
        times, _, features = visual
        for label, sample, weight in row["deliveries"]["original"]["articulation"]:
            i = _index(times, sample - FIT_LAG)
            if weight >= .50 and i is not None:
                profiles[label].append(features[i])
                speakers[label].add(row["speaker"])
    counts = {label: {"events": len(values), "speakers": len(speakers[label])} for label, values in profiles.items()}
    labels = tuple(label for label in LABELS if counts[label]["events"] >= 12 and counts[label]["speakers"] >= 4)
    if len(labels) < 3:
        raise ValueError("The visual fit lacks enough observed shape classes")
    data = np.vstack([profiles[label] for label in labels])
    midpoint, scale = data.mean(axis=0), np.maximum(1e-6, data.std(axis=0))
    groups = [(np.vstack(profiles[label]) - midpoint) / scale for label in labels]
    centroids = np.stack([group.mean(axis=0) for group in groups])
    # Equal weight per class and fixed shrinkage limit speaker- and
    # recording-specific directions in the shared covariance.
    residuals = [group - center for group, center in zip(groups, centroids, strict=True)]
    covariance = np.mean([residual.T @ residual / len(residual) for residual in residuals], axis=0)
    inverse = np.linalg.solve(.8 * covariance + .2 * np.eye(data.shape[1]), centroids.T)
    bias = -.5 * np.sum(centroids * inverse.T, axis=1)
    return {"shape": shape, "counts": counts, "labels": list(labels), "midpoint": midpoint.round(10).tolist(),
            "scale": scale.round(10).tolist(), "coefficients": inverse.round(10).tolist(), "bias": bias.round(10).tolist()}


def _log_support(features, classifier):
    normalized = (features - classifier["midpoint"]) / classifier["scale"]
    logits = normalized @ classifier["coefficients"] + classifier["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    chance = np.mean(np.exp(logits), axis=1, keepdims=True)
    return logits - np.log(chance)


def compatible(frames, sounds, classifier):
    visual = visual_features(frames, shape=classifier["shape"])
    if visual is None:
        return []
    times, indices, features = visual
    labels = classifier["labels"]
    support = _log_support(features, classifier)
    answer = support.argmax(axis=1)
    events = []
    for label, sample, weight in sounds:
        if label not in labels:
            continue
        picture = [_index(times, sample - lag) for lag in LAGS]
        if any(i is None for i in picture):
            continue
        scores = [round(float(np.clip(support[i, labels.index(label)], -MAX_LOG_SUPPORT, MAX_LOG_SUPPORT)
                              * weight) * timing.SCORE_UNITS) for i in picture]
        events.append((sample, label, scores, [indices[i] for i in picture], [labels[answer[i]] for i in picture]))
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
