"""Confirm writing ranges against an economic-report reference when it fits."""

from __future__ import annotations

import math
from statistics import median

from app.evidence.passage_interval_selection import (
    combine_responses,
    interval_candidates,
    length_bucket,
    normalize,
    probability_log_odds,
    select_intervals,
)

REPORT = "economic_report"
VIEWS = ("isolated", "context")


def _quantile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def _buckets(sentences):
    counts = [row[2] for row in sentences]
    return {
        "isolated": [length_bucket(count) for count in counts],
        "context": [length_bucket(sum(counts[max(0, index - 1):index + 2])) for index in range(len(counts))],
    }


def _responses(readings, buckets, reference):
    return {
        view: [normalize(probability_log_odds(value), *reference[view][bucket])
               for value, bucket in zip(readings[view], buckets[view], strict=True)]
        for view in VIEWS
    }


def _profile(sentences, standardized):
    counts = [row[2] for row in sentences]
    middle = max(1, median(counts))
    values = [math.log(len(counts)), math.log(middle), _quantile(counts, .10) / middle,
              _quantile(counts, .90) / middle, sum(end - start for start, end, _ in sentences) / max(1, sum(counts))]
    for view in VIEWS:
        lower = _quantile(standardized[view], .25)
        values.extend((lower, median(standardized[view]) - lower))
    return values


def _nearest(vector, examples, scale):
    distances = sorted(math.sqrt(math.fsum(((value - reference) / step)**2
                                           for value, reference, step in zip(vector, example, scale, strict=True))
                                 / len(scale)) for example in examples)[:3]
    return math.fsum(distances) / len(distances)


def review(sentences, readings, pooled, calibration):
    """Return report ranges, or None when the document is outside that reference."""
    if readings is None:
        return None
    buckets = _buckets(sentences)
    vector = _profile(sentences, _responses(readings, buckets, pooled))
    profile = calibration["profile"]
    report = _nearest(vector, profile["examples"][REPORT], profile["scale"])
    prose = _nearest(vector, profile["examples"]["historical_prose"], profile["scale"])
    if report >= prose or report > profile["radius"][REPORT] or prose < 1.5 * report:
        return None
    scores = _responses(readings, buckets, calibration["scales"])
    combined = combine_responses(scores["isolated"], scores["context"])
    return select_intervals(interval_candidates(combined), calibration["point"], len(sentences))


def confirm(current, reference):
    if reference is None:
        return current
    return tuple(
        (start, stop, value, basis)
        for start, stop, value, basis in current
        if any(2 * (min(stop, right) - max(start, left)) >= min(stop - start, right - left)
               for left, right, _ in reference)
    )
