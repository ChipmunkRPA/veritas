"""Rank short passages using the responses from overlapping native windows."""

from __future__ import annotations

import math
from statistics import mean, median

from app.evidence.passage_interval_selection import probability_log_odds

MAX_WIDTH = 3
MAX_FINDINGS = 2
SCOPE = 4


def _window_edges(count):
    return [(max(0, index - 1), min(count, index + 2)) for index in range(count)]


def _sum(prefix, first, stop):
    return prefix[stop] - prefix[first]


def _slope(occupancy, responses):
    center = mean(occupancy)
    den = math.fsum((item - center)**2 for item in occupancy)
    return math.fsum((item - center) * value for item, value in zip(occupancy, responses, strict=True)) / (den + .15)


def features(sentences, readings):
    """Return a row for every one-to-three sentence interval."""
    count = len(sentences)
    edges = _window_edges(count)
    isolated, context = ([probability_log_odds(value) for value in readings[key]] for key in ("isolated", "context"))
    backgrounds = median(isolated), median(context)
    word_sums = [0]
    for _, _, words in sentences:
        word_sums.append(word_sums[-1] + max(words, 1))
    for first in range(count):
        for stop in range(first + 1, min(first + MAX_WIDTH, count) + 1):
            input_words = _sum(word_sums, first, stop)
            isolated_mean = mean(isolated[first:stop])
            inside = [context[k] for k in range(max(0, first - 1), min(count, stop + 1))
                      if edges[k][0] <= first and edges[k][1] >= stop]
            # For three sentences the middle window is the only complete view.
            context_mean = mean(inside)
            typical = (
                isolated_mean,
                min(isolated[first:stop]),
                isolated_mean - backgrounds[0],
                mean(context[first:stop]),
                min(inside),
                context_mean - backgrounds[1],
                math.log1p(input_words),
                float(stop - first),
            )

            left_edge, right_edge = max(0, first - SCOPE), min(count, stop + SCOPE)
            neighborhood = context[left_edge:right_edge]
            exposure = []
            for a, b in edges[left_edge:right_edge]:
                left, right = max(a, first), min(b, stop)
                shared = _sum(word_sums, left, right) if left < right else 0
                exposure.append(shared / _sum(word_sums, a, b))
            before = [value for value, (_, b) in zip(neighborhood, edges[left_edge:right_edge], strict=True) if b <= first]
            after = [value for value, (a, _) in zip(neighborhood, edges[left_edge:right_edge], strict=True) if a >= stop]
            contrast_left = context_mean - (median(before) if before else backgrounds[1])
            contrast_right = context_mean - (median(after) if after else backgrounds[1])
            shape = (_slope(exposure, neighborhood), min(contrast_left, contrast_right),
                     max(contrast_left, contrast_right), contrast_left * contrast_right)
            yield first, stop, typical, shape


def candidates(sentences, readings, head, *, use_overlap):
    if readings is None:
        return []
    result = []
    for first, stop, ordinary, overlap in features(sentences, readings):
        measurements = ordinary + overlap if use_overlap else ordinary
        score = math.fsum((value - center) * coefficient / scale
                          for value, center, scale, coefficient in zip(
                              measurements, head["center"], head["scale"], head["coefficients"], strict=True,
                          ))
        result.append((first, stop, score))
    return result


def select(proposals, cutoff):
    selected = []
    for first, stop, score in sorted(proposals, key=lambda row: (-round(row[2], 10), row[1] - row[0], row[0])):
        if score <= cutoff:
            continue
        if all(stop <= other_first or first >= other_stop for other_first, other_stop, _ in selected):
            selected.append((first, stop, score))
            if len(selected) == MAX_FINDINGS:
                break
    return tuple(sorted(selected))
