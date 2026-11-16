"""Portable arithmetic for proposing disjoint writing passages."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median

LONGEST = 40
CAP = 6


def probability_log_odds(value):
    bounded = min(1 - 1e-6, max(1e-6, value))
    return math.log(bounded / (1 - bounded))


def length_bucket(value):
    return 0 if value < 16 else 1 if value < 33 else 2


def normalize(value, center, scale):
    return max(-4., min(4., (value - center) / scale))


def combine_responses(isolated, context):
    fitted = list(isolated)
    constraints = []
    for i, reading in enumerate(context):
        indices = range(max(0, i-1), min(len(fitted), i+2))
        if reading is not None and all(isolated[j] is not None for j in indices):
            constraints.append((reading, indices))
    for _ in range(20):
        step = [0. if anchor is None else .55 * (anchor - old) for anchor, old in zip(isolated, fitted, strict=True)]
        for reading, indices in constraints:
            adjustment = (reading - sum(fitted[j] for j in indices) / len(indices)) / len(indices)
            for j in indices:
                step[j] += adjustment
        fitted = [None if item is None else max(-4., min(4., item + .4 * delta))
                  for item, delta in zip(fitted, step, strict=True)]
    return fitted


def interval_candidates(values):
    candidates = []
    for start, opening in enumerate(values):
        if opening is None:
            continue
        nearby_left = values[start - 1] if start else None
        score_sum, positive = 0., 0
        for end in range(start + 1, min(start + LONGEST, len(values)) + 1):
            closing = values[end - 1]
            if closing is None:
                break
            score_sum += closing
            positive += closing > 0
            width = end - start
            if 2 * positive < width:
                continue
            nearby_right = values[end] if end < len(values) else None
            left = max(0., opening - nearby_left) if nearby_left is not None else 0.
            right = max(0., closing - nearby_right) if nearby_right is not None else 0.
            score = score_sum / math.sqrt(width) + .15 * (min(left, 3.) + min(right, 3.))
            candidates.append((start, end, score))
    return candidates


def select_intervals(proposals, point, count):
    endings = defaultdict(list)
    for start, end, score in proposals:
        if score > point:
            endings[end].append((start, score))
    # Each cell stores the best cumulative excess over the point for the
    # prefix and available range count, with one untreated sentence between ranges.
    table = [[(0., ()) for _ in range(CAP + 1)] for _ in range(count + 1)]
    for end in range(1, count + 1):
        for length in range(1, CAP + 1):
            best = table[end-1][length]
            for start, score in endings[end]:
                previous, spans = table[max(0, start-1)][length-1]
                added = previous + score - point
                if added > best[0]:
                    best = added, (*spans, (start, end, score))
            table[end][length] = best
    return table[count][CAP][1]


def lower_quarter(values):
    eligible = sorted(value for value in values if value is not None)
    return eligible[math.ceil(len(eligible) / 4) - 1] if eligible else None


def center_on_document(values, reference):
    baseline = lower_quarter(values)
    if baseline is None:
        return values
    shift = max(-1., min(1., baseline - reference))
    return [None if value is None else max(-4., min(4., value - shift)) for value in values]


def flank_adjusted_candidates(values):
    candidates = []
    for start, end, score in interval_candidates(values):
        before = values[max(0, start - 5):start]
        after = values[end:end + 5]
        if min(len(before), len(after)) >= 2 and None not in (*before, *after):
            contrast = sum(values[start:end]) / (end - start) - max(median(before), median(after))
            score += math.sqrt(end - start) * min(2., max(0., contrast)) / 3
        candidates.append((start, end, score))
    return candidates


def add_disjoint_ranges(primary, secondary):
    """Preserve the existing selection and add separate ranges up to its display limit."""
    chosen = [(first, stop, response, "pooled") for first, stop, response in primary]
    for first, stop, response in sorted(secondary, key=lambda row: (-row[2], row[0], row[1])):
        if len(chosen) >= CAP:
            break
        if all(stop < a or b < first for a, b, _, _ in chosen):
            chosen.append((first, stop, response, "document_baseline"))
    return sorted(chosen)
