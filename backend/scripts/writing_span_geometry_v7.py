"""Source-character geometry for the controlled span-learning comparison."""

from __future__ import annotations

import math

import numpy as np


def require(value, reason):
    if not value:
        raise ValueError(reason)


def truth_spans(text, labels):
    require(len(text) == len(labels), "Every extracted character needs a provenance slot")
    spans, start, last = [], None, None
    for i, char in enumerate(text):
        if char.isspace():
            continue
        require(type(labels[i]) is int and labels[i] in (0, 1), "Non-whitespace provenance is missing")
        if labels[i] == 1:
            if start is None:
                start = i
            last = i + 1
        elif start is not None:
            spans.append([start, last])
            start = last = None
    if start is not None:
        spans.append([start, last])
    return spans


def native_coverage(text, offsets):
    covered = np.zeros(len(text), bool)
    for a, b in offsets:
        require(type(a) is type(b) is int and 0 <= a <= b <= len(text), "Native offset exceeds source")
        covered[a:b] = True
    required = np.asarray([not c.isspace() for c in text])
    require(required.any() and np.all(covered[required]), "Native token coverage omits source characters")
    return covered


def token_targets(text, labels, offsets):
    native_coverage(text, offsets)
    target, weight = [], []
    for a, b in offsets:
        values = [labels[i] for i in range(a, b) if not text[i].isspace()]
        require(all(type(v) is int and v in (0, 1) for v in values), "Unknown token provenance")
        target.append(sum(values) / len(values) if values else 0.0)
        weight.append(len(values))
    require(sum(weight) > 0, "Whitespace is not evidence")
    return np.asarray(target, np.float32), np.asarray(weight, np.float32)


def absolute_interval(span, length):
    a, b = map(float, span)
    require(
        length > 0 and math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= 1,
        "Finite positive normalized interval required",
    )
    # This is the defined readout of stored float32 endpoints, not a display
    # rounding convention or an inferred exact boundary of intellectual authorship.
    return [max(0, math.floor(a * length)), min(length, math.ceil(b * length))]


def proposals_mask(text, proposals, cut):
    require(type(cut) in (int, float) and math.isfinite(cut), "Finite fixed point required")
    mask = np.zeros(len(text), bool)
    for p in proposals:
        a, b, score = p["start"], p["end"], p["score"]
        require(type(a) is type(b) is int and 0 <= a <= b <= len(text), "Proposal exceeds extracted source")
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "Finite proposal response required",
        )
        if score >= cut:
            mask[a:b] = True
    return mask & np.asarray([not c.isspace() for c in text])


def metrics(text, labels, mask):
    require(mask.dtype == np.bool_ and mask.shape == (len(text),), "Exact source-character mask required")
    truth_spans(text, labels)
    ai = np.asarray([not c.isspace() and y == 1 for c, y in zip(text, labels, strict=True)])
    human = np.asarray([not c.isspace() and y == 0 for c, y in zip(text, labels, strict=True)])
    tp, fp, fn = int((mask & ai).sum()), int((mask & human).sum()), int((~mask & ai).sum())
    return {
        "ai_characters": tp + fn,
        "human_characters": int(human.sum()),
        "located_ai_characters": tp,
        "highlighted_human_characters": fp,
        "missed_ai_characters": fn,
        "highlighted_characters": tp + fp,
        "character_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "character_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }


def genuine_point(rows):
    require(
        rows and all(r["role"] == "point" and r["label"] == 0 and r["status"] == "ok" for r in rows),
        "Only complete genuine point controls may choose a point",
    )
    groups = {}
    for row in rows:
        values = [p["score"] for p in row["proposals"] if p["end"] > p["start"]]
        require(values and all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Missing genuine proposals")
        groups.setdefault(row["group"], []).append(max(values))
    require(
        len(groups) == 32 and all(len(v) == 4 for v in groups.values()),
        "All32 genuine groups and four ordinary compositions required",
    )
    values = sorted((max(v) for v in groups.values()), reverse=True)
    cut = max(0.001, math.nextafter(values[1], math.inf))
    return {
        "cut": cut,
        "source_groups": 32,
        "group_budget": 1,
        "groups_highlighted": sum(v >= cut for v in values),
        "uses_ai_or_diagnostic": False,
    }
