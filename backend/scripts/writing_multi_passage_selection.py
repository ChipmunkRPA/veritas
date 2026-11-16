# SPDX-License-Identifier: MIT
"""Compare independent sentence alerts and disjoint ranges on pinned responses."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_multi_passage_panel as source

ARMS = ("context_centers", "isolated_centers", "isolated_intervals", "combined_intervals")
POINTS = {"tighter": .90, "wider": .75}
LONGEST = 40
CAP = 6
DESIGN = {
    "score_views": "Current three-sentence context and isolated sentence responses. Standardize each view by genre and "
                   "input length against ordinary validation sentences only.",
    "reconstruction": "Solve the three-sentence averaging approximation by 20 bounded gradient steps with a separate "
                      "isolated-score penalty. Compare interval selection with and without this reconstruction.",
    "intervals": "Combine sustained standardized responses and smaller bonuses for any upward transition at either outer edge. "
                 "Require at least half the sentences to support the interval. Select disjoint ranges that maximize the sum "
                 "of their scores above the point, with at least one unselected sentence between ranges. Use up to six ranges.",
    "points": "Calibrate each method independently on ordinary work maxima over both layout variants. "
              "Use fixed 90th and 75th percentile points, stratified by genre.",
    "truth": "One-to-one matching with at least half of the labeled non-whitespace characters and 0.30 intersection over union.",
}


def _quantile(rows, fraction):
    return sorted(rows)[max(0, math.ceil(len(rows) * fraction) - 1)]


def _logit(value):
    bounded = min(1 - 1e-6, max(1e-6, value))
    return math.log(bounded / (1 - bounded))


def _width(value):
    return 0 if value < 16 else 1 if value < 33 else 2


def _inputs(variant, view):
    sentences = variant["sentences"]
    for i, value in enumerate(variant[view]):
        if value is None:
            yield None
            continue
        span = [sentences[i]] if view == "isolated" else sentences[max(0, i-1):i+2]
        yield _width(sum(item[-1] for item in span)), _logit(value)


def _normalizers(works):
    found = defaultdict(list)
    for work in works:
        if work["split"] != "validation":
            continue
        for variant in work["variants"]:
            assert variant["version"] == "ordinary"
            for view in ("context", "isolated"):
                for item in _inputs(variant, view):
                    if item is not None:
                        bucket, score = item
                        found[work["genre"], view, bucket].append(score)
    return {key: (median(values), max(.25, _quantile(values, .90) - median(values)))
            for key, values in found.items()}


def _normalized(variant, genre, stats, view):
    adapted = []
    for item in _inputs(variant, view):
        if item is None:
            adapted.append(None)
        else:
            length, score = item
            values = stats.get((genre, view, length))
            adapted.append(None if values is None else max(-4., min(4., (score - values[0]) / values[1])))
    return adapted


def _reconstruct(isolated, context):
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


def _ranges(values, *, runs):
    if not runs:
        return [(i, i + 1, score) for i, score in enumerate(values) if score is not None]
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


def _select_centers(proposals, point):
    selected = []
    for start, end, score in proposals:
        if score <= point:
            continue
        if selected and selected[-1][1] == start:
            previous, _, reading = selected[-1]
            selected[-1] = previous, end, max(reading, score)
        else:
            selected.append((start, end, score))
    return sorted(sorted(selected, key=lambda row: (-row[2], row[0]))[:CAP])


def _select_ranges(proposals, point, count):
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


def proposals(variant, genre, stats):
    isolated = _normalized(variant, genre, stats, "isolated")
    context = _normalized(variant, genre, stats, "context")
    return {
        ARMS[0]: _ranges(context, runs=False),
        ARMS[1]: _ranges(isolated, runs=False),
        ARMS[2]: _ranges(isolated, runs=True),
        ARMS[3]: _ranges(_reconstruct(isolated, context), runs=True),
    }


def _choice(arm, variant, candidates, point):
    if arm.endswith("centers"):
        selected = _select_centers(candidates, point)
    else:
        selected = _select_ranges(candidates, point, len(variant["sentences"]))
    readings = []
    for start, end, value in selected:
        first, last = variant["sentences"][start], variant["sentences"][end - 1]
        readings.append({"sentences": [start, end], "native": [first[0], last[1]],
                         "compact": [first[2], last[3]], "response": round(value, 9)})
    return readings


def _overlap(a, b):
    inside = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inside if 2 * inside >= b[1] - b[0] and 10 * inside >= 3 * union else 0


def _match(nominations, targets):
    matches = {(i, j): value for i, selected in enumerate(nominations) for j, target in enumerate(targets)
               if (value := _overlap(selected["compact"], target))}
    results = [()]
    for i, j in matches:
        results.append(((i, j),))
        results.extend(((i, j), (other_i, other_j)) for other_i, other_j in matches if other_i != i and other_j > j)
    return max(results, key=lambda row: (len(row), sum(matches[item] for item in row)))


def _summarize(works, decisions, arm, point):
    report = {}
    for genre in sorted(set(work["genre"] for work in works)):
        specific = [work for work in works if work["genre"] == genre]
        by_layout = {}
        stable = 0
        for layout in ("published", "neutral"):
            stats = {key: 0 for key in ("files", "labeled_passages", "located", "both_passages", "ordinary_flags",
                                        "unmatched_flags", "complete_native_readings", "unconfounded")}
            ordinary_groups = set()
            for work in specific:
                number = work["document"]
                case = decisions[number][arm][point][layout]
                stats["files"] += 1
                stats["labeled_passages"] += len(case["targets"])
                stats["located"] += len(case["hits"])
                stats["both_passages"] += len(case["hits"]) == 2
                stats["ordinary_flags"] += bool(case["ordinary"])
                stats["unmatched_flags"] += len(case["mixed"]) - len(case["hits"])
                stats["complete_native_readings"] += case["complete"]
                stats["unconfounded"] += len(set(case["hits"]) - set(case["ordinary_at_edit"]))
                if case["ordinary"]:
                    ordinary_groups.add(work["group_sha256"])
                if layout == "published":
                    other = decisions[number][arm][point]["neutral"]
                    stable += len(set(case["hits"]) & set(other["hits"]))
            by_layout[layout] = stats | {"ordinary_groups_flagged": len(ordinary_groups),
                                        "source_groups": len({work["group_sha256"] for work in specific})}
        report[genre] = by_layout | {"located_in_both_layouts": stable}
    return report


def evaluate(measured):
    works = measured["works"]
    stats = _normalizers(works)
    ordinary_maxima = defaultdict(list)
    for work in works:
        if work["split"] != "validation":
            continue
        genre = work["genre"]
        per_layout = [proposals(variant, genre, stats) for variant in work["variants"]]
        for arm in ARMS:
            values = [score for plan in per_layout for _, _, score in plan[arm]]
            assert values
            ordinary_maxima[genre, arm].append(max(values))
    points = {key: {label: _quantile(values, fraction) for label, fraction in POINTS.items()}
              for key, values in ordinary_maxima.items()}
    held = [work for work in works if work["split"] == "test"]
    decisions = {}
    for work in held:
        genre = work["genre"]
        by_layout = {layout: {v["version"]: (v, proposals(v, genre, stats)) for v in work["variants"] if v["layout"] == layout}
                     for layout in ("published", "neutral")}
        choices = {}
        for arm in ARMS:
            choices[arm] = {}
            for point, value in points[genre, arm].items():
                choices[arm][point] = {}
                for layout, pair in by_layout.items():
                    ordinary, ordinary_candidates = pair["ordinary"]
                    mixed, mixed_candidates = pair["mixed"]
                    regular = _choice(arm, ordinary, ordinary_candidates[arm], value)
                    rewritten = _choice(arm, mixed, mixed_candidates[arm], value)
                    target = [row["compact"] for row in mixed["truth"]]
                    counterpart = ordinary["matched_targets"]
                    hits = _match(rewritten, target)
                    views = ("context",) if arm == ARMS[0] else ("isolated", "context") if arm == ARMS[3] else ("isolated",)
                    choices[arm][point][layout] = {
                        "ordinary": regular, "mixed": rewritten, "targets": target,
                        "hits": [j for _, j in hits],
                        "ordinary_at_edit": [j for j, candidate in enumerate(counterpart)
                                             if any(_overlap(item["compact"], candidate) for item in regular)],
                        "complete": all(item is not None for variant in (ordinary, mixed)
                                        for view in views for item in variant[view]),
                    }
        decisions[work["document"]] = choices
    result = {arm: {point: _summarize(held, decisions, arm, point) for point in POINTS} for arm in ARMS}
    return {
        "design": DESIGN, "normalizers": {"/".join((g, v, str(b))): list(value) for (g, v, b), value in stats.items()},
        "points": {"/".join(key): value for key, value in points.items()},
        "decisions": decisions, "summary": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    document = source.read(args.receipt)
    outcome = evaluate(document)
    if args.write:
        source.save(args.write, document | {"comparison": outcome})
    else:
        assert outcome == document["comparison"], "Published writing outcomes differ from their inputs"
    for arm, points in outcome["summary"].items():
        for label, work in points.items():
            print(arm, label, work)


if __name__ == "__main__":
    main()
