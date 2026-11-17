# SPDX-License-Identifier: Apache-2.0
"""Count masked-word proposals against the app's existing writing evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from app.bench.writing_entropy_prominence import compact_range, matched, max_distinct, overlap, quantile

BOUNDS = (24, 48, 96, 193)
ARMS = ("masked_words", "surrounding_context", "nearby_masked_words", "paired_context")
POINTS = {"strict": .95, "wider": .90}
UNIT = 10_000


def measurements(case):
    readings = {row[0]: row[1:6] for row in case["readings"]}
    for first in sorted(readings):
        for width in (1, 2, 3):
            stop = first + width
            if any(index not in readings for index in range(first, stop)):
                break
            entries = [readings[index] for index in range(first, stop)]
            tokens = sum(entry[0] for entry in entries)
            if not BOUNDS[0] <= tokens < BOUNDS[-1]:
                continue
            count = sum(entry[1] for entry in entries)
            full = round(sum(words * both for _, words, _, both, _ in entries) / count)
            alone = round(sum(words * isolated for _, words, _, _, isolated in entries) / count)
            key = f"{width}/{next(i for i, bound in enumerate(BOUNDS[1:]) if tokens < bound)}"
            yield (first, stop), key, (-full, full - alone)


def calibrate_scales(ordinary):
    bins = defaultdict(lambda: defaultdict(list))
    for case in ordinary:
        for _, key, raw in measurements(case):
            bins[key][case["group"]].append(raw)
    scales = {}
    for key, groups in sorted(bins.items()):
        if len(groups) < 6:
            continue
        stats = [[], []]
        for rows in groups.values():
            for dim in range(2):
                line = [raw[dim] for raw in rows]
                stats[dim].append((median(line), quantile(line, .8), quantile(line, .2)))
        scales[key] = [[round(median(item[0] for item in values)),
                        max(1, round(median(item[1] for item in values) - median(item[2] for item in values)))]
                       for values in stats]
    return scales


def scaled(case, scales):
    return {span: tuple(round(UNIT * (value - loc) / size)
                        for value, (loc, size) in zip(raw, scales[key], strict=True))
            for span, key, raw in measurements(case) if key in scales}


def paired_weights(cases, scales):
    paired = defaultdict(lambda: [[], []])
    for case in cases:
        group = paired[case["group"]]
        for bounds, values in scaled(case, scales).items():
            if case["role"] == "ordinary":
                group[0].append(values)
            elif any(overlap(compact_range(case["sentences"], bounds), span) for span in case["truth"]):
                group[1].append(values)
    margins = [[], []]
    for ordinary, changed in paired.values():
        if ordinary and changed:
            for dim in range(2):
                margins[dim].append(mean(row[dim] for row in changed) - mean(row[dim] for row in ordinary))
    if min(map(len, margins)) < 8:
        raise ValueError("Insufficient paired sources for the masked-context combination")
    difference = [median(values) for values in margins]
    norm = sum(map(abs, difference))
    return [round(UNIT * value / norm) for value in difference] if norm else [0, 0]


def candidates(case, scales, weights):
    values = scaled(case, scales)
    options = {arm: [] for arm in ARMS}
    for (first, stop), (direct, context) in values.items():
        options["masked_words"].append((direct, first, stop))
        options["surrounding_context"].append((context, first, stop))
        options["paired_context"].append((round((weights[0] * direct + weights[1] * context) / UNIT), first, stop))
        left = [values[i, i+1][0] for i in range(max(0, first - 3), first) if (i, i+1) in values]
        right = [values[i, i+1][0] for i in range(stop, stop + 3) if (i, i+1) in values]
        if len(left) >= 2 and len(right) >= 2:
            options["nearby_masked_words"].append((round(direct - max(median(left), median(right))), first, stop))
    return {arm: (max_distinct(rows, case["existing"]), bool(rows)) for arm, rows in options.items()}


def points(controls, scales, weights):
    scores = defaultdict(dict)
    for case in controls:
        for arm, (best, _) in candidates(case, scales, weights).items():
            if best is not None:
                previously = scores[arm].get(case["group"])
                scores[arm][case["group"]] = best[0] if previously is None else max(previously, best[0])
    if any(len(scores[arm]) < 8 for arm in ARMS):
        raise ValueError("The local masked comparison has too few ordinary controls")
    return {policy: {arm: quantile(list(scores[arm].values()), share) for arm in ARMS}
            for policy, share in POINTS.items()}


def summarize(cases, scales, weights, limits, *, match=None):
    count = match or (lambda intervals, truth: len(matched(intervals, truth)))
    totals = defaultdict(Counter)
    for case in cases:
        baseline = [compact_range(case["sentences"], interval) for interval in case["existing"]]
        truth = case["truth"]
        initial = count(baseline, truth)
        proposals = candidates(case, scales, weights)
        for policy, thresholds in limits.items():
            for arm, (primary, assessed) in proposals.items():
                chosen = primary is not None and primary[0] > thresholds[arm]
                added = [compact_range(case["sentences"], primary[1:])] if chosen else []
                final = count(baseline + added, truth)
                for genre in ("all", case["genre"]):
                    row = totals[f"{policy}/{arm}/{genre}"]
                    row.update(files=1, measured=assessed, available=primary is not None,
                               baseline_assessed=case["incumbent_assessed"])
                    if case["role"] == "ordinary":
                        row.update(ordinary=1, baseline_flags=bool(baseline), added_flags=bool(added) and not baseline,
                                   ordinary_with_candidate=bool(added))
                    else:
                        row.update(rewrites=1, edits=len(truth), baseline_hits=initial,
                                   new_hits=final - initial, misplaced_additions=bool(added) and final == initial)
    return {name: dict(row) for name, row in sorted(totals.items())}


def choose(results):
    options = []
    for policy in POINTS:
        for arm in ARMS:
            row = results[f"{policy}/{arm}/all"]
            new = row["new_hits"]
            false = row["added_flags"]
            misplaced = row["misplaced_additions"]
            utility = 2 * new - 2 * false - misplaced
            options.append(((utility, new, -false, -misplaced, policy == "strict"), policy, arm))
    key, policy, arm = max(options)
    return {"policy": policy, "arm": arm, "utility": key[0]}
