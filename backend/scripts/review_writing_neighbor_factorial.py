# SPDX-License-Identifier: Apache-2.0
"""Recount the anonymous neighboring-sentence trial with only the standard library."""

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

SCHEMA = "chipmunk.writing-neighbor-factorial/1"
METHODS = ("current", "isolated", "agreement")
ORDINARY = {"original", "archive_revision"}
FACTUAL = {"preserved", "changed", "unclear"}
SENTENCES, INTERIOR = 12, range(2, 10)
CROSSINGS = (("original", "neighbor_only", "focal"), ("focal_only", "both", "focal"),
             ("original", "focal_only", "neighbor"), ("neighbor_only", "both", "neighbor"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(reading, words, reference):
    value = min(.999999, max(.000001, reading / 1_000_000))
    center, scale = reference[0 if words < 16 else 1 if words < 33 else 2]
    return round(min(4., max(-4., (math.log(value / (1 - value)) - center) / scale)), 8)


def best(values):
    rank = sorted(zip(values, INTERIOR, strict=True), reverse=True)
    return rank[0] if rank[0][0] > rank[1][0] else None


def readings(row, references):
    words, raw = row["words"], row["readings_ppm"]
    isolated = best([normalized(raw["isolated"][i], words[i], references["isolated"]) for i in INTERIOR])
    shared = best([normalized(raw["shared_endpoints"][i - 2], words[0] + words[i] + words[-1], references["context"])
                   for i in INTERIOR])
    agreement = ((min(isolated[0], shared[0]), isolated[1])
                 if isolated and shared and isolated[1] == shared[1] else None)
    return {"isolated": isolated, "agreement": agreement}


def interval_match(weights, span, focus):
    start, stop = span
    return start <= focus < stop and 10 * weights[focus] >= 3 * sum(weights[start:stop])


def match_sites(weights, ranges, truth):
    options = [[None, *(focus for focus in truth if interval_match(weights, span, focus))] for span in ranges]
    return max((len({focus for focus in assignment if focus is not None}) for assignment in product(*options)), default=0)


def audit_attempts(record, members):
    summaries, qualified = {}, {}
    for arm, entries in record.items():
        observed, best_labels = set(), {}
        statuses, facts = Counter(), Counter()
        for work, position, attempt, outcome, factual in entries:
            key = work, position, attempt
            require(work in members and members[work]["split"] == "held" and key not in observed
                    and position in {"focal", "neighbor"} and type(attempt) is int and attempt in {0, 1},
                    "Invalid generation-attempt identity")
            observed.add(key)
            statuses[outcome] += 1
            if outcome == "eligible":
                require(factual in FACTUAL, "Eligible generation attempt has no factual review")
                facts[factual] += 1
                label = (factual != "preserved", attempt, factual)
                prior = best_labels.get((work, position))
                if prior is None or label < prior:
                    best_labels[(work, position)] = label
            else:
                require(factual is None, "Ineligible attempts cannot carry a factual label")
        summaries[arm] = {"attempts": len(entries), "eligible": statuses["eligible"],
                          "rejected": dict(statuses - Counter({"eligible": statuses["eligible"]})), "factual": dict(facts)}
        if arm == "target_only":
            qualified = {key: value[2] for key, value in best_labels.items()}
    return summaries, qualified


def recount(packet):
    require(packet["schema"] == SCHEMA, "Unsupported writing trial receipt")
    groups = {row["work"]: row for row in packet["groups"]}
    require(len(groups) == len(packet["groups"]), "Repeated work group")
    generation, qualified = audit_attempts(packet["generation_attempts"], groups)
    files, layouts = {}, defaultdict(dict)
    for row in packet["rows"]:
        work, role = row["work"], row["role"]
        keys = ("words", "nonspace_chars", "sentence_ids")
        require(work in groups and role not in layouts[work] and all(len(row[key]) == SENTENCES for key in keys)
                and all(type(n) is int and n > 0 for key in keys for n in row[key]), "Invalid sentence geometry")
        measured = row["readings_ppm"]
        require(set(measured) == {"isolated", "rolling", "shared_endpoints"}
                and all(len(series) == (8 if kind == "shared_endpoints" else SENTENCES) and
                        all(type(n) is int and 0 <= n <= 1_000_000 for n in series) for kind, series in measured.items()),
                "Invalid measured readings")
        ranges = row["current_ranges"]
        require(len(ranges) <= 6 and all(type(start) is int and type(stop) is int and 0 <= start < stop <= SENTENCES
                                         for start, stop in ranges), "Invalid current range")
        layouts[work][role] = row
        files[(work, role)] = readings(row, packet["normalization"])

    for work, group in groups.items():
        require(group["split"] in {"development", "held"} and group["focal"] in INTERIOR
                and group["neighbor"] in INTERIOR and abs(group["focal"] - group["neighbor"]) == 1,
                "A source group lacks an interior adjacent pair")
        require(group["generator"] in ({"A", "B"} if group["split"] == "held" else {None}), "Invalid generator cohort")
        roles = layouts[work]
        available = {name: qualified[(work, name)] for name in ("focal", "neighbor") if (work, name) in qualified}
        expected_roles = ORDINARY | {name + "_only" for name in available}
        if len(available) == 2:
            expected_roles.add("both")
        require(set(roles) == expected_roles, "Wrong number or types of source deliveries")
        for role, row in roles.items():
            edited = ({group["focal"]: "archive"} if role == "archive_revision" else
                      {group[name]: label for name, label in available.items() if role in {name + "_only", "both"}})
            changed = {i for i, (left, right) in enumerate(zip(roles["original"]["sentence_ids"], row["sentence_ids"], strict=True))
                       if left != right}
            require(edited == dict(row["truth"]) and set(edited) == changed, "Sentence identity differs from recorded edit truth")

    development = {work for work, row in groups.items() if row["split"] == "development"}
    held = set(groups) - development
    allowed = min(len(development), 1 + sum(any(r["current_ranges"] for r in layouts[work].values()) for work in development))
    points = {}
    for method in METHODS[1:]:
        maxima = sorted((max((files[work, role][method] or (-1e9, None))[0] for role in layouts[work])
                         for work in development), reverse=True)
        points[method] = maxima[allowed] if allowed < len(maxima) else -1e9

    counts = {name: Counter() for name in METHODS}
    sites, ordinary, independent_truth = defaultdict(set), defaultdict(set), set()
    scores = {name: Counter() for name in METHODS[1:]}
    variant_count, strict_single_count, generated_files, full_groups, complete_strict_groups = 0, 0, 0, 0, 0
    changed = Counter()
    maximum = Counter()
    matched_readings = 0
    for work in sorted(held):
        group, roles = groups[work], layouts[work]
        for left, right, retained in CROSSINGS:
            if left not in roles or right not in roles:
                continue
            focus = group[retained]
            before, after = roles[left], roles[right]
            require(before["sentence_ids"][focus] == after["sentence_ids"][focus], "Crossed comparison changed its target sentence")
            matched_readings += 1
            for key, index in (("isolated", focus), ("rolling", focus), ("shared_endpoints", focus - 2)):
                delta = abs(before["readings_ppm"][key][index] - after["readings_ppm"][key][index])
                changed[key] += delta > 1
                maximum[key] = max(maximum[key], delta)

        for role, row in roles.items():
            picks = files[work, role]
            spans = {name: [(pick[1], pick[1] + 1)] if (pick := picks[name]) and pick[0] > points[name] else []
                     for name in METHODS[1:]}
            spans["current"] = row["current_ranges"]
            if role in ORDINARY:
                for name, selection in spans.items():
                    if selection:
                        ordinary[name].add(work)
                        counts[name][role + "_alert_files"] += 1
                continue
            truth = dict(row["truth"])
            strict = {i for i, status in truth.items() if status == "preserved"}
            independent_truth.update((work, i) for i in truth)
            variant_count += len(truth)
            generated_files += 1
            strict_single_count += len(truth) == len(strict) == 1
            full_groups += role == "both"
            complete_strict_groups += role == "both" and len(strict) == 2
            other = group["neighbor"] if role == "focal_only" else group["focal"] if role == "neighbor_only" else None
            for name, selection in spans.items():
                hits = match_sites(row["nonspace_chars"], selection, truth)
                counts[name].update(useful=hits, wrong=len(selection) - hits,
                                    strict_single_hits=hits if len(truth) == len(strict) == 1 else 0,
                                    displaced_to_unchanged_partner=(other is not None and any(
                                        interval_match(row["nonspace_chars"], span, other) for span in selection)))
                sites[name].update((work, i) for i in truth
                                   if any(interval_match(row["nonspace_chars"], span, i) for span in selection))
                if name in picks:
                    pick = picks[name]
                    scores[name].update(unique_rank=bool(pick), correct_without_point=bool(pick and pick[1] in truth))

    for method in METHODS:
        counts[method].update(ordinary_groups=len(ordinary[method]), independent_sites=len(sites[method]))
    return {"generation": generation, "development_groups": len(development), "held_groups": len(held),
            "held_generated_files": generated_files, "edit_variant_positions": variant_count,
            "independent_edit_sites": len(independent_truth), "meaning_preserving_single_edit_files": strict_single_count,
            "full_factorial_groups": full_groups, "fully_meaning_preserving_factorial_groups": complete_strict_groups,
            "comparison": {name: dict(counts[name]) for name in METHODS}, "points": points,
            "independent_sites_by_generator": {method: dict(Counter(groups[work]["generator"] for work, _ in sites[method]))
                                               for method in METHODS},
            "ranks_before_point": {name: dict(scores[name]) for name in METHODS[1:]},
            "added_ordinary_groups_by_agreement": len(ordinary["agreement"] - ordinary["current"]),
            "neighbor_crossings": {"matched": matched_readings, "changed": dict(changed), "max_absolute_delta_ppm": dict(maximum)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.receipt.read_text())
    summary = recount(packet)
    require(summary == packet["summary"], "Aggregate differs from the anonymous measurement ledger")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
