# SPDX-License-Identifier: Apache-2.0
"""Recalculate source-bound semantic-continuity decisions from a text-free receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-semantic-continuity-transfer/1"
PANEL = {"fit": 38, "kubernetes": 31, "elife": 54}
FAMILIES = {"granite", "olmo", "qwen", "smol2"}
ARMS = ("neighbors_only", "flank_contrast")
SHA = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ranks(case):
    prior_end = 0
    count = len(case["spans"])
    require(count == len(case["nonspace"]) == len(case["sentence_sha256"]), "Incomplete source roster")
    for (start, end), letters, pin in zip(case["spans"], case["nonspace"], case["sentence_sha256"], strict=True):
        require(all(type(value) is int for value in (start, end, letters))
                and prior_end <= start < end and 0 < letters <= end - start and SHA.fullmatch(pin), "Invalid sentence source range")
        prior_end = end
    result = {arm: [] for arm in ARMS}
    require(case["focuses"] == [row["index"] for row in case["semantic"]]
            == sorted(set(case["focuses"])), "Semantic responses do not match eligible foci")
    for row in case["semantic"]:
        index = row["index"]
        require(type(index) is int and 2 <= index < count - 2, "A focus lacks four neighboring sentences")
        left, right, inner_flanks, outer_flanks = [float.fromhex(value) for value in row["cosine_hex"]]
        require(all(math.isfinite(value) and -1.000001 <= value <= 1.000001
                    for value in (left, right, inner_flanks, outer_flanks)), "Invalid cosine response")
        neighbor_mean = (left + right) / 2
        result["neighbors_only"].append((1 - neighbor_mean, index))
        result["flank_contrast"].append(((inner_flanks + outer_flanks) / 2 - neighbor_mean, index))
    require(len(case["contexts"]) == count, "Native context roster differs")
    for index, context in enumerate(case["contexts"]):
        require(SHA.fullmatch(context["sha256"]) and context["sentences"] == list(range(max(0, index - 1), min(count, index + 2))),
                "Native context ranges differ")
    if case["truth"] is not None:
        start, end = case["truth"]
        known = case["target_nonspace"]
        overlap = case["target_overlap"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= prior_end
                and type(known) is int and 0 < known <= end - start and sum(overlap) == known
                and len(overlap) == count and all(type(x) is int and 0 <= x <= min(letters, max(0, min(b, end) - max(a, start)))
                    for x, letters, (a, b) in zip(overlap, case["nonspace"], case["spans"], strict=True)),
                "Known edit does not match its saved source overlap")
    require(case["role"] not in FAMILIES or case["truth"] is not None, "Generated case lacks a known edit range")
    return {arm: sorted(values, reverse=True) for arm, values in result.items()}


def useful(case, indices):
    intersection = sum(case["target_overlap"][index] for index in indices)
    predicted = sum(case["nonspace"][index] for index in indices)
    target = case["target_nonspace"]
    return 2 * intersection >= target and 10 * intersection >= 3 * (predicted + target - intersection)


def report(items, arm, point, primary):
    totals = Counter({key: 0 for key in ("generated", "generated_useful", "generated_wrong_only", "generated_unmeasured",
                                         "generated_no_location", "ordinary", "ordinary_flagged", "ordinary_unmeasured", "incumbent_useful",
                                         "additional_useful", "ordinary_groups_flagged", "incumbent_ordinary_groups",
                                         "additional_ordinary_groups", "additional_works")})
    ordinary, prior_ordinary, additions, gains = set(), set(), set(), set()
    by_family = defaultdict(Counter)
    ordinary_roles = defaultdict(Counter)
    for row, ranking in items:
        arm_rows = ranking[arm]
        winner = arm_rows[0] if arm_rows and (len(arm_rows) == 1 or arm_rows[0][0] > arm_rows[1][0]) else None
        flagged = winner is not None and winner[0] > point
        prior = [context["sentences"] for context in row["contexts"] if primary[context["sha256"]] >= .5][:6]
        if row["role"] in FAMILIES:
            previous = any(useful(row, indices) for indices in prior)
            correct = flagged and useful(row, [winner[1]])
            covered = any(useful(row, [index]) for _, index in arm_rows)
            totals.update(generated=1, generated_useful=correct, generated_wrong_only=flagged and not correct,
                          generated_unmeasured=not covered, generated_no_location=not flagged, incumbent_useful=previous,
                          additional_useful=correct and not previous)
            by_family[row["role"]].update(total=1, useful=correct, incumbent=previous, added=correct and not previous,
                                          target_unmeasured=not covered)
            if correct and not previous:
                additions.add(row["group"])
                gains.add(row["role"])
        else:
            totals.update(ordinary=1, ordinary_flagged=flagged, ordinary_unmeasured=not arm_rows)
            ordinary_roles[row["role"]].update(total=1, candidate=flagged, incumbent=bool(prior), unmeasured=not arm_rows)
            if flagged:
                ordinary.add(row["group"])
            if prior:
                prior_ordinary.add(row["group"])
    totals.update(ordinary_groups_flagged=len(ordinary), incumbent_ordinary_groups=len(prior_ordinary),
                  additional_ordinary_groups=len(ordinary - prior_ordinary), additional_works=len(additions))
    return dict(totals) | {"additional_families": sorted(gains), "families": {key: dict(value) for key, value in by_family.items()},
                           "ordinary_by_role": {key: dict(value) for key, value in ordinary_roles.items()}}


def replay(receipt):
    require(receipt["schema"] == SCHEMA and all(SHA.fullmatch(pin) for pin in receipt["commitments"].values()), "Unknown or unbound receipt")
    rows = receipt["cases"]
    require(len({(r["panel"], r["group"], r["role"]) for r in rows}) == len(rows), "Repeated work variant")
    counts = Counter(row["panel"] for row in rows)
    held_roles = Counter(row["role"] for row in rows if row["panel"] == "plos")
    require(set(counts) == {*PANEL, "plos"} and all(counts[p] == n for p, n in PANEL.items())
            and held_roles == Counter(original=10, ordinary=10, **receipt["admission"])
            and receipt["attempts"] == {"granite": 30, "olmo": 30}, "Unexpected frozen cohort")
    require(set(receipt["admission"]) <= {"granite", "olmo"}, "Unexpected generator family")
    groups = {row["group"] for row in rows if row["panel"] == "plos"}
    require(len(groups) == 10 and len(receipt["publisher"]) == 10 and groups == {row["group"] for row in receipt["publisher"]},
            "Held publication identities differ")
    require(receipt["native_source_audit"] == {"publisher_originals": 10, "cases": 20 + sum(receipt["admission"].values()),
            "sentences": 8 * (20 + sum(receipt["admission"].values())), "native_contexts": 8 * (20 + sum(receipt["admission"].values())),
            "generated_target_locations": sum(receipt["admission"].values())}, "Native source replay roster differs")
    for source in receipt["publisher"]:
        require(source["doi"] == f"10.1371/journal.pbio.{source['group']}" and SHA.fullmatch(source["xml_sha256"])
                and source["license"].removeprefix("https://").removeprefix("http://").rstrip("/") == "creativecommons.org/licenses/by/4.0",
                "Publisher identity, commitment or license differs")
    require(all(SHA.fullmatch(row["text_sha256"]) for row in rows), "Missing input source hash")
    require(not ({r["text_sha256"] for r in rows if r["panel"] == "plos"} & {r["text_sha256"] for r in rows if r["panel"] != "plos"}),
            "Held and development source texts overlap")
    primary = {pin: float.fromhex(value) for pin, value in receipt["incumbent"].items()}
    require(set(primary) == {ctx["sha256"] for row in rows for ctx in row["contexts"]}
            and all(0 <= v <= 1 for v in primary.values()), "Incumbent response roster differs")
    panels = defaultdict(list)
    for row in rows:
        panels[row["panel"]].append((row, ranks(row)))
    thresholds = {}
    results = {}
    for arm in ARMS:
        maxima = {}
        for row, ranks_ in panels["fit"]:
            if row["role"] not in FAMILIES and ranks_[arm]:
                group = row["group"]
                maxima[group] = max(maxima.get(group, -math.inf), ranks_[arm][0][0])
        require(len(maxima) >= 2, "Insufficient genuine groups for the frozen review point")
        point = sorted(maxima.values(), reverse=True)[1]
        thresholds[arm] = {"groups": len(maxima), "point_hex": point.hex()}
        results[arm] = {panel: report(items, arm, point, primary) for panel, items in panels.items() if panel != "fit"}
    held = results["flank_contrast"]["plos"]
    advances = (all(receipt["admission"].get(family, 0) >= 4 for family in ("granite", "olmo"))
                and held["additional_works"] >= 2 and held["additional_families"] == ["granite", "olmo"]
                and held["additional_ordinary_groups"] <= 1)
    return {"points": thresholds, "results": results, "supports_further_development": advances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data = args.receipt.read_bytes()
    receipt = json.loads(data)
    result = replay(receipt)
    require(result == receipt["result"], "Published outcomes differ from replay")
    print(json.dumps({"receipt_sha256": hashlib.sha256(data).hexdigest(), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
