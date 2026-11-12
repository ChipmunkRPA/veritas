"""Recompute the published phrase-smoothing dispositions from scalar evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-adaptive-word-transfer/1"
ARMS = ("unsmoothed", "fixed_3", "adaptive")
GENERATORS = {"qwen", "smol2"}
CONTROL = {"original", "ordinary"}
BUDGETS = (0, 1, 2, 4)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def useful(overlap):
    if overlap is None:
        return False
    common, union, target = overlap
    require(all(type(value) is int for value in overlap) and 0 <= common <= target <= union and target > 0,
            "Invalid native character-overlap counts")
    return 10 * common >= 3 * union and 2 * common >= target


def summarize(rows, points):
    output, details = {}, {}
    for name in ARMS:
        counts = Counter()
        flagged_ordinary, useful_rows, candidates = set(), set(), []
        for index, row in enumerate(rows):
            generated = row["kind"] in GENERATORS
            require(generated or row["kind"] in CONTROL, "Unknown trial case")
            require(generated == (row["target"] is not None), "Ground truth differs from role")
            counts["generated" if generated else "ordinary"] += 1
            candidate = row["methods"][name]
            require(bool(row["candidate_count"]) == (candidate is not None), "Incomplete region search")
            if candidate is None:
                counts["unassessed"] += 1
                continue
            score = candidate["score"]
            start, end = candidate["span"]
            require(math.isfinite(score) and type(candidate["unique"]) is bool and 0 <= start < end,
                    "Invalid phrase score or location")
            require(generated == (candidate["overlap"] is not None), "Missing generated location arithmetic")
            located = useful(candidate["overlap"])
            flag = candidate["unique"] and score > points[name]
            require(flag is candidate["flag"], "Flag does not follow the frozen point and tie rule")
            counts["top_useful"] += located
            if candidate["unique"]:
                candidates.append((score, row["group"], generated, located))
            if not flag:
                continue
            counts["flagged_generated" if generated else "flagged_ordinary"] += 1
            if located:
                useful_rows.add(index)
            if not generated:
                flagged_ordinary.add(row["group"])
        at_budget = {str(budget): 0 for budget in BUDGETS}
        for point in sorted({score for score, *_ in candidates}):
            accepted = [candidate for candidate in candidates if candidate[0] >= point]
            false_groups = len({group for _, group, generated, _ in accepted if not generated})
            for budget in BUDGETS:
                if false_groups <= budget:
                    key = str(budget)
                    at_budget[key] = max(at_budget[key], sum(located for _, _, _, located in accepted))
        output[name] = dict(counts) | {"useful": len(useful_rows), "false_groups": len(flagged_ordinary),
                                       "matched_false_group_budgets": at_budget}
        details[name] = useful_rows, flagged_ordinary
    previous, prior_false = details["unsmoothed"]
    adaptive, current_false = details["adaptive"]
    new_groups = {rows[index]["group"] for index in adaptive - previous}
    generators = {rows[index]["kind"] for index in adaptive - previous}
    additional = len(current_false - prior_false)
    return {"arms": output, "added_works": len(new_groups), "added_generators": sorted(generators),
            "lost_useful_cases": len(previous - adaptive), "new_ordinary_work_flags": additional,
            "advances": len(new_groups) >= 2 and generators == GENERATORS and not previous - adaptive and additional <= 1}


def audit(packet):
    require(packet["schema"] == SCHEMA and packet["widths"] == [1, 3, 7], "Unsupported study")
    selection = packet["selection"]
    points, groups = selection["points"], selection["group_maxima"]
    require(len(groups) == 11 and set(points) == set(ARMS), "Incorrect ordinary selection role")
    for name in ARMS:
        require(all(math.isfinite(group[name]) for group in groups) and
                points[name] == sorted(group[name] for group in groups)[-2], "Frozen selection point differs")
    require(len(selection["radii"]) == 3 and all(math.isfinite(x) and x >= 0 for x in selection["radii"]),
            "Invalid compatibility radii")
    output = {}
    require(set(packet["held"]) == {"kubernetes", "pep"}, "Missing held cohort")
    for name, panel in packet["held"].items():
        cases = panel["cases"]
        kinds = Counter(case["kind"] for case in cases)
        roster = {(case["group"], case["kind"]) for case in cases}
        require(len(roster) == len(cases) and kinds["original"] == kinds["ordinary"], "Held roster differs")
        require(all((group, control) in roster for group, _ in roster for control in CONTROL),
                "Missing matched ordinary case")
        require(set(panel["selected_widths"]) <= {"1", "3", "7"} and
                all(type(count) is int and count > 0 for count in panel["selected_widths"].values()),
                "Incorrect adaptive-width tally")
        output[name] = summarize(cases, points)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--emit", type=Path, help="Add independently computed summaries to a new public receipt")
    args = parser.parse_args()
    packet = json.loads(args.receipt.read_text())
    sibling = Path(__file__).with_name("writing_adaptive_word_transfer.py")
    require(hashlib.sha256(sibling.read_bytes()).hexdigest() == packet["runner_sha256"], "Scoring runner differs")
    results = audit(packet)
    if args.emit:
        with args.emit.open("x") as output:
            json.dump(packet | {"summary": results}, output, indent=2)
            output.write("\n")
    else:
        require(packet["summary"] == results, "Public summary differs")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
