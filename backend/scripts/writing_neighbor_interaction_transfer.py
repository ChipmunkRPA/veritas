"""Replay sentence localization after discounting changes driven by neighboring context."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.review_writing_context_agreement import FILES, load
from scripts.writing_omission_attribution import (
    decisions,
    digest,
    log_odds,
    points,
    require,
    responses,
    verify_panel,
)

SCHEMA = "chipmunk.writing-neighbor-interaction/1"
ARMS = ("omission", "side_agreement", "triple_agreement")
PROTOCOL = {
    "inputs": "Use the isolated focus and left and right sentences, both adjacent pairs and the full native triple from the pinned checkpoint.",
    "side_agreement": "Discount the smaller left and right focus-omission effects by their absolute difference.",
    "triple_agreement": "Also subtract the absolute difference between the measured triple log odds and its estimate from the two adjacent pairs minus the isolated focus. All terms are normalized by the square root of focus words over 24.",
    "baseline": "Reuse the raw two-sided omission and both incumbent context and position checks at their measured source locations.",
    "choice": "Select one unique strongest complete native center strictly above the second-highest maximum of 14 ordinary F1000Research works, with a floor at zero. Count both dated versions under each work maximum.",
    "advance": "Add at least three exact locations on different held works across both publishers and both generators, with at most one additional flagged ordinary work per publisher and no more than two wrong locations per additional exact location. Require fresh confirmation before serving.",
}


def cases(name, measured):
    panel = measured[name]
    verify_panel(name, panel)
    old_cases = {(row["group"], row["variant"]): row for row in load(name)["cases"]
                 if name != "point" or row["role"] == "calibration"}
    require(len(old_cases) == len(panel["cases"]), "The two source ledgers contain different cases")
    for case in panel["cases"]:
        old = old_cases.pop((case["group"], case["variant"]))
        require(all(case[key] == old[key] for key in ("positions", "nonspace", "truth", "text_sha256")),
                "The two source ledgers contain different positions or input text")
        former = responses(case, panel["inputs"])
        position = {row["idx"]: row for row in old["review"]}
        readings = {}
        for index, left_pair, right_pair, left, center, right in case["centers"]:
            source = position[index]
            triple = old["native_current"][index]
            observed = source["responses"]
            require(source["span"] == case["positions"][index][:2]
                    and [row["input_sha256"] for row in observed] == [left_pair, triple["input_sha256"], right_pair],
                    "The measured pairs and triple differ from the recorded native spans")
            if index not in former or triple["score"] is None:
                continue
            lc, cr, l, c, r = (log_odds(panel["inputs"][key][1]) for key in (left_pair, right_pair, left, center, right))
            lcr = log_odds(float(triple["score"]).hex())
            scale = math.sqrt(case["positions"][index][3] / 24)
            omission = min(lc - l, cr - r) / scale
            require(omission == former[index][0], "The previous omission score did not replay")
            side_agreement = omission - abs((lc - l) - (cr - r)) / scale
            triple_agreement = side_agreement - abs(lcr - (lc + cr - c)) / scale
            readings[index] = dict(zip(ARMS, (omission, side_agreement, triple_agreement), strict=True))
        yield case, readings


def freeze(measured):
    maxima = defaultdict(list)
    for case, readings in cases("point", measured):
        require(case["truth"] is None and readings, "The operating point requires complete historical ordinary work")
        maxima[case["group"]].extend(readings.values())
    require(len(maxima) == 14, "Unexpected number of operating-point works")
    by_work = {group: {arm: max(row[arm] for row in rows) for arm in ARMS} for group, rows in maxima.items()}
    cutoffs = {arm: max(0, sorted(row[arm] for row in by_work.values())[-2]) for arm in ARMS}
    require(cutoffs["omission"] == points(measured["point"])["shared_drop"], "Previous omission point differs")
    return {"schema": SCHEMA, "protocol": PROTOCOL, "inputs": {name: pair[1] for name, pair in FILES.items()},
            "implementation_sha256": digest(Path(__file__)), "work_maxima": by_work, "cutoffs": cutoffs}


def evaluate(name, measured, cutoffs):
    counts = {arm: Counter() for arm in ARMS}
    ordinary, additions, families = ({arm: set() for arm in ARMS} for _ in range(3))
    existing_ordinary = set()
    rows = []
    for case, readings in cases(name, measured):
        group, variant, truth, incumbent = (case[key] for key in ("group", "variant", "truth", "incumbent"))
        existing_hit = incumbent["context_useful"] or incumbent["position_useful"]
        if truth is None and (incumbent["context_candidates"] or incumbent["position_candidate"] is not None):
            existing_ordinary.add(group)
        measures = {}
        for arm in ARMS:
            ranking = sorted(((row[arm], index) for index, row in readings.items()), reverse=True)
            unique = ranking and (len(ranking) == 1 or ranking[0][0] > ranking[1][0])
            maximum, index = ranking[0] if ranking else (None, None)
            span = case["positions"][index][:2] if unique else None
            selected = span if unique and maximum > cutoffs[arm] else None
            measures[arm] = {"maximum": maximum, "unique_top": span, "selected": selected}
            out = counts[arm]
            out.update({"files": 1, "files_assessed": bool(readings), "centers_measured": len(readings)})
            if truth is None:
                out.update({"ordinary_files": 1, "ordinary_flags": selected is not None})
                if selected is not None:
                    ordinary[arm].add(group)
                continue
            exact = selected == truth
            out.update({"edits": 1, "target_assessed": any(case["positions"][i][:2] == truth for i in readings),
                        "target_unique_top": span == truth, "exact_locations": exact, "existing_locations": existing_hit,
                        "combined_locations": existing_hit or exact, "wrong_locations": selected is not None and not exact})
            if exact and not existing_hit:
                additions[arm].add(group)
                families[arm].add(variant)
        rows.append({"group": group, "variant": variant, "truth": truth, "measures": measures})
    previous = decisions(measured[name], {"shared_drop": cutoffs["omission"], "shared_excess": 0})["shared_drop"]
    require(counts["omission"]["combined_locations"] == previous["counts"]["combined_useful"],
            "The historical omission comparator changed")
    summaries = {}
    for arm in ARMS:
        summaries[arm] = dict(counts[arm]) | {"additional_works": sorted(additions[arm]),
            "additional_generators": sorted(families[arm]), "existing_ordinary_works": len(existing_ordinary),
            "combined_ordinary_works": len(existing_ordinary | ordinary[arm]),
            "additional_ordinary_works": sorted(ordinary[arm] - existing_ordinary)}
    return {"summary": summaries, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score", "review"))
    parser.add_argument("freeze", type=Path)
    parser.add_argument("receipt", nargs="?", type=Path)
    args = parser.parse_args()
    measured = load("omission")
    frozen = freeze(measured)
    if args.action == "freeze":
        with args.freeze.open("x") as handle:
            print(json.dumps(frozen, indent=2, sort_keys=True), file=handle)
        return
    require(json.loads(args.freeze.read_bytes()) == frozen, "The operating-point freeze does not replay")
    if args.receipt is None:
        parser.error("score and review require a receipt")
    results = {panel: evaluate(panel, measured, frozen["cutoffs"]) for panel in ("f1000", "elife")}
    advanced = {}
    for arm in ARMS[1:]:
        panels = [result["summary"][arm] for result in results.values()]
        added = sum(len(row["additional_works"]) for row in panels)
        advanced[arm] = (added >= 3 and all(row["additional_works"] for row in panels)
                         and len({x for row in panels for x in row["additional_generators"]}) == 2
                         and all(len(row["additional_ordinary_works"]) <= 1 for row in panels)
                         and sum(row["wrong_locations"] for row in panels) <= 2 * added)
    result = {"schema": SCHEMA, "freeze_sha256": digest(args.freeze), "results": results, "advance": advanced}
    if args.action == "score":
        with args.receipt.open("x") as handle:
            print(json.dumps(result, indent=2, sort_keys=True), file=handle)
    else:
        require(json.loads(args.receipt.read_bytes()) == result, "Sentence predictions or results do not replay")
    print(json.dumps({"advance": advanced, "summary": {name: panel["summary"] for name, panel in results.items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
