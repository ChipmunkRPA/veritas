# SPDX-License-Identifier: Apache-2.0
"""Recompute fresh actor repeat-speech localization and false alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import review_video_lipsync_repeat_motion as motion

CARRIERS, CONTROL, POSITIVE, SHA256 = motion.CARRIERS, motion.CONTROL, motion.POSITIVE, motion.SHA256
require = motion.require

SCHEMA = "chipmunk.video-lipsync-repeat-fresh-actors/1"
OUTCOMES = {"selected", "outside_shared_duration", "outside_source_geometry", "outside_source_clock", "after_selection"}


def summarize(receipt):
    lock = receipt["freeze"]
    require(lock["schema"] == SCHEMA and lock["point"] == .65 and lock["grid"] == [.2, .35, .5, .65], "Wrong frozen method")
    require(lock["planned_groups"] == 16 and lock["choices"] == 4 and lock["previous_actors"] == 31,
            "Wrong frozen actor or source count")
    require(lock["gate"] == {"added": 6, "groups": 4, "lost": 0, "ordinary_added": 1, "synchronized": 0,
                             "coverage": .75}, "Wrong frozen admission requirements")
    intake = receipt["qualification"]
    require([r["group"] for r in intake] == list(range(201, 217)), "Planned actor groups differ")
    source_hashes = set()
    selected = {}
    for row in intake:
        choices = row["choices"]
        require(len(choices) == 4 and all(c["outcome"] in OUTCOMES for c in choices), "Source choices differ")
        for c in choices:
            h = c["sha256"]
            require(SHA256.fullmatch(h) and h not in source_hashes, "Non-unique publisher source")
            source_hashes.add(h)
            group = row["group"]
            if c["outcome"] == "selected":
                require(group not in selected and row.get("selected") == h, "Mismatched selected source")
                selected[group] = h
            elif c["outcome"] == "after_selection":
                require(group in selected, "Inconsistent source order")
            else:
                require(group not in selected, "Candidate retried after selection")
        require(("selected" in row) == (row["group"] in selected), "Source selection differs")
    groups = receipt["rows"]
    require(len(groups) == len(selected) and {r["group"]: r["source_sha256"] for r in groups} == selected,
            "Assessed source set differs")
    for i, group in enumerate(groups):
        require(group["donor_group"] == groups[(i + 1) % len(groups)]["group"] and group["group"] != group["donor_group"],
                "The separate voice control must use a distinct actor")
        cases = group["cases"]
        require(len(cases) == 2 * (len(POSITIVE) + len(CONTROL)) and
                {(c["carrier"], c["role"]) for c in cases} == {(v, role) for v in CARRIERS for role in (*POSITIVE, *CONTROL)},
                "Missing or repeated matched delivery")
        for case in cases:
            motion.validate(case, min(lock["grid"]))
    results = {carrier: motion.summarize(groups, carrier, lock["point"], lock["gate"]) for carrier in CARRIERS}
    for result in results.values():
        result["positive_planned"] = lock["planned_groups"] * len(POSITIVE)
        result["controls_planned"] = lock["planned_groups"] * len(CONTROL)
        result["gate"]["coverage"] = result["totals"].get("measured", 0) / (result["positive_planned"] + result["controls_planned"]) >= lock["gate"]["coverage"]
        result["passes"] = all(result["gate"].values())
    return {"groups_planned": len(intake), "groups_assessed": len(groups), "carriers": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    result = summarize(receipt)
    require(receipt["summary"] == result, "Published outcomes differ from portable recount")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
