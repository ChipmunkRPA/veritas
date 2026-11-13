# SPDX-License-Identifier: Apache-2.0
"""Recompute public conversational recording repeat-speech outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import review_video_lipsync_repeat_motion as motion

SCHEMA = "chipmunk.video-lipsync-repeat-public-speech/1"
OUTCOMES = {"selected", "outside_shared_duration", "outside_source_geometry", "insufficient_speech", "after_selection"}


def summarize(receipt):
    lock = receipt["freeze"]
    motion.require(lock["schema"] == SCHEMA and lock["point"] == .65, "Wrong conversational test point")
    intake = receipt["qualification"]
    motion.require([r["group"] for r in intake] == list(range(301, 305)) and
                   len({r["source_sha256"] for r in intake if motion.SHA256.fullmatch(r["source_sha256"])}) == 4,
                   "Wrong public conversational source roster")
    selected = {}
    for row in intake:
        attempts = row["attempts"]
        motion.require(len(attempts) == 3 and len({a["offset"] for a in attempts}) == 3
                       and all(a["outcome"] in OUTCOMES for a in attempts), "Invalid qualification order")
        accepted = [a for a in attempts if a["outcome"] == "selected"]
        if accepted:
            motion.require(len(accepted) == 1 and row["selected"] == accepted[0]["offset"]
                           and motion.SHA256.fullmatch(row["prepared_sha256"]), "Selected source segment differs")
            selected[row["group"]] = row["prepared_sha256"]
        else:
            motion.require("selected" not in row and "prepared_sha256" not in row, "Unexpected source selection")
    groups = receipt["rows"]
    motion.require(len(groups) == len(selected) and {row["group"]: row["source_sha256"] for row in groups} == selected,
                   "Selected sources and scored sources differ")
    for i, row in enumerate(groups):
        donor = groups[(i + 1) % len(groups)]["group"]
        motion.require(row["donor_group"] == donor and donor != row["group"], "Separate voice source differs")
        cases = row["cases"]
        expected = {(carrier, role) for carrier in motion.CARRIERS for role in (*motion.POSITIVE, *motion.CONTROL)}
        motion.require(len(cases) == len(expected) and {(c["carrier"], c["role"]) for c in cases} == expected,
                       "Incomplete matched cases")
        for case in cases:
            motion.validate(case, .2)
    return {"groups_planned": len(intake), "groups_assessed": len(groups),
            "carriers": {carrier: motion.summarize(groups, carrier, lock["point"], lock["gate"]) for carrier in motion.CARRIERS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_bytes())
    counts = summarize(record)
    motion.require(counts == record["summary"], "Public speech results differ from portable recount")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
