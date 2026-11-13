"""Recount trimmed H.264 alignment and region results without detector libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-trimmed-reencode-transfer/1"
ROLES = {"large", "small", "local_texture", "reencoded", "global_grade", "wrong_source", "whole_frame_shift"}
EDITS = {"large", "small", "local_texture"}
ANCHORS = ("before", "during", "after")
SHA = re.compile("[0-9a-f]{64}")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def alignment(votes, count, offset_count):
    check([vote["offset"] for vote in votes] == list(range(offset_count)), "Missing candidate offsets")
    ranks = []
    for vote in votes:
        check(len(vote["flanks"]) == 2 and all(len(side) == 2 for side in vote["flanks"]), "Missing motion readings")
        levels = []
        for side in vote["flanks"]:
            for i, item in enumerate(side):
                active, value = item["active"], item["similarity"]
                check(type(active) is int and 0 <= active <= (count - (2 + 2 * i)) // 4
                      and math.isfinite(value) and -1.000001 <= value <= 1.000001
                      and (value == -1. if active < 4 else True), "Invalid motion reading")
            levels.append(sum(item["similarity"] for item in side) / 2)
        ranks.append((sum(levels) / 2, vote["offset"], levels, vote["flanks"]))
    first, second, *others = sorted(ranks, key=lambda row: -row[0])
    score, offset, levels, flanks = first
    lead = score - second[0]
    if score < .67 or lead < .045 or min(levels) < .60:
        return None
    if any(other[2][i] > levels[i] + .03 for other in (second, *others) for i in range(2)):
        return None
    return {"earlier_start": offset, "lead": round(lead, 6), "flanks": [
        {"moving_pairs": [item["active"] for item in side], "mean_similarity": round(level, 6)}
        for side, level in zip(flanks, levels, strict=True)]}


def verify_measurement(row, result):
    start, stop = result["frames"]
    check(5 <= start < stop <= row["source_frames"][1] - 5 and stop - start in (8, 12, 16, 20),
          "Invalid selected interval")
    check(set(result["anchors"]) == set(ANCHORS), "Missing source pictures")
    offset = row["alignment"]["earlier_start"]
    times = []
    for role, index in zip(ANCHORS, (start - 1, (start + stop) // 2, stop), strict=True):
        sample = result["anchors"][role]
        check(sample["frames"] == [index + offset, index] and all(SHA.fullmatch(v) for v in sample["rgb_sha256"]),
              "Source pictures differ from both clip positions")
        times.append([tick * Fraction(base) for tick, base in zip(sample["pts"], row["time_bases"], strict=True)])
    check(all(a - times[0][0] == b - times[0][1] for a, b in times), "Paired sample clocks differ")
    crop = result["crop"]
    check(0 <= crop["x"] < crop["x"] + crop["w"] <= 720 and 0 <= crop["y"] < crop["y"] + crop["h"] <= 480,
          "Crop is outside the source")
    check(math.isfinite(result["score"]) and result["score"] > 1.5712549522005277, "Selected result did not reach its point")
    if row["role"] in EDITS:
        a, b = row["truth"]
        pixels = row["pixels"]
        area = crop["w"] * crop["h"]
        check(0 <= pixels["intersection"] <= min(area, pixels["truth"]), "Invalid crop overlap")
        return (Fraction(max(0, min(b, stop) - max(a, start)), max(b, stop) - min(a, start)) >= Fraction(1, 2)
                and Fraction(pixels["intersection"], pixels["truth"] + area - pixels["intersection"]) >= Fraction(1, 10))
    return False


def verify_split(document, roster):
    by_source = {item["group"]: i for i, item in enumerate(roster)}
    seen = set()
    results = {role: Counter() for role in sorted(ROLES)}
    for row in document["rows"]:
        source, role = row["source"], row["role"]
        check(source in by_source and row["index"] == by_source[source] and role in ROLES
              and (source, role) not in seen, "Unknown or duplicate test case")
        seen.add((source, role))
        old_count, new_count = row["source_frames"]
        lead, tail = row["trim"]
        check(old_count == 52 and new_count == old_count - lead - tail and 3 <= min(lead, tail) <= max(lead, tail) <= 6
              and len(row["files_sha256"]) == 2 and all(SHA.fullmatch(v) for v in row["files_sha256"]),
              "Media commitments or trim differ")
        truth = [15 + row["index"] % 3 - lead, 31 + row["index"] % 4 - lead] if role in EDITS else None
        check(row["truth"] == truth and row["legacy_offset"] is None, "Authored range or prior outcome differs")
        cell = results[role]
        cell.update(files=1, legacy=0)
        measured = alignment(row["motion_votes"], new_count, lead + tail + 1)
        check(measured == row["alignment"], "Saved alignment differs from motion readings")
        if measured is None:
            check("selection" not in row, "Unaligned videos cannot have a region")
            continue
        cell.update(aligned=1, correct_offset=measured["earlier_start"] == lead,
                    assessed=row["status"] in {"no_local_range", "local_difference"})
        check(("selection" in row) == (row["status"] == "local_difference"), "Range disagrees with status")
        if "selection" in row:
            localized = verify_measurement(row, row["selection"])
            cell["flags"] += 1
            if role in EDITS:
                cell["localized"] += localized
    check(seen == {(source, role) for source in by_source for role in ROLES}, "Incomplete split")
    check(results == document["counts"], "Saved totals do not replay")
    return results


def passes(totals):
    related = EDITS | {"reencoded", "global_grade"}
    return (all(totals[role]["localized"] >= floor for role, floor in (("large", 7), ("small", 7), ("local_texture", 5)))
            and all(totals[role]["aligned"] == totals[role]["correct_offset"] for role in related)
            and totals["wrong_source"]["aligned"] == totals["whole_frame_shift"]["flags"] == 0
            and sum(totals[role]["flags"] for role in ("reencoded", "global_grade")) <= 2
            and totals["reencoded"]["assessed"] >= 8)


def digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def review(document):
    freeze, development, held = (document[key] for key in ("freeze", "development", "held"))
    check(all(section["schema"] == SCHEMA for section in (freeze, development, held)), "Unknown schema")
    check(len(freeze["sources"]["development"]) == 4 and len(freeze["sources"]["held"]) == 11
          and len({row["group"] for rows in freeze["sources"].values() for row in rows}) == 15,
          "Unexpected source split")
    check(development["split"] == "development" and held["split"] == "held"
          and development["freeze_sha256"] == held["freeze_sha256"] == digest(freeze)
          and held["development_sha256"] == digest(development), "Freeze commitment differs")
    first = verify_split(development, freeze["sources"]["development"])
    second = verify_split(held, freeze["sources"]["held"])
    return {"development": first, "held": second, "passes": passes(second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    document = (json.loads(args.ledger.read_bytes()) if args.assemble is None else {
        key: json.loads((args.assemble / (key + ".json")).read_bytes()) for key in ("freeze", "development", "held")
    })
    result = review(document)
    if args.assemble is not None:
        with args.ledger.open("x") as output:
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
