"""Review local-context speech decisions without access to audio or model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "chipmunk.audio-local-context-transfer-review/1"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def useful(found, expected):
    a, b = found
    c, d = expected
    overlap = max(0, min(b, d) - max(a, c))
    return overlap / (d - c) >= .5 and overlap / (max(b, d) - min(a, c)) >= .3


def totals(rows, arm, point):
    assessed = [row for row in rows if row[arm] is not None]
    found = [row for row in assessed if row["truth"] and useful(row[arm][1], row["truth"])]
    flagged = [row for row in assessed if row[arm][0] > point]
    located = [row for row in found if row[arm][0] > point]
    controls = [row for row in flagged if row["truth"] is None]
    families = {row["family"] for row in located}
    return {"localized": len(located), "misplaced": sum(row["truth"] is not None for row in flagged) - len(located),
            "unassessed": len(rows) - len(assessed),
            "unassessed_synthetic": sum(row["truth"] is not None and row[arm] is None for row in rows),
            "useful_before_point": len(found), "false_files": len(controls),
            "isolated_background_flags": sum(row["arm"] == "background_only" for row in controls),
            "false_groups": len({row["group"] for row in controls}),
            "localized_families": {family: sum(row["family"] == family for row in located) for family in families}}


def verify(document):
    check(document["schema"] == SCHEMA, "Unexpected receipt schema")
    development, held = document["development"], document["held"]
    dev_samples = {row["pcm_sha256"] for row in development}
    held_samples = {row["pcm_sha256"] for row in held}
    check(len(development) == 28 and len(held) == len(held_samples) == 52 and len(dev_samples) == 24
          and not dev_samples.intersection(held_samples), "Unexpected source size or overlap")
    check(sum(row["truth"] is not None for row in held) == 12
          and sum(row["arm"] == "background_only" for row in held) == 4, "Held control roster differs")
    controls = [row["candidate"][0] for row in development
                if not row["arm"].startswith("synthetic") and row["candidate"] is not None]
    points = document["points"]
    check(max(controls) == points["candidate"] and document["assessed_development_controls"] == len(controls)
          and points["incumbent"] == document["freeze"]["incumbent_point"], "Operating point or coverage differs")
    for row in development + held:
        check(0 <= row["admitted_frames"] <= 299 and 0 <= row["examined_ranges"] <= 56, "Invalid frame geometry")
        result = row["candidate"]
        check((result is None) == (row["examined_ranges"] == 0), "Range and coverage contradict")
        if result is not None:
            a, b = result[1]
            check(a >= 6400 and b <= 89600 and a % 1280 == 0 and b - a == 12800, "Candidate range is outside its grid")
        if result := row.get("incumbent"):
            check(result[1] in ([16000, 48000], [48000, 80000]), "Incumbent range is outside its grid")
    fixed = {arm: totals(held, arm, point) for arm, point in points.items()}
    curve = {}
    for arm in points:
        control_groups = {row["group"] for row in held if row["truth"] is None and row[arm] is not None}
        descending = sorted((max(r[arm][0] for r in held if r["group"] == group and r["truth"] is None and r[arm] is not None)
                             for group in control_groups), reverse=True)
        curve[arm] = {str(i): totals(held, arm, cutoff) for i, cutoff in enumerate(descending[:4])}
    new, old = fixed["candidate"], fixed["incumbent"]
    passed = (new["localized"] - old["localized"] >= 2 and len(new["localized_families"]) >= 2
              and new["false_groups"] - old["false_groups"] <= 1 and new["isolated_background_flags"] == 0)
    return {"fixed": fixed, "equal_false_groups": curve, "promotion_gate": passed}


def package(path):
    raw = path.read_bytes()
    native = json.loads(raw)
    groups = {}
    held = [{k: v for k, v in row.items() if k not in ("id", "group")}
            | {"group": groups.setdefault(("held", row["group"]), len(groups))} for row in native["held"]]
    development = [{k: v for k, v in row.items() if k not in ("id", "group")}
                   | {"group": groups.setdefault(("development", row["group"]), len(groups))} for row in native["development"]]
    # Development truth also records where ordinary gain/resampling was applied.
    check(all(bool(row["truth"]) == row["arm"].startswith("synthetic") for row in held),
          "Synthetic labels and exact target intervals differ")
    return {"schema": SCHEMA, "freeze": native["freeze"], "points": native["points"],
            "assessed_development_controls": sum(not row["arm"].startswith("synthetic") and row["candidate"] is not None
                                                 for row in development),
            "development": development, "held": held, "summary": native["summary"],
            "native_receipt_sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-native", type=Path)
    args = parser.parse_args()
    try:
        document = package(args.from_native) if args.from_native else json.loads(args.receipt.read_bytes())
        check(document["summary"] == verify(document), "Independent coverage or decisions differ")
        if args.from_native:
            args.receipt.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print("Verified 80 source records, frozen points, exact range geometry, control groups and useful locations.")
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
