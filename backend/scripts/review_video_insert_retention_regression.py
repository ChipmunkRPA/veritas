# SPDX-License-Identifier: Apache-2.0
"""Replay retained-picture decisions on the older single-insertion confirmation set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import video_insert_retention_transfer as guard
from scripts import video_temporal_insert_confirmation as confirmation
from scripts import video_temporal_insert_native as native

SCHEMA = "chipmunk.video-insert-retention-regression/1"
BASELINE = native.ROOT / "docs/evaluation-panels/video-temporal-insert-confirmation-2026-11-13.json"
PLAN = {
    "source": "Every previously reported range from the published 72-comparison single-insertion panel.",
    "scope": "Historical accuracy regression. Retain the exact prior proposal and classify with the already frozen guard.",
    "bounds": "No changes to selected operating points or prior range edges.",
    "guard": guard.METHOD,
}


def baseline():
    receipt = json.loads(BASELINE.read_bytes())
    confirmation.review(receipt)
    return receipt


def support_rows(old):
    for group in old["groups"]:
        for row in group["rows"]:
            if row["paired"]["bounds"] is not None:
                yield group["example"], row


def rerun(cache, preflight):
    """Rebuild the original renders and read the retained sections on their exact native clocks."""
    from unittest.mock import patch

    cases = confirmation.plan()["cases"]
    found, rejected = confirmation.read_sources(cache, cases, preflight=preflight)
    native.source.require(not rejected and len(found) == len(cases), "Prior source commitments did not reproduce")
    before = confirmation.compare

    def compare(earlier_path, submitted_path, earlier, submitted, *, verify_app):
        row = before(earlier_path, submitted_path, earlier, submitted, verify_app=verify_app)
        if bounds := row["paired"]["bounds"]:
            support = guard.local_stats(earlier[1], submitted[1], bounds)
            row["retention"] = {"support": support, "kept": guard.outcomes(support)["both"]}
        return row

    with patch.object(confirmation, "compare", compare):
        return [confirmation.measure_group(cache, case, found[case["example"]], cases[(index + 1) % len(cases)],
                                           verify_app=False) for index, case in enumerate(cases)]


def assemble(local_rows):
    old = baseline()
    observed = iter((group["example"], item) for group in local_rows for item in group["rows"] if item["paired"]["bounds"] is not None)
    rows = []
    for (expected_example, expected), (example, row) in zip(support_rows(old), observed, strict=True):
        native.source.require(example == expected_example and all(row[field] == expected[field] for field in ("role", "truth", "paired")),
                              "Native refresh failed to reproduce a published original range")
        supports = row["retention"]["support"]
        kept = row["retention"]["kept"]
        native.source.require(guard.outcomes(supports)["both"] == kept, "Wrong guard outcome")
        rows.append({"group": example, "role": row["role"], "render": expected["render"], "bounds": row["paired"]["bounds"],
                     "support": supports, "kept": kept})
    receipt = {"schema": SCHEMA, "plan": PLAN, "baseline_sha256": native.file_sha(BASELINE),
               "compared_pairs": sum(len(group["rows"]) for group in old["groups"]), "rows": rows, "counts": summarize(rows)}
    review(receipt)
    return receipt


def summarize(rows):
    counts = {}
    for row in rows:
        counts.setdefault(row["role"], Counter()).update(prior_reports=1, kept=int(row["kept"]), withheld=int(not row["kept"]))
    return counts


def review(receipt):
    require = native.source.require
    require(receipt["schema"] == SCHEMA and receipt["plan"] == PLAN and
            receipt["baseline_sha256"] == native.file_sha(BASELINE), "Wrong original point or receipt")
    old = baseline()
    require(receipt["compared_pairs"] == sum(len(group["rows"]) for group in old["groups"]), "Missing prior comparisons")
    for record, (group, row) in zip(receipt["rows"], support_rows(old), strict=True):
        require((record["group"], record["role"], record["render"], record["bounds"]) ==
                (group, row["role"], row["render"], row["paired"]["bounds"]), "Changed original proposed range")
        require(len(record["support"]) == 2 and all(
            0 <= side["median"] <= 16 and 0 <= side["alternate_gain"] <= 16 for side in record["support"]
        ) and type(record["kept"]) is bool and guard.outcomes(record["support"])["both"] == record["kept"],
                "Invalid retained-picture result")
    counts = summarize(receipt["rows"])
    require(receipt["counts"] == counts, "Incorrect historical outcome totals")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--native-log", type=Path)
    modes.add_argument("--native-cache", type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.native_log or args.native_cache:
        if args.native_cache and not args.preflight:
            parser.error("--native-cache requires --preflight for the disjoint-source verification")
        observed = (rerun(args.native_cache, args.preflight) if args.native_cache else
                    [json.loads(line) for line in args.native_log.read_text(encoding="utf-8").splitlines()])
        receipt = assemble(observed)
        with args.receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    else:
        receipt = json.loads(args.receipt.read_bytes())
    print(json.dumps(review(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
