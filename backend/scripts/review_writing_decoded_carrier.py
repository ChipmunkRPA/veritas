# SPDX-License-Identifier: MIT
"""Recount saved writing carrier outcomes without source prose or model weights."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.verify_writing_entropy_prominence import matching

CARRIERS = {"odt", "hidden-odt", "rtf", "hidden-rtf"}


def _validate_locations(row, carrier):
    groups = carrier["source_positions"]
    if len(groups) != len(row["selected"]):
        raise ValueError("Carrier omitted a selected writing range")
    for group, span in zip(groups, row["selected"], strict=True):
        cursor, end = span["decoded"]
        block = -1
        for place in group:
            start, stop = place["range"]
            index = place["paragraph_index"]
            ranges = place["source"]
            if (index <= block or start != cursor or not start < stop <= end
                    or not ranges or any(not 0 <= a < b for a, b in ranges)
                    or any(a < previous for (_, previous), (a, _) in zip(ranges, ranges[1:]))):
                raise ValueError("Carrier has invalid source positions")
            block, cursor = index, stop + 2
        if cursor - 2 != end:
            raise ValueError("Carrier positions omit part of a selected passage")


def review(receipt):
    if receipt["schema"] != "chipmunk.writing-decoded-carrier-transfer/1":
        raise ValueError("Unrecognized writing carrier receipt")
    counts = defaultdict(Counter)
    witnessed = defaultdict(set)
    for row in receipt["cases"]:
        panel, role = row["panel"], row["role"]
        group = (panel, row["group"])
        if role in witnessed[group] or set(row["carriers"]) != CARRIERS or not 12 <= row["sentence_count"] <= 128:
            raise ValueError("Repeated case or incomplete carrier scope")
        witnessed[group].add(role)
        for carrier in row["carriers"].values():
            _validate_locations(row, carrier)
        selected = [span["compact"] for span in row["selected"]]
        truth = row["truth"]
        hits = matching(selected, truth)
        ordinary_alert = int(bool(selected) and not truth)
        counts[panel].update(cases=1, labels=len(truth), localized=hits, assessed=1,
                             ordinary_cases=int(not truth), ordinary_alerts=ordinary_alert,
                             selected_ranges=len(selected), native_exact_cases=len(CARRIERS),
                             selected_carrier_ranges=len(selected) * len(CARRIERS),
                             historic_focus_alerts=matching(selected, [row["natural_edit"]]) if row["natural_edit"] else 0,
                             unmatched_mixed_ranges=len(selected) - hits if truth else 0)
        counts[panel][f"{role}_alerts"] += ordinary_alert
    counts = {panel: dict(values) for panel, values in counts.items()}
    if counts != receipt["totals"]:
        raise ValueError("Recorded writing carrier counts differ from their ranges")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt) as stream:
        print(json.dumps(review(json.load(stream)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
