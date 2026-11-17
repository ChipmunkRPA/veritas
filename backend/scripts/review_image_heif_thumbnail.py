# SPDX-License-Identifier: Apache-2.0
"""Recount HEIC edit locations and control flags using the standard library."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from itertools import combinations


def selected_area(boxes):
    # The served rectangles usually do not intersect. Inclusion-exclusion
    # also covers a smaller rectangle lying inside a rounded native bound.
    area = 0
    for size in range(1, len(boxes) + 1):
        for items in combinations(boxes, size):
            left, top = (max(b[key] for b in items) for key in ("x", "y"))
            right = min(b["x"] + b["w"] for b in items)
            bottom = min(b["y"] + b["h"] for b in items)
            area += (-1) ** (size + 1) * max(0, right - left) * max(0, bottom - top)
    return area


def review(document):
    assert document["schema"] == "chipmunk.heif-thumbnail-validation/1"
    rows = document["rows"]
    sites, totals = defaultdict(set), defaultdict(Counter)
    conditions = {"sharp", "feather", "ordinary", "refreshed_edit", "refreshed_tone", "refreshed_blur", "refreshed_copy"}
    for row in rows:
        key = row["source"], row["edge"]
        assert row["edge"] in (192, 320)
        assert row["condition"] in conditions - sites[key]
        sites[key].add(row["condition"])
        reading = row["reading"]
        regions, (height, width) = reading["regions"], row["shape"]
        assert len(regions) <= 3 and all(0 <= b["x"] < b["x"] + b["w"] <= width
                                        and 0 <= b["y"] < b["y"] + b["h"] <= height for b in regions)
        assert reading["status"] == "comparable" or not regions
        summary = totals[f'{row["family"]}-{row["edge"]}']
        if reading["status"] != "comparable":
            summary["unassessed"] += 1
        if row["condition"] in ("sharp", "feather"):
            assert row["mismatch"] and len(row["truth_sha256"]) == 64
            summary["mismatched"] += 1
            truth, selected, overlap = (reading.get(key, 0) for key in ("truth", "selected", "intersection"))
            assert selected == selected_area(regions) and 0 <= overlap <= min(truth, selected) and truth <= width * height
            if truth and overlap / truth >= .50 and overlap / (truth + selected - overlap) >= .30:
                summary["located"] += 1
            elif regions:
                summary["misplaced"] += 1
        else:
            assert not row["mismatch"] and row["truth_sha256"] is None
            summary["controls"] += 1
            summary["control_flags"] += bool(regions)
    assert len({r["file"] for r in rows}) == len(rows)
    assert all(values == conditions for values in sites.values())
    assert totals == document["counts"]
    return {"source_photos": len({key[0] for key in sites}), "cases": len(rows), "groups": totals,
            "camera_controls": len(document["cameras"]), "format_controls": len(document["external"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt")
    args = parser.parse_args()
    opener = gzip.open if args.receipt.endswith(".gz") else open
    with opener(args.receipt, "rt") as source:
        print(json.dumps(review(json.load(source)), sort_keys=True))
