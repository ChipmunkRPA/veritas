# SPDX-License-Identifier: MIT
"""Recount the held stretched-image comparison from its public numeric receipt."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-reference-axis-resize-confirmation/1"


def _recount(confirmation):
    plan, rows = confirmation["plan"], confirmation["rows"]
    groups = {group["group"]: group for group in plan["groups"]}
    if (len(groups) != 12 or len({group["original_rgb_sha256"] for group in groups.values()}) != 12
            or len({group["credit"]["AuthorProfileURL"] for group in groups.values()}) != 12):
        raise ValueError("The evaluation needs twelve independent source groups")
    expected = {(case["group"], case["role"], carrier) for case in plan["cases"] for carrier in case["formats"]}
    if len(rows) != len(expected) or {(row["group"], row["role"], row["carrier"]) for row in rows} != expected:
        raise ValueError("Deliveries differ from the frozen cases")
    totals = {}
    for row in rows:
        role, carrier = row["role"], row["carrier"]
        before, after = row["base"], row["followup"]
        w, h = row["submitted_size"]
        rw, rh = row["earlier_size"]
        group = groups.get(row["group"])
        if group and (row["earlier_size"] != group["size"] or row["submitted_size"] != group["submitted_size"]):
            raise ValueError("Delivery differs from its source dimensions")
        counts = totals.setdefault(carrier, {}).setdefault(role, Counter())
        counts.update(files=1, compared=after["assessed"], exact=row["exact_replayed"], marked_regions=len(after["regions"]))
        if role in plan["valid_local_regions"]:
            if before != after or row["exact_replayed"]:
                raise ValueError("Exact clean verification changed the local measurements")
            target = row["truth_pixels"]
            useful = False
            for region in after["regions"]:
                x, y, width, height = region["box"]
                native = region["earlier"]
                if (native["x"] != x * rw // w or native["y"] != y * rh // h
                        or native["x"] + native["w"] != ((x + width) * rw + w - 1) // w
                        or native["y"] + native["h"] != ((y + height) * rh + h - 1) // h):
                    raise ValueError("Region does not map to the recorded coordinates in the earlier copy")
                selected, overlap = region["selected"], region["intersection"]
                if not 0 <= overlap <= min(target, selected):
                    raise ValueError("Region overlap is inconsistent with changed pixels")
                useful |= 2 * overlap >= target and 10 * overlap >= 3 * (selected + target - overlap)
            counts["useful"] += useful
        elif role in plan["unchanged_roles"] or role == "flat_equal":
            if not row["exact_replayed"] or not after["assessed"] or after["regions"] or after["exact_pixels"] != w * h:
                raise ValueError("A clean pair lacks an exact full-image match")
        elif after["assessed"] or after["regions"] or row["exact_replayed"]:
            raise ValueError("An incompatible control produced comparison evidence")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema") != SCHEMA:
        raise ValueError("Unsupported evaluation receipt")
    print(json.dumps(_recount(report["confirmation"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
