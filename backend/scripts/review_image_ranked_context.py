"""Recount archived image ranking and localization without a model or source media."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

ARMS = ("absolute", "local_contrast")
CARRIERS = ("png", "jpeg90")
EDITS = {"target", "confined"}
CONTROLS = {"base", "blur", "tone"}
AUTHORED = {"ordinary_picture", "ordinary_blur", "ordinary_tone"}


def recount(receipt):
    if receipt["schema"] != "chipmunk.image-ranked-context/1":
        raise ValueError("Unrecognized image ranking comparison")
    for name, item in receipt["points"].items():
        maxima = item["source_maxima"]
        if len(maxima) != 8 or not all(math.isfinite(x) for x in maxima) or item["point"] != sorted(maxima)[-2]:
            raise ValueError(f"Operating point differs from calibration maxima for {name}")

    expected = set()
    for bank, sources in (("prior", 20), ("followup", 11)):
        for prefix, count, roles in (("publisher", sources, EDITS | CONTROLS), ("ordinary", 12, AUTHORED)):
            expected.update(product([bank], [f"{prefix}:{n}" for n in range(count)], roles, CARRIERS, ARMS))
    tally = {}
    for record in receipt["rows"]:
        bank, group, role, carrier, arm = (record[k] for k in ("bank", "group", "role", "carrier", "arm"))
        expected.remove((bank, group, role, carrier, arm))
        point = receipt["points"][f"{carrier}:{arm}"]["point"]
        candidates = record["candidates"]
        if record["point"] != point or len(candidates) != 3:
            raise ValueError("A case changed its locked point or ranking budget")

        width, height = record["extent"]
        findings = []
        earlier = []
        for score, box, area, label, overlap in candidates:
            x, y, right, bottom = box
            if (not math.isfinite(score) or any(type(v) is not int for v in (*box, area, label, overlap))
                    or not (0 <= x < right <= width and 0 <= y < bottom <= height)
                    or area != (right - x) * (bottom - y) or not 0 <= overlap <= min(area, label)
                    or label > width * height or (label > 0) != (role in EDITS)):
                raise ValueError("Invalid region or truth count")
            for previous, (a, b, c, d), truth in earlier:
                if score > previous or label != truth or not (right <= a or c <= x or bottom <= b or d <= y):
                    raise ValueError("Candidates are out of rank order or overlap")
            earlier.append((score, box, label))
            union = area + label - overlap
            useful = label > 0 and overlap * 2 >= label and overlap * 10 >= union * 3
            findings.append((score > point, useful))

        stats = tally.setdefault(f"{bank}:{carrier}:{arm}:{role}", Counter(inputs=0))
        stats["inputs"] += 1
        for limit in (1, 3):
            shown = findings[:limit]
            stats[f"rank_correct_{limit}"] += any(useful for _, useful in shown)
            stats[f"correct_{limit}"] += any(accept and useful for accept, useful in shown)
            stats[f"file_alerts_{limit}"] += any(accept for accept, _ in shown)
            wrong = sum(accept and not useful for accept, useful in shown)
            stats[f"wrong_regions_{limit}"] += wrong
            if role in EDITS:
                stats[f"wrong_files_{limit}"] += bool(wrong)

    if expected or tally != receipt["summary"]:
        raise ValueError("The reported outcomes do not match the complete paired roster")
    return tally


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as ledger:
        receipt = json.load(ledger)
    totals = recount(receipt)
    for carrier, arm in product(CARRIERS, ARMS):
        target = [stats for key, stats in totals.items() if key.endswith(f":{carrier}:{arm}:target")]
        controls = [stats for key, stats in totals.items() if f":{carrier}:{arm}:ordinary_" in key]
        for limit in (1, 3):
            correct = sum(stats[f"correct_{limit}"] for stats in target)
            unmatched = sum(stats[f"wrong_regions_{limit}"] for stats in target)
            ordinary = sum(stats[f"file_alerts_{limit}"] for stats in controls)
            print(f"{carrier} {arm} top {limit}: {correct}/31 direct locations, {unmatched} wrong boxes on edits, "
                  f"{ordinary}/72 authored controls flagged")


if __name__ == "__main__":
    main()
