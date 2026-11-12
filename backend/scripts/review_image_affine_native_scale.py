"""Recompute the exact large-image copy evaluation counts without detector imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{64}")
CONTROLS = ("published", "global_tone", "local_tone", "resize")


def read(path: Path):
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, note):
    if not ok:
        raise ValueError(note)


def inspect(run: Path):
    freeze, aggregate, rows = (read(run / name) for name in ("freeze.json", "aggregate.json", "measurements.json"))
    require(sha(run / "freeze.json") == aggregate["freeze_sha256"]
            and sha(run / "measurements.json") == aggregate["measurements_sha256"]
            and sha(run / "plan.json") == freeze["roster_sha256"]
            and sha(run / "acquisition.json") == freeze["acquisition_sha256"], "Run commitments differ")
    sources = {row["slot"]: row for row in freeze["sources"]}
    require(len(sources) == aggregate["sources"] and len(rows) == aggregate["views"], "Source or view count differs")
    for row in sources.values():
        require(sha(run / row["name"]) == row["sha256"], "Original image bytes differ")
    arms, deliveries, treatments = (tuple(freeze[key]) for key in ("arms", "formats", "treatments"))
    expected = set(product(sources, deliveries, treatments))
    counts = defaultdict(Counter)
    matched, ordinary = defaultdict(set), defaultdict(set)
    pair_sides = 0
    for row in rows:
        slot, delivery, treatment = (row[key] for key in ("slot", "format", "treatment"))
        require((slot, delivery, treatment) in expected, "Missing, duplicate or unexpected input view")
        expected.remove((slot, delivery, treatment))
        require(all(SHA.fullmatch(value) for value in (row["input_sha256"], *row["truth_sha256"]))
                and len(row["truth_sha256"]) == 2, "Image or exact truth commitment is missing")
        status, proposed, results = (row[key] for key in ("status", "proposals", "results"))
        require(type(proposed) is int and 0 <= proposed <= 3 and (
            (status == "assessed" and set(results) == set(arms))
            or (status == "unassessed_match_budget" and not results and proposed == 0)
        ), "Unexpected result or proposal state")
        for arm in arms:
            pairs = results.get(arm, [])
            require(len(pairs) <= proposed, "More pair responses than distinct proposals")
            useful = False
            for pair in pairs:
                require(len(pair["overlap"]) == 2 and len(pair["mask_sha256"]) == 2
                        and all(SHA.fullmatch(value) for value in pair["mask_sha256"]), "Incomplete pair mask evidence")
                hits = []
                for selected, truth, intersect in pair["overlap"]:
                    require(all(type(value) is int for value in (selected, truth, intersect))
                            and 300 <= selected <= sources[slot]["size"][0] * sources[slot]["size"][1]
                            and 0 <= intersect <= min(selected, truth)
                            and (truth == 0) == (treatment in CONTROLS), "Invalid native overlap counts")
                    iou = Fraction(intersect, selected + truth - intersect)
                    recall = Fraction(intersect, truth) if truth else 0
                    hits.append(iou >= Fraction(3, 10) and recall >= Fraction(1, 2))
                    pair_sides += 1
                useful |= all(hits)
            counts[arm, delivery, treatment].update(views=1, unassessed=status != "assessed",
                                                     proposals=proposed > 0, flagged=bool(pairs), useful=useful)
            if useful:
                matched[arm, delivery].add(slot)
            if treatment in CONTROLS and pairs:
                ordinary[arm, delivery].add(slot)
    require(not expected, "Frozen views were omitted")

    results = {}
    passed = []
    for arm in arms:
        results[arm] = {}
        for fmt in deliveries:
            table = {treatment: dict(counts[arm, fmt, treatment]) for treatment in treatments}
            groups = len(matched[arm, fmt])
            controls = len(ordinary[arm, fmt])
            advances = groups >= 6 and table["feathered"]["useful"] >= 2 and table["published"]["flagged"] <= 1 and controls <= 2
            results[arm][fmt] = {"treatments": table, "useful_groups": groups, "ordinary_groups": controls, "passes": advances}
        if all(results[arm][fmt]["passes"] for fmt in deliveries):
            passed.append(arm)
    require(results == aggregate["results"], "Published comparison counts differ")
    selected = max(passed, key=lambda arm: (
        sum(counts[arm, fmt, edit]["useful"] for fmt in deliveries for edit in ("sharp", "feathered")),
        -sum(counts[arm, fmt, control]["flagged"] for fmt in deliveries for control in CONTROLS),
        -arms.index(arm),
    )) if passed else None
    require(selected == aggregate["selected"], "Published selection differs from the frozen rule")
    return {"sources": len(sources), "views": len(rows), "verified_pair_sides": pair_sides,
            "selected": selected, "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run), sort_keys=True))


if __name__ == "__main__":
    main()
