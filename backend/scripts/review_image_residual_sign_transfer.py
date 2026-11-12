# SPDX-License-Identifier: Apache-2.0
"""Rebuild deliveries and read out native overlap without running a detector."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import image_edge_spread_transfer as panel

ARMS = ("fine_triplets", "wide_triplets", "two_scale")
BASELINE = "format_reference"


def selected_mask(shape, proposal, picked):
    mask = np.zeros(shape, bool)
    if picked:
        x0, y0, x1, y1 = proposal["box"]
        if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
            raise ValueError("Recorded box outside the original pixels")
        mask[y0:y1, x0:x1] = True
    return mask


def hits(measurement):
    selected, target, overlap = (measurement[key] for key in ("reported", "target", "overlap"))
    return bool(target and Fraction(overlap, target) >= Fraction(1, 2)
                and Fraction(overlap, target + selected - overlap) >= Fraction(3, 10))


def summarize(rows, fmt, arm):
    applicable = [row for row in rows if row["format"] == fmt]
    count = Counter()
    all_groups, new_groups = set(), set()
    for row in applicable:
        current, baseline = (row["measurements"][method] for method in (arm, BASELINE))
        if row["role"] in panel.POSITIVE:
            count["unassessed_positives"] += row["proposals"][arm] is None
            if hits(current):
                if row["role"] == "edited":
                    count["sharp"] += 1
                else:
                    count["feathered"] += 1
                    count["extra_feathered"] += not hits(baseline)
            continue
        count["unassessed_controls"] += row["proposals"][arm] is None
        if current["reported"]:
            count["ordinary"] += 1
            count["original"] += row["role"] == "original"
            all_groups.add(row["slot"])
            if not baseline["reported"]:
                count["extra_ordinary"] += 1
                new_groups.add(row["slot"])
    count["ordinary_groups"] = len(all_groups)
    count["extra_ordinary_groups"] = len(new_groups)
    return count


def review(folder, rows, points, summary):
    keys = {(row["slot"], row["generator"], row["role"], row["format"]): row for row in rows}
    counts = Counter()
    ranked = {fmt: {arm: dict.fromkeys(panel.POSITIVE, 0) for arm in ARMS} for fmt in panel.FORMATS}
    for case, source, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(source, case["box"]).items():
            truth = np.count_nonzero(pixels != source["original"], axis=2) > 0 if role in panel.POSITIVE else np.zeros_like(known)
            truth_sha = hashlib.sha256(truth.tobytes()).hexdigest()
            target = int(truth.sum())
            for fmt in panel.FORMATS:
                row = keys.pop((case["slot"], case["generator"], role, fmt))
                body, _ = panel.paired.encoded(pixels, fmt)
                panel.require(hashlib.sha256(body).hexdigest() == row["input_sha256"] and truth_sha == row["truth_sha256"],
                              "Rebuilt delivery or native truth differs")
                counts["verified_views"] += 1
                for arm in (*ARMS, BASELINE):
                    proposal = row["proposals"][arm]
                    if proposal and arm in ARMS:
                        candidate = min(max(0., band["core_to_flank"] - band["among_flanks"]) for band in proposal["bands"])
                        panel.require(candidate == proposal["score"], "Stored score does not match its band evidence")
                    picked = (proposal["selected"] if arm == BASELINE else proposal["score"] > points[fmt][arm]["point"]) if proposal else False
                    mask = selected_mask(truth.shape, proposal, picked)
                    selected = int(mask.sum())
                    overlap = int(np.count_nonzero(mask & truth))
                    measurement = row["measurements"][arm]
                    panel.require(hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"][arm]
                                  and (selected, target, overlap) == tuple(measurement[key] for key in ("reported", "target", "overlap")), "Independent native overlap differs")
                    counts["verified_masks"] += 1
                    if role in panel.POSITIVE and arm in ARMS and proposal:
                        best = selected_mask(truth.shape, proposal, True)
                        ranked[fmt][arm][role] += hits({"reported": int(best.sum()), "target": target,
                                                       "overlap": int(np.count_nonzero(best & truth))})
    panel.require(not keys, "Extra or duplicate rows")
    for fmt in panel.FORMATS:
        for arm in (*ARMS, BASELINE):
            expected = Counter(summary[fmt][arm])
            panel.require(expected == summarize(rows, fmt, arm), "Published summary differs from independent readback")
            counts["verified_summaries"] += 1
    return counts, ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("earlier", "later", "run", "output"):
        parser.add_argument(label, type=Path)
    args = parser.parse_args()
    frozen = panel.read(args.run / "freeze.json")
    aggregate = panel.read(args.run / "aggregate.json")
    points = panel.read(args.run / "points.json")
    panel.require(panel.sha(args.run / "freeze.json") == aggregate["freeze_sha256"] == points["freeze_sha256"]
                  and panel.sha(args.run / "points.json") == aggregate["points_sha256"], "Method or points commitment differs")
    for table in points["points"].values():
        for row in table.values():
            panel.require(len(row["source_maxima"]) == 8 and row["point"] == max(.002, sorted(row["source_maxima"])[-2]),
                          "Fixed research point differs")
    total = Counter()
    highest_ranked = {}
    for label, folder in (("earlier", args.earlier), ("later", args.later)):
        for name, sha in frozen["intake_sha256"][label].items():
            panel.require(panel.sha(folder / name) == sha, "Published source qualification differs")
        journal = args.run / f"{label}-rows.json"
        panel.require(panel.sha(journal) == aggregate["journal_sha256"][label], "Held measurements differ")
        counts, highest_ranked[label] = review(folder, panel.read(journal), points["points"], aggregate["results"][label])
        total.update(counts)
    panel.require(total["verified_views"] == aggregate["views"], "Unexpected view count")
    decision = {}
    for fmt in panel.FORMATS:
        arms = [by_format[fmt]["two_scale"] for by_format in aggregate["results"].values()]
        decision[fmt] = min(row["extra_feathered"] for row in arms) >= 2 and (
            max(row["extra_ordinary"] for row in arms) <= 4 and max(row["extra_ordinary_groups"] for row in arms) <= 3)
    panel.require(decision == aggregate["two_scale_advances"], "Recorded decision differs")
    panel.write(args.output, {**total, "status": "verified", "aggregate_sha256": panel.sha(args.run / "aggregate.json"),
                             "two_scale_advances": decision, "highest_ranked_without_point": highest_ranked})
    print(dict(total))


if __name__ == "__main__":
    main()
