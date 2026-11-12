# SPDX-License-Identifier: Apache-2.0
"""Read back frozen contour regions and source-pixel overlaps without detector calls."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_contour_continuation_transfer as study
from scripts import review_image_residual_sign_transfer as arithmetic

panel = study.panel


def readback(folder, journal, points):
    keys = {panel._key(row): row for row in journal}
    panel.require(len(keys) == len(journal) == 288, "Missing or repeated holdout views")
    counts = Counter()
    sources = list(panel.paired.inputs(folder))
    for index, (case, images, _) in enumerate(sources):
        source, box = images["original"], case["box"]
        other = sources[(index + 1) % len(sources)][1]["original"]
        variants = panel.native_variants(images, box) | study.conventional(source, other, box)
        for role, pixels in variants.items():
            changes = np.count_nonzero(pixels != source, axis=2) > 0
            truth = changes if role in panel.POSITIVE else np.zeros_like(changes)
            for fmt in panel.FORMATS:
                row = keys.pop((case["slot"], case["generator"], role, fmt))
                payload, _ = panel.paired.encoded(pixels, fmt)
                panel.require(hashlib.sha256(payload).hexdigest() == row["input_sha256"]
                              and hashlib.sha256(truth.tobytes()).hexdigest() == row["truth_sha256"]
                              and hashlib.sha256(changes.tobytes()).hexdigest() == row["actual_changes_sha256"],
                              "Delivered input or change footprint differs")
                counts["views"] += 1
                for arm, proposal in row["proposals"].items():
                    if proposal and arm in study.ARMS:
                        sided = sorted(value for value in proposal["side_excess"] if value is not None)
                        panel.require(len(sided) >= 3 and proposal["score"] == max(0., sided[-3]),
                                      "Reported score differs from the supported sides")
                    picked = (proposal["selected"] if arm == study.BASELINE
                              else proposal["score"] > points[fmt][arm]["point"]) if proposal else False
                    mask = arithmetic.selected_mask(truth.shape, proposal, picked)
                    observed = row["measurements"][arm]
                    panel.require(hashlib.sha256(mask.tobytes()).hexdigest() == row["mask_sha256"][arm]
                                  and (mask.sum(), truth.sum(), np.count_nonzero(mask & truth)) ==
                                  tuple(observed[key] for key in ("reported", "target", "overlap")), "Recorded overlap differs")
                    counts["masks"] += 1
                    if role in panel.POSITIVE and arm in study.ARMS:
                        ranked = arithmetic.selected_mask(truth.shape, proposal, bool(proposal))
                        values = row["rank_geometry"][arm]
                        panel.require((ranked.sum(), truth.sum(), np.count_nonzero(ranked & truth)) ==
                                      tuple(values[key] for key in ("reported", "target", "overlap")), "Rank overlap differs")
                        counts["unthresholded_rank_checks"] += 1
    panel.require(not keys, "Unexpected holdout rows")
    return counts


def compare_summary(journal, aggregate):
    for fmt in panel.FORMATS:
        rows = [row for row in journal if row["format"] == fmt]
        for arm in (*study.ARMS, study.BASELINE):
            def measurement(row, key=arm):
                return row["measurements"][key]
            hits = Counter(row["role"] for row in rows if arithmetic.hits(measurement(row)))
            flags = [row for row in rows if row["role"] not in panel.POSITIVE and measurement(row)["reported"]]
            reported = aggregate[fmt][arm]
            panel.require((hits["edited"], hits["feathered_edit"], len(flags), len({row["slot"] for row in flags})) ==
                          tuple(reported[key] for key in ("sharp", "feathered", "ordinary", "ordinary_groups")),
                          "Published accuracy counts differ from verified rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("earlier", "later", "run", "output"):
        parser.add_argument(label, type=Path)
    args = parser.parse_args()
    frozen = panel.read(args.run / "freeze.json")
    result = panel.read(args.run / "aggregate.json")
    points = panel.read(args.run / "points.json")
    panel.require(panel.sha(args.run / "freeze.json") == result["freeze_sha256"] == points["freeze_sha256"]
                  and panel.sha(args.run / "points.json") == result["points_sha256"], "Freeze or point differs")
    for table in points["points"].values():
        for row in table.values():
            panel.require(len(row["source_maxima"]) == 8 and row["point"] == max(.10, sorted(row["source_maxima"])[-2]),
                          "Point differs from the frozen rule")
    totals = Counter()
    for name, folder in (("earlier", args.earlier), ("later", args.later)):
        for file, sha in frozen["intake_sha256"][name].items():
            panel.require(panel.sha(folder / file) == sha, "Source qualification differs")
        journal_path = args.run / f"{name}-rows.json"
        panel.require(panel.sha(journal_path) == result["journal_sha256"][name], "Holdout journal differs")
        journal = panel.read(journal_path)
        totals.update(readback(folder, journal, points["points"]))
        compare_summary(journal, result["results"][name])
    panel.require(totals["views"] == result["views"], "Readback incomplete")
    panel.write(args.output, {**totals, "aggregate_sha256": panel.sha(args.run / "aggregate.json"), "status": "verified"})
    print(dict(totals))


if __name__ == "__main__":
    main()
