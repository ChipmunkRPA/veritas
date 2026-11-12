# SPDX-License-Identifier: Apache-2.0
"""Independently rebuild the held deliveries and regional measurements."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_edge_spread_transfer as panel
from scripts.review_image_multidepth_transfer import box_for_region
from scripts.review_image_residual_sign_transfer import hits, selected_mask

ARMS = ("binary_residual", "matched_processing", "format_reference")
BASELINE = ARMS[-1]
FILES = ("plan.json", "attempts.jsonl", "result.json")


def measure(truth, mask):
    reported, target, overlap = int(mask.sum()), int(truth.sum()), int(np.count_nonzero(truth & mask))
    return {"reported": reported, "target": target, "overlap": overlap,
            "iou": overlap / (reported + target - overlap) if target else None,
            "recall": overlap / target if target else None}


def replay(folder, rows, thresholds, outcomes):
    lookup = {panel._key(row): row for row in rows}
    panel.require(len(lookup) == len(rows), "Held rows contain a duplicate")
    keys = ("sharp", "feathered", "ordinary", "original", "added_feathered", "added_controls", "ungated_useful")
    counts = {fmt: {arm: Counter(dict.fromkeys(keys, 0)) for arm in ARMS} for fmt in panel.FORMATS}
    false_sources = {fmt: {arm: set() for arm in ARMS} for fmt in panel.FORMATS}
    verified = 0
    for case, native, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(native, case["box"]).items():
            truth = np.any(pixels != native["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            for fmt in panel.FORMATS:
                row = lookup.pop((case["slot"], case["generator"], role, fmt))
                delivered, _ = panel.paired.encoded(pixels, fmt)
                panel.require(hashlib.sha256(delivered).hexdigest() == row["input_sha256"]
                              and hashlib.sha256(truth.tobytes()).hexdigest() == row["truth_sha256"],
                              "Held bytes or native truth differ")
                baseline = row["proposals"][BASELINE]
                bmask = selected_mask(truth.shape, baseline, bool(baseline and baseline["selected"]))
                baseline_measure = measure(truth, bmask)
                for arm in ARMS:
                    saved = row["proposals"][arm]
                    record = counts[fmt][arm]
                    if arm == BASELINE:
                        actual = baseline_measure
                    else:
                        area = box_for_region(saved["grid_region"], truth.shape)
                        rectangle = selected_mask(truth.shape, {"box": area}, True)
                        ranked = hits(measure(truth, rectangle))
                        panel.require(math.isfinite(saved["score"]) and ranked == saved["ungated_useful"],
                                      "Ungated geometry or response invalid")
                        record["ungated_useful"] += ranked
                        actual = measure(truth, rectangle & (saved["score"] > thresholds[fmt][arm]["point"]))
                    panel.require(actual == row["measurements"][arm], "Native overlap or selected area differs")
                    if role in panel.POSITIVE and hits(actual):
                        record["sharp" if role == "edited" else "feathered"] += 1
                        record["added_feathered"] += role == "feathered_edit" and not hits(baseline_measure)
                    elif role in panel.CONTROLS and actual["reported"]:
                        record["ordinary"] += 1
                        record["original"] += role == "original"
                        if not baseline_measure["reported"]:
                            record["added_controls"] += 1
                            false_sources[fmt][arm].add(case["slot"])
                verified += 1
    panel.require(not lookup and verified == len(rows), "Held roster is incomplete")
    for fmt in panel.FORMATS:
        for arm in ARMS:
            row = dict(counts[fmt][arm]) | {"added_control_sources": len(false_sources[fmt][arm])}
            panel.require(row == outcomes[fmt][arm], "Saved outcome differs from independent summary")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "earlier", "later", "fresh"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    files = {name: args.run / f"{name}.json" for name in ("freeze", "fit", "result")}
    freeze, fit, result = (panel.read(files[name]) for name in files)
    frozen = panel.sha(files["freeze"])
    panel.require(frozen == fit["freeze_sha256"] == result["freeze_sha256"]
                  and panel.sha(files["fit"]) == result["fit_sha256"], "Study or fitted decision commitment differs")
    for arm in ARMS[:-1]:
        panel.require(panel.sha(args.run / f"{arm}.safetensors") == fit["weight_sha256"][arm], "Fitted weights differ")
    for per_arm in fit["points"].values():
        for item in per_arm.values():
            maxima = item["source_maxima"]
            panel.require(len(maxima) == 8 and all(math.isfinite(value) for value in maxima)
                          and item["point"] == max(.25, sorted(maxima)[-2]), "Control-derived point differs")
    counts = {}
    for label in ("earlier", "later", "fresh"):
        folder = getattr(args, label)
        panel.require({file: panel.sha(folder / file) for file in FILES} == freeze["intakes"][label],
                      "Held intake differs")
        journal = args.run / f"{label}-rows.json"
        panel.require(panel.sha(journal) == result["rows_sha256"][label], "Held observations differ")
        counts[label] = replay(folder, panel.read(journal), fit["points"], result["results"][label])
    panel.require(counts == result["view_counts"], "Held denominators differ")
    qualified = freeze["qualified_fresh_by_family"]
    advance = {}
    panels = [result["results"][label] for label in ("earlier", "later", "fresh")]
    for fmt in panel.FORMATS:
        best = [section[fmt]["matched_processing"] for section in panels]
        binary = [section[fmt]["binary_residual"] for section in panels]
        advance[fmt] = (min(row["added_feathered"] for row in best[:2]) >= 2 and best[-1]["added_feathered"] >= 1
                        and all(a["feathered"] >= b["feathered"] for a, b in zip(best, binary, strict=True))
                        and max(row["added_controls"] for row in best) <= 4
                        and max(row["added_control_sources"] for row in best) <= 3
                        and sum(qualified.values()) >= 6 and min(qualified.values()) >= 2)
    panel.require(advance == result["advances"], "Advancement decision differs")
    total = sum(counts.values())
    panel.write(args.run / "audit.json", {"result_sha256": panel.sha(files["result"]), "status": "verified",
                "exact_deliveries": counts, "native_overlap_records": total * len(ARMS), "advance": advance})
    print(total, "held input and truth hashes,", total * len(ARMS), "region outcomes verified", flush=True)


if __name__ == "__main__":
    main()
