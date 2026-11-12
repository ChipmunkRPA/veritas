"""Verify resized crop receipts and both native coordinates without assuming alignment."""
from __future__ import annotations

import argparse
import io
from collections import Counter
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_crop_resize
from scripts import image_reference_crop_resize_transfer as trial

panel = trial.panel
FIELDS = ("x", "y", "w", "h")


def region_bounds(region, crop, size):
    rx = Fraction(crop["w"], size[0])
    ry = Fraction(crop["h"], size[1])
    x0 = floor(region["x"] * rx)
    y0 = floor(region["y"] * ry)
    x1 = ceil((region["x"] + region["w"]) * rx)
    y1 = ceil((region["y"] + region["h"]) * ry)
    return {"x": crop["x"] + x0, "y": crop["y"] + y0, "w": x1 - x0, "h": y1 - y0}


def review(intake, label, journal):
    keys = {(row["source"], row["generator"], row["role"], row["format"]): row for row in journal}
    panel.require(len(keys) == len(journal), "Duplicate journal rows")
    sources = trial.sources(intake, label)
    counts = Counter()
    for i, (slot, generator, images, target) in enumerate(sources):
        donor = sources[(i + 1) % len(sources)][2]["original"]
        for role, pixels, truth, expected in trial.cases(slot, images, target, donor):
            for fmt in panel.FORMATS:
                row = keys.pop((slot, generator, role, fmt))
                body, decoded = panel.paired.encoded(pixels, fmt)
                panel.require(panel.paired.shared.digest(body) == row["input_sha256"]
                              and panel.paired.shared.digest(truth.tobytes()) == row["truth_sha256"]
                              and expected == row["truth"], "Source delivery or authored geometry differs")
                with Image.open(io.BytesIO(body)) as carrier:
                    result, mask = reference_crop_resize.compare(images["original"], np.asarray(decoded), carrier)
                observed = result.get("reference_crop_bounds")
                panel.require(result["status"] == row["status"] and result["regions"] == row["regions"]
                              and observed == row["geometry_support"], "Candidate receipt differs")
                counts["verified_views"] += 1
                if observed:
                    panel.require({key: observed[key] for key in FIELDS} == row["crop"], "Reported crop differs")
                    counts["wrong_or_incompatible_crop" if row["crop"] != expected else "exact_crop"] += 1
                    for region in row["regions"]:
                        panel.require(region["reference_region"] == region_bounds(region, row["crop"], decoded.size),
                                      "Earlier crop disagrees with independent rational coordinates")
                        counts["mapped_regions"] += 1
                if mask is not None:
                    panel.require(panel.paired.shared.digest(mask.tobytes()) == row["mask_sha256"], "Native selection mask differs")
                    counts["verified_masks"] += 1
                else:
                    panel.require(row["mask_sha256"] is None, "Unassessed view has a mask")
                    mask = np.zeros_like(truth)
                pixels_selected = int(np.count_nonzero(mask))
                target_count = int(np.count_nonzero(truth))
                overlap = int(np.count_nonzero(mask & truth))
                metrics = row["candidate"]
                panel.require((pixels_selected, target_count, overlap) == tuple(metrics[key] for key in ("reported", "target", "overlap")),
                              "Independent overlap counts differ")
    panel.require(not keys, "Journal has extra cases")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier", "later", "run", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen = panel.read(args.run / "freeze.json")
    aggregate = panel.read(args.run / "confirm-summary.json")
    panel.require(panel.sha(args.run / "freeze.json") == aggregate["freeze_sha256"], "Freeze and outcomes differ")
    for path, digest in frozen["code"].items():
        panel.require(panel.sha(panel.ROOT / path) == digest, "Frozen method code differs")
    cv2.setNumThreads(1)
    totals = Counter()
    for label, intake in (("earlier", args.earlier), ("later", args.later)):
        for name, digest in frozen["intakes"][label].items():
            panel.require(panel.sha(intake / name) == digest, "Frozen source intake differs")
        journal = args.run / f"{label}.json"
        panel.require(panel.sha(journal) == aggregate["journal_sha256"][label], "Outcome journal differs")
        counts = review(intake, label, panel.read(journal))
        totals.update(counts)
        print(label, dict(counts), flush=True)
    panel.require(totals["verified_views"] == 480, "Incomplete holdout review")
    panel.write(args.output, {"status": "verified", "aggregate_sha256": panel.sha(args.run / "confirm-summary.json"), **totals})


if __name__ == "__main__":
    main()
