# SPDX-License-Identifier: Apache-2.0
"""Find two independent exact RGB copy pairs without picking among competing crops."""

from __future__ import annotations

from app.detectors.image import exact_reflected_copy, exact_small_copy, exact_turned_copy
from app.evidence.image_copy_similarity_binding import PNG_COPY_TRANSFORMS, disjoint_copy_regions


def analyse(rgb):
    direct = exact_small_copy.analyse(rgb, allow_multiple=True)
    if direct["status"] not in {"supported_pair", "no_pair"}:
        return direct
    pairs = [{**row, "transform": "same_orientation"} for row in direct["pairs"]]
    seeds = exact_turned_copy._unique_seeds(rgb)
    for detector in (exact_turned_copy, exact_reflected_copy):
        checked = detector.analyse(rgb, allow_multiple=True, seeds=seeds)
        if checked["status"] not in {"supported_pair", "no_pair"}:
            return checked
        for row in checked["pairs"]:
            if len(pairs) == 2 or any(not disjoint_copy_regions(row["pair"], seen["pair"]) for seen in pairs):
                return {"status": "ambiguous_pairs", "pairs": []}
            pairs.append({"pair": row["pair"], "measurement": row["measurement"],
                          "transform": row["transform"] if "transform" in row else PNG_COPY_TRANSFORMS[row["turns"]]})
    if len(pairs) < 2:
        return {"status": "no_second_pair", "pairs": []}
    if not disjoint_copy_regions(*(item["pair"] for item in pairs)):
        return {"status": "ambiguous_pairs", "pairs": []}
    return {"status": "supported_pair", "pairs": sorted(pairs, key=lambda row: (sorted(row["pair"]), row["transform"]))}
