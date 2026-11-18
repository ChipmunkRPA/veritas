# SPDX-License-Identifier: Apache-2.0
"""Find independent exact RGB copy pairs without picking among competing crops."""

from __future__ import annotations

from itertools import combinations

from app.detectors.image import exact_reflected_copy, exact_small_copy, exact_turned_copy
from app.evidence.image_copy_similarity_binding import (
    PNG_COPY_TRANSFORMS,
    THREE_EXACT_SIZE,
    disjoint_copy_regions,
    three_exact_profile,
)


def analyse(rgb, *, allow_three=False):
    direct = exact_small_copy.analyse(rgb, allow_multiple=True)
    if direct["status"] not in {"supported_pair", "no_pair"}:
        return direct
    pairs = [{**row, "transform": "same_orientation"} for row in direct["pairs"]]
    capacity = 3 if allow_three and rgb.shape[:2] == THREE_EXACT_SIZE else 2
    seeds = exact_turned_copy._unique_seeds(rgb)
    for detector in (exact_turned_copy, exact_reflected_copy):
        checked = detector.analyse(rgb, allow_multiple=True, seeds=seeds)
        if checked["status"] not in {"supported_pair", "no_pair"}:
            return checked
        for row in checked["pairs"]:
            if len(pairs) == capacity or any(not disjoint_copy_regions(row["pair"], seen["pair"]) for seen in pairs):
                return {"status": "ambiguous_pairs", "pairs": []}
            pairs.append({"pair": row["pair"], "measurement": row["measurement"],
                          "transform": row["transform"] if "transform" in row else PNG_COPY_TRANSFORMS[row["turns"]]})
    if len(pairs) < 2:
        return {"status": "no_second_pair", "pairs": []}
    if len(pairs) == 3 and not three_exact_profile(rgb.shape[:2], (pair["transform"] for pair in pairs)):
        return {"status": "ambiguous_pairs", "pairs": []}
    if not all(disjoint_copy_regions(a["pair"], b["pair"]) for a, b in combinations(pairs, 2)):
        return {"status": "ambiguous_pairs", "pairs": []}
    return {"status": "supported_pair", "pairs": sorted(pairs, key=lambda row: (sorted(row["pair"]), row["transform"]))}
