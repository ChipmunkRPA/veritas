# SPDX-License-Identifier: Apache-2.0
"""Recount native JPEG90 reflection measurements with the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

if __package__:
    from .review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require
else:
    from review_image_regional_jpeg90_carrier import EDITED, HASH, measure, require

SCHEMA = "chipmunk.image-regional-jpeg90-reflection/1"
ARMS = ("jpeg75", "direct", "reencoded", "reflected_hard", "reflected_contrast")
FLOOR = 82
DESIGN = {
    "source": "Repeat only files with retained native JPEG90 proposals. Use the frozen 200-delivery main/external bank for calibration and the separate 88-delivery F5 bank for confirmation.",
    "transform": "Reverse columns of decoded JPEG90 RGB without another JPEG encode. Align the repeated raw response by reversing its columns. Keep native JPEG90 reliability and exact component membership.",
    "hard": "Retain original components with at least half of their native pixels selected by the reflected view using the existing global area gate and component cutoff.",
    "contrast": "Use the median repeated response on the component minus the median on an eight-pixel outside ring. Divide by the larger of .02 or the outside interdecile range. Quantize to 12-bit steps, exclude unreliable pixels, other native components and the first two outside pixels. Require at least 32 ring pixels and at least one quarter of component area. Retain the original component on a strict exceedance.",
    "calibration": "Use the third-highest per-source ordinary maximum on the first two banks. Do not select a cutoff from confirmation, and compare the separate hard arm unchanged.",
    "confirmation": "Preserve all useful native JPEG90 edits and both previously localized source photos, at most two extra ordinary flags from one group compared with JPEG75, and at most two extra misplaced edited highlights compared with JPEG75. Do not use confirmation to change the rule.",
    "confidence_weight": 0,
}


def contrast(component):
    values = component["contrast"]
    if values is None:
        return None
    inside, outside_low, outside, outside_high, ring_count = values
    require(all(type(n) is int for n in values) and 0 <= inside <= 4095
            and 0 <= outside_low <= outside <= outside_high <= 4095
            and ring_count >= 32 and 4 * ring_count >= component["area"], "Invalid response quantiles or ring support")
    return Fraction(inside - outside, max(FLOOR, outside_high - outside_low))


def selected(component, arm, point):
    if arm == "reflected_hard":
        return component["hard_overlap"] * 2 >= component["area"]
    value = contrast(component)
    return value is not None and value > point


def cases(receipt, development, confirmation):
    lock = receipt["freeze"]
    require(lock["schema"] == SCHEMA and lock["design"] == DESIGN, "Frozen reflection design differs")
    expected = [*development["rows"], *confirmation["freeze"]["rows"]]
    require(len(expected) == len(receipt["rows"]) == 288, "Expected both complete source banks")
    for index, (row, original) in enumerate(zip(receipt["rows"], expected, strict=True)):
        held = index >= len(development["rows"])
        if held:
            group = original["group"]
            metrics = confirmation["rows"][index - len(development["rows"])]
            base = {"jpeg75": metrics["measures"]["native_q75"], "direct": metrics["measures"]["direct_q90"],
                    "reencoded": metrics["measures"]["agreement"]}
            mask = metrics["mask_sha256"]["direct_q90"]
        else:
            group = list(dict.fromkeys(item["group"] for item in development["rows"])).index(original["group"]) + 1
            base = {"jpeg75": original["q75"], "direct": original["q90"]}
            mask = original["q90_mask_sha256"]
        h, w = original["geometry" if held else "shape"]
        require(row["index"] == index and row["group"] == group and row["input_sha256"] == original["q90_sha256"]
                and row["truth_sha256"] == original["truth_sha256"] and row["native_mask_sha256"] == mask,
                "Source order, mask or truth binding differs")
        target = base["direct"]["reference_positive_pixels"]
        ordinary = original["role"] not in EDITED
        components = row["components"]
        require(isinstance(components, list) and len(components) <= 2 and bool(components) == bool(row["reflection"]),
                "Unexpected reflected candidates")
        require(sum(c["area"] for c in components) == base["direct"]["predicted_positive_pixels"]
                and sum(c["intersection"] for c in components) == base["direct"]["intersection"], "Component totals differ")
        if row["reflection"]:
            require(all(HASH.fullmatch(value) for value in row["reflection"].values()), "Missing reflection commitments")
        for c in components:
            require(all(type(c[key]) is int for key in ("area", "intersection", "hard_overlap")) and c["area"] > 0
                    and 0 <= c["intersection"] <= min(target, c["area"]) and 0 <= c["hard_overlap"] <= c["area"],
                    "Invalid component or overlap")
            x0, y0, x1, y1 = c["bounds"]
            require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and c["area"] <= (x1 - x0) * (y1 - y0)
                    and HASH.fullmatch(c["mask_sha256"]), "Invalid exact component bounds or commitment")
            contrast(c)
        yield row, base, target, h * w, held, ordinary, original["family"], original.get("panel", "held")


def development_maxima(rows, development):
    source = development["rows"]
    require(len(rows) == len(source) == 200, "Incomplete development rows")
    groups = {}
    for row, original in zip(rows, source, strict=True):
        if original["role"] in EDITED:
            continue
        values = [value for component in row["components"] if (value := contrast(component)) is not None]
        if values:
            group = row["group"]
            groups[group] = max([*values, groups.get(group, Fraction(-4095, FLOOR))])
    require(len(groups) >= 4, "Insufficient independent ordinary maxima")
    return sorted(groups.values(), reverse=True)[2], groups


def summarize(cases, cutoff):
    panels = {}
    for row, base, target, pixels, held, ordinary, family, panel in cases:
        for arm in ARMS[3:]:
            retained = [component for component in row["components"] if selected(component, arm, cutoff)]
            base[arm] = {"reference_positive_pixels": target, "predicted_positive_pixels": sum(c["area"] for c in retained),
                         "intersection": sum(c["intersection"] for c in retained), "pixels": pixels}
        panel_totals = panels.setdefault(panel, {})
        for arm, metrics in base.items():
            flag, useful = measure(metrics, pixels, not ordinary)
            group = row["group"]
            stats = panel_totals.setdefault(arm, {"counts": Counter(), "useful_groups": set(), "ordinary_groups": set(),
                                                  "useful_families": Counter(), "lost": 0, "gained": 0})
            stats["counts"].update({"ordinary": 1, "flags": flag} if ordinary else
                                   {"edited": 1, "useful": useful, "wrong": flag and not useful})
            if ordinary and flag:
                stats["ordinary_groups"].add(group)
            if useful:
                stats["useful_groups"].add(group)
                stats["useful_families"][family] += 1
            if not ordinary:
                before = measure(base["direct"], pixels, True)[1]
                stats["lost"] += before and not useful
                stats["gained"] += not before and useful
    held = panels["held"]
    passes = {}
    for name in ARMS[3:]:
        original, direct, candidate = (held[key] for key in ("jpeg75", "direct", name))
        passes[name] = (candidate["lost"] == 0 and len(candidate["useful_groups"] & direct["useful_groups"]) >= 2
                        and candidate["counts"]["flags"] <= original["counts"]["flags"] + 2
                        and len(candidate["ordinary_groups"] - original["ordinary_groups"]) <= 1
                        and candidate["counts"]["wrong"] <= original["counts"]["wrong"] + 2)
    for panel in panels.values():
        for arm in panel.values():
            arm["useful_groups"] = len(arm["useful_groups"])
            arm["ordinary_groups"] = len(arm["ordinary_groups"])
    return {"panels": panels, "confirmation_passes": passes}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "development", "confirmation"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    receipt, development, confirmation = (json.loads(path.read_bytes()) for path in (args.receipt, args.development, args.confirmation))
    require(receipt["freeze"]["prior_sha256"] == {name: digest(getattr(args, name)) for name in ("development", "confirmation")},
            "Baseline ledgers differ")
    rows = list(cases(receipt, development, confirmation))
    cutoff, maxima = development_maxima(receipt["rows"][:200], development)
    require(receipt["calibration"] == {"cutoff": [cutoff.numerator, cutoff.denominator],
                                       "ordinary_group_maxima": {str(k): [v.numerator, v.denominator] for k, v in maxima.items()}},
            "Calibration does not match development")
    result = summarize(rows, cutoff)
    require(result == receipt["summary"], "Reflection outcome differs")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
