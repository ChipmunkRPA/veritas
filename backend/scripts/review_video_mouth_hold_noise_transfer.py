# SPDX-License-Identifier: Apache-2.0
"""Recompute paired outcomes and development decisions without media or weights."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-hold-noise-transfer/1"
SCENES = ("rain", "train", "traffic", "competing")
VIEWS = ("raw", "band_balance", "flank_whitening")
SHA = re.compile(r"[0-9a-f]{64}")
RAW = .78


def _require(ok, text):
    if not ok:
        raise ValueError(text)


def _pairs(receipt):
    plan = receipt["plan"]
    _require(plan["schema"] == SCHEMA and len(plan["groups"]) == 26
             and len({row["source_id"] for row in plan["groups"]}) == 26
             and len(plan["windows"]) == 4 and all(b - a == 6400 for a, b in plan["windows"])
             and sum(group["split"] == "development" for group in plan["groups"]) == 8,
             "Incomplete source or four-window plan")
    data = {}
    for row in receipt["rows"]:
        key = row["group"], row["side"], row["scene"]
        _require(key not in data and SHA.fullmatch(row["pcm_sha256"]) and len(row["windows"]) == 4
                 and all(set(values) == set(VIEWS)
                         and all(type(number) in (float, int) and math.isfinite(number) and 0 <= number <= 1.000001
                                 for number in values.values()) for values in row["windows"]),
                 "Invalid or duplicated acoustic measurement")
        data[key] = row
    for i, group in enumerate(plan["groups"]):
        _require(group["group"] == i and group["split"] == ("development" if i < 8 else "held"),
                 "The group order differs from the locked split")
        for side in ("original", "variant"):
            original = data.pop((i, side, "clean"))
            _require(original["background_id"] is None, "Clean excerpts must have no input noise")
            for scene in SCENES:
                noisy = data.pop((i, side, scene))
                background = data.pop((i, side, "background_" + scene))
                _require(noisy["background_id"] == background["background_id"]
                         == ((0 if i < 8 else 4) + i % 4 if scene == "competing" else (0 if i < 8 else 2) + i % 2),
                         "A matched noise excerpt is outside its split")
                for position, (clean, mixed, alone) in enumerate(zip(original["windows"], noisy["windows"],
                                                                    background["windows"], strict=True)):
                    yield group["split"], i, side, scene, background["background_id"], position, clean, mixed, alone
    _require(not data, "Extra unbound acoustic results")


def _counts(pairs, view, threshold):
    stats = Counter({"clean_supported": 0, "noise_recoveries": 0, "raw_noisy_supported": 0,
                     "added_noise_controls": 0, "raw_noise_controls": 0, "mapped_background_alone": 0,
                     "confounded_additions": 0})
    scenes = {scene: Counter({"clean_supported": 0, "raw_noisy_supported": 0, "noise_recoveries": 0,
                               "added_noise_controls": 0, "confounded_additions": 0}) for scene in SCENES}
    added_backgrounds, raw_backgrounds, added_groups = set(), set(), set()
    for _, group, _, scene, background, position, clean, mixed, alone in pairs:
        baseline, baseline_background = mixed["raw"] >= RAW, alone["raw"] >= RAW
        extra = threshold is not None and mixed[view] >= threshold and not baseline
        extra_background = threshold is not None and alone[view] >= threshold and not baseline_background
        if baseline_background:
            raw_backgrounds.add((scene, background, position))
            scenes[scene]["raw_noise_controls"] += 1
        if extra_background:
            added_backgrounds.add((scene, background, position))
            scenes[scene]["added_noise_controls"] += 1
        stats["mapped_background_alone"] += 1
        if clean["raw"] < RAW:
            scenes[scene]["added_mixture_without_clean_periodicity"] += extra
        elif baseline:
            scenes[scene]["clean_supported"] += 1
            scenes[scene]["raw_noisy_supported"] += 1
        else:
            scenes[scene]["clean_supported"] += 1
            if extra:
                if baseline_background or extra_background:
                    scenes[scene]["confounded_additions"] += 1
                else:
                    scenes[scene]["noise_recoveries"] += 1
                    added_groups.add(group)
    for counts in scenes.values():
        stats.update(counts)
    return dict(stats) | {"added_background_windows": len(added_backgrounds),
                          "raw_background_windows": len(raw_backgrounds),
                          "recovered_groups": len(added_groups), "scenes": {name: dict(counts) for name, counts in scenes.items()}}


def summarize(receipt):
    pairs = tuple(_pairs(receipt))
    folds = {key: [row for row in pairs if row[0] == key] for key in ("development", "held")}
    result = {"raw": {key: _counts(rows, "raw", None) for key, rows in folds.items()}, "views": {}}
    for view in VIEWS[1:]:
        grid = [(point, _counts(folds["development"], view, point)) for point in receipt["plan"]["threshold_grid"]]
        eligible = [(point, stats) for point, stats in grid
                    if stats["noise_recoveries"] > 0 and stats["added_background_windows"] <= 1
                    and stats.get("added_mixture_without_clean_periodicity", 0) <= 1 and stats["confounded_additions"] == 0]
        point = max(eligible, key=lambda entry: (entry[1]["noise_recoveries"], entry[0]))[0] if eligible else None
        held = _counts(folds["held"], view, point)
        advances = (held["noise_recoveries"] >= 6 and held["recovered_groups"] >= 4 and held["added_background_windows"] <= 1
                    and held["confounded_additions"] == 0 and held.get("added_mixture_without_clean_periodicity", 0) <= 2
                    and sum(bool(row["noise_recoveries"]) for row in held["scenes"].values()) >= 2)
        result["views"][view] = {"selected": point, "development_grid": [[p, stats] for p, stats in grid],
                                 "held": held, "advances": advances}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    receipt = json.loads(parser.parse_args().receipt.read_bytes())
    result = summarize(receipt)
    _require(result == receipt["summary"], "The stored review differs from measured comparisons")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
