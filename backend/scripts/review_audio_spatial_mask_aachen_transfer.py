"""Verify anonymous Aachen scalar evidence with Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "chipmunk.aachen-spatial-voice/1"
WINDOWS = ((16000, 48000), (48000, 80000))
ROOMS = ("stairway", "aula_carolina")
ARMS = ("mono", "earlier_either", "directional")
POINTS = {"mono": 5.356589748054317, "earlier_either": 6.760727907655432, "directional": 7.068065580780201}
GENERATORS = {"cosyvoice", "elevenlab", "gptsovits", "ljjets", "xttsv2", "yourtts"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def decision(row, arm):
    values = row["reads"][arm]
    require(isinstance(values, list), "Missing score views")
    if arm == "mono":
        require(len(values) == 1 and values[0] is not None, "Missing native mono arm")
    elif arm == "earlier_either":
        require(len(values) in (0, 2), "Malformed prior-separator views")
    else:
        require(len(values) == len(row["modes"]) and len(values) in (0, 2, 3), "Malformed directional views")
    accepted = []
    for view in values:
        if view is None:
            continue
        require(len(view) == len(WINDOWS), "Unexpected window count")
        for window, reading in zip(WINDOWS, view, strict=True):
            require(set(reading) == {"score"}, "Unexpected public voice payload")
            value = reading["score"]
            if value is not None:
                require(type(value) in (int, float) and math.isfinite(value), "Invalid score")
                accepted.append((value, -window[0], window))
    if not accepted:
        return "unassessed", None
    score, _, window = max(accepted)
    if score <= POINTS[arm]:
        return "no_flag", window
    if not row["truth"]:
        return "ordinary", window
    for start, end in row["truth"]:
        intersection = max(0, min(end, window[1]) - max(start, window[0]))
        union = max(end, window[1]) - min(start, window[0])
        if intersection * 10 >= union * 3 and intersection * 2 >= end - start:
            return "located", window
    return "wrong", window


def check(receipt):
    freeze, rows = receipt["freeze"], receipt["responses"]
    require(receipt["schema"] == freeze["schema"] == SCHEMA, "Unknown receipt schema")
    require(freeze["windows"] == [list(window) for window in WINDOWS] and freeze["points"] == POINTS, "Operating points changed")
    require(hashlib.sha256(canonical(freeze)).hexdigest() == receipt["freeze_sha256"], "Freeze commitment differs")
    require(receipt["exact_alternate_fft_replay"] == len(rows), "Incomplete native replay")
    require([row["case"] for row in rows] == list(range(len(rows))), "Missing or reordered response")
    require({row["room"] for row in rows} == set(ROOMS), "Missing room cohort")
    for row in rows:
        require(set(row) == {"case", "group", "room", "kind", "family", "truth", "modes", "reads"}, "Unexpected public row payload")
        require(bool(row["truth"]) == (row["kind"] == "publisher_edit") == bool(row["family"]), "Invalid truth role")
        require(row["kind"] in ("publisher_edit", "genuine", "matched_genuine", "ordinary_same_reader", "background_only"),
                "Invalid case role")
        require(all(type(a) is int and type(b) is int and 0 <= a < b <= 96000 for a, b in row["truth"])
                and all(previous[1] <= following[0] for previous, following in zip(row["truth"], row["truth"][1:])),
                "Invalid native truth range")
        require(all(type(mode) is int and -22 <= mode <= 22 for mode in row["modes"]), "Invalid lag modes")
        require(set(row["reads"]) == set(ARMS), "Incomplete comparison arms")
    distinct = {row["group"] for row in rows if row["kind"] != "background_only"}
    require(len(distinct) == freeze["foreground_and_genuine_speakers"], "Incomplete native foreground groups")
    require({row["family"] for row in rows if row["truth"]} == GENERATORS, "Generator coverage differs")
    require(len({row["group"] for row in rows}) == len(rows) == 25, "Delivered source group count differs")

    layouts, additions = {}, {}
    for cohort in (*ROOMS, "pooled"):
        population = [row for row in rows if cohort in (row["room"], "pooled")]
        table, locations, ordinary, background = {}, {}, {}, {}
        for arm in ARMS:
            choices = [(row, *decision(row, arm)) for row in population]
            locations[arm] = {row["group"] for row, status, _ in choices if status == "located"}
            ordinary[arm] = {row["group"] for row, status, _ in choices if status == "ordinary"}
            background[arm] = {row["group"] for row, status, _ in choices if status == "ordinary" and row["kind"] == "background_only"}
            unassessed = sum(status == "unassessed" for _, status, _ in choices)
            table[arm] = {"assessed": len(choices) - unassessed, "unassessed": unassessed, "located": len(locations[arm]),
                          "wrong_edit_windows": sum(status == "wrong" for _, status, _ in choices),
                          "flagged_ordinary_groups": len(ordinary[arm]), "flagged_background_only_groups": len(background[arm])}
        gained = locations["directional"] - locations["mono"]
        table["additive"] = {"located": len(locations["mono"] | locations["directional"]), "added_locations": len(gained),
                             "added_ordinary_groups": len(ordinary["directional"] - ordinary["mono"]),
                             "added_background_groups": len(background["directional"] - background["mono"]),
                             "flagged_ordinary_groups": len(ordinary["mono"] | ordinary["directional"])}
        table["totals"] = {"cases": len(population), "edits": sum(bool(row["truth"]) for row in population),
                           "ordinary_groups": len({row["group"] for row in population if not row["truth"]})}
        if cohort == "pooled":
            additions = {row["group"]: {"room": row["room"], "family": row["family"], "truth": row["truth"],
                                        "window": list(decision(row, "directional")[1])}
                         for row in population if row["group"] in gained and row["truth"]}
        layouts[cohort] = table
    gates = {"two_edits": len(additions) >= 2, "two_families": len({row["family"] for row in additions.values()}) >= 2,
             "ordinary_per_room": all(layouts[room]["additive"]["added_ordinary_groups"] <= 1 for room in ROOMS),
             "background_per_room": all(layouts[room]["additive"]["added_background_groups"] == 0 for room in ROOMS)}
    summary = {"layouts": layouts, "added_locations": additions, "gates": gates, "passed": all(gates.values())}
    require(summary == receipt["summary"], "Independent scalar readout disagrees with the receipt")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.receipt.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
