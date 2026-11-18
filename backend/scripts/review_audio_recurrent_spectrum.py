# SPDX-License-Identifier: MIT
"""Recount the recurrent-spectrum Opus study without models or recordings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import median

RATE = 16_000
STEP = 8_000
WINDOW = 12_000
SCHEMA = "chipmunk.audio-recurrent-spectrum-opus/1"
KINDS = {"ordinary", "edited", "background"}


def require(condition, text):
    if not condition:
        raise ValueError(text)


def match(windows, labels):
    ranked = []
    for start, stop in labels:
        candidates = []
        for a, b in windows:
            overlap = min(stop, b) - max(start, a)
            span = (stop - start) + (b - a) - overlap
            if 2 * overlap >= stop - start and 10 * overlap >= 3 * span:
                candidates.append((-overlap / span, (a, b)))
        ranked.append(sorted(candidates))
    assigned = {}

    def place(label, visited):
        for _, window in ranked[label]:
            if window not in visited:
                visited.add(window)
                owner = assigned.get(window)
                if owner is None or place(owner, visited):
                    assigned[window] = label
                    return True
        return False

    for label in range(len(labels)):
        place(label, set())
    return set(assigned.values())


def selected(row, lane, point):
    bit = {"wider": 2, "strict": 4}[point]
    values = [(start, flags[lane]) for start, *flags in row["windows"]]
    return {(offset, offset + WINDOW) for (a, first), (b, second) in pairwise(values)
            if b == a + STEP and first & second & bit for offset in (a, b)}


def validate(rows, *, complete_grid=False):
    require(len({r["id"] for r in rows}) == len(rows), "Repeated delivery ID")
    require(len({r["source_sha256"] for r in rows}) == len(rows), "Repeated original PCM")
    for row in rows:
        n, pts, pre_skip, granule, packets, trim = row["clock"]
        require(row["kind"] in KINDS and bool(row["truth"]) == (row["kind"] == "edited"), "Invalid source role")
        require(n > 0 and pts == 3 * n == granule - pre_skip and pre_skip == 312
                and granule + trim == 960 * packets and 0 <= trim < 960, "Invalid Opus presentation")
        for field in ("source_sha256", "decoded_pcm_sha256", "file_sha256"):
            value = row[field]
            require(len(value) == 64 and len(bytes.fromhex(value)) == 32, "Invalid source digest")
        for intervals in (row["truth"], row.get("baseline", ()), row.get("short_raw", ())):
            require(all(0 <= a < b <= n for a, b in intervals), "Recorded interval is outside its source")
        starts = [start for start, *_ in row["windows"]]
        schedule = [a for a in range(0, n - WINDOW + 1, STEP) if a % (2 * RATE) in (0, STEP)]
        require(starts == sorted(set(starts)) and set(starts) <= set(schedule), "Incorrect sample schedule")
        require(not complete_grid or starts == schedule, "Incomplete confirmation grid")
        for _, raw, candidate in row["windows"]:
            require(raw in (0, 1, 3, 7) and candidate in (0, 1, 3, 7) and (not candidate or raw & 1),
                    "A reading exceeds a point without its admission or wider point")


def pilot_summary(rows):
    out = defaultdict(Counter)
    for row in rows:
        base = set(map(tuple, row["baseline"]))
        older = base | set(map(tuple, row["short_raw"]))
        arms = {"historical": base, "historical_plus_short_raw": older}
        for point in ("strict", "wider"):
            arms[f"historical_plus_masked_{point}"] = base | selected(row, 1, point)
            arms[f"historical_plus_short_raw_and_masked_{point}"] = older | selected(row, 1, point)
            arms[f"historical_plus_equal_grid_raw_{point}"] = base | selected(row, 0, point)
        for arm, intervals in arms.items():
            labels = match(intervals, row["truth"])
            out[f"{arm}/{row['kind']}"].update(cases=1, labels=len(row["truth"]), located=len(labels),
                                               flagged=bool(intervals), model_locations=bool(intervals - base))
    return {key: dict(value) for key, value in sorted(out.items())}


def confirmation_summary(rows):
    groups, totals, new_clean = defaultdict(dict), defaultdict(Counter), []
    for row in rows:
        require(row["kind"] not in groups[row["group"]], "Repeated type in a matched group")
        groups[row["group"]][row["kind"]] = row
    for group, triple in sorted(groups.items()):
        require(set(triple) == KINDS and len({value["room"] for value in triple.values()}) == 1
                and len({value["clock"][0] for value in triple.values()}) == 1, "Incomplete matched source triple")
        edited = triple["edited"]
        for point in ("strict", "wider"):
            proposals = {kind: [selected(row, lane, point) for lane in (0, 1)] for kind, row in triple.items()}
            for pair in proposals.values():
                pair.append(pair[0] | pair[1])
            baseline = match(proposals["edited"][0], edited["truth"])
            for index, arm in enumerate(("raw", "masked", "union")):
                results = match(proposals["edited"][index], edited["truth"])
                added = results - baseline
                controls = proposals["ordinary"][index] | proposals["background"][index]
                clean = [i for i in sorted(added) if not match(controls, [edited["truth"][i]])]
                totals[f"{point}/{arm}"].update(groups=1, labeled=len(edited["truth"]), located=len(results),
                    gained=len(added), lost=len(baseline - results), newly_located_without_control=len(clean),
                    ordinary_flags=bool(proposals["ordinary"][index]), background_flags=bool(proposals["background"][index]),
                    extra_ordinary_flags=bool(proposals["ordinary"][index] and not proposals["ordinary"][0]))
                if clean:
                    new_clean.append({"group": group, "family": edited["family"], "point": point, "arm": arm,
                        "labels": [edited["truth"][i] for i in clean],
                        "proposals": [list(interval) for interval in sorted(proposals["edited"][index])]})
    return {key: dict(value) for key, value in sorted(totals.items())}, new_clean


def component_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == "background":
            continue
        first, *labels = row["component_db"]
        require(len(labels) == len(row["truth"]), "Missing component comparison")
        groups[row["kind"]].append(first)
        groups["edited_intervals"].extend(labels)
    return {kind: {"count": len(values), "foreground_db": round(median(r[0] for r in values), 6),
                   "background_db": round(median(r[1] for r in values), 6),
                   "ratio_change_db": round(median(r[0] - r[1] for r in values), 6)}
            for kind, values in sorted(groups.items())}


def review(value):
    require(value["schema"] == SCHEMA, "Wrong capsule schema")
    calibration, pilot, confirmation = (value[key] for key in ("calibration", "pilot", "confirmation"))
    validate(pilot)
    validate(confirmation, complete_grid=True)
    for key in ("group", "room", "source_sha256", "file_sha256"):
        require(not ({row[key] for row in pilot} & {row[key] for row in confirmation}), f"Source roles share {key}")
    require(not (set(calibration["groups"]) & {row["group"] for row in pilot + confirmation})
            and calibration["room"] not in {row["room"] for row in pilot + confirmation}, "Calibration and held roles overlap")
    calls = {kind: [sum(bool(w[lane] & 1) for row in rows for w in row["windows"]) for lane in (1, 2)]
             for kind, rows in (("pilot", pilot), ("confirmation", confirmation))}
    for lane in ("raw", "candidate"):
        values = sorted(calibration["maxima"][lane]["ordinary"])
        bg = max(calibration["maxima"][lane]["background"])
        require(len(values) == 7 and calibration["points"][lane] == {
            "strict": max(bg, values[-1]), "wider": max(bg, values[-2]), "ordinary_groups": 7}, "Frozen calibration mismatch")
    confirmation_counts, controlled_locations = confirmation_summary(confirmation)
    output = {"pilot": pilot_summary(pilot), "confirmation": confirmation_counts, "calls": calls,
              "component_audit": component_summary(confirmation), "controlled_locations": controlled_locations,
              "confirmation_generator_families": sorted({row["family"] for row in confirmation if row["truth"]})}
    require(len(confirmation) == 33 and len(pilot) == 18 and len(output["confirmation_generator_families"]) == 6,
            "Unexpected panel scope")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    original = args.receipt.read_bytes()
    value = json.loads(gzip.decompress(original))
    result = review(value)
    require(result == value["expected"], "Recount differs from the published comparison")
    print(json.dumps({"capsule_sha256": hashlib.sha256(original).hexdigest(), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
