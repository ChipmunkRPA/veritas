# SPDX-License-Identifier: Apache-2.0
"""Recount a paired video soundtrack transfer using standard Python."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-soundtrack-cadence/2"
RATES = (24, 25)
CARRIERS = ("PCM", "AAC")
ROLES = ("earlier", "later", "untouched", "gain", "filter", "constant", "picture_only",
         "synchronized_picture", "synchronized_mouth", "alternate_speech")
EDITS = ROLES[:2]
DESIGN = {
    "probe": "Four seconds of pinned public recording delivered as 1280x720 H.264/4:2:0 at 24 or 25 fps.",
    "picture": "Frozen 80x60 and 160x90 grids and the original picture points.",
    "timing": "Two 200 ms shifts inside a 960 ms interval, with separate files for each edited delivery "
              "and the untouched reference. Use the existing waveform search.",
    "pass": "For each rate both carriers must locate at least half of planned edits and flag zero controls. "
            "Localization needs IoU of at least 0.60 and the signed movement within 16 samples.",
}
POINTS = {"whole_q65": 2.5, "whole_q90": 4.0, "local_five_frame_median": 6.27}
SHA = re.compile(r"[a-f0-9]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def edit_interval(group):
    start = 23040 + (group % 5) * 640
    return [start, start + 15360]


def _pair(case, span):
    match = case["sound"]
    if match is None:
        return False
    a, b = match["submitted_samples"]
    c, d = match["earlier_samples"]
    shift = match["shift_samples"]
    require(all(type(n) is int for n in (a, b, c, d, shift)) and a - c == b - d == shift
            and 0 <= min(a, c) < max(b, d) <= 65024 and b - a >= 4480
            and 1920 <= abs(shift) <= 5760 and .86 <= match["minimum_correlation"] <= 1
            and match["supporting_windows"] == len(match["window_start_samples"])
            and all(len(value) == 2 for value in match["same_time_support"].values()), "Invalid measured audio match")
    expected = -3200 if case["role"] == "earlier" else 3200
    left, right = span
    return abs(shift - expected) <= 16 and 5 * (min(b, right) - max(a, left)) >= 3 * (max(b, right) - min(a, left))


def review(doc):
    freeze, rows = doc["freeze"], doc["rows"]
    require(freeze["schema"] == SCHEMA and freeze["design"] == DESIGN and freeze["points"] == POINTS
            and freeze["rates"] == list(RATES) and len(freeze["sources"]) == 14
            and len({s["sha256"] for s in freeze["sources"]}) == 14, "Invalid frozen comparison")
    plans = {source["group"]: source for source in freeze["sources"]}
    require(len(rows) == len(plans) * len(RATES) and {(r["group"], r["rate"]) for r in rows}
            == {(group, rate) for group in plans for rate in RATES}, "A rate or original is missing")
    totals = {(rate, carrier): Counter() for rate in RATES for carrier in CARRIERS}
    for row in rows:
        group, rate = row["group"], row["rate"]
        require(SHA.fullmatch(row["source_sha256"]) and row["source_sha256"] == plans[group]["sha256"]
                and row["pictures"] == 4 * rate and row["edit_samples"] == edit_interval(group),
                "Source, cadence or intervention changed")
        require(len(row["cases"]) == len(ROLES) * len(CARRIERS) and {(c["role"], c["carrier"]) for c in row["cases"]}
                == {(role, carrier) for role in ROLES for carrier in CARRIERS}, "A carrier or control is missing")
        for carrier in CARRIERS:
            by_role = {case["role"]: case for case in row["cases"] if case["carrier"] == carrier}
            same = by_role["untouched"]
            require(same["earlier_pcm_sha256"] == same["submitted_pcm_sha256"]
                    and all(by_role[role]["submitted_pcm_sha256"] != same["earlier_pcm_sha256"] for role in EDITS)
                    and all(case.get("earlier_pcm_sha256", same["earlier_pcm_sha256"]) == same["earlier_pcm_sha256"]
                            for case in by_role.values()), "Edited or untouched audio differs from its fixed role")
        for case in row["cases"]:
            role = case["role"]
            count = totals[rate, case["carrier"]]
            edited = role in EDITS
            count["planned_edits" if edited else "planned_controls"] += 1
            if "unavailable" in case:
                require(role == "synchronized_mouth" and row["mouth_box"] is None
                        and case["unavailable"] == "mouth_geometry", "Unexpected unmeasured case")
                count["unavailable_controls"] += 1
                continue
            require(all(SHA.fullmatch(case[key]) for key in ("delivery_sha256", "earlier_pcm_sha256", "submitted_pcm_sha256"))
                    and set(case["picture"]) == {"coarse", "fine"}, "A measured delivery lost its source binding")
            for grid in case["picture"].values():
                require(set(grid) == set(POINTS) and all(math.isfinite(value) and 0 <= value <= 255
                        for value in grid.values()), "Invalid measured picture comparison")
            qualifies = all(grid[name] <= point for grid in case["picture"].values() for name, point in POINTS.items())
            hit = _pair(case, row["edit_samples"])
            count["picture_qualified_edits" if edited else "picture_qualified_controls"] += qualifies
            if edited:
                count["useful"] += qualifies and hit
                count["misplaced"] += qualifies and case["sound"] is not None and not hit
            else:
                count["control_flags"] += qualifies and case["sound"] is not None
    out = {f"{rate}/{carrier}": dict(stats) for (rate, carrier), stats in totals.items()}
    out["admit"] = [rate for rate in RATES if all(
        2 * totals[rate, carrier]["useful"] >= totals[rate, carrier]["planned_edits"]
        and not totals[rate, carrier]["control_flags"] for carrier in CARRIERS)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    doc = json.loads(parser.parse_args().receipt.read_text())
    result = review(doc)
    require(result == doc["outcome"], "The saved outcome does not replay")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
