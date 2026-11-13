# SPDX-License-Identifier: Apache-2.0
"""Verify the CFA PNG transfer's geometry and outcomes using the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SIZES = (32, 64, 128)
PHASES = ("even", "x_odd", "y_odd", "both_odd")
BLENDS = ("sharp", "feather")
ORDINARY = ("original", "crop_odd", "global_gain", "local_fractional", "resize", "eight_bit", "alpha")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def measure(row, point):
    result = Counter(files=1)
    status = row["status"]
    result[status] += 1
    if status != "assessed":
        require(status in {"unsupported_png", "no_repeatable_stencil"} and "box" not in row,
                "An unassessed upload contains a box")
        return result
    box = row["box"]
    if box is not None:
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 1536
                and all(coordinate % 32 == 0 for coordinate in box)
                and 64 <= x1 - x0 <= 192 and 64 <= y1 - y0 <= 192,
                "Native box geometry is outside the frozen scope")
        box_pixels = (x1 - x0) * (y1 - y0)
    else:
        box_pixels = 0
    overlap = row["overlap"]
    truth, intersection = row["truth_pixels"], overlap["intersection"]
    require(0 <= intersection <= min(truth, box_pixels) and overlap["predicted_positive_pixels"] == box_pixels
            and overlap["reference_positive_pixels"] == truth, "Overlap counts are impossible")
    union = truth + box_pixels - intersection
    require(overlap["iou"] == (intersection / union if union else 1.)
            and overlap["recall"] == (intersection / truth if truth else None), "Overlap ratios differ")
    flag = row["score"] > point
    useful = bool(truth and intersection * 10 >= 3 * union and 2 * intersection >= truth)
    result.update(flags=int(flag), useful=int(flag and useful), ungated_useful=int(useful))
    return result


def verify(run, roster, receipt):
    freeze_path = run / "freeze.json"
    result_path = run / "held.json"
    freeze, result, official, sources = map(read, (freeze_path, result_path, receipt, roster))
    require(sha(freeze_path) == result["freeze_sha256"] == official["freeze_sha256"]
            and sha(result_path) == official["held_sha256"] and sha(roster) == freeze["roster_sha256"] == official["roster_sha256"],
            "Source or result commitments differ")
    require(freeze["point"] == official["point"] == .25 and all(r["license"] == "CC0-1.0" for r in sources),
            "Frozen point or license differs")
    cameras = {r["id"] for r in sources}
    edited = {f"{size}_{phase}_{blend}" for size in SIZES for phase in PHASES for blend in BLENDS}
    cases = {(c, r) for c in cameras for r in edited.union(ORDINARY)}
    summary = defaultdict(Counter)
    for row in result["rows"]:
        role = row["role"]
        key = row["camera_id"], role
        require(key in cases and (row["truth_pixels"] > 0) == (role in edited), "Unexpected role or source")
        cases.remove(key)
        reading = measure(row, freeze["point"])
        for name, value in reading.items():
            summary[role][name] += value
    require(not cases and len(cameras) == official["held_sources"] == 10 and len(result["rows"]) == official["exact_held_replays"] == 310,
            "Missing sealed cases")
    ordinary = sum(c["flags"] for name, c in summary.items() if name in ORDINARY or "_even_" in name)
    strong = all(summary[f"128_{phase}_sharp"]["useful"] >= 8 for phase in PHASES[1:])
    incompatible = all(summary[r]["assessed"] == 0 for r in ("resize", "eight_bit", "alpha"))
    require(official["held"] == {"roles": dict(summary), "ordinary_highlights": ordinary,
                                  "admission": strong and ordinary == 0 and incompatible}, "Receipt decision differs")
    print("Verified all 310 camera/role rows, integer localization overlaps and the frozen decision.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    verify(args.run, args.roster, args.receipt)
