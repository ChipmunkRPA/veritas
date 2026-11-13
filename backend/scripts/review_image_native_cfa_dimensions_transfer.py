# SPDX-License-Identifier: Apache-2.0
"""Audit all frozen camera geometry and control decisions without any media or image libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = ("original", "even_copy", "global_crop", "local_fractional", "resized", "phase_shift", "phase_shift_feather")
POSITIVE = ("phase_shift", "phase_shift_feather")
GEOMETRIES = {"current_landscape": [1536, 2048], "large_landscape": [1728, 2304], "portrait": [2048, 1536],
              "square": [1792, 1792], "wide": [1344, 2368]}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def measure(row, shape, point):
    box = row["box"]
    if box:
        height, width = shape
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and all(i % 32 == 0 for i in box),
                "Reported box is outside the tested source grid")
        selected = (x1 - x0) * (y1 - y0)
    else:
        selected = 0
    pixels = row["pixels"]
    truth, intersection = pixels["reference_positive_pixels"], pixels["intersection"]
    require(pixels["predicted_positive_pixels"] == selected and 0 <= intersection <= min(truth, selected),
            "Impossible changed pixel overlap")
    union = selected + truth - intersection
    require(pixels["iou"] == (intersection / union if union else 1.)
            and pixels["recall"] == (intersection / truth if truth else None), "Incorrect stored overlap ratios")
    require((truth > 0) == (row["role"] in (*POSITIVE, "even_copy", "local_fractional")), "Wrong changed-pixel support")
    assessed = row["status"] == "assessed"
    flagged = row["score"] > point
    require(assessed or (not box and not flagged and row["status"] == "no_repeatable_stencil"), "Unsupported file has a result")
    useful = flagged and bool(truth and intersection * 10 >= union * 3 and 2 * intersection >= truth)
    return Counter(files=1, assessed=int(assessed), flags=int(flagged), useful=int(useful))


def audit(held_path, roster, receipt, *, lock_path=None):
    held, cameras, official = map(read, (held_path, roster, receipt))
    require(held["freeze_sha256"] == official["freeze_sha256"] and sha(held_path) == official["held_sha256"]
            and sha(roster) == official["roster_sha256"], "Frozen receipts differ")
    scopes = GEOMETRIES
    point = official["point"]
    if lock_path is not None:
        lock = read(lock_path)
        require(sha(lock_path) == official["freeze_sha256"] and lock["roster_sha256"] == official["roster_sha256"]
                and lock["point"] == point and lock["geometries"] == scopes, "Native run freeze differs")
    require(point == .25 and all(r["license"] == "CC0-1.0" for r in cameras), "Method or terms differ")
    require(len(scopes) == 5 and len(cameras) == 10 and len(held["rows"]) == official["exact_replay_rows"] == 350,
            "Unexpected cohort size")
    expected = {(r["id"], shape, role) for r in cameras for shape in scopes for role in ROLES}
    observed = {shape: {role: Counter(files=0, assessed=0, flags=0, useful=0) for role in ROLES} for shape in scopes}
    for row in held["rows"]:
        camera, shape, role = row["camera_id"], row["geometry"], row["role"]
        require((camera, shape, role) in expected, "Unexpected camera, shape or role")
        expected.remove((camera, shape, role))
        observed[shape][role].update(measure(row, scopes[shape], point))
    require(not expected, "Missing cases")
    outcomes = {shape: {"roles": counts, "passed": all(counts[role]["useful"] >= 7 for role in POSITIVE)
                        and sum(counts[role]["flags"] for role in ROLES if role not in POSITIVE) == 0
                        and counts["resized"]["assessed"] == 0}
                for shape, counts in observed.items()}
    require(outcomes == official["held"], "Gate outcomes differ")
    print("Verified 350 geometry, overlap and operating-point decisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="Published scalar ledger or native run directory")
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.measurements.is_dir():
        audit(args.measurements / "held.json", args.roster, args.receipt, lock_path=args.measurements / "freeze.json")
    else:
        audit(args.measurements, args.roster, args.receipt)
