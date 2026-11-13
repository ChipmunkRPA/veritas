# SPDX-License-Identifier: Apache-2.0
"""Recompute the held rotated-stencil decisions from scalar commitments."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROTATIONS = ("rotate_bilinear", "rotate_bicubic")
POSITIVES = ("generated_sharp", "generated_feathered", "generated_linear_promote")
UNSUPPORTED = ("resized", "eight_bit")
ROLES = ("original", "blur", "tone", "copy_even", "diagonal_shift", "quantized_shift", *ROTATIONS, *POSITIVES, *UNSUPPORTED)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(row):
    proposal, pattern = row["proposal"], row["rotated_pattern"]
    if proposal is None:
        require(not row["preselected"] and pattern is None, "Unassessed case has evidence")
        return False, False, False
    box = proposal["box"]
    overlap = proposal["pixels"]
    if box:
        x0, y0, x1, y1 = box
        dx, dy = proposal["grid"]
        require(0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 1536 and dx in (0, 16) and dy in (0, 16)
                and all((v - dx) % 32 == 0 for v in (x0, x1)) and all((v - dy) % 32 == 0 for v in (y0, y1)),
                "Source or grid coordinates differ")
        area = (x1 - x0) * (y1 - y0)
    else:
        area = 0
    hit, truth = overlap["intersection"], row["truth_pixels"]
    union = area + truth - hit
    require(overlap["predicted_positive_pixels"] == area and overlap["reference_positive_pixels"] == truth
            and 0 <= hit <= min(area, truth) and overlap["iou"] == (hit / union if union else 1.)
            and overlap["recall"] == (hit / truth if truth else None), "Overlap differs from recorded native pixels")
    baseline = bool(box and proposal["score"] > 2.4346873354224834
                    and not any(min(axis) > .70 for axis in proposal["parity_fit"]))
    require(baseline == row["preselected"] == (pattern is not None), "Old interpolation decision differs")
    useful = baseline and bool(truth and hit * 10 >= union * 3 and hit * 2 >= truth)
    withheld = False
    if pattern:
        zero, peak, angle = pattern["zero"], pattern["peak"], pattern["angle"]
        require(0 <= zero <= 4 and 0 <= peak <= 4 and 1.75 <= abs(angle) <= 6 and angle * 4 == int(angle * 4),
                "Rotated carrier response outside frozen sweep")
        withheld = peak >= .08 and peak > 1.35 * zero
        require(withheld == pattern["selected"], "Rotated pattern decision differs")
    return baseline, useful, withheld


def audit(measurements, roster, receipt):
    ledger_path = measurements / "scored.json" if measurements.is_dir() else measurements
    ledger, sources, official = map(load, (ledger_path, roster, receipt))
    require(ledger["freeze_sha256"] == official["freeze_sha256"] and sha(ledger_path) == official["scored_sha256"]
            and sha(roster) == official["roster_sha256"], "Frozen file commitments differ")
    if measurements.is_dir():
        freeze_path = measurements / "freeze.json"
        freeze = load(freeze_path)
        require(sha(freeze_path) == official["freeze_sha256"] and freeze["roster_sha256"] == official["roster_sha256"],
                "Native freeze differs")
    require(len(ledger["rows"]) == official["exact_native_replays"] == 104 and len(sources) == 8
            and all(row["license"] == "CC0-1.0" for row in sources), "Source commitments differ")
    expected = {(s["id"], role) for s in sources for role in ROLES}
    roles = {role: Counter(files=0, assessed=0, unsupported_format=0, no_repeatable_stencil=0, old_flags=0,
                           new_flags=0, old_useful=0, new_useful=0, withheld=0) for role in ROLES}
    for row in ledger["rows"]:
        role = row["role"]
        key = row["key"], role
        require(key in expected and (row["truth_pixels"] == 0) == (role in (*UNSUPPORTED, "original")),
                "Unknown held role or wrong truth")
        expected.remove(key)
        old, useful, withheld = measure(row)
        roles[role][row["status"]] += 1
        roles[role].update(files=1, old_flags=int(old), old_useful=int(useful), withheld=int(withheld),
                           new_flags=int(old and not withheld), new_useful=int(useful and not withheld))
    require(not expected, "Missing held roles")
    rotations = sum(roles[r]["old_flags"] for r in ROTATIONS)
    removed = sum(roles[r]["withheld"] for r in ROTATIONS)
    advances = sum(roles[p]["new_useful"] for p in POSITIVES[:2]) >= 8
    advances = (advances and all(roles[p]["new_useful"] == roles[p]["old_useful"] for p in POSITIVES)
                and removed >= 4 and 2 * removed >= rotations and roles["original"]["new_flags"] == 0
                and not any(roles[r]["assessed"] for r in UNSUPPORTED))
    require(official["measurements"] == {"roles": roles, "preselected_rotations": rotations, "withheld_rotations": removed,
                                         "advancement": advances}, "Frozen gate arithmetic differs")
    print("Verified 104 source-coordinate results, exact overlap and rotated-carrier decisions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="Published scalar JSON or native scratch directory")
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    audit(args.measurements, args.roster, args.receipt)
