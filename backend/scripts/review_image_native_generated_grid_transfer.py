# SPDX-License-Identifier: Apache-2.0
"""Verify the native generated grid study from its scalar ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = ("original", "even_copy", "blur", "local_tone", "diagonal_cubic", "rotation_bilinear",
         "quantized_256", "quantized_257", "generated_sharp", "generated_feathered", "generated_linear_promote",
         "resized", "eight_bit")
POSITIVE = ("generated_sharp", "generated_feathered", "generated_linear_promote")
UNSUPPORTED = ("resized", "eight_bit")
POINT = 2.4346873354224834
VETO = .70


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def arm(row, name):
    measured = row[name]
    if measured is None:
        return False, False, False
    xoff, yoff = measured["grid"]
    box = measured["box"]
    area = 0
    if box is not None:
        x0, y0, x1, y1 = box
        require(xoff in (0, 16) and yoff in (0, 16) and 0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 1536
                and all((x - xoff) % 32 == 0 for x in (x0, x1)) and all((y - yoff) % 32 == 0 for y in (y0, y1)),
                "Grid offset or native rectangle is outside scope")
        area = (x1 - x0) * (y1 - y0)
    pixels = measured["pixels"]
    truth, common = row["truth_pixels"], pixels["intersection"]
    require(pixels["predicted_positive_pixels"] == area and pixels["reference_positive_pixels"] == truth
            and 0 <= common <= min(area, truth), "Impossible pixel overlap")
    union = area + truth - common
    require(pixels["iou"] == (common / union if union else 1.)
            and pixels["recall"] == (common / truth if truth else None), "Invalid overlap ratios")
    veto = box is not None and any(all(v > VETO for v in axis) for axis in measured["parity_fit"])
    flagged = measured["score"] > POINT and not veto
    useful = flagged and bool(truth and 10 * common >= 3 * union and 2 * common >= truth)
    promotion = flagged and any(item["core"] is not None and item["ring"] is not None
                                and all(x >= .95 for x in item["core"]) and all(x <= .70 for x in item["ring"])
                                for item in measured["promotion"])
    return flagged, useful, promotion


def review(row_path, roster, receipt, *, lock_path=None):
    ledger, sources, published = map(read, (row_path, roster, receipt))
    require(ledger["freeze_sha256"] == published["freeze_sha256"]
            and digest(row_path) == published["scored_sha256"]
            and digest(roster) == published["roster_sha256"], "Frozen receipts differ")
    if lock_path is not None:
        frozen = read(lock_path)
        require(digest(lock_path) == published["freeze_sha256"] and frozen["roster_sha256"] == published["roster_sha256"]
                and frozen["design"] == published["design"], "Native run freeze differs")
    require(len(ledger["rows"]) == published["exact_native_replays"] == 104, "Wrong frozen input scope")
    cases = {(r["id"], role) for r in sources for role in ROLES}
    require(len(cases) == 104 and all(r["license"] == "CC0-1.0" for r in sources), "Unexpected source terms or identity")
    scores = {role: Counter(files=0, unsupported_format=0, no_repeatable_stencil=0, assessed=0,
                            base_flags=0, base_useful=0, shifted_flags=0, shifted_useful=0,
                            shifted_promotion=0) for role in ROLES}
    for row in ledger["rows"]:
        role = row["role"]
        key = row["key"], role
        require(key in cases and (row["truth_pixels"] == 0) == (role in (*UNSUPPORTED, "original")),
                "Unknown input or wrong exact mask")
        cases.remove(key)
        out = scores[role]
        require(row["status"] in {"unsupported_format", "no_repeatable_stencil", "assessed"}, "Unknown input status")
        out["files"] += 1
        out[row["status"]] += 1
        for name in ("base", "shifted"):
            flagged, useful, promotion = arm(row, name)
            out[f"{name}_flags"] += flagged
            out[f"{name}_useful"] += useful
            if name == "shifted":
                out["shifted_promotion"] += promotion
    require(not cases, "Some cases are missing")
    ordinary = [c for r,c in scores.items() if r not in (*POSITIVE, *UNSUPPORTED)]
    previous, current = (sum(c[f"{name}_flags"] for c in ordinary) for name in ("base", "shifted"))
    passed = all(scores[r]["shifted_useful"] >= scores[r]["base_useful"] + 2 for r in POSITIVE[:2])
    passed = passed and current <= previous + 1 and not scores["original"]["shifted_flags"]
    passed = bool(passed and not any(scores[r]["assessed"] for r in UNSUPPORTED))
    safe = not any(scores[r]["shifted_promotion"] for r in POSITIVE)
    require(published["measurements"] == {"by_role": scores, "ordinary_base_flags": previous,
                                           "ordinary_shifted_flags": current, "shifted_grid_passed": passed,
                                           "promotion_safe_as_veto": safe}, "Gate decision differs")
    print("Verified 104 frozen rows and exact geometry, overlap and gate arithmetic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="Published scalar ledger or native run directory")
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.measurements.is_dir():
        review(args.measurements / "scored.json", args.roster, args.receipt,
               lock_path=args.measurements / "freeze.json")
    else:
        review(args.measurements, args.roster, args.receipt)
