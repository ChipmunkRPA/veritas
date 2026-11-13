"""Recount the local rotation trial from scalar region and overlap records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.native-rotation-location/1"
POSITIVE = {"bilinear_rotation", "bicubic_rotation", "feathered_rotation"}
ROLES = POSITIVE | {"untouched", "full_rotation", "translation", "quantized_translation", "blur", "tone",
                    "generated_sharp", "generated_feathered", "resized", "eight_bit"}
STATUSES = {"unsupported", "no_supported_region", "ordinary_axis_interpolation", "no_rotation_support", "local_rotation"}
SHA = re.compile("[a-f0-9]{64}")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def review(ledger, roster):
    value = json.loads(ledger.read_bytes())
    sources = json.loads(roster.read_bytes())
    check(value["freeze"]["schema"] == SCHEMA
          and value["freeze"]["roster_sha256"] == hashlib.sha256(roster.read_bytes()).hexdigest()
          and len(sources) == len({row["model"].casefold() for row in sources}) == 13
          and all(row["license"] == "CC0-1.0" for row in sources), "Unexpected source roster")
    pending = {(row["id"], role) for row in sources for role in ROLES}
    totals = {role: Counter(files=0, flagged=0, useful=0, assessed=0) for role in sorted(ROLES)}
    for row in value["rows"]:
        role, status, box, reading = (row[key] for key in ("role", "status", "box", "readout"))
        key = row["camera"], role
        check(key in pending and status in STATUSES
              and all(SHA.fullmatch(row[field]) for field in ("input_sha256", "truth_sha256")),
              "Unexpected or duplicated input")
        pending.remove(key)
        flag = status == "local_rotation"
        if reading is not None:
            zero, peak, angle = (reading[key] for key in ("zero", "peak", "angle"))
            check(all(math.isfinite(v) and 0 <= v <= 4 for v in (zero, peak))
                  and 1.75 <= abs(angle) <= 6 and angle * 4 == int(angle * 4)
                  and reading["selected"] == flag == (peak >= .08 and peak > 1.35 * zero)
                  and status in {"local_rotation", "no_rotation_support"}, "Rotation decision differs")
        check((reading is not None) == (status in {"local_rotation", "no_rotation_support"})
              and flag == bool(box) and (role == "eight_bit") == (status == "unsupported"), "Coverage differs")
        size = 0
        if box is not None:
            x, y, x1, y1 = box
            check(all(type(v) is int and v % 16 == 0 for v in box) and 0 <= x < x1 <= 2048
                  and 0 <= y < y1 <= 1536 and all(n in (64, 96, 128, 160, 192) for n in (x1 - x, y1 - y)),
                  "Selected box lies outside the tested native grid")
            size = (x1 - x) * (y1 - y)
        overlap = row["overlap"]
        truth, hit = overlap["reference_positive_pixels"], overlap["intersection"]
        check(0 <= hit <= min(size, truth) and 0 <= truth <= 128**2 and overlap["predicted_positive_pixels"] == size,
              "Impossible overlap")
        check(bool(truth) == (role not in {"untouched", "full_rotation", "resized", "eight_bit"}), "Incorrect truth extent")
        union = truth + size - hit
        check(overlap["iou"] == (hit / union if union else 1.) and overlap["recall"] == (hit / truth if truth else None),
              "Saved overlap differs")
        totals[role].update(files=1, flagged=flag, assessed=status not in {"unsupported", "no_supported_region"},
                            useful=flag and role in POSITIVE and 2 * hit >= truth and 10 * hit >= 3 * union)
    check(not pending and totals == value["counts"], "Missing cases or incorrect totals")
    sharp = sum(totals[key]["useful"] for key in POSITIVE - {"feathered_rotation"})
    passes = (sharp >= 10 and totals["feathered_rotation"]["useful"] >= 4
              and sum(totals[role]["flagged"] for role in ROLES - POSITIVE) <= 1
              and not any(totals[r]["flagged"] for r in ("untouched", "generated_sharp", "generated_feathered"))
              and totals["eight_bit"]["assessed"] == 0)
    check(value["passes"] is passes, "Gate differs")
    return {"rows": len(value["rows"]), "counts": totals, "passes": passes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("roster", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.ledger, args.roster), sort_keys=True))
