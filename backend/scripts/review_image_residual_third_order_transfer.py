"""Review the frozen small-image-edit measurements using only the standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from scripts import image_residual_third_order_transfer as study

PLAN_SHA256 = "394f06fe59c4b46145464e297048ddebb633e39c994187d69bc6645eaea33962"
ARM_WIDTHS = {"second_order": 28, "third_order": 28, "joint": 56}
MAX_RECEIPT_BYTES = 2_000_000


def review(receipt):
    """Check the frozen roster, validate proposals, and recompute advancement by format."""
    freeze = receipt["freeze"]
    plan = json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()
    study.require(hashlib.sha256(plan).hexdigest() == PLAN_SHA256, "Frozen 108-source image plan differs")
    study.require(freeze["schema"] == study.SCHEMA and freeze["design"] == study.DESIGN,
                  "Recorded method differs from the runner")
    study.require(freeze["runner_sha256"] == hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
                  "Historical measuring code differs from the frozen runner")
    weights = receipt["weights"]
    study.require(set(weights) == set(study.FORMATS), "Missing delivery-specific fit")
    for arms in weights.values():
        study.require(set(arms) == set(ARM_WIDTHS), "Missing comparison fit")
        for arm, vector in arms.items():
            study.require(isinstance(vector, list) and len(vector) == ARM_WIDTHS[arm]
                          and all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                          "Invalid fitted feature dimensions or weights")

    for part in ("calibration", "tgif", "bgreal"):
        for row in receipt[part]:
            rows, columns = (side // study.CELL for side in row["shape"])
            for arm in study.ARMS:
                record = row["checks"][arm]
                study.require(isinstance(record, dict) and isinstance(record.get("box"), list)
                              and len(record["box"]) == 4
                              and all(type(x) is int and x % study.CELL == 0 for x in record["box"]),
                              "Missing native-grid candidate")
                left, top, right, bottom = (side // study.CELL for side in record["box"])
                width, height = right - left, bottom - top
                study.require(width in study.BOX_SIDES and height in study.BOX_SIDES
                              and max(width, height) <= 2 * min(width, height)
                              and 100 * width * height <= 12 * rows * columns
                              and 2 <= left < right <= columns - 2 and 2 <= top < bottom <= rows - 2,
                              "Region differs from the frozen box search")

    summary = study.audit(receipt)
    limits = study.DESIGN["advance"]
    gates = {}
    for format_ in study.FORMATS:
        gates[format_] = {}
        for arm in study.ARMS:
            key = f"{format_}/{arm}"
            tgif, bgreal = (summary[part][key] for part in ("tgif", "bgreal"))
            gates[format_][arm] = {
                "added_feathered_tgif_groups": tgif["new_feathered_groups"] >= limits["TGIF_feathered_groups"],
                "added_feathered_bgreal_groups": bgreal["new_feathered_groups"] >= limits["BGREAL_feathered_groups"],
                "ordinary_groups": max(tgif["new_ordinary_groups"], bgreal["new_ordinary_groups"])
                <= limits["max_extra_ordinary_groups_each"],
                "misplaced_edits": max(tgif["misplaced"], bgreal["misplaced"]) <= limits["max_misplaced_each"],
            }
    return {
        "source_groups": {part: len({row["source"] for row in rows}) for part, rows in freeze["inputs"].items()},
        "fixed_plan_sha256": PLAN_SHA256,
        "summary": summary,
        "gates": gates,
        "advancing": [f"{format_}/{arm}" for format_, arms in gates.items() for arm, results in arms.items()
                      if all(results.values())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.receipt.suffix == ".gz" else open
    with opener(args.receipt, "rb") as handle:
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    study.require(len(raw) <= MAX_RECEIPT_BYTES, "Image receipt exceeds the review input limit")
    print(json.dumps(review(json.loads(raw)), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
