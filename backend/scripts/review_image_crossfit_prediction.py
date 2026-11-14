"""Recount the held blind image-prediction comparison without numerical libraries."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts import image_residual_third_order_transfer as geometry

PLAN_SHA256 = "b0aa0351082b436aac77d69d4a2499524a43559c17d1cd28a20209268f865080"
ARMS = {"fixed": 9, "crossfit": 9, "joint": 18}
PART_COUNTS = {"fit": 24, "calibration": 24, "tgif": 48, "bgreal": 12}
MAX_BYTES = 2_000_000
require = geometry.require


def population(rows):
    return {row["source"] for row in rows}


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        value = handle.read(MAX_BYTES + 1)
    require(len(value) <= MAX_BYTES, "Replay input exceeds 2 MB")
    return json.loads(value)


def review(receipt, comparison, runner):
    lock = receipt["freeze"]
    serialized = (json.dumps(lock, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(geometry.digest(serialized) == PLAN_SHA256 and geometry.digest(runner) == lock["script_sha256"],
            "Frozen code or image source plan differs")
    require(set(lock["inputs"]) == set(PART_COUNTS), "Missing a frozen roster")
    parts = {part: population(rows) for part, rows in lock["inputs"].items()}
    require({part: len(groups) for part, groups in parts.items()} == PART_COUNTS
            and len(set.union(*parts.values())) == sum(PART_COUNTS.values()), "Source split or source count differs")

    require(set(receipt["weights"]) == set(geometry.FORMATS), "Missing delivery-specific fit")
    for arms in receipt["weights"].values():
        require(arms.keys() == ARMS.keys() and all(len(values) == ARMS[arm]
                and all(type(value) in (int, float) and math.isfinite(value) for value in values)
                for arm, values in arms.items()), "Invalid fitted measurements")

    maxima = defaultdict(lambda: defaultdict(float))
    for part in ("calibration", "tgif", "bgreal"):
        for index, (row, planned) in enumerate(zip(receipt[part], lock["inputs"][part], strict=True)):
            require({key: row[key] for key in planned} == planned, "Delivery differs from its frozen record")
            h, w = planned["shape"]
            expected = set(ARMS)
            if part != "calibration":
                expected.add("incumbent")
                peer = comparison[part][index]
                require({key: peer[key] for key in planned} == planned
                        and peer["checks"]["incumbent"] == row["checks"]["incumbent"], "Earlier comparison differs")
            require(set(row["checks"]) == expected, "Missing case comparison")
            for arm, candidate in row["checks"].items():
                if arm == "incumbent":
                    continue
                score = candidate["score"]
                require(type(score) in (int, float) and math.isfinite(score), "Invalid candidate response")
                box = candidate["box"]
                require(len(box) == 4 and all(type(x) is int and x % geometry.CELL == 0 for x in box), "Invalid image grid")
                x0, y0, x1, y1 = (x // geometry.CELL for x in box)
                width, height = x1 - x0, y1 - y0
                require(width in geometry.BOX_SIDES and height in geometry.BOX_SIDES
                        and max(width, height) <= 2 * min(width, height)
                        and 100 * width * height <= 12 * (w // 8) * (h // 8)
                        and 2 <= x0 < x1 <= w // 8 - 2 and 2 <= y0 < y1 <= h // 8 - 2, "Invalid candidate rectangle")
                area = (box[2] - box[0]) * (box[3] - box[1])
                truth = row["changed_pixels"] if row["role"] in geometry.POSITIVE else 0
                counts = candidate["overlap"]
                require(counts["target"] == truth and counts["reported"] == area
                        and max(0, area + truth - h * w) <= counts["overlap"] <= min(area, truth), "Impossible overlap counts")
                if part == "calibration":
                    scores = maxima[row["format"], arm]
                    scores[row["source"]] = max(scores[row["source"]], score)

    points = {fmt: {arm: sorted(maxima[fmt, arm].values())[-2] for arm in ARMS} for fmt in geometry.FORMATS}
    require(points == receipt["points"], "Operating point disagrees with a frozen source maximum")
    # The geometric outcome rules are shared with the prior comparison.
    geometry.ARMS = tuple(ARMS)
    counted = {part: geometry.summarize(receipt[part], points) for part in ("tgif", "bgreal")}
    require(counted == receipt["summary"], "Held outcomes did not reproduce")
    gate = lock["design"]["advance"]
    passing = []
    for fmt in geometry.FORMATS:
        for arm in ARMS:
            key = f"{fmt}/{arm}"
            first, second = counted["tgif"][key], counted["bgreal"][key]
            if (first["new_feathered_groups"] >= gate["main_new_feathered_sources"]
                    and second["new_feathered_groups"] >= gate["cross_dataset_new_feathered_sources"]
                    and all(row["new_ordinary_groups"] <= gate["max_new_control_groups_each"]
                            and row.get("misplaced", 0) <= gate["max_misplaced_each"] for row in (first, second))):
                passing.append(key)
    require(passing == receipt["advance"], "Advancement decision did not reproduce")
    return {"source_groups": PART_COUNTS, "summary": counted, "advanced": passing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("image_crossfit_prediction.py").read_bytes()
    record = read(args.receipt)
    require(geometry.digest(args.comparison.read_bytes()) == record["freeze"]["baseline_sha256"], "Earlier comparison hash differs")
    print(json.dumps(review(record, read(args.comparison), runner), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
