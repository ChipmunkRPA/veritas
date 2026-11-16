"""Recount timing and spatial accuracy for the fixed and aligned crop checks."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
from statistics import median

from scripts.review_video_cropped_reuse import source_clock
from scripts.review_video_reflected_sequence_panel import _paired_overlap

ARMS = ("fixed", "centered", "landmarks", "combined")
ROLES = (
    "center_reuse", "offset_reuse", "between_grid_reuse", "offset_reuse_tandem",
    "original", "in_place_crop", "moving_crop", "held_crop", "straight_repeat", "original_tandem",
    "center_fractional_reuse", "center_fractional_reuse_tandem", "between_grid_reuse_tandem",
)
SIZE = (192, 108, 192, 108)


def geometry_error(prediction, truth, dimensions):
    width, height = dimensions
    errors = [abs(found / (used * prediction["unit"]) - round(label * native) / native) * used
              for found, label, used, native in zip(prediction["crop_box"], truth["crop_box"], SIZE,
                                                   (width, height, width, height), strict=True)]
    return max(errors)


def row_results(row):
    arms = row["results"]
    ranked = [name for name in ("centered", "landmarks") if arms[name]]
    chosen = "fixed" if arms["fixed"] else max(
        ranked, key=lambda name: (len(range(arms[name]["ranges"][0][0], arms[name]["ranges"][0][1] + 1)),
                                  len(arms[name]["support"])), default=None,
    )
    assert row["selected"] == chosen
    return arms | {"combined": arms[chosen] if chosen else None}


def recount(suite, media):
    plan = suite["plan"]
    dimensions = plan["dims"]
    limit = 225 if dimensions == [640, 360] else 200
    assert dimensions in ([640, 360], [1280, 720]) and plan["minimum_iou"] == .5
    summary, geometry, timing = defaultdict(Counter), defaultdict(list), defaultdict(list)
    checked, delivered = set(), set()
    for row in plan["cases"]:
        identifier = row["source"], row["role"]
        assert identifier not in delivered
        delivered.add(identifier)
        truth = row["truth"]
        timing["fixed"].append(row["fixed_s"])
        timing["aligned"].append(row["aligned_s"])
        if truth:
            summary[row["role"]]["known"] += 1
        else:
            summary[row["role"]]["controls"] += 1
        results = row_results(row)
        for method, prediction in results.items():
            if prediction is None:
                continue
            (a, b), (c, d) = prediction["ranges"]
            assert 0 <= a < b < c < d < limit and b - a == d - c
            changes = prediction["support"]
            assert len(changes) >= 8 and 5 * len(changes) >= 3 * (b - a) and changes[0] == 0 and changes[-1] == b - a - 1
            assert all(0 < stop - start <= 3 for start, stop in zip(changes, changes[1:]))
            x, y, right, bottom = prediction["crop_box"]
            assert prediction["unit"] in (1, 32) and 0 <= x < right <= 192 * prediction["unit"]
            assert 0 <= y < bottom <= 108 * prediction["unit"] and prediction["wider_range"] in {"first", "second"}
            assert all((stop - start) * Fraction(row["time_base"]) == Fraction(b - a, 25) for start, stop in prediction["pts"])
            useful = bool(truth and _paired_overlap(prediction["ranges"], truth["ranges"]))
            assert prediction["useful"] == useful
            summary[row["role"]][f"{method}_flags"] += 1
            if useful:
                summary[row["role"]][f"{method}_locate"] += 1
                if prediction["ranges"] == truth["ranges"]:
                    summary[row["role"]][f"{method}_exact"] += 1
                if prediction["wider_range"] == truth["wider_range"]:
                    geometry[method].append(geometry_error(prediction, truth, dimensions))
            if media:
                if identifier not in checked:
                    basis, ticks = source_clock(media / row["delivery"], row["sha256"])
                    checked.add(identifier)
                assert row["time_base"] == basis
                assert prediction["pts"] == [[ticks[a], ticks[b]], [ticks[c], ticks[d]]]
                summary[row["role"]]["verified_endpoint_appearances"] += 4
    if plan.get("alignment_plan"):
        assert delivered == set(product((source["id"] for source in plan["sources"]), ROLES))
    return {
        "counts": dict(summary),
        "geometry_working_pixels": {method: {"correct_direction": len(errors), "median": round(median(errors), 3),
                                              "maximum": round(max(errors), 3)} for method, errors in geometry.items()},
        "seconds": {method: {"median": round(median(readings), 4), "maximum": max(readings)} for method, readings in timing.items()},
        "distinct_deliveries_read_back": len(checked),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    options = parser.parse_args()
    data = options.receipt.read_bytes()
    document = json.loads(gzip.decompress(data) if options.receipt.suffix == ".gz" else data)
    suites = document.get("panels", [document])
    print(json.dumps({
        suite["plan"]["profile"]: recount(suite, options.media / suite["run"] if options.media and "run" in suite else options.media)
        for suite in suites
    }, indent=2, sort_keys=True))
