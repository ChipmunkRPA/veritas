"""Recompute the complete JPEG tile main-bank counts from anonymous pixel statistics."""
import argparse
import json
from collections import Counter
from pathlib import Path

METHODS = ("current_luma", "native", "native_plus_tiles")


def summarize(rows):
    if len({row["case"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case")
    totals, groups = {}, {}
    for method in METHODS:
        counts = Counter()
        flagged_groups = set()
        for row in rows:
            target, predicted, overlap = row["pixels"][method]
            if not 0 <= overlap <= min(target, predicted) or max(target, predicted) > row["image_pixels"]:
                raise ValueError("Impossible mask counts")
            edited = row["role"] in {"sharp", "feathered"}
            if edited != (target > 0) or target != row["pixels"]["native"][0]:
                raise ValueError("Edit and mask truth differ")
            useful = edited and 10 * overlap >= 3 * (target + predicted - overlap) and 2 * overlap >= target
            for family in ("all", row["family"]):
                prefix = f"{family}/{row['role'] if edited else 'ordinary'}"
                counts[f"{prefix}/files"] += 1
                if edited:
                    counts[f"{prefix}/useful"] += useful
                    counts[f"{prefix}/wrong"] += predicted > 0 and not useful
                else:
                    counts[f"{prefix}/flags"] += predicted > 0
            if not edited and predicted:
                flagged_groups.add(row["group"])
        totals[method] = dict(counts)
        groups[method] = len(flagged_groups)
    return totals, groups


def verify(receipt):
    rows = receipt["anonymous_pixel_counts"]
    report = receipt["main_bank"]
    totals, groups = summarize(rows)
    if (totals != report["aggregates"] or groups != report["ordinary_groups"]
            or len(rows) != report["completed_views"]
            or len({row["group"] for row in rows}) != report["complete_source_groups"]
            or sum(row["tile_forwards"] for row in rows) != report["actual_tile_forwards"]
            or len(rows) * len(METHODS) != report["verified_recorded_pixel_statistics"]):
        raise ValueError("Main-bank totals do not match the rows")
    return {"completed_views": len(rows), "ordinary_groups": groups, "aggregates": totals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))
