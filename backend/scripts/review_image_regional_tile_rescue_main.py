"""Recompute JPEG tile study counts from anonymous pixel statistics."""
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
    if receipt.get("schema") == "chipmunk.image-regional-tile-rescue-complete/1":
        return verify_full(receipt)
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


def useful(pixels):
    target, predicted, overlap = pixels
    return target > 0 and 10 * overlap >= 3 * (target + predicted - overlap) and 2 * overlap >= target


def verify_full(receipt):
    rows = receipt["anonymous_pixel_counts"]
    main, external = ([r for r in rows if r["panel"] == name] for name in ("retained_b", "external"))
    if ([r["case"] for r in rows] != list(range(1, 201)) or rows != main + external
            or [len(main), len(external)] != [136, 64] or receipt["source_roles"] != {"retained_b": 136, "external": 64}
            or receipt["source_counts"] != {"retained_b": 15, "external": 7}
            or receipt["view_count"] != len(rows) or receipt["exact_source_mask_replays"] != len(rows)
            or {r["group"] for r in main} != set(range(1, 16))
            or {r["group"] for r in external} != set(range(16, 23))):
        raise ValueError("Incomplete or overlapping frozen source banks")
    counts = {}
    for name, group in (("retained_b", main), ("external", external), ("all", rows)):
        summarize(group)
        counts[name] = {method: Counter() for method in METHODS}
        ordinary_groups = {method: set() for method in METHODS}
        for row in group:
            tiles = row["tile_forwards"]
            native, candidate = (row["pixels"][method] for method in ("native", "native_plus_tiles"))
            if (type(tiles) is not int or not 0 <= tiles <= 8 or (native[1] and (native != candidate or tiles))
                    or (not native[1] and not 2 <= tiles <= 8)):
                raise ValueError("Tile fallback must preserve native regions and obey its input budget")
            for method in METHODS:
                measured = row["pixels"][method]
                field = f"{row['family']}/{row['role']}"
                counts[name][method][f"{field}/files"] += 1
                if row["role"] in {"sharp", "feathered"}:
                    localized = useful(measured)
                    counts[name][method][f"{field}/useful"] += localized
                    counts[name][method][f"{field}/wrong"] += bool(measured[1]) and not localized
                else:
                    counts[name][method][f"{field}/flags"] += bool(measured[1])
                    if measured[1]:
                        ordinary_groups[method].add(row["group"])
        counts[name]["ordinary_groups"] = {method: len(values) for method, values in ordinary_groups.items()}
        counts[name]["ordinary_groups_added_by_tiles"] = len(ordinary_groups["native_plus_tiles"] - ordinary_groups["native"])
        counts[name]["rescue_forwards"] = sum(r["tile_forwards"] for r in group)
    gained, wrong = set(), 0
    for row in rows:
        native, candidate = (row["pixels"][method] for method in ("native", "native_plus_tiles"))
        if row["role"] in {"sharp", "feathered"} and not native[1] and candidate[1]:
            wrong += not useful(candidate)
            if useful(candidate) and row["role"] == "feathered" and row["family"] == "lama":
                gained.add(row["group"])
    paired = {"added_feathered_lama_groups": len(gained), "additional_wrong_edit_files": wrong,
              "additional_ordinary_groups": counts["all"]["ordinary_groups_added_by_tiles"]}
    gate = receipt["design"]["advance"]
    passed = (len(gained) >= gate["feathered_lama_groups"] and wrong <= gate["additional_wrong_edit_files"]
              and paired["additional_ordinary_groups"] <= gate["additional_ordinary_groups"])
    if (counts != receipt["results"] or paired != receipt["paired"] or passed != receipt["passes"]):
        raise ValueError("Two-bank receipt disagrees with independent counts")
    return {"completed_views": len(rows), "paired": paired, "passes": passed, "results": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))
