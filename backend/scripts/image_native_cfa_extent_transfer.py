# SPDX-License-Identifier: Apache-2.0
"""Evaluate extending camera interpolation crops with adjacent source evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from app.evidence.image_interpolation_binding import extent_region
from scripts import image_native_cfa_fine_transfer as fine

SCHEMA = "chipmunk.image-native-interpolation-extent/1"
WIDTH = 9
POINTS = {"inner_maximum": 0.75, "outer_minimum": 0.80, "difference_minimum": 0.20}
WIDE = ("32_feather", "48_feather", "64_feather")
read, require = fine.read, fine.require


def reference(split):
    return fine.PANELS / f"image-native-cfa-fine-{split}-2026-11-15.json"


def inspect(rgb, truth, original):
    from app.detectors.image.native_cfa_fine import POINT, refine_extent

    seed = original["fine"]
    if seed is None or seed["response"] <= POINT:
        return None
    reading = refine_extent(rgb, original["phase"], seed["region"])
    if reading is None:
        return None
    x, y, right, bottom = reading["region"]
    return reading | {"intersection": int(truth[y:bottom, x:right].sum())}


def measure(split, cache, archive):
    from app.detectors.image import native_cfa_fine
    from scripts.image_native_cfa_png_transfer import png_body

    require(POINTS == {"inner_maximum": native_cfa_fine.EDGE_INTERIOR_MAX,
                       "outer_minimum": native_cfa_fine.EDGE_EXTERIOR_MIN,
                       "difference_minimum": native_cfa_fine.EDGE_CHANGE_MIN}, "The frozen edge points differ")
    baseline = read(reference(split))
    expected = {(row["id"], row["role"]): row for row in baseline["rows"]}
    measurements = {"prior": [], "wider": []}
    args = SimpleNamespace(phase=split, archive=archive, cache=cache)
    for group, index, photo in fine.native_photos(args):
        arms = (("prior", 6),) if split == "development" else (("prior", 6), ("wider", WIDTH))
        for arm, width in arms:
            deliveries = fine.cases(photo, index, split, feather_width=width)
            for role, rgb, truth in deliveries:
                if arm == "wider" and role not in WIDE:
                    continue
                original = expected[group["id"], role] if arm == "prior" else fine.readout(rgb, truth)
                if original["fine"] and original["fine"]["response"] > baseline["point"]:
                    require(fine.digest(png_body(rgb)) == original["png_sha256"]
                            and fine.digest(truth.tobytes()) == original["truth_sha256"], "A retained source or truth mask differs")
                reading = inspect(rgb, truth, original)
                result = {"id": group["id"], "role": role, "extent": reading}
                if arm == "wider":
                    result["input"] = original
                measurements[arm].append(result)
            print(split, arm, group["id"], flush=True)
    reading = {"schema": SCHEMA, "split": split, "reference_sha256": fine.file_hash(reference(split)),
               "points": POINTS, "wider_feather_pixels": WIDTH, "measurements": measurements}
    reading["summary"] = review(reading)
    return reading


def useful(item, changed):
    if item is None or not changed:
        return False
    x, y, right, bottom = item["region"]
    intersection = item["intersection"]
    union = changed + (right - x) * (bottom - y) - intersection
    return 2 * intersection >= changed and 10 * intersection >= 3 * union


def checked_extent(result, old, changed):
    require(old is not None and old["region"] == result["inner_region"], "The refinement changed the inner region")
    region = extent_region({key: result[key] for key in ("inner_region", "edges")})
    a, b, c, d = region
    require(region == result["region"] and 0 <= a < c <= 2048 and 0 <= b < d <= 1536
            and old["intersection"] <= result["intersection"] <= min(changed, (c-a)*(d-b)), "The extent is not bound to its source")


def review(record):
    split = record["split"]
    baseline = read(reference(split))
    require(record["schema"] == SCHEMA and record["reference_sha256"] == fine.file_hash(reference(split))
            and record["points"] == POINTS and record["wider_feather_pixels"] == WIDTH, "The frozen comparison differs")
    expected = {(row["id"], row["role"]): row for row in baseline["rows"]}
    totals = {}
    for arm, measurements in record["measurements"].items():
        rows = (list(expected) if arm == "prior" else [] if split == "development"
                else [key for key in expected if key[1] in WIDE])
        require(arm in {"prior", "wider"} and [(r["id"], r["role"]) for r in measurements] == rows, "The evaluation roster differs")
        counts = {role: Counter() for role in dict.fromkeys(key[1] for key in rows)}
        for key, result in zip(rows, measurements, strict=True):
            original = expected[key] if arm == "prior" else result["input"]
            changed = original["changed"]
            seed = original["fine"] if original["fine"] and original["fine"]["response"] > baseline["point"] else None
            old = original["coarse"] or seed
            extent = result["extent"]
            if extent:
                checked_extent(extent, seed, changed)
            before, after = useful(old, changed), useful(extent or old, changed)
            counts[key[1]].update(files=1, flagged=int(old is not None), previous_useful=int(before), refined_useful=int(after),
                                  expanded=int(extent is not None), gained=int(after and not before), lost=int(before and not after),
                                  exact=int(extent is not None and extent["intersection"] == changed and
                                  (extent["region"][2]-extent["region"][0])*(extent["region"][3]-extent["region"][1]) == changed))
        totals[arm] = {role: dict(value) for role, value in counts.items()}
    require(set(totals) == {"prior", "wider"} and ("summary" not in record or totals == record["summary"]), "Outcomes do not reproduce")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "transfer", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(read(args.receipt)), sort_keys=True, indent=2))
    else:
        if args.cache is None or args.archive is None:
            parser.error("Native replay requires --cache and --archive")
        result = measure(args.phase, args.cache, args.archive)
        args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
