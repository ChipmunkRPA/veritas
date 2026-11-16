"""Verify image commitments and distinguish missed regions from cutoff misses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from scripts.image_region_rank_compare import exact_mask, require, verify
from scripts.image_region_rank_inputs import POSITIVE
from scripts.video_face_temporal_fit import read, save, sha


def signature(path):
    from PIL import Image

    with Image.open(path, formats=["JPEG"]) as image:
        small = image.convert("L").resize((17, 17), Image.Resampling.LANCZOS)
        values = list(small.getdata())
    horizontal = [values[y * 17 + x] > values[y * 17 + x + 1] for y in range(16) for x in range(16)]
    vertical = [values[y * 17 + x] > values[(y + 1) * 17 + x] for y in range(16) for x in range(16)]
    return int("".join("1" if value else "0" for value in horizontal + vertical), 2)


def inputs(media, output):
    from PIL import Image

    manifest = media / "manifest.json"
    plan = read(manifest)
    groups = {row["tag"]: row for row in plan["groups"]}
    signatures = {}
    masks = set()
    for row in plan["images"]:
        image, truth = (media / row[key] for key in ("file", "mask"))
        group = groups[row["group"]]
        assert sha(image) == row["sha256"] and sha(truth) == row["mask_sha256"]
        with Image.open(image, formats=["JPEG"]) as pixels, Image.open(truth, formats=["PNG"]) as labels:
            assert list(pixels.size[::-1]) == group["shape"] and pixels.size == labels.size and pixels.mode == "RGB"
            histogram = labels.convert("L").histogram()
            assert histogram[255] == row["changed_pixels"] and sum(histogram[1:255]) == 0
            assert row["role"] in POSITIVE or not row["changed_pixels"]
            if row["role"] == "sharp":
                assert row["changed_pixels"] == group["first_changed_pixels"]
        masks.add(row["mask"])
        if row["role"] == "ordinary" and row["quality"] == 90:
            signatures[row["group"]] = signature(image)
    require(set(signatures) == set(groups), "An ordinary-source thumbnail is missing")
    compared = [(a, b, (left ^ right).bit_count()) for (a, left), (b, right) in combinations(signatures.items(), 2)
                if groups[a]["split"] != groups[b]["split"]]
    near = [(a, b, distance) for a, b, distance in compared if distance <= 16]
    result = {"manifest_sha256": sha(manifest), "verified_encoded_inputs": len(plan["images"]), "verified_masks": len(masks),
              "thumbnail_scheme": "512 gradient bits on a 17 by 17 grayscale thumbnail",
              "compared_cross_role_source_pairs": len(compared), "closest_bit_distance": min(row[2] for row in compared),
              "near_cross_role_candidates": near}
    save(output, result)
    print(json.dumps(result, sort_keys=True))


def diagnostics(study, result, output):
    import numpy as np
    from PIL import Image
    from scripts.image_region_rank_compare import all_scores, integral, layout, load_grid, window
    from scripts.image_region_rank_features import CELL

    verify(result)
    paths = study / "deliveries"
    frozen = result["freeze"]
    audit = read(study / "input-audit.json")
    require(audit["manifest_sha256"] == frozen["manifest_sha256"], "Independent input audit covered different files")
    coverage = Counter()
    audited = 0
    panels = result["held"]
    for row in panels:
        truth_path = paths / row["mask"]
        with Image.open(truth_path, formats=["PNG"]) as bitmap:
            labeled = bitmap.convert("L")
            for arm in row["checks"].values():
                assert labeled.crop(tuple(arm["box"])).histogram()[255] == arm["intersection"]
                audited += 1
        if row["role"] not in POSITIVE:
            continue
        quality = str(row["quality"])
        selected = frozen["chosen"][quality]
        truth = exact_mask(paths, row)
        geometry = layout(tuple(row["shape"]))
        boxes, _, size, _ = geometry
        hits = window(integral(truth), boxes * CELL)
        oracle = (2 * hits >= row["changed_pixels"]) & (10 * hits >= 3 * (row["changed_pixels"] + size * CELL**2 - hits))
        group = f"{row['split']}/{quality}/{row['role']}"
        coverage[group, "planned"] += 1
        if not np.any(oracle):
            coverage[group, "no_candidate_geometry"] += 1
            continue
        grid = load_grid(study / ("rank-" + row["split"]), row)
        response = all_scores(grid, geometry, frozen["heads"][quality][selected])
        cutoff = frozen["levels"][f"{quality}/{selected}"]["wider"]
        if not np.any(response[oracle] > cutoff):
            coverage[group, "useful_candidate_below_point"] += 1
        elif not oracle[int(np.argmax(response))]:
            coverage[group, "useful_candidate_outranked"] += 1
        else:
            coverage[group, "useful_candidate_selected"] += 1
    result["independent_audit"] = audit | {"native_candidate_intersections_verified": audited,
                                          "failure_analysis": {"/".join(k): v for k, v in sorted(coverage.items())}}
    save(output, result)
    print(json.dumps({"outcome": verify(result), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "native_candidate_intersections_verified": audited}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inputs", "release"))
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "inputs":
        inputs(args.path, args.output)
    else:
        diagnostics(args.path, read(args.path / "rank-result.json.gz"), args.output)
