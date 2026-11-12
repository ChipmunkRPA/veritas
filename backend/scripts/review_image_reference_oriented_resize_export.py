# SPDX-License-Identifier: Apache-2.0
"""Verify an earlier-copy comparison, including rotated crops and screenshot insets."""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_image_reference_oriented_crop_export as previous  # noqa: E402

existing = previous.existing
SCHEMA = "chipmunk.image-reference-comparison/8"
SCREENSHOT_SCHEMA = "chipmunk.image-reference-comparison/9"
KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
           "lanczos": Image.Resampling.LANCZOS, "opencv_area": None}


def _projection(comparison, ref, sub, carrier):
    import cv2
    import numpy as np

    require = existing._require
    require(comparison["status"] == "comparable" and comparison["coordinate_space"] == "exif-oriented-submitted"
            and "alignment" not in comparison and carrier.format == "PNG" and carrier.mode == "RGB"
            and min(*ref.size, *sub.size) >= 160 and max(ref.size) <= 768 and max(sub.size) <= 845,
            "Unsupported oriented resize scope")
    proof = comparison["exact_oriented_resize"]
    operation = existing.ORIENTATIONS[proof["transform"]]
    inverse = {Image.Transpose.ROTATE_90: Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_270: Image.Transpose.ROTATE_90}.get(operation, operation)
    submitted = np.asarray(sub)
    textures = list(existing._textured_boxes(sub))
    with ref.transpose(operation) as oriented:
        box = existing._bounds(proof["projected_crop"], oriented.size)
        left, top, right, bottom = box
        size = right-left, bottom-top
        sw, sh = sub.size
        require(existing._bounds(proof["native_crop"], ref.size) == existing._orientation_bounds(box, oriented.size, inverse),
                "Earlier overlap does not match the recorded orientation")
        require(size not in (oriented.size, sub.size) and min(size) >= 160
                and 100*size[0]*size[1] >= 55*ref.width*ref.height
                and all(round(.72*d) <= actual <= round(1.17*d) for d, actual in zip(size, sub.size, strict=True))
                and abs((sw/size[0])/(sh/size[1])-1) <= .008, "Unsupported earlier crop or resize geometry")
        with oriented.crop(box) as cropped:
            matches = []
            for label, kernel in KERNELS.items():
                projected = (cv2.resize(np.asarray(cropped), sub.size, interpolation=cv2.INTER_AREA) if kernel is None
                             else np.asarray(cropped.resize(sub.size, kernel)))
                difference = (projected != submitted).any(axis=2)
                matched = difference.size - int(difference.sum())
                if 100*matched < 65*difference.size:
                    continue
                witnesses = [[r, c] for r, c, (x, y, right, bottom) in textures if not difference[y:bottom, x:right].any()]
                quadrants = {(r > 2, c > 2) for r, c in witnesses if r != 2 and c != 2}
                if len(witnesses) < 12 or min(len({p[0] for p in witnesses}), len({p[1] for p in witnesses})) < 4 or len(quadrants) != 4:
                    continue
                rows, cols = np.nonzero(difference)
                if len(cols) and 5*(cols.max()+1-cols.min())*(rows.max()+1-rows.min()) > 2*difference.size:
                    continue
                matches.append((label, existing._hash(projected.tobytes()), difference.tobytes(), matched, witnesses))
        oriented_size = oriented.size
    require(bool(matches) and [item[0] for item in matches] == proof["kernels"]
            and all(item[1:] == matches[0][1:] for item in matches), "Reproduced resize kernels differ or yield competing projections")
    _, rgb_sha, difference, matched, witnesses = matches[0]
    require(proof["projected_rgb_sha256"] == rgb_sha and proof["difference_mask_sha256"] == existing._hash(difference)
            and proof["matched_pixels"] == matched and proof["matched_tiles"] == witnesses,
            "Exact oriented resize measurements differ from the originals")
    return box, oriented_size, inverse, difference


def _anchor_cells(ref, before, after, sub):
    original, screenshot = ref.load(), sub.load()
    samples = ((0, 0), (12, 5), (2, 12))
    anchors = {(x, y): tuple(original[x + ox, y + oy] for ox, oy in samples)
               for y in range(0, ref.height - 12, 12) for x in range(0, ref.width - 12, 12)}
    unique = Counter(anchors.values())
    dx, dy = after[0] - before[0], after[1] - before[1]
    width, height = before[2] - before[0], before[3] - before[1]
    cells = Counter()
    for (x, y), colors in anchors.items():
        if (unique[colors] != 1 or not before[0] <= x < before[2] - 12 or not before[1] <= y < before[3] - 12
                or tuple(screenshot[x + dx + ox, y + dy + oy] for ox, oy in samples) != colors):
            continue
        cells[(y - before[1]) * 5 // height, (x - before[0]) * 5 // width] += 1
    return [[row, col, count] for (row, col), count in sorted(cells.items()) if count >= 3]


def _screenshot_assets(stack, comparison, assets, ref, sub, carrier):
    require = existing._require
    require(comparison["status"] == "comparable" and comparison["coordinate_space"] == "exif-oriented-submitted"
            and "alignment" not in comparison and carrier.format == "PNG" and carrier.mode == "RGB"
            and carrier.getexif().get(274, 1) == 1 and ref.size != sub.size
            and min(*ref.size, *sub.size) >= 160 and max(ref.size) <= 1024 and max(sub.size) <= 1280,
            "Unsupported screenshot inset scope")
    proof = comparison["embedded_native_photo"]
    before = existing._bounds(proof["reference_bounds"], ref.size)
    after = existing._bounds(proof["submitted_bounds"], sub.size)
    width, height = after[2] - after[0], after[3] - after[1]
    area = width * height
    require((before[2] - before[0], before[3] - before[1]) == (width, height) and min(width, height) >= 160
            and .16 <= area / (sub.width * sub.height) <= .73
            and min(after[0], after[1], sub.width - after[2], sub.height - after[3]) >= 8,
            "Reported photo placement exceeds the supported bounds")
    with ref.crop(before) as earlier, sub.crop(after) as later, ImageChops.difference(earlier, later) as rgb:
        red, green, blue = rgb.split()
        with red, green, blue, ImageChops.lighter(red, green) as rg, ImageChops.lighter(rg, blue) as strongest:
            difference = strongest.point([0] + [1] * 255)
        with difference:
            exact = difference.histogram()[0]
            rectangle = difference.getbbox()
            require(exact >= .75 * area and (rectangle is None
                    or (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1]) <= .40 * area)
                    and proof["exact_pixels"] == exact and proof["native_difference_sha256"] == existing._hash(difference.tobytes()),
                    "Native inset pixel differences do not match the originals")
            with Image.new("L", sub.size) as full:
                full.paste(difference, after[:2])
                native = full.tobytes()
    cells = _anchor_cells(ref, before, after, sub)
    quadrants = {(row > 2, col > 2) for row, col, _ in cells if row != 2 and col != 2}
    require(proof["witness_cells"] == cells and len(cells) >= 12 and len(quadrants) == 4
            and min(len({cell[i] for cell in cells}) for i in (0, 1)) >= 4, "Inset matches differ from the originals")
    dx, dy = before[0] - after[0], before[1] - after[1]
    return previous._assets(stack, comparison, assets, ref, sub, (before, ref.size, None, native),
                            map_submitted_box=lambda box: existing._translation(box, dx, dy),
                            minimum=max(48, round(area * .0004)))


def review(export: Path, earlier: Path, submitted: Path) -> dict:
    document = json.loads(existing._read(export, existing.MAX_JSON_BYTES), object_pairs_hook=existing._record)
    existing._require(type(document) is dict and set(document) == {"comparison", "pngs"}, "Expected a comparison JSON export")
    comparison, assets = document["comparison"], document["pngs"]
    schema = comparison["schema"]
    if schema not in {SCHEMA, SCREENSHOT_SCHEMA}:
        return previous.review(export, earlier, submitted)
    existing._require(type(assets) is dict, "Expected PNG assets indexed by filename")
    with ExitStack() as stack:
        ref = existing._source(stack, existing._read(earlier, existing.MAX_SOURCE_BYTES), comparison["reference"])
        submitted_bytes = existing._read(submitted, existing.MAX_SOURCE_BYTES)
        sub = existing._source(stack, submitted_bytes, comparison["submitted"])
        carrier = stack.enter_context(Image.open(io.BytesIO(submitted_bytes), formats=["PNG", "JPEG"]))
        if schema == SCREENSHOT_SCHEMA:
            proof_field = "embedded_native_photo"
            retained = _screenshot_assets(stack, comparison, assets, ref, sub, carrier)
        else:
            proof_field = "exact_oriented_resize"
            projection = _projection(comparison, ref, sub, carrier)
            retained = previous._assets(stack, comparison, assets, ref, sub, projection,
                                        map_submitted_box=lambda box: existing._resize_bounds(
                                            box, ref, sub, comparison[proof_field]["projected_crop"],
                                        ))
    # Revisit the same bounded proposal search and preceding comparison paths.
    # The bytewise projection and evidence assets above have their own checks.
    from app.evidence.image_reference import inspect

    result = inspect(earlier, submitted, allow_proportional_resize=True, allow_oriented_crop=True,
                     allow_oriented_resize=True, allow_embedded_photo=schema == SCREENSHOT_SCHEMA)
    existing._require(result.get("schema") == schema and all(result[key] == comparison[key]
                      for key in ("reference", "submitted", proof_field)),
                      "Comparison differs from the unique supported search result")
    return {"reported_status": "comparable", "source_hashes_verified": 2,
            "crops_verified": 2 * len(comparison["regions"]), "mask_pixels_counted": retained}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "earlier", "submitted"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        result = review(args.export, args.earlier, args.submitted)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, Image.DecompressionBombError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
