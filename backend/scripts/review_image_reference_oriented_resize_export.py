# SPDX-License-Identifier: Apache-2.0
"""Verify an earlier-copy comparison, including a rotated and resized crop."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_image_reference_oriented_crop_export as previous  # noqa: E402

existing = previous.existing
SCHEMA = "chipmunk.image-reference-comparison/8"
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


def review(export: Path, earlier: Path, submitted: Path) -> dict:
    document = json.loads(existing._read(export, existing.MAX_JSON_BYTES), object_pairs_hook=existing._record)
    existing._require(type(document) is dict and set(document) == {"comparison", "pngs"}, "Expected a comparison JSON export")
    comparison, assets = document["comparison"], document["pngs"]
    if comparison["schema"] != SCHEMA:
        return previous.review(export, earlier, submitted)
    existing._require(type(assets) is dict, "Expected PNG assets indexed by filename")
    with ExitStack() as stack:
        ref = existing._source(stack, existing._read(earlier, existing.MAX_SOURCE_BYTES), comparison["reference"])
        submitted_bytes = existing._read(submitted, existing.MAX_SOURCE_BYTES)
        sub = existing._source(stack, submitted_bytes, comparison["submitted"])
        carrier = stack.enter_context(Image.open(io.BytesIO(submitted_bytes), formats=["PNG", "JPEG"]))
        projection = _projection(comparison, ref, sub, carrier)
        retained = previous._assets(stack, comparison, assets, ref, sub, projection,
                                    map_submitted_box=lambda box: existing._resize_bounds(
                                        box, ref, sub, comparison["exact_oriented_resize"]["projected_crop"],
                                    ))
    # Revisit the same bounded proposal search and preceding comparison paths.
    # The bytewise projection and evidence assets above have their own checks.
    from app.evidence.image_reference import inspect

    result = inspect(earlier, submitted, allow_proportional_resize=True, allow_oriented_crop=True, allow_oriented_resize=True)
    existing._require(result.get("schema") == SCHEMA and all(result[key] == comparison[key]
                      for key in ("reference", "submitted", "exact_oriented_resize")),
                      "Oriented resize comparison differs from the unique supported search result")
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
