# SPDX-License-Identifier: Apache-2.0
"""Verify an earlier-copy export against both sources, including an oriented native crop."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_image_reference_export as existing  # noqa: E402
from review_image_reference_oriented_crop_transfer import offsets, projections  # noqa: E402

SCHEMA = "chipmunk.image-reference-comparison/7"
MAX_REFERENCE_SIDE = 1024
MAX_REVIEW_OFFSETS = 128


def _components(difference: bytes, width: int, height: int) -> list[tuple[tuple[int, int, int, int], list[int]]]:
    remaining = bytearray(difference)
    minimum = max(48, round(width * height * .0004))
    components = []
    for start, selected in enumerate(remaining):
        if not selected:
            continue
        pending = [start]
        remaining[start] = 0
        left, top = width, height
        right = bottom = 0
        for index in pending:
            y, x = divmod(index, width)
            left, top = min(left, x), min(top, y)
            right, bottom = max(right, x + 1), max(bottom, y + 1)
            for row in range(max(0, y - 1), min(height, y + 2)):
                for col in range(max(0, x - 1), min(width, x + 2)):
                    adjacent = row * width + col
                    if remaining[adjacent]:
                        remaining[adjacent] = 0
                        pending.append(adjacent)
        if len(pending) >= minimum:
            components.append(((left, top, right, bottom), pending))
    return sorted(components, key=lambda item: -len(item[1]))


def _projection(comparison, ref, sub, carrier):
    require = existing._require
    submitted_area, earlier_area = sub.width * sub.height, ref.width * ref.height
    require(comparison["status"] == "comparable" and comparison["coordinate_space"] == "exif-oriented-submitted"
            and "alignment" not in comparison and carrier.format == "PNG" and carrier.mode == "RGB"
            and max(ref.size) <= MAX_REFERENCE_SIDE and min(sub.size) >= 160
            and 55 * earlier_area <= 100 * submitted_area <= 90 * earlier_area, "Unsupported oriented crop scope")
    boxes = list(existing._textured_boxes(sub))
    for operation in (None, *existing.ORIENTATIONS.values()):
        with ref.copy() if operation is None else ref.transpose(operation) as oriented:
            require(len(offsets(oriented, sub, boxes)) <= MAX_REVIEW_OFFSETS, "Too many competing crop offsets")
    matches = list(projections(ref, sub))
    require(len(matches) == 1 and matches[0][0] in existing.ORIENTATIONS, "The crop orientation and placement must be unique")
    transform, crop, projected_size, rgb_sha, difference, witnesses = matches[0]
    proof = comparison["exact_oriented_crop"]
    operation = existing.ORIENTATIONS[transform]
    inverse = {Image.Transpose.ROTATE_90: Image.Transpose.ROTATE_270,
               Image.Transpose.ROTATE_270: Image.Transpose.ROTATE_90}.get(operation, operation)
    require(proof["transform"] == transform and existing._bounds(proof["projected_crop"], projected_size) == crop
            and existing._bounds(proof["native_crop"], ref.size) == existing._orientation_bounds(crop, projected_size, inverse)
            and proof["matched_pixels"] == submitted_area - sum(difference) and proof["matched_tiles"] == witnesses
            and proof["projected_rgb_sha256"] == rgb_sha and proof["difference_mask_sha256"] == existing._hash(difference),
            "Exact oriented crop measurements differ from the originals")
    return crop, projected_size, inverse, difference


def _assets(stack, comparison, assets, ref, sub, projection, *, map_submitted_box=None):
    require = existing._require
    crop, projected_size, inverse, difference = projection
    components = _components(difference, *sub.size)
    regions = comparison["regions"]
    kept = min(len(components), existing.MAX_REGIONS)
    require(type(regions) is list and len(regions) == kept
            and comparison["additional_regions"] == len(components) - kept, "Difference region selection differs from the native mask")
    if not regions:
        require(not assets and "mask" not in comparison, "An empty comparison contains unbound evidence")
        return 0

    mask = existing._image(stack, assets, comparison["mask"], "difference-mask.png", "L", sub.size)
    mask_pixels = mask.tobytes()
    expected_mask = bytearray(len(difference))
    areas = [len(indices) for _, indices in components[:kept]]
    for index, (region, area) in enumerate(zip(regions, areas, strict=True), 1):
        box = existing._bounds(region, sub.size)
        earlier_box = existing._bounds(region["reference_region"], ref.size)
        shifted = (map_submitted_box(box) if map_submitted_box else tuple(value + crop[i % 2] for i, value in enumerate(box)))
        require(earlier_box == existing._orientation_bounds(shifted, projected_size, inverse)
                and region["selected_pixels"] == area, "Retained region coordinates or count differ")
        component = next((item for item in components if item[0] == box and len(item[1]) == area
                          and all(mask_pixels[pixel] == 255 for pixel in item[1])), None)
        require(component is not None, "The retained difference mask does not contain the exact component")
        components.remove(component)
        for pixel in component[1]:
            expected_mask[pixel] = 255
        for label, image, source_box in (("reference", ref, earlier_box), ("submitted", sub, box)):
            x, y, right, bottom = source_box
            png = existing._image(stack, assets, region[f"{label}_crop"], f"{label}-{index:02d}.png", "RGB", (right - x, bottom - y))
            with image.crop(source_box) as original:
                require(png.tobytes() == original.tobytes(), "Native evidence crop differs from the source")
        width, height = box[2] - box[0], box[3] - box[1]
        overlay = existing._image(stack, assets, region["submitted_overlay"], f"submitted-overlay-{index:02d}.png", "RGBA", (width, height))
        with Image.new("RGBA", (width, height)) as expected, mask.crop(box) as local:
            expected.paste((255, 173, 46, 144), (0, 0, width, height), local)
            require(overlay.tobytes() == expected.tobytes(), "Native evidence overlay differs from the mask")
    require(mask_pixels == expected_mask and not assets, "Difference mask or evidence inventory differs")
    return sum(areas)


def review(export: Path, earlier: Path, submitted: Path) -> dict:
    document = json.loads(existing._read(export, existing.MAX_JSON_BYTES), object_pairs_hook=existing._record)
    existing._require(type(document) is dict and set(document) == {"comparison", "pngs"}, "Expected a comparison JSON export")
    comparison, assets = document["comparison"], document["pngs"]
    if comparison["schema"] != SCHEMA:
        return existing.review(export, earlier, submitted)
    existing._require(type(assets) is dict, "Expected PNG assets indexed by filename")
    with ExitStack() as stack:
        ref = existing._source(stack, existing._read(earlier, existing.MAX_SOURCE_BYTES), comparison["reference"])
        submitted_bytes = existing._read(submitted, existing.MAX_SOURCE_BYTES)
        sub = existing._source(stack, submitted_bytes, comparison["submitted"])
        carrier = stack.enter_context(Image.open(io.BytesIO(submitted_bytes), formats=["PNG", "JPEG"]))
        projection = _projection(comparison, ref, sub, carrier)
        retained = _assets(stack, comparison, assets, ref, sub, projection)
    return {"reported_status": "comparable", "source_hashes_verified": 2,
            "crops_verified": 2 * len(comparison["regions"]), "mask_pixels_counted": retained}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "earlier", "submitted"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        result = review(args.export, args.earlier, args.submitted)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, Image.DecompressionBombError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
