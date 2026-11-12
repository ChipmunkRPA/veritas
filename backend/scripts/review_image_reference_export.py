"""Verify the crops, mask and geometry of an earlier-copy browser export offline."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import stat
import sys
from contextlib import ExitStack
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

MAX_SOURCE_BYTES = 60 * 1024 ** 2
MAX_PNG_BYTES = 32 * 1024 ** 2
MAX_JSON_BYTES = 4 * ((MAX_PNG_BYTES + 2) // 3) + 1024 ** 2
MAX_PIXELS = 4_000_000
MAX_REGIONS = 5
RESIZE_SCHEMA = "chipmunk.image-reference-comparison/3"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(path, limit):
    _require(stat.S_ISREG(path.stat().st_mode), "Expected a regular local file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    _require(0 < len(data) <= limit, "An input is empty or exceeds the local review size limit")
    return data


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _record(pairs):
    record = dict(pairs)
    _require(len(record) == len(pairs), "Comparison JSON contains duplicate fields")
    return record


def _source(stack, data, record):
    _require(record["sha256"] == _hash(data), "Original source SHA-256 differs")
    raw = stack.enter_context(Image.open(io.BytesIO(data), formats=["PNG", "JPEG"]))
    _require(getattr(raw, "n_frames", 1) == 1 and raw.width * raw.height <= MAX_PIXELS,
             "Unsupported original source dimensions or frames")
    with ImageOps.exif_transpose(raw) as oriented:
        image = stack.enter_context(oriented.convert("RGB"))
    dimensions = record["width"], record["height"]
    _require(all(type(item) is int for item in dimensions) and image.size == dimensions,
             "Original source dimensions differ after orientation")
    return image


def _bounds(region, size):
    x, y, w, h = (region[field] for field in ("x", "y", "w", "h"))
    _require(all(type(item) is int for item in (x, y, w, h)), "Coordinates must be integers")
    right, bottom = x + w, y + h
    _require(0 <= x < right <= size[0] and 0 <= y < bottom <= size[1], "Coordinates are outside the recorded original")
    return x, y, right, bottom


def _contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _translation(bounds, dx, dy):
    x, y, right, bottom = bounds
    return x + dx, y + dy, right + dx, bottom + dy


def _same_pixels(reference, submitted):
    if reference.size != submitted.size:
        return False
    with ImageChops.difference(reference, submitted) as difference:
        return not difference.getbbox()


def _number(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


def _check_alignment_support(alignment, overlap, earlier, reference, submitted, dx, dy):
    width, height = overlap[2] - overlap[0], overlap[3] - overlap[1]
    source_area = max(reference.width * reference.height, submitted.width * submitted.height)
    _require(min(width, height) >= 160 and 5 * width * height >= 4 * source_area,
             "Reported alignment does not meet the overlap requirement")
    tiles = alignment["supporting_tiles"]
    features = alignment["matching_features"]
    _require(type(features) is int and features >= 18 and alignment["tested_tiles"] == 9
             and type(tiles) is list and 6 <= len(tiles) <= 9, "Reported alignment has insufficient recorded support")
    positions, side_lengths = set(), set()
    for tile in tiles:
        x, y, size = tile["x"], tile["y"], tile["size"]
        bounds = _bounds({"x": x, "y": y, "w": size, "h": size}, submitted.size)
        fractional = tile["fractional_offset"]
        _require(_contains(overlap, bounds) and _contains(earlier, _translation(bounds, dx, dy))
                 and 32 <= size <= 96 and type(tile["dx"]) is int and type(tile["dy"]) is int
                 and (tile["dx"], tile["dy"]) == (dx, dy) and _number(tile["correlation"], .83, 1)
                 and _number(tile["next_peak_difference"], .012, 2)
                 and type(fractional) is list and len(fractional) == 2 and all(_number(n, -.35, .35) for n in fractional),
                 "A recorded alignment tile falls outside the comparison geometry or support bounds")
        positions.add((x, y))
        side_lengths.add(size)
    _require(len(positions) == len(tiles) and len(side_lengths) == 1 and len({x for x, _ in positions}) == 3
             and len({y for _, y in positions}) == 3, "Reported alignment tiles do not span the three-by-three grid")


def _geometry(comparison, reference, submitted):
    schema = comparison["schema"]
    same_pixels = _same_pixels(reference, submitted)
    if schema == "chipmunk.image-reference-comparison/1":
        _require("alignment" not in comparison and comparison["coordinate_space"] == "exif-oriented-original",
                 "The unshifted record has an unexpected coordinate space")
        expected_status = ("comparable" if min(reference.size) >= 64 else "unsupported_dimensions") if same_pixels else (
            "integer_alignment_unresolved")
        _require(not comparison["regions"] and comparison["status"] == expected_status,
                 "The unshifted browser comparison conflicts with the source pixels or status")
        return 0, 0, (0, 0, submitted.width, submitted.height)
    _require(schema == "chipmunk.image-reference-comparison/2" and comparison["coordinate_space"] == "exif-oriented-submitted",
             "Unsupported comparison schema or coordinate space")
    _require(not same_pixels and comparison["status"] in {"comparable", "unaligned", "insufficient_common_background",
                                                           "uncertain_delivery", "partial_jpeg_delivery_unassessed"},
             "The aligned browser comparison conflicts with the source pixels or status")
    alignment = comparison["alignment"]
    _require(alignment["method"] == "corroborated_integer_translation", "Unsupported alignment method")
    offset = alignment["reference_minus_submitted"]
    dx, dy = offset["x"], offset["y"]
    _require(type(dx) is int and type(dy) is int, "Alignment offsets must be integers")
    overlap = _bounds(alignment["submitted_overlap"], submitted.size)
    earlier = _bounds(alignment["reference_overlap"], reference.size)
    expected = (max(0, -dx), max(0, -dy), min(submitted.width, reference.width - dx), min(submitted.height, reference.height - dy))
    _require(overlap == expected and earlier == _translation(overlap, dx, dy), "Overlap geometry conflicts with the source offsets")
    _check_alignment_support(alignment, overlap, earlier, reference, submitted, dx, dy)
    return dx, dy, overlap


def _resize_replay(comparison, ref, sub, carrier):
    """Recompute v3 directly from the two sources, separate from evidence generation."""
    try:
        import numpy as np
        from cv2 import error as OpenCVError

        from app.detectors.image import reference_alignment, reference_resize
    except ImportError as error:
        raise ValueError("Use the backend Python environment and run this reviewer as a module for resize exports") from error
    _require(comparison["coordinate_space"] == "exif-oriented-submitted" and ref.size != sub.size
             and "alignment" not in comparison, "Resize comparison has unexpected source dimensions or coordinates")
    earlier, submitted = np.asarray(ref), np.asarray(sub)
    try:
        aligned, _ = reference_alignment.compare(earlier, submitted, carrier)
        _require(aligned["status"] == "integer_alignment_unresolved", "Resize comparison must follow the integer alignment check")
        expected, mask = reference_resize.compare(earlier, submitted, carrier)
    except OpenCVError as error:
        raise ValueError("The image comparison could not be recomputed") from error
    measured_fields = {"status", "regions", "additional_regions", "supported_resize_kernels", "framing_check",
                       "submitted_over_reference_xy"}
    measured = {field: value for field, value in comparison.items() if field in measured_fields}
    measured["regions"] = [{key: value for key, value in region.items()
                            if key not in {"reference_crop", "submitted_crop", "submitted_overlay"}}
                           for region in measured["regions"]]
    _require(json.dumps(measured, sort_keys=True) == json.dumps(expected, sort_keys=True),
             "Resize comparison differs from recomputed status, framing or regions")
    return mask


def _resize_bounds(box, reference, submitted):
    left, top, right, bottom = box
    x_scale = Fraction(reference.width, submitted.width)
    y_scale = Fraction(reference.height, submitted.height)
    return floor(left * x_scale), floor(top * y_scale), ceil(right * x_scale), ceil(bottom * y_scale)


def _image(stack, blobs, asset, filename, mode, dimensions):
    _require(asset["file"] == filename and filename in blobs, "A named evidence PNG is missing or inconsistent")
    source = blobs.pop(filename)
    _require(type(source) is str and len(source) <= 4 * ((MAX_PNG_BYTES + 2) // 3), "Invalid encoded evidence PNG")
    data = base64.b64decode(source, validate=True)
    _require(_hash(data) == asset["sha256"], "Evidence PNG SHA-256 differs")
    image = stack.enter_context(Image.open(io.BytesIO(data), formats=["PNG"]))
    _require(image.size == dimensions == (asset["width"], asset["height"]) and image.mode == mode and image.n_frames == 1,
             "Evidence PNG dimensions, pixel format or frame count differ")
    return image


def review(export, earlier, submitted):
    """Check exact source and crop bytes, reported mask geometry and v3 scoring."""
    document = json.loads(_read(export, MAX_JSON_BYTES), object_pairs_hook=_record)
    _require(type(document) is dict and set(document) == {"comparison", "pngs"}, "Expected the browser comparison JSON")
    comparison, assets = document["comparison"], document["pngs"]
    _require(type(assets) is dict, "Expected PNG assets indexed by filename")
    regions = comparison["regions"]
    _require(type(regions) is list and len(regions) <= MAX_REGIONS, "Invalid difference region count")
    status = comparison["status"]
    _require(status in {"comparable", "integer_alignment_unresolved", "different_dimensions", "unsupported_dimensions",
                        "unaligned", "insufficient_common_background", "uncertain_delivery", "partial_jpeg_delivery_unassessed",
                        "unsupported_resize_geometry", "same_framing_unresolved", "resize_delivery_unresolved"},
             "Unknown comparison status")
    _require(not regions or status == "comparable", "An unassessed comparison contains difference regions")
    with ExitStack() as stack:
        ref = _source(stack, _read(earlier, MAX_SOURCE_BYTES), comparison["reference"])
        submitted_bytes = _read(submitted, MAX_SOURCE_BYTES)
        sub = _source(stack, submitted_bytes, comparison["submitted"])
        resized = comparison["schema"] == RESIZE_SCHEMA
        if resized:
            carrier = stack.enter_context(Image.open(io.BytesIO(submitted_bytes), formats=["PNG", "JPEG"]))
            expected_mask = _resize_replay(comparison, ref, sub, carrier)
            overlap = (0, 0, sub.width, sub.height)
        else:
            dx, dy, overlap = _geometry(comparison, ref, sub)
        if not regions:
            _require(not assets and "mask" not in comparison, "An empty comparison contains unbound image evidence")
            return {"reported_status": status, "source_hashes_verified": 2, "crops_verified": 0, "mask_pixels_counted": 0}
        mask = _image(stack, assets, comparison["mask"], "difference-mask.png", "L", sub.size)
        mask_counts = mask.histogram()
        _require(mask_counts[0] + mask_counts[255] == sub.width * sub.height, "Difference mask is not binary")
        if resized:
            _require(expected_mask is not None and mask.tobytes() == (expected_mask.astype("uint8") * 255).tobytes(),
                     "Difference mask pixels differ from the recomputed resize comparison")
        covered = stack.enter_context(Image.new("L", sub.size))
        drawn = ImageDraw.Draw(covered)
        retained = 0
        for index, region in enumerate(regions, 1):
            box = _bounds(region, sub.size)
            _require(_contains(overlap, box), "Region extends outside the recorded overlap")
            earlier_box = _bounds(region.get("reference_region", region), ref.size)
            expected_box = _resize_bounds(box, ref, sub) if resized else _translation(box, dx, dy)
            _require(earlier_box == expected_box, "Region coordinates conflict with the source mapping")
            width, height = box[2] - box[0], box[3] - box[1]
            count = region["selected_pixels"]
            with mask.crop(box) as local_mask:
                _require(type(count) is int and 0 < count <= local_mask.histogram()[255], "Region pixel count differs from the mask")
                _require(local_mask.getbbox() == (0, 0, width, height), "Region does not bound its mask pixels")
                if "submitted_overlay" in region:
                    overlay = _image(stack, assets, region["submitted_overlay"], f"submitted-overlay-{index:02d}.png",
                                     "RGBA", (width, height))
                    with Image.new("RGBA", (width, height)) as expected:
                        expected.paste((255, 173, 46, 144), (0, 0, width, height), local_mask)
                        _require(overlay.tobytes() == expected.tobytes(), "Difference overlay pixels differ from the native mask")
            retained += count
            drawn.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
            for field, original, original_box in (("reference", ref, earlier_box), ("submitted", sub, box)):
                x, y, right, bottom = original_box
                png = _image(stack, assets, region[f"{field}_crop"], f"{field}-{index:02d}.png", "RGB", (right - x, bottom - y))
                with original.crop(original_box) as expected, ImageChops.difference(png, expected) as difference:
                    _require(not difference.getbbox(), "Evidence crop pixels differ from the original source")
        with ImageChops.subtract(mask, covered) as outside:
            _require(retained == mask_counts[255] and not outside.getbbox(), "Mask pixels fall outside the declared regions or count")
    _require(not assets, "Unexpected evidence PNGs are present")
    return {"reported_status": status, "source_hashes_verified": 2, "crops_verified": 2 * len(regions),
            "mask_pixels_counted": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("export", "earlier", "submitted"):
        parser.add_argument(field, type=Path)
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
