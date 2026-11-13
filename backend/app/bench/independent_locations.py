"""Compute exact localization counts from private, source-coordinate annotations."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

from app.bench import independent

TRUTH_SCHEMA = "chipmunk.independent-location-truth/1"
MAX_REGIONS = 256
MAX_MASK_SIDE = 16_384
MAX_MASK_PIXELS = 64_000_000
MASK_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,123}\.pbm")


def intervals(regions):
    """Return the sorted disjoint union of half-open integer intervals."""
    merged = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    return merged


def measures(truth, reported):
    """Measure target, reported and intersection extents without expanding pixels."""
    def width(spans):
        return sum(end - start for start, end in spans)

    def overlap(first, second):
        i = j = area = 0
        while i < len(first) and j < len(second):
            start, end = first[i]
            left, right = second[j]
            area += max(0, min(end, right) - max(start, left))
            if end <= right:
                i += 1
            else:
                j += 1
        return area

    if not any(len(region) == 4 for region in (*truth, *reported)):
        first, second = intervals(truth), intervals(reported)
        return width(first), width(second), overlap(first, second)

    edges = sorted({edge for x0, _, x1, _ in (*truth, *reported) for edge in (x0, x1)})
    total = measured = matched = 0
    for left, right in pairwise(edges):
        first = intervals((y0, y1) for x0, y0, x1, y1 in truth if x0 <= left < x1)
        second = intervals((y0, y1) for x0, y0, x1, y1 in reported if x0 <= left < x1)
        columns = right - left
        total += columns * width(first)
        measured += columns * width(second)
        matched += columns * overlap(first, second)
    return total, measured, matched


def _regions(value, extent):
    if not isinstance(value, list) or len(value) > MAX_REGIONS:
        raise ValueError(f"Use a list of at most {MAX_REGIONS} locations per input")
    axes = len(extent)
    for item in value:
        if (not isinstance(item, list) or len(item) != 2 * axes or any(type(x) is not int for x in item)
                or any(not 0 <= item[i] < item[i + axes] <= limit for i, limit in enumerate(extent))):
            raise ValueError("Location must have positive extent within the source")
    return value


def _mask(value, extent, directory):
    if len(extent) != 2 or directory is None:
        raise ValueError("Image masks need image extent and an explicitly supplied mask directory")
    width, height = extent
    if max(extent) > MAX_MASK_SIDE or width * height > MAX_MASK_PIXELS:
        raise ValueError("Image mask exceeds the supported dimensions")
    if (not isinstance(value, dict) or value.keys() != {"file", "sha256"}
            or not isinstance(value["file"], str) or not MASK_FILE.fullmatch(value["file"])
            or not independent._hash(value["sha256"])):
        raise ValueError("Image mask needs a PBM basename and SHA-256")
    path = directory / value["file"]
    if path.is_symlink() or not path.is_file():
        raise ValueError("Image mask must be a regular file in the supplied directory")
    header = f"P4\n{width} {height}\n".encode("ascii")
    row_bytes = (width + 7) // 8
    length = len(header) + row_bytes * height
    with path.open("rb") as handle:
        body = handle.read(length + 1)
    if (len(body) != length or not body.startswith(header)
            or hashlib.sha256(body).hexdigest() != value["sha256"]):
        raise ValueError("Image mask bytes, dimensions or SHA-256 differ")
    padding = 8 * row_bytes - width
    rows = tuple(int.from_bytes(body[start:start + row_bytes], "big")
                 for start in range(len(header), length, row_bytes))
    if padding and any(row & ((1 << padding) - 1) for row in rows):
        raise ValueError("Image mask must have zero row padding bits")
    return tuple(row >> padding for row in rows)


def _location(row, extent, directory):
    if "mask" in row:
        return _mask(row["mask"], extent, directory)
    return _regions(row["regions"], extent)


def _area(truth, reported, extent):
    if not isinstance(truth, tuple) and not isinstance(reported, tuple):
        return measures(truth, reported)

    def packed(location):
        if isinstance(location, tuple):
            return location
        width, height = extent
        rows = [0] * height
        for x0, y0, x1, y1 in location:
            bits = ((1 << (x1 - x0)) - 1) << (width - x1)
            for row in range(y0, y1):
                rows[row] |= bits
        return rows

    target = detected = intersection = 0
    for first, second in zip(packed(truth), packed(reported), strict=True):
        target += first.bit_count()
        detected += second.bit_count()
        intersection += (first & second).bit_count()
    return target, detected, intersection


def _truth_rows(document, expected, manifest_hash, split):
    if (not isinstance(document, dict) or document.get("schema") != TRUTH_SCHEMA
            or document.get("manifest_sha256") != manifest_hash or document.get("split") != split
            or not isinstance(document.get("cases"), list)):
        raise ValueError("Location truth must commit to the exact manifest and split")
    rows = {}
    keys = {"id", "input_sha256", "extent"}
    for row in document["cases"]:
        if (not isinstance(row, dict) or row.keys() not in (keys | {"regions"}, keys | {"mask"})
                or not isinstance(row["id"], str) or row["id"] not in expected or row["id"] in rows):
            raise ValueError("Unexpected or duplicate planned location truth")
        case = expected[row["id"]]
        extent = row["extent"]
        axes = 2 if case["modality"] == "image" else 1
        if (row["input_sha256"] != case["input_sha256"] or not isinstance(extent, list) or len(extent) != axes
                or any(type(n) is not int or n <= 0 for n in extent)):
            raise ValueError("Location truth differs from the declared input or source extent")
        source_units = extent[0] * extent[1] if axes == 2 else extent[0]
        if source_units != case["localization_truth"]["source_units"]:
            raise ValueError("Location truth source extent differs from the manifest")
        rows[row["id"]] = row
    if rows.keys() != expected.keys():
        raise ValueError("Every planned localization needs exact location truth")
    return rows


def _count_locations(manifest, truth, response, mask_directory, *, retain_locations=False):
    audit = independent.audit_manifest(manifest)
    split = response.get("split")
    if split not in independent.SPLITS or response.get("manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("Response does not match the declared manifest and split")
    expected = {row["id"]: row for row in manifest["cases"]
                if row["split"] == split and "localization_truth" in row}
    if not expected:
        raise ValueError("The split has no planned localizations")
    directory = mask_directory.resolve(strict=True) if mask_directory is not None else None
    if directory is not None and not directory.is_dir():
        raise ValueError("Image mask directory is not a directory")
    rows = _truth_rows(truth, expected, audit["manifest_sha256"], split)
    if not isinstance(response.get("predictions"), list):
        raise ValueError("Missing planned predictions")
    result = deepcopy(response)
    exact_locations = {}
    for row in result["predictions"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Expected a response with a case ID")
        if row["id"] not in rows:
            if "localization" in row:
                raise ValueError("Unexpected localization outcome without planned truth")
            continue
        location = row.get("localization")
        if not isinstance(location, dict) or location.get("status") not in independent.STATUSES:
            raise ValueError("Every planned localization needs an explicit outcome")
        truth_row = rows[row["id"]]
        extent = truth_row["extent"]
        reference = _location(truth_row, extent, directory)
        target_units = (sum(value.bit_count() for value in reference) if isinstance(reference, tuple)
                        else measures(reference, [])[0])
        if target_units != expected[row["id"]]["localization_truth"]["target_units"]:
            raise ValueError("Location truth union differs from the declared target extent")
        if location["status"] != "ok":
            if location.keys() != {"status", "reason"}:
                raise ValueError("Unassessed locations need a reason without coordinates")
            continue
        if location.keys() not in ({"status", "regions"}, {"status", "mask"}):
            raise ValueError("Measured locations need exact regions or an image mask")
        detected = _location(location, extent, directory)
        _, reported, overlap = _area(reference, detected, extent)
        if retain_locations:
            exact_locations[row["id"]] = (extent, location, directory)
        row["localization"] = {"status": "ok", "reported_units": reported, "overlap_units": overlap}
    independent.audit_receipt(manifest, result, split)
    return result, exact_locations


def count_locations(manifest, truth, response, *, mask_directory: Path | None = None):
    """Convert local annotations to a portable receipt while stripping location coordinates."""
    return _count_locations(manifest, truth, response, mask_directory)[0]
