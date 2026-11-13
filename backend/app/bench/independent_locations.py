"""Compute exact localization counts from private, source-coordinate annotations."""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise

from app.bench import independent

TRUTH_SCHEMA = "chipmunk.independent-location-truth/1"
MAX_REGIONS = 256


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


def _truth_rows(document, expected, manifest_hash, split):
    if (not isinstance(document, dict) or document.get("schema") != TRUTH_SCHEMA
            or document.get("manifest_sha256") != manifest_hash or document.get("split") != split
            or not isinstance(document.get("cases"), list)):
        raise ValueError("Location truth must commit to the exact manifest and split")
    rows = {}
    for row in document["cases"]:
        if (not isinstance(row, dict) or row.keys() != {"id", "input_sha256", "extent", "regions"}
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
        regions = _regions(row["regions"], extent)
        if measures(regions, [])[0] != case["localization_truth"]["target_units"]:
            raise ValueError("Location truth union differs from the declared target extent")
        rows[row["id"]] = (extent, regions)
    if rows.keys() != expected.keys():
        raise ValueError("Every planned localization needs exact location truth")
    return rows


def count_locations(manifest, truth, response):
    """Convert local annotations to a portable receipt while stripping location coordinates."""
    audit = independent.audit_manifest(manifest)
    split = response.get("split")
    if split not in independent.SPLITS or response.get("manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("Response does not match the declared manifest and split")
    expected = {row["id"]: row for row in manifest["cases"]
                if row["split"] == split and "localization_truth" in row}
    if not expected:
        raise ValueError("The split has no planned localizations")
    rows = _truth_rows(truth, expected, audit["manifest_sha256"], split)
    if not isinstance(response.get("predictions"), list):
        raise ValueError("Missing planned predictions")
    result = deepcopy(response)
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
        if location["status"] != "ok":
            if location.keys() != {"status", "reason"}:
                raise ValueError("Unassessed locations need a reason without coordinates")
            continue
        if location.keys() != {"status", "regions"}:
            raise ValueError("Measured locations need exact regions")
        extent, reference = rows[row["id"]]
        _, reported, overlap = measures(reference, _regions(location["regions"], extent))
        row["localization"] = {"status": "ok", "reported_units": reported, "overlap_units": overlap}
    independent.audit_receipt(manifest, result, split)
    return result
