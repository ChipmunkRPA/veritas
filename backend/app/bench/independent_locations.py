"""Compute exact localization counts from private, source-coordinate annotations."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

from app.bench import independent

TRUTH_SCHEMA = "chipmunk.independent-location-truth/1"
EVENTS_SCHEMA = "chipmunk.independent-location-events/1"
EVENT_COMPARISON_SCHEMA = "chipmunk.independent-event-comparison/1"
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


def _count_locations(manifest, truth, response, mask_directory, *, retain_locations=False, collect_events=None):
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
            if collect_events is not None:
                collect_events(expected[row["id"]], reference, None)
            continue
        if location.keys() not in ({"status", "regions"}, {"status", "mask"}):
            raise ValueError("Measured locations need exact regions or an image mask")
        detected = _location(location, extent, directory)
        _, reported, overlap = _area(reference, detected, extent)
        if collect_events is not None:
            collect_events(expected[row["id"]], reference, detected)
        if retain_locations:
            exact_locations[row["id"]] = (extent, location, directory)
        row["localization"] = {"status": "ok", "reported_units": reported, "overlap_units": overlap}
    independent.audit_receipt(manifest, result, split)
    return result, exact_locations


def count_locations(manifest, truth, response, *, mask_directory: Path | None = None):
    """Convert local annotations to a portable receipt while stripping location coordinates."""
    return _count_locations(manifest, truth, response, mask_directory)[0]


def _event_assignment(reference, detected, minimum_iou, minimum_recall):
    # Eligible pairs form a bipartite graph. Reassign earlier matches when
    # needed so that a broad proposal cannot block two narrower matches.
    edges = []
    for target in reference:
        compatible = []
        for index, proposal in enumerate(detected):
            target_units, reported_units, overlap = measures([target], [proposal])
            if overlap / (target_units + reported_units - overlap) >= minimum_iou and overlap / target_units >= minimum_recall:
                compatible.append(index)
        edges.append(compatible)
    assigned = {}

    def assign(target, visited):
        for proposal in edges[target]:
            if proposal in visited:
                continue
            visited.add(proposal)
            if proposal not in assigned or assign(assigned[proposal], visited):
                assigned[proposal] = target
                return True
        return False

    for target in range(len(reference)):
        assign(target, set())
    return edges, assigned


def _event_membership(reference, proposals, minimum_iou, minimum_recall):
    edges, assigned = _event_assignment(reference, proposals, minimum_iou, minimum_recall)
    matched = set(assigned.values())
    # Starting at an unmatched target, an alternating path can swap which
    # endpoint is located without changing the maximum number of matches.
    optional = {target for target, options in enumerate(edges) if options and target not in matched}
    frontier = list(optional)
    while frontier:
        target = frontier.pop()
        for proposal in edges[target]:
            other = assigned[proposal]  # A reachable unused proposal would improve the maximum matching.
            if other not in optional:
                optional.add(other)
                frontier.append(other)
    return len(matched), matched - optional, optional


def _event_limits(minimum_iou, minimum_recall):
    if any(type(value) not in (float, int) or not 0 < value <= 1 for value in (minimum_iou, minimum_recall)):
        raise ValueError("Minimum event IoU and recall must each be greater than zero and at most one")


def _unique_events(reference):
    if len({tuple(event) for event in reference}) != len(reference):
        raise ValueError("Event truth contains identical duplicate annotations")


def count_events(manifest, truth, response, *, minimum_iou, minimum_recall, mask_directory: Path | None = None):
    """Summarize separate annotations and proposals without emitting their coordinates."""
    _event_limits(minimum_iou, minimum_recall)
    grouped = defaultdict(Counter)
    groups = defaultdict(set)

    def collect(case, reference, detected):
        stratum = case["modality"], case["condition"], case["label"]
        group = grouped[stratum]
        group["planned"] += 1
        groups[stratum].add(case["group"])
        if isinstance(reference, tuple) or isinstance(detected, tuple):
            group["unavailable_instance_geometry"] += 1
            return
        _unique_events(reference)
        group["target_events"] += len(reference)
        if detected is None:
            group["unassessed"] += 1
            return
        group["assessed"] += 1
        distinct_proposals = list(dict.fromkeys(map(tuple, detected)))
        _, assigned = _event_assignment(reference, distinct_proposals, minimum_iou, minimum_recall)
        matched = len(assigned)
        group["reported_events"] += len(detected)
        group["matched_events"] += matched
        group["complete_positive_inputs"] += bool(reference) and matched == len(reference)

    _count_locations(manifest, truth, response, mask_directory, collect_events=collect)
    strata = []
    fields = ("planned", "assessed", "unassessed", "unavailable_instance_geometry", "target_events",
              "reported_events", "matched_events", "complete_positive_inputs")
    for (modality, condition, label), counts in sorted(grouped.items()):
        strata.append({"modality": modality, "condition": condition, "label": label,
                       "source_groups": len(groups[modality, condition, label]),
                       **{key: counts[key] for key in fields},
                       "missed_events": counts["target_events"] - counts["matched_events"],
                       "unmatched_proposals": counts["reported_events"] - counts["matched_events"]})
    return {"schema": EVENTS_SCHEMA, "manifest_sha256": independent.digest(manifest),
            "truth_sha256": independent.digest(truth), "response_sha256": independent.digest(response),
            "split": response["split"], "minimum_iou": minimum_iou, "minimum_recall": minimum_recall,
            "matching": "maximum one-to-one matches", "strata": strata}


def _record_events(counts, reference, reported, minimum_iou, minimum_recall):
    if reported is None:
        counts["unassessed"] += 1
        return set(), set()
    proposals = sorted({tuple(item) for item in reported})
    matches, certain, optional = _event_membership(reference, proposals, minimum_iou, minimum_recall)
    counts.update(assessed=1, flagged_inputs=bool(reported), reported_events=len(reported), matched_events=matches,
                  unmatched_proposals=len(reported) - matches, complete_positive_inputs=bool(reference) and matches == len(reference))
    return certain, optional


def compare_events(manifest, truth, baseline, candidate, *, minimum_iou, minimum_recall, mask_directory: Path | None = None):
    """Compare exact edit coverage, including the union of two sets of proposals."""
    _event_limits(minimum_iou, minimum_recall)
    prior = {}

    def capture(case, reference, detected):
        prior[case["id"]] = detected

    _count_locations(manifest, truth, baseline, mask_directory, collect_events=capture)
    if candidate.get("split") != baseline["split"]:
        raise ValueError("Both event readings must use the same split")
    strata = defaultdict(lambda: (set(), Counter(), defaultdict(Counter), Counter()))

    def collect(case, reference, detected):
        earlier = prior[case["id"]]
        key = case["modality"], case["condition"], case["label"]
        groups, totals, methods, transitions = strata[key]
        groups.add(case["group"])
        totals["planned"] += 1
        if any(isinstance(reading, tuple) for reading in (reference, earlier, detected)):
            totals["unavailable_instance_geometry"] += 1
            return
        _unique_events(reference)
        totals["target_events"] += len(reference)
        before, before_optional = _record_events(methods["baseline"], reference, earlier, minimum_iou, minimum_recall)
        after, after_optional = _record_events(methods["candidate"], reference, detected, minimum_iou, minimum_recall)
        combined = None
        if earlier is not None or detected is not None:
            combined = sorted({tuple(item) for item in (*(earlier or []), *(detected or []))})
        _record_events(methods["combined"], reference, combined, minimum_iou, minimum_recall)
        uncertain = before_optional | after_optional
        transitions.update(ambiguous_match=len(uncertain), retained=len((before & after) - uncertain),
                           baseline_only=len((before - after) - uncertain), candidate_only=len((after - before) - uncertain),
                           missed_by_both=len(reference) - len(before | after | uncertain))

    _count_locations(manifest, truth, candidate, mask_directory, collect_events=collect)
    fields = ("assessed", "unassessed", "flagged_inputs", "reported_events", "matched_events", "unmatched_proposals", "complete_positive_inputs")
    comparisons = []
    for (modality, condition, label), (groups, totals, methods, transitions) in sorted(strata.items()):
        readings = {name: {**{key: methods[name][key] for key in fields},
                           "missed_events": totals["target_events"] - methods[name]["matched_events"]}
                    for name in ("baseline", "candidate", "combined")}
        comparisons.append({"modality": modality, "condition": condition, "label": label, "source_groups": len(groups),
                            "planned": totals["planned"], "target_events": totals["target_events"],
                            "unavailable_instance_geometry": totals["unavailable_instance_geometry"], **readings,
                            "event_transitions": {key: transitions[key] for key in (
                                "retained", "baseline_only", "candidate_only", "missed_by_both", "ambiguous_match")}})
    return {"schema": EVENT_COMPARISON_SCHEMA, "manifest_sha256": independent.digest(manifest),
            "truth_sha256": independent.digest(truth), "baseline_sha256": independent.digest(baseline),
            "candidate_sha256": independent.digest(candidate), "split": baseline["split"],
            "minimum_iou": minimum_iou, "minimum_recall": minimum_recall, "strata": comparisons}
