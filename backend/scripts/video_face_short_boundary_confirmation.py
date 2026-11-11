# SPDX-License-Identifier: Apache-2.0
"""Prospective check of the fixed one-second observed face-range refinement."""
from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

import cv2

from scripts import video_face_native_boundary as boundary
from scripts import video_face_texture_external as external
from scripts import video_face_texture_return as common

SOURCE_DIR = common.ROOT / "backend/data/independent-evaluation/video-common-sources-v89"
METADATA = (SOURCE_DIR / "plan.json", SOURCE_DIR / "permissions/rtfs-videos.csv")
SPAN_LIMIT = Fraction(1)
POINT = common.face_appearance.REVIEW_POINT
CAMERA_BOUNDS = (Fraction(3), Fraction(6))


def metadata():
    """Join the published original filename with its exact video tags."""
    with METADATA[1].open() as stream:
        by_filename = {row["filename"]: {item.strip().casefold() for item in row["tags"].split(";")}
                       for row in csv.DictReader(stream)}
    by_group = {}
    for group in common.read(METADATA[0])["groups"]:
        if group["source_family"] != "paired_face_swap":
            continue
        names = [Path(asset["source_path"]).name for asset in group["assets"]]
        matches = [by_filename[name] for name in names if name in by_filename]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous original video metadata")
        by_group[group["group"]] = {"tags": matches[0], "url": group["source_video_url"],
                                     "donors": set(group["donor_source_keys"] + group["donor_filenames"])}
    return by_group


def freeze(output, archive, texture_run, boundary_run):
    media = archive / "video-native-inserts-v95"
    prior_files = (texture_run / "freeze.json", boundary_run / "freeze.json", texture_run / "external-freeze.json")
    earlier = [common.read(path) for path in prior_files]
    old_indexes = {entry["index"] for study in earlier[:2] for entry in study["roster"]}
    old_camera_paths = {entry["relative"] for entry in earlier[2]["sources"] if entry["type"] == "camera"}
    source_metadata = metadata()
    past_metadata = [source_metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]]
                     for i in old_indexes]
    old_urls = {row["url"] for row in past_metadata}
    old_donors = set.union(*(row["donors"] for row in past_metadata))
    options = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        source = common.read(path)
        index = source["index"]
        if source["role"] != "confirmation" or index in old_indexes:
            continue
        tags = source_metadata[source["group"]]["tags"]
        if "head rotation" not in tags or "child" in tags:
            continue
        if source_metadata[source["group"]]["url"] in old_urls or source_metadata[source["group"]]["donors"] & old_donors:
            raise RuntimeError("Prior source or known donor overlaps held group")
        verify = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
        if not verify.exists():
            continue
        receipt_sha = common.digest(path)
        review = common.read(verify)
        if review["status"] != "complete" or review["private_group_receipt_sha256"] != receipt_sha:
            continue
        cases = source["cases"]
        if {case["condition"] for case in cases} != set(common.CONTROL_CONDITIONS + common.SWAP_CONDITIONS):
            raise RuntimeError("Incomplete moving source group")
        table = cases[0]["frame_table"]
        if table["width"] < 1280 or table["height"] < 720 or len(table["pts"]) < 144:
            continue
        case_dir = media / f"group-{index:03d}"
        if any((case_dir / case["file"]).stat().st_size != case["bytes"] for case in cases):
            continue
        options.append(("gesticulation" not in tags, sum(case["bytes"] for case in cases), index,
                        receipt_sha, common.digest(verify), source["group"]))
    if len(options) < 6:
        raise RuntimeError("Fewer than six disjoint head-rotation source groups")
    roster = []
    used_urls, used_donors = set(), set()
    for no_gesticulation, _, index, receipt, review, group in sorted(options)[:6]:
        row = source_metadata[group]
        if row["url"] in used_urls or used_donors & row["donors"]:
            raise RuntimeError("Known source links overlap within the prospective panel")
        used_urls.add(row["url"])
        used_donors.update(row["donors"])
        roster.append({"type": "moving", "index": index, "receipt_sha256": receipt, "review_sha256": review,
                       "head_rotation": True, "gesticulation": not no_gesticulation})
    camera = []
    for path in sorted((archive / external.CAMERA_DIR).glob("slot-*-attempt-01/qualification.json")):
        relative = str(path.relative_to(archive))
        if relative in old_camera_paths:
            continue
        receipt = common.read(path)
        by_condition = receipt["cases"]
        if set(by_condition) != {"unchanged", "chroma_conversion_only", "recompression_only"}:
            continue
        if any((path.parent / row["file"]).stat().st_size != row["bytes"] for row in by_condition.values()):
            continue
        camera.append((sum(row["bytes"] for row in by_condition.values()), relative, path))
    if len(camera) < 2:
        raise RuntimeError("Fewer than two disjoint ordinary camera groups")
    roster.extend({"type": "camera", "relative": relative, "sha256": common.digest(path)}
                  for _, relative, path in sorted(camera)[:2])
    common.write(output / "freeze.json", {
        "schema": "chipmunk.video-face-short-boundary-confirmation-freeze/1",
        "worker_sha256": common.digest(Path(__file__)), "bindings": common.bindings(),
        "boundary_worker_sha256": common.digest(Path(boundary.__file__)),
        "camera_worker_sha256": common.digest(Path(external.__file__)),
        "prior_studies_sha256": [common.digest(path) for path in prior_files],
        "metadata_sha256": [common.digest(path) for path in METADATA],
        "roster": roster,
        "selection": "six unused confirmation groups with head rotation tag, prefer gesticulation then smaller exact byte size; two smallest unused camera groups",
        "rule": "current review point unchanged. Run frozen 12Hz paired-edge refinement only when flagged coarse end minus start <=1.000s. Never add a file flag.",
        "moving_conditions": "two publisher swap families, ordinary light/blur/time-step, original; nominal 0.75s and 1.50s centered at 3s plus entire native 1/3–2/3 insert",
        "camera_conditions": "genuine original, chroma-only and recompression-only; nominal 0.75s and 1.50s centered at 4.5s plus actual 3–6s ordinary edit",
        "readout": "compare all flags, coverage, overlap at 0.20/0.50/0.75, onset/end error, gains/losses by case and source, and timing of original ordinary flags",
        "product_boundary": "timing only. A local pilot would need gains across sources and no material loss of correct ranges or sharpening of ordinary false flags. Review before serving.",
    })


def eligible(coarse):
    return coarse is not None and coarse["score"] > POINT and (
        Fraction(coarse["end"]) - Fraction(coarse["start"]) <= SPAN_LIMIT
    )


def exact_span(times, center, duration):
    return tuple(next(i for i, t in enumerate(times) if t >= edge)
                 for edge in (center - duration / 2, center + duration / 2))


def match_cases(folder, cases, *, camera):
    original = cases["unchanged" if camera else "original"]
    raw = original["frame_table"]
    tb = Fraction(raw["time_base"])
    times = [(i - raw["pts"][0]) * tb for i in raw["pts"]]
    center = Fraction(9, 2) if camera else Fraction(3)
    intervals = {str(duration): exact_span(times, center, duration) for duration in common.DURATIONS}
    if camera:
        intervals["native_long"] = tuple(next(i for i, t in enumerate(times) if t >= p) for p in CAMERA_BOUNDS)
    else:
        intervals["native_long"] = (len(times) // 3, 2 * len(times) // 3)
    if not all(intervals["native_long"][0] <= a < b <= intervals["native_long"][1]
               for a, b in intervals.values()):
        raise RuntimeError("Short intervention falls outside verified native change")
    return original, times, intervals


def compare_group(folder, cases, camera, yunet, sface):
    original, times, intervals = match_cases(folder, cases, camera=camera)
    coarse = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
    dense = common.schedule(times, boundary.CADENCE)
    wanted = {i + delta for i in coarse for delta in (-1, 0, 1)} | set(dense)
    table, plain, decoded = common.source_features(folder / original["file"], original, wanted, wanted, yunet, sface)
    rows = []

    def compare(condition, duration, truth, inputs):
        result = common.appearance(coarse, table, times, inputs)
        proposed = boundary.refined(result, dense, times, inputs) if eligible(result) else None
        rows.append({"condition": condition, "duration": duration, "truth": truth, "current": result, "refined": proposed})

    compare("original", None, None, plain)
    lo, hi = intervals["native_long"]
    within = {i for i in wanted if lo <= i < hi}
    for name, case in cases.items():
        if name in {"original", "unchanged"}:
            continue
        expected = case["expected_rgb_sha256"]
        if any(expected[i] == original["expected_rgb_sha256"][i] for i in range(lo, hi)):
            raise RuntimeError("Altered RGB is absent from a declared native frame")
        observed, changed, count = common.source_features(folder / case["file"], case, within, wanted, yunet, sface)
        if common.relative_times(observed) != times:
            raise RuntimeError("Matched variant source clock differs")
        decoded += count
        for duration, (a, b) in intervals.items():
            inputs = {i: value for i, value in plain.items() if not a <= i < b}
            inputs.update({i: value for i, value in changed.items() if a <= i < b})
            compare(name, duration, [str(times[a]), str(times[b])], inputs)
    return rows, decoded


def verify_worker(output):
    frozen = common.read(output / "freeze.json")
    if (frozen["worker_sha256"] != common.digest(Path(__file__))
            or frozen["boundary_worker_sha256"] != common.digest(Path(boundary.__file__))
            or frozen["camera_worker_sha256"] != common.digest(Path(external.__file__))
            or frozen["bindings"] != common.bindings()
            or frozen["metadata_sha256"] != [common.digest(path) for path in METADATA]):
        raise RuntimeError("Frozen code, weights, source metadata or media tools changed")
    return frozen


def run(output, archive):
    frozen = verify_worker(output)
    cv2.setNumThreads(2)
    yunet = common.load_yunet(common.FACE_FILE)
    sface = cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    start = time.monotonic()
    decoded = 0
    with (output / "scored.jsonl").open("x") as file:
        for ordinal, pin in enumerate(frozen["roster"], 1):
            camera = pin["type"] == "camera"
            if camera:
                folder, cases = external.unpack(pin, archive)
                lookup = {case["condition"]: {**case, "expected_rgb_sha256": case["rgb_sha256"]} for case in cases}
            else:
                index = pin["index"]
                path = common.DATA / f"group-{index:03d}-complete.json"
                review = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
                if common.digest(path) != pin["receipt_sha256"] or common.digest(review) != pin["review_sha256"]:
                    raise RuntimeError("Frozen source receipt or independent command review changed")
                folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
                lookup = {case["condition"]: case for case in common.read(path)["cases"]}
            rows, n = compare_group(folder, lookup, camera, yunet, sface)
            decoded += n
            file.write(json.dumps({"group": ordinal, "cohort": pin["type"], "native_samples": n, "rows": rows},
                                  sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            print(f"{ordinal}/{len(frozen['roster'])} groups, {decoded} verified native samples", flush=True)
    common.write(output / "run.json", {"native_samples": decoded, "seconds": time.monotonic() - start,
                                       "max_self_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def add_boundary_metrics(stats, timing, current, chosen, truth):
    left, right = map(Fraction, truth)
    for arm, value in (("current", current), ("policy", chosen)):
        overlap = common.overlap(value, truth)
        for level in (Fraction(1, 5), Fraction(1, 2), Fraction(3, 4)):
            stats[f"{arm}_iou_at_least_{level}"] += overlap >= level
        start, end = (Fraction(value[k]) for k in ("start", "end"))
        timing[arm].append((overlap, abs(start - left), abs(end - right)))
    a, b = common.overlap(current, truth), common.overlap(chosen, truth)
    stats["improved_overlap"] += b > a
    stats["worsened_overlap"] += b < a
    stats["lost_half_overlap"] += a >= Fraction(1, 2) and b < Fraction(1, 2)
    return b > a, b < a


def readout(output):
    frozen = verify_worker(output)
    groups = [json.loads(line) for line in (output / "scored.jsonl").read_text().splitlines()]
    if len(groups) != len(frozen["roster"]):
        raise RuntimeError("Planned held groups are incomplete")
    swaps, controls = defaultdict(Counter), defaultdict(Counter)
    timing, by_group = defaultdict(lambda: defaultdict(list)), defaultdict(Counter)
    for ordinal, (group, pin) in enumerate(zip(groups, frozen["roster"], strict=True), 1):
        cohort = group["cohort"]
        if group["group"] != ordinal or cohort != pin["type"] or len(group["rows"]) != (16 if cohort == "moving" else 7):
            raise RuntimeError("Held roster order or row count is inconsistent")
        statuses = defaultdict(set)
        for row in group["rows"]:
            result, refined = row["current"], row["refined"]
            flagged = result is not None and result["score"] > POINT
            allowed = eligible(result)
            if refined is not None and (not allowed or refined["score"] != result["score"]):
                raise RuntimeError("Refinement altered the appearance flag or violated the fixed gate")
            swap = row["condition"] in common.SWAP_CONDITIONS
            duration = row["duration"]
            stats = swaps[duration] if swap else controls[cohort]
            stats["views"] += 1
            stats["unassessed"] += result is None
            stats["flagged_identically"] += flagged
            stats["eligible_by_coarse_span"] += allowed
            stats["refined"] += refined is not None
            stats["refined_range_changed"] += refined is not None and (
                refined["start"], refined["end"]) != (result["start"], result["end"])
            key = duration if swap else cohort
            if flagged:
                statuses[key].add("flagged")
            if allowed:
                statuses[key].add("eligible")
            if not swap or not flagged:
                continue
            improved, worsened = add_boundary_metrics(stats, timing[duration], result, refined or result, row["truth"])
            if improved:
                statuses[key].add("improved")
            if worsened:
                statuses[key].add("worsened")
        for key, events in statuses.items():
            by_group[key].update(events)
    common.write(output / "aggregate.json", {
        "schema": "chipmunk.video-face-short-boundary-confirmation-result/1",
        "held_moving_groups": sum(pin["type"] == "moving" for pin in frozen["roster"]),
        "tagged_head_rotation_groups": sum(pin.get("head_rotation", False) for pin in frozen["roster"]),
        "tagged_gesticulation_groups": sum(pin.get("gesticulation", False) for pin in frozen["roster"]),
        "held_camera_groups": sum(pin["type"] == "camera" for pin in frozen["roster"]),
        "by_swap_duration": swaps, "controls": controls, "source_groups_by_outcome": by_group,
        "timing": {duration: {arm: dict(zip(("median_iou", "median_onset_error_s", "median_end_error_s"),
                          (float(median(col)) for col in zip(*rows)))) for arm, rows in measurements.items()}
                   for duration, measurements in timing.items()},
        "run": common.read(output / "run.json"),
        "commitments": {name: common.digest(output / name) for name in ("freeze.json", "scored.jsonl", "run.json")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--texture-run", type=Path)
    parser.add_argument("--boundary-run", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output, args.archive, args.texture_run, args.boundary_run)
    elif args.stage == "run":
        run(args.output, args.archive)
    else:
        readout(args.output)
