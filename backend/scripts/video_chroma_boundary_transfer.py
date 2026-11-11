# SPDX-License-Identifier: Apache-2.0
"""Measure multiscale local color transitions on archived video with exact clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from scripts import video_selected_face_boundary as boundary

prior = boundary.prior
common = boundary.common
require = boundary.require
ARCS = tuple(boundary.ARCS.items())
OFFSETS = (-.08, 0., .08)
SPEC = {
    "cadence_hz": 4,
    "core_centers": [3, 5, 7, 8],
    "flank_centers_per_side": 2,
    "core_quorum": .60,
    "angular_site_degrees": [-6, -4, -2, 0, 2, 4, 6],
    "spatial_quorum": 4,
    "native_scales": [.025, .05],
    "minimum_scale_pixels": [3., 5.],
    "chroma_stabilizer": 1.5,
    "minimum_direction_cosine": .5,
    "max_face_center_step": .65,
    "max_face_sample_gap_seconds": .55,
}
CAMERA_ROLES = ("unchanged", "chroma_conversion_only", "recompression_only")
ADVANCE = "Replicate only with at least 12 of 24 correctly localized swaps (IoU >= 0.50) across four held sources, at most two false flags across all held genuine views, and at least 80% assessed coverage in swaps and controls separately."
POLICY = "Standalone research localization. A non-flagged or unassessed interval carries no originality inference. No overall confidence contribution."


def source_pin(index, archive):
    path = common.DATA / f"group-{index:03d}-complete.json"
    review = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
    data, verified = common.read(path), common.read(review)
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    sha = common.digest(path)
    require(verified["status"] == "complete" and verified["private_group_receipt_sha256"] == sha
            and set(c["condition"] for c in data["cases"]) == set(common.CONTROL_CONDITIONS + common.SWAP_CONDITIONS)
            and all((folder / c["file"]).stat().st_size == c["bytes"] for c in data["cases"]), "Incomplete native group")
    return {"index": index, "receipt_sha256": sha, "review_sha256": common.digest(review)}


def camera_pin(path, archive):
    data = common.read(path)
    require(data["model_calls"] == 0 and set(data["cases"]) == set(CAMERA_ROLES)
            and all((path.parent / case["file"]).is_file() and case["verified_frames"] == data["frame_count"]
                    for case in data["cases"].values()), "Camera qualification is incomplete")
    return {"qualification": str(path.relative_to(archive)), "qualification_sha256": common.digest(path)}


def bindings():
    return {**common.bindings(), **{str(Path(worker.__file__).relative_to(common.ROOT)): common.digest(Path(worker.__file__))
        for worker in (boundary, prior, prior.comparison)}, "worker_sha256": common.digest(Path(__file__))}


def freeze(folder, archive, prior_runs):
    require(archive.resolve() == prior.camera_saved.ARCHIVE, "Exact retained archive required")
    frozen_plans = [common.read(run / "freeze.json") for run in prior_runs]
    excluded = {row["index"] for doc in frozen_plans for row in doc.get("roster", doc.get("rtfs", [])) if "index" in row}
    require(len(excluded) >= 20 and set(prior.GROUPS).issubset(excluded), "Prior localization exposure ledger is incomplete")
    metadata = prior.comparison.metadata()
    earlier = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    urls = {m["url"] for m in earlier}
    donors = set.union(*(m["donors"] for m in earlier))
    prospective = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        m = metadata[doc["group"]]
        if doc["role"] != "selection" or doc["index"] in excluded or "child" in m["tags"]:
            continue
        if m["url"] in urls or donors & m["donors"]:
            continue
        # File size is metadata for CPU scheduling. No response participates in selection.
        prospective.append((sum(c["bytes"] for c in doc["cases"]), doc["index"], m))
    held = []
    for _, index, m in sorted(prospective):
        if m["url"] in urls or donors & m["donors"]:
            continue
        held.append(source_pin(index, archive))
        urls.add(m["url"])
        donors.update(m["donors"])
        if len(held) == 6:
            break
    require(len(held) == 6, "Six disjoint prospective matched swap groups required")
    development = [source_pin(i, archive) for i in prior.GROUPS[:4]]
    dev_camera = prior.camera_saved.PREPARATION / "slot-09-attempt-01/qualification.json"
    known_camera = {common.read(dev_camera)["source_sha256"]}
    for doc in frozen_plans:
        for entry in doc.get("roster", []):
            if entry.get("type") == "camera":
                known_camera.add(common.read(archive / entry["relative"])["source_sha256"])
    camera = []
    for path in sorted(prior.camera_native.PREPARATION.glob("slot-*/qualification.json")):
        data = common.read(path)
        if data["slot"] == 8 or data["source_sha256"] in known_camera:
            continue
        camera.append(camera_pin(path, archive))
        known_camera.add(data["source_sha256"])
        if len(camera) == 2:
            break
    require(len(camera) == 2, "Two source-disjoint archived camera groups required")
    common.write(folder / "freeze.json", {
        "schema": "chipmunk.video-chroma-boundary-transfer-freeze/1", "bindings": bindings(),
        "prior_localization_freezes": [common.digest(run / "freeze.json") for run in prior_runs],
        "source_metadata_sha256": [common.digest(path) for path in prior.comparison.METADATA],
        "development": {"rtfs": development, "camera": [camera_pin(dev_camera, archive)]},
        "held": {"rtfs": held, "camera": camera}, "specification": SPEC, "arcs": dict(ARCS), "offsets": list(OFFSETS),
        "threshold": "Maximum assessed development-control score, strictly greater than, sealed before holdout.",
        "roster_selection": "Previously scored RTFS selection-role sources, smallest archived bytes first, excluding known localization sources and donor aliases. Fixed camera slots in archive order, excluding prior tested source IDs and coverage slot08.",
        "localization": "Independently scan fixed native-clock ranges and report the highest same-arc change above both flanks. Evaluate half-open edit IoU separately from flagging.",
        "policy": POLICY, "pilot_advancement": ADVANCE,
    })


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == bindings() and lock["specification"] == SPEC
            and lock["arcs"] == dict(ARCS) and lock["offsets"] == list(OFFSETS), "Method bindings differ from freeze")
    return lock


def contrast(lab, locations, normals, width, crop_origin):
    """Measure the central chroma step and compare identical parallel offset steps."""
    # Each band's mean spans three normal positions, using native pixel interpolation.
    distances = np.asarray([center + side + delta for center in (-3., 0., 3.) for side in (-1., 1.)
                            for delta in (-.25, 0., .25)], np.float32) * width
    xy = locations[:, None] + normals[:, None] * distances[None, :, None] - crop_origin
    h, w = lab.shape[:2]
    valid = np.all((xy >= 1.) & (xy < np.asarray([w - 2, h - 2])), axis=(1, 2))
    if np.count_nonzero(valid) < SPEC["spatial_quorum"]:
        return None
    xy = np.ascontiguousarray(xy[valid], np.float32)
    colors = cv2.remap(lab, xy[:, :, 0].copy(), xy[:, :, 1].copy(), cv2.INTER_LINEAR).reshape(-1, 3, 2, 3, 2).mean(axis=3)
    changes = colors[:, :, 1] - colors[:, :, 0]
    step = changes[:, 1]
    sizes = np.linalg.norm(changes, axis=2)
    direction = np.median(step, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= .5:
        return 0.
    aligned = (step @ direction) / np.maximum(.5, norm * sizes[:, 1]) >= SPEC["minimum_direction_cosine"]
    if np.count_nonzero(aligned) < SPEC["spatial_quorum"]:
        return 0.
    comparison = np.maximum(sizes[:, 0], sizes[:, 2])
    stabilize = SPEC["chroma_stabilizer"]
    log_ratios = np.log((sizes[:, 1] + stabilize) / (comparison + stabilize))
    return max(0., float(np.median(log_ratios[aligned])))


def describe_frame(rgb, yunet):
    faces = common.detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=48)
    if len(faces) != 1:
        return None
    face = faces[0]
    pose = boundary.geometry(face)
    if pose is None:
        return None
    eyes, horizontal, vertical, _ = pose
    center = eyes + .16 * face.h * vertical
    x, y = max(0, face.x - face.w // 2), max(0, face.y - face.h // 2)
    x2, y2 = min(rgb.shape[1], face.x + int(1.5 * face.w)), min(rgb.shape[0], face.y + int(1.5 * face.h))
    lab = cv2.cvtColor(rgb[y:y2, x:x2], cv2.COLOR_RGB2LAB)[:, :, 1:].astype(np.float32)
    scale_widths = [max(low, face.w * scale) for scale, low in zip(SPEC["native_scales"], SPEC["minimum_scale_pixels"], strict=True)]
    arcs = {}
    for name, degrees in ARCS:
        angles = np.radians(degrees + np.asarray(SPEC["angular_site_degrees"]))
        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        sites = center + .44 * face.w * cos * horizontal + .39 * face.h * sin * vertical
        normals = cos * horizontal + sin * vertical
        for number, offset in enumerate(OFFSETS):
            locations = sites + normals * offset * face.w
            responses = [contrast(lab, locations, normals, width, np.asarray([x, y])) for width in scale_widths]
            if any(v is None for v in responses):
                continue
            points = locations.astype(int).tolist()
            arcs[f"{name}/{number}"] = {"score": min(responses), "native_polyline": points}
    return {"face_xywh": list(face.box()), "arcs": arcs}


def bounded_track(indices, times, features):
    visible = [(i, features[i]["face_xywh"]) for i in indices if features.get(i) is not None]
    for (a, first), (b, second) in zip(visible, visible[1:]):
        if times[b] - times[a] > SPEC["max_face_sample_gap_seconds"]:
            return False
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        delta = math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
        if delta > SPEC["max_face_center_step"] * min(aw, bw):
            return False
    return True


def scan(times, schedule, features):
    best = None
    assessed = 0
    for length in SPEC["core_centers"]:
        needed = math.ceil(length * SPEC["core_quorum"])
        for start in range(2, len(schedule) - length - 1):
            stop = start + length
            before, core, after = schedule[start - 2:start], schedule[start:stop], schedule[stop:stop + 2]
            if not bounded_track(before + core + after, times, features):
                continue
            measurements = [defaultdict(list) for _ in range(3)]
            for role, block in zip(measurements, (before, core, after), strict=True):
                for index in block:
                    if features.get(index) is None:
                        continue
                    for arc, item in features[index]["arcs"].items():
                        role[arc].append((item["score"], index))
            window_assessed = False
            for arc, middle in measurements[1].items():
                left, right = measurements[0].get(arc), measurements[2].get(arc)
                if not left or not right or len(middle) < needed:
                    continue
                window_assessed = True
                representative = sorted(middle, reverse=True)
                score = max(0., representative[needed - 1][0] - max(median(v for v, _ in left), median(v for v, _ in right)))
                key = (-score, times[core[0]], length, arc)
                if best is not None and key >= best[0]:
                    continue
                anchors = (left[-1], representative[0], right[0])
                best = (key, {"score": score, "start": str(times[core[0]]), "end": str(times[after[0]]),
                              "arc": arc, "available_centers": [len(left), len(middle), len(right)],
                              "anchors": [{"index": i, "time": str(times[i]), "arc_score": value,
                                           "native_polyline": features[i]["arcs"][arc]["native_polyline"]}
                                          for value, i in anchors]})
            assessed += window_assessed
    return {"best": best[1] if best else None, "measured_windows": assessed}


def sample(path, expected_sha, source, rgb, indices, yunet):
    require(common.digest(path) == expected_sha, "Native video source bytes differ from qualification")
    table = common.probe_frames(path, timeout_s=30)
    expected = [Fraction(pts) * Fraction(source["time_base"]) for pts in source["pts"]]
    require(not table.truncated and not table.interpolated and table.rotation == 0
            and (table.width, table.height) == (source["width"], source["height"])
            and [pts * table.tb for pts in table.pts] == expected, "Native presentation clock differs")
    features = {}
    for frame in common.decode_frames(path, table, sorted(indices), timeout_s=90):
        index = frame.index
        require(frame.pts_verified and index not in features and hashlib.sha256(frame.rgb).hexdigest() == rgb[index],
                "Native picture or decoder timestamp differs from qualification")
        features[index] = describe_frame(frame.rgb, yunet)
    require(features.keys() == set(indices), "Exact sample roster incomplete")
    return common.relative_times(table), features


def rtfs_group(pin, archive, yunet, *, development):
    index = pin["index"]
    path = common.DATA / f"group-{index:03d}-complete.json"
    review = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
    require(common.digest(path) == pin["receipt_sha256"] and common.digest(review) == pin["review_sha256"], "Native receipt differs")
    cases = {item["condition"]: item for item in common.read(path)["cases"]}
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    original, times, intervals = prior.comparison.match_cases(folder, cases, camera=False)
    schedule = common.schedule(times, SPEC["cadence_hz"])
    native = intervals["native_long"]
    inside = [i for i in schedule if native[0] <= i < native[1]]

    def case_features(case, wanted):
        observed, values = sample(folder / case["file"], case["sha256"], case["frame_table"], case["expected_rgb_sha256"], wanted, yunet)
        require(observed == times, "Paired case clock differs from source")
        return values

    before = case_features(original, schedule)
    rows = [{"condition": "original", "duration": None, "truth": None, **scan(times, schedule, before)}]
    sampled = len(schedule)
    for name, case in cases.items():
        if name == "original" or development and name in common.SWAP_CONDITIONS:
            continue
        require(set(inside) <= set(case["actual_changed_frames"]), "Native intervention pixels differ from receipt")
        alternate = case_features(case, inside)
        sampled += len(alternate)
        for duration in ("3/4", "native_long"):
            start, stop = intervals[duration]
            features = {i: alternate[i] if start <= i < stop else value for i, value in before.items()}
            truth = [str(times[start]), str(times[stop])] if name in common.SWAP_CONDITIONS else None
            rows.append({"condition": name, "duration": duration, "truth": truth, **scan(times, schedule, features)})
    return rows, sampled


def camera_group(pin, archive, yunet):
    path = archive / pin["qualification"]
    require(common.digest(path) == pin["qualification_sha256"], "Camera qualification differs")
    document = common.read(path)
    table = {"width": document["width"], "height": document["height"],
             "time_base": document["time_base"], "pts": document["relative_pts"]}
    tb = Fraction(table["time_base"])
    times = [(pts - table["pts"][0]) * tb for pts in table["pts"]]
    schedule = common.schedule(times, SPEC["cadence_hz"])
    rows = []
    for name in CAMERA_ROLES:
        case = document["cases"][name]
        observed, values = sample(path.parent / case["file"], case["sha256"], table, case["rgb_sha256"], schedule, yunet)
        require(observed == times, "Camera delivery has a different presentation clock")
        rows.append({"condition": name, "duration": None, "truth": None, **scan(times, schedule, values)})
    return rows, len(schedule) * len(rows)


def run(folder, archive, phase):
    lock = frozen(folder)
    if phase == "held":
        sealed_point(folder)
    require(archive.resolve() == prior.camera_saved.ARCHIVE, "Exact retained archive required")
    cv2.setNumThreads(2)
    yunet = common.load_yunet(common.FACE_FILE)
    start = time.monotonic()
    journal = folder / f"{phase}.jsonl"
    samples = 0
    with journal.open("x") as stream:
        for family, pins in lock[phase].items():
            for number, pin in enumerate(pins, 1):
                rows, count = (rtfs_group(pin, archive, yunet, development=phase == "development")
                               if family == "rtfs" else camera_group(pin, archive, yunet))
                stream.write(json.dumps({"family": family, "group": f"{family}_{number}", "rows": rows, "verified_frames": count},
                                        sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                print(phase, family, number, "verified", count, "frames", flush=True)
                samples += count
    common.write(folder / f"{phase}-run.json", {"verified_frames": samples, "elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def result_groups(folder, phase):
    journal = folder / f"{phase}.jsonl"
    receipt = common.read(folder / f"{phase}-run.json")
    groups = boundary.journal(journal)
    require(common.digest(journal) == receipt["journal_sha256"]
            and sum(group["verified_frames"] for group in groups) == receipt["verified_frames"], "Incomplete or altered journal")
    expected = [3, *([7] * 4)] if phase == "development" else [3, 3, *([11] * 6)]
    require([len(group["rows"]) for group in groups] == expected, "Unexpected frozen roster")
    return groups, receipt


def seal(folder):
    frozen(folder)
    groups, run = result_groups(folder, "development")
    rows = [row for group in groups for row in group["rows"]]
    require(not any(row["truth"] for row in rows), "Only genuine controls may set the review point")
    scores = [row["best"]["score"] for row in rows if row["best"]]
    require(len(scores) >= 20, "Insufficient assessed development controls to seal a point")
    common.write(folder / "point.json", {"schema": "chipmunk.video-chroma-boundary-point/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "development_journal_sha256": run["journal_sha256"],
        "assessed_controls": len(scores), "all_controls": len(rows), "point": max(scores)})


def sealed_point(folder):
    sealed = common.read(folder / "point.json")
    require(sealed["freeze_sha256"] == common.digest(folder / "freeze.json")
            and sealed["development_journal_sha256"] == common.digest(folder / "development.jsonl"), "Point does not bind to development")
    return sealed


def outcome(row, point):
    selected = row["best"]
    if selected is None:
        return "unassessed"
    if selected["score"] <= point:
        return "measured_no_flag"
    if not row["truth"]:
        return "false_flag"
    return "localized" if common.overlap(selected, row["truth"]) >= Fraction(1, 2) else "wrong_interval"


def readout(folder):
    frozen(folder)
    point = sealed_point(folder)
    development, dev_run = result_groups(folder, "development")
    held, held_run = result_groups(folder, "held")
    rows = [row for group in development for row in group["rows"]]
    scores = [row["best"]["score"] for row in rows if row["best"]]
    require(max(scores) == point["point"] and len(scores) == point["assessed_controls"], "Sealed point does not replay")
    totals, conditions, durations = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    localized_groups, false_groups = set(), set()
    for group in held:
        for row in group["rows"]:
            result = outcome(row, point["point"])
            cohort = "known_swap" if row["truth"] else group["family"]
            totals[cohort][result] += 1
            conditions[row["condition"]][result] += 1
            if row["truth"]:
                durations[row["duration"]][result] += 1
            if result == "localized":
                localized_groups.add(group["group"])
            if result == "false_flag":
                false_groups.add(group["group"])
    swaps = totals["known_swap"]
    controls = totals["rtfs"] + totals["camera"]
    passed = (swaps["localized"] >= 12 and len(localized_groups) >= 4 and controls["false_flag"] <= 2
              and swaps["unassessed"] <= 4 and controls["unassessed"] <= 9)
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-chroma-boundary-transfer/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "sealed_point_sha256": common.digest(folder / "point.json"),
        "development": point, "held": {name: dict(values) for name, values in totals.items()},
        "held_by_condition": {name: dict(values) for name, values in conditions.items()},
        "swap_by_duration": {name: dict(values) for name, values in durations.items()},
        "groups_with_localization": len(localized_groups), "groups_with_false_flags": len(false_groups),
        "pilot_advances": passed, "policy": POLICY, "pilot_advancement": ADVANCE,
        "runtime": {"development": dev_run, "held": held_run}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "seal", "held", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, default=prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.archive, args.prior)
    elif args.stage in {"development", "held"}:
        run(args.folder, args.archive, args.stage)
    elif args.stage == "seal":
        seal(args.folder)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
