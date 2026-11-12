# SPDX-License-Identifier: Apache-2.0
"""Prospective control of incumbent face-review flags that coincide with paired scene cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.detectors.video import sampling, shots
from scripts import video_short_return_secondary_transfer as prior

trial, common, require = prior.trial, prior.common, prior.require
archive_io = prior.margin.archive_io
SPANS = tuple(map(Fraction, ("3/8", "1/2", "3/4")))
SLOP = Fraction(1, 4)
MIN_IOU = Fraction(1, 2)
CAMERA = prior.margin.CAMERA_ROOTS[1]
RULE = "Suppress an incumbent range only when both ends are within 0.25 s of an exact paired cut, overlap is at least 0.50, and the pre/post and first/last middle pictures are similar by the unchanged app shot metrics. Never suppress on one cut or unavailable context."
GATE = "Preserve every useful swap localization and suppress at least four and at least half of incumbent two-cut false ranges, across at least two source pairs. Preserve all single-cut flags and require at least six measured incumbent two-cut flags."
SELECTION = "Six new complete 1080p25 RTFS groups, motion tags prioritized, then smallest archived bytes. Exclude all known source links and donor keys from nine previous localization rosters. Use the one remaining v206 matched camera recording after prior-source and reserved-source exclusion."


def bindings():
    return {**prior.bindings(), **{str(Path(module.__file__).relative_to(common.ROOT)): common.digest(Path(module.__file__))
        for module in (sampling, shots)}, "backend/scripts/video_incumbent_cut_guard_transfer.py": common.digest(Path(__file__))}


def freeze(output, archive, secondary, earlier, texture_external):
    old = prior.frozen(secondary)
    readout = common.read(secondary / "aggregate.json")
    paths = [directory / "freeze.json" for directory in earlier]
    hashes = [common.digest(path) for path in paths]
    require(archive.resolve() == trial.prior.camera_saved.ARCHIVE
            and len(hashes) == len(set(hashes)) == 8
            and set(hashes) == {old["prior_margin_freeze_sha256"], *old["prior_freeze_sha256"]}
            and common.digest(texture_external) == old["texture_external_freeze_sha256"]
            and readout["freeze_sha256"] == common.digest(secondary / "freeze.json"), "Exact exposure ledger is required")
    previous = [old, *(common.read(path) for path in paths)]
    excluded = set.union(*(trial.previous_rtf_sources(doc) for doc in previous))
    require(len(excluded) >= 58, "Prior source roster is incomplete")
    metadata = trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    eligible = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        i, row = doc["index"], metadata[doc["group"]]
        if i in excluded or row["url"] in urls or row["donors"] & donors or "child" in row["tags"]:
            continue
        clock = doc["cases"][0]["frame_table"]
        times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
        if (clock["width"], clock["height"]) == (1920, 1080) and times == [Fraction(j, 25) for j in range(150)]:
            motion = bool(row["tags"] & {"gesticulation", "head rotation"})
            eligible.append((not motion, sum(case["bytes"] for case in doc["cases"]), i, row))
    rtfs, motion_sources = [], 0
    for still, _, i, row in sorted(eligible):
        if row["url"] in urls or row["donors"] & donors:
            continue
        rtfs.append(archive_io.source_pin(i, archive))
        motion_sources += not still
        urls.add(row["url"])
        donors.update(row["donors"])
        if len(rtfs) == 6:
            break
    require(len(rtfs) == 6 and motion_sources >= 4, "Insufficient motion-tagged and source-disjoint RTFS groups")
    known_camera = {archive / name for plan in (*previous, common.read(texture_external))
                    for name in prior.margin.camera_paths(plan)}
    known_camera.add(archive / prior.margin.CAMERA_ROOTS[0] / "slot-08-attempt-01/qualification.json")
    camera_ids = {common.read(path)["source_sha256"] for path in known_camera}
    camera = []
    for path in sorted((archive / CAMERA).glob("slot-*/qualification.json")):
        doc = common.read(path)
        if doc["source_sha256"] not in camera_ids and all(c["status"] == "prepared" for c in doc["cases"].values()):
            camera.append(archive_io.camera_pin(path, archive))
            camera_ids.add(doc["source_sha256"])
    require(len(camera) == 1, "Expected the one remaining complete camera group")
    common.write(output / "freeze.json", {"schema": "chipmunk.video-incumbent-cut-guard-freeze/1", "bindings": bindings(),
        "incumbent_point": trial.prior.POINT, "content_point": shots.CONTENT_THRESHOLD, "histogram_point": shots.HISTOGRAM_THRESHOLD,
        "end_slop_s": str(SLOP), "minimum_cut_overlap": str(MIN_IOU), "spans_s": list(map(str, SPANS)),
        "rule": RULE, "advancement": GATE, "selection": SELECTION, "rtfs": rtfs, "motion_tagged_groups": motion_sources,
        "camera": camera, "cut_pairs": [[0, 1], [2, 3], [4, 5]], "prior_freeze_sha256": hashes,
        "prior_secondary_freeze_sha256": common.digest(secondary / "freeze.json"),
        "prior_secondary_result_sha256": common.digest(secondary / "aggregate.json"),
        "texture_external_freeze_sha256": common.digest(texture_external)})


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == bindings() and lock["incumbent_point"] == trial.prior.POINT
            and lock["content_point"] == shots.CONTENT_THRESHOLD and lock["histogram_point"] == shots.HISTOGRAM_THRESHOLD
            and lock["end_slop_s"] == str(SLOP) and lock["minimum_cut_overlap"] == str(MIN_IOU)
            and lock["spans_s"] == list(map(str, SPANS)), "Frozen thresholds or implementation changed")
    return lock


def scene_difference(a, b):
    thumb, histogram = a
    other, other_histogram = b
    return float(np.abs(thumb - other).mean()), 1. - float(np.minimum(histogram, other_histogram).sum())


def is_cut(delta):
    return delta[0] >= shots.CONTENT_THRESHOLD and delta[1] >= shots.HISTOGRAM_THRESHOLD


def is_same_scene(delta):
    return delta[0] < shots.CONTENT_THRESHOLD and delta[1] < shots.HISTOGRAM_THRESHOLD


def cut_guard(incumbent, times, scene):
    if incumbent is None or incumbent["score"] <= trial.prior.POINT:
        return None
    a, b = (Fraction(incumbent[key]) for key in ("start", "end"))
    entries = [i for i in range(1, len(times)) if abs(times[i] - a) <= SLOP and i in scene and i - 1 in scene
               and is_cut(scene_difference(scene[i - 1], scene[i]))]
    exits = [i for i in range(1, len(times)) if abs(times[i] - b) <= SLOP and i in scene and i - 1 in scene
             and is_cut(scene_difference(scene[i - 1], scene[i]))]
    candidates = []
    for first in entries:
        for last in exits:
            if last - first < 2:
                continue
            c, d = times[first], times[last]
            iou = max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou < MIN_IOU:
                continue
            flank = scene_difference(scene[first - 1], scene[last])
            middle = scene_difference(scene[first], scene[last - 1])
            if is_same_scene(flank) and is_same_scene(middle):
                entry = scene_difference(scene[first - 1], scene[first])
                exit_ = scene_difference(scene[last - 1], scene[last])
                candidates.append((-iou, first, last, entry, exit_, flank, middle))
    if not candidates:
        return None
    neg_iou, first, last, entry, exit_, flank, middle = min(candidates)
    return {"indices": [first, last], "span": [str(times[first]), str(times[last])], "iou": str(-neg_iou),
            "entry": entry, "exit": exit_, "flank": flank, "middle": middle}


def samples(table, duration):
    count = min(40, max(16, int(round(duration * 4))))
    centers = {i for i in sampling.uniform_indices(len(table), count) if 0 < i < len(table) - 1}
    return centers, {i + delta for i in centers for delta in (-1, 0, 1)}


def acquire(path, case, expected, indices, face_indices, models):
    require(common.digest(path) == case["sha256"], "Source bytes differ from qualification")
    table = common.probe_frames(path, timeout_s=30)
    tb = Fraction(expected["time_base"])
    require(not table.truncated and not table.interpolated and not table.rotation
            and (table.width, table.height) == (expected["width"], expected["height"])
            and [i * table.tb for i in table.pts] == [i * tb for i in expected["pts"]], "Source clock differs")
    measured, features = {}, {}
    yunet, recognizer = models
    for frame in common.decode_frames(path, table, sorted(indices), timeout_s=110):
        i = frame.index
        require(frame.pts_verified and i not in features and hashlib.sha256(frame.rgb).hexdigest() == case["rgb_sha256"][i],
                "Native RGB or picture clock differs from its receipt")
        features[i] = shots._features(frame.rgb)
        if i in face_indices:
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            faces = common.detect_faces(bgr, detector=yunet, min_size=48)
            if len(faces) == 1:
                face = faces[0]
                detection = common.face_appearance.face_row(face.landmarks)
                detection[0, :4], detection[0, 14] = face.box(), face.score
                measured[i] = common.face_appearance.unit(recognizer.feature(recognizer.alignCrop(bgr, detection)))
    require(features.keys() == indices, "Incomplete native decode")
    return table, measured, features


def evaluate(table, centers, features, scene):
    times = common.relative_times(table)
    tracks = {"single": [SimpleNamespace(frame_index=i) for i in sorted(features)]}
    windows = common.face_appearance.windows(centers, tracks, features, table)
    incumbent = None
    if windows:
        highest = windows[0]
        indices = highest["centers"]
        incumbent = {"score": highest["response"], "start": str(times[indices[2]]), "end": str(times[indices[-3]])}
    return {"incumbent": incumbent, "cut_support": cut_guard(incumbent, times, scene)}


def assemble(original, alternate, a, b):
    return {**{i: row for i, row in original.items() if i < a or i >= b}, **{i: row for i, row in alternate.items() if a <= i < b}}


def rtfs_group(pin, archive, models):
    index = pin["index"]
    require(archive_io.source_pin(index, archive) == pin, "Source receipt or review changed")
    doc = common.read(common.DATA / f"group-{index:03d}-complete.json")
    cases = {case["condition"]: case for case in doc["cases"]}
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    base, times, spans = trial.prior.comparison.match_cases(folder, cases, camera=False)
    intervals = [(d, *trial.prior.comparison.exact_span(times, Fraction(3), d)) for d in SPANS]
    whole = set(range(len(times)))
    altered = set.union(*(set(range(a, b)) for _, a, b in intervals))
    a, b = spans["native_long"]
    require(altered <= set(range(a, b)), "Synthetic edit leaves the qualified native region")
    centers, face_positions = samples(times, 6.)

    def load(case, indices):
        pinned = {**case, "rgb_sha256": case["expected_rgb_sha256"]}
        return acquire(folder / case["file"], pinned, case["frame_table"], indices, face_positions, models)

    table, primary, scene = load(base, whole)
    rows = [{"condition": "original", "duration": None, "truth": None, **evaluate(table, centers, primary, scene)}]
    for name, case in cases.items():
        if name == "original":
            continue
        _, alternate, change = load(case, altered)
        for duration, start, stop in intervals:
            truth = [str(times[start]), str(times[stop])] if name in common.SWAP_CONDITIONS else None
            rows.append({"condition": name, "duration": str(duration), "truth": truth,
                         **evaluate(table, centers, assemble(primary, alternate, start, stop), assemble(scene, change, start, stop))})
    return rows, len(whole) + (len(cases) - 1) * len(altered), (table, centers, primary, scene)


def camera_group(pin, archive, models):
    path = archive / pin["qualification"]
    require(archive_io.camera_pin(path, archive) == pin, "Camera qualification changed")
    doc = common.read(path)
    times = [i * Fraction(doc["time_base"]) for i in doc["relative_pts"]]
    duration = float(times[-1] - times[0] + Fraction(doc["last_duration"]) * Fraction(doc["time_base"]))
    centers, face_positions = samples(times, duration)
    expected = {"width": doc["width"], "height": doc["height"], "time_base": doc["time_base"], "pts": doc["relative_pts"]}
    rows = []
    for name in archive_io.CAMERA_ROLES:
        case = doc["cases"][name]
        table, features, scene = acquire(path.parent / case["file"], case, expected, set(range(len(times))), face_positions, models)
        rows.append({"condition": name, "duration": None, "truth": None, **evaluate(table, centers, features, scene)})
    return rows, len(times) * len(rows)


def cut_group(left, right):
    rows = []
    for (table, centers, face, scene), (other, _, alternate, change) in ((left, right), (right, left)):
        times = common.relative_times(table)
        require(times == common.relative_times(other) and (table.width, table.height) == (other.width, other.height),
                "Genuine cuts require a common exact picture clock")
        for duration in SPANS:
            a, b = trial.prior.comparison.exact_span(times, Fraction(3), duration)
            rows.append({"condition": "two_cut", "duration": str(duration), "truth": None, "cut_span": [str(times[a]), str(times[b])],
                         **evaluate(table, centers, assemble(face, alternate, a, b), assemble(scene, change, a, b))})
        a = times.index(Fraction(3))
        rows.append({"condition": "single_cut", "duration": None, "truth": None, "cut_at": str(times[a]),
                     **evaluate(table, centers, assemble(face, alternate, a, len(times)), assemble(scene, change, a, len(times)))})
    return rows


def run(folder, archive):
    lock = frozen(folder)
    cv2.setNumThreads(2)
    models = (common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), ""))
    journal = folder / "scored.jsonl"
    originals = []
    verified, start = 0, time.monotonic()
    with journal.open("x") as stream:
        def emit(family, number, rows, frames):
            nonlocal verified
            verified += frames
            stream.write(json.dumps({"family": family, "group": f"{family}_{number}", "rows": rows,
                                     "verified_frames": frames}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(family, number, "verified", frames, "frames", flush=True)

        for number, pin in enumerate(lock["rtfs"], 1):
            rows, frames, original = rtfs_group(pin, archive, models)
            originals.append(original)
            emit("rtfs", number, rows, frames)
        for number, (left, right) in enumerate(lock["cut_pairs"], 1):
            emit("cut", number, cut_group(originals[left], originals[right]), 0)
        for number, pin in enumerate(lock["camera"], 1):
            emit("camera", number, *camera_group(pin, archive, models))
    common.write(folder / "run.json", {"elapsed_s": time.monotonic() - start, "verified_frames": verified,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def readout(folder):
    frozen(folder)
    journal = folder / "scored.jsonl"
    groups, runtime = archive_io.boundary.journal(journal), common.read(folder / "run.json")
    require(runtime["journal_sha256"] == common.digest(journal)
            and [len(group["rows"]) for group in groups] == [*([16] * 6), 8, 8, 8, 3]
            and sum(group["verified_frames"] for group in groups) == runtime["verified_frames"], "Incomplete score journal")
    totals, durations = defaultdict(Counter), defaultdict(Counter)
    groups_suppressed = defaultdict(set)
    for group in groups:
        for row in group["rows"]:
            truth, selected = row["truth"], row["incumbent"]
            status = trial.classify(selected, trial.prior.POINT, truth)
            name = "swap" if truth else row["condition"] if group["family"] == "cut" else group["family"]
            counts = totals[name]
            counts[f"current_{status}"] += 1
            if (support := row["cut_support"]) is not None:
                require(status in {"localized", "wrong_interval", "false_flag"}, "Unflagged incumbent was suppressed")
                counts[f"suppressed_{status}"] += 1
                groups_suppressed[name].add(group["group"])
                if name == "two_cut" and support["span"] == row["cut_span"]:
                    counts["suppressed_with_exact_genuine_cut_boundaries"] += 1
            else:
                counts[f"retained_{status}"] += 1
            if row["duration"]:
                key = f"{name}@{row['duration']}"
                durations[key][f"current_{status}"] += 1
                if support:
                    durations[key][f"suppressed_{status}"] += 1
    cuts, swaps = totals["two_cut"], totals["swap"]
    passed = (cuts["current_false_flag"] >= 6 and cuts["suppressed_false_flag"] >= 4
              and 2 * cuts["suppressed_false_flag"] >= cuts["current_false_flag"] and len(groups_suppressed["two_cut"]) >= 2
              and not swaps["suppressed_localized"] and not totals["single_cut"]["suppressed_false_flag"])
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-incumbent-cut-guard-transfer/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "rule": RULE, "advancement": GATE,
        "totals": totals, "by_duration": durations, "suppressed_source_groups": {key: len(v) for key, v in groups_suppressed.items()},
        "pilot_advances": passed, "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--secondary-run", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--texture-external", type=Path)
    parser.add_argument("--archive", type=Path, default=trial.prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.secondary_run is None or args.texture_external is None:
            parser.error("freeze requires --secondary-run and --texture-external")
        freeze(args.folder, args.archive, args.secondary_run, args.prior, args.texture_external)
    elif args.stage == "run":
        run(args.folder, args.archive)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
