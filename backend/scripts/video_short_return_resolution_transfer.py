# SPDX-License-Identifier: Apache-2.0
"""Replay the serving short-face return on paired 720p and 360p pictures."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2

from app.detectors.video import face_return
from scripts import video_short_return_secondary_transfer as prior

trial, common, require = prior.trial, prior.common, prior.require
archive_io = prior.margin.archive_io
ARCHIVE = trial.prior.camera_saved.ARCHIVE
PRIOR = ("video-short-return-secondary-2026-11-11", "video-incumbent-cut-guard-2026-11-11")
SIZES = ((1280, 720), (640, 360))
DURATIONS = (Fraction(3, 8), Fraction(1, 2))
CENTER = Fraction(3)
RULE = "Change only the resolution condition; preserve the current threshold, guards, winner, duplicate handling and incumbent."
GATE = "Per view size: at least six added correct 0.375-second ranges across three source groups; at least 80% assessed for swaps and controls separately; at most two added false ordinary ranges and zero added cut ranges."


def bindings():
    return {**common.bindings(), **{str(Path(module.__file__).relative_to(common.ROOT)): common.digest(Path(module.__file__))
            for module in (trial, trial.prior.comparison, archive_io, face_return)}, "worker_sha256": common.digest(Path(__file__)),
            "opencv": cv2.__version__}


def freeze(folder, archive):
    require(archive.resolve() == ARCHIVE, "The retained archive is required")
    paths = [archive / name / "freeze.json" for name in PRIOR]
    exposed = set.union(*(trial.previous_rtf_sources(common.read(path)) for path in paths))
    require(len(exposed) == 12, "The two preceding short-return and cut panels must have twelve groups")
    metadata = trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in exposed]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    candidates = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        document = common.read(path)
        i, row = document["index"], metadata[document["group"]]
        if i in exposed or row["url"] in urls or row["donors"] & donors or "child" in row["tags"] or document["role"] == "confirmation":
            continue
        clock = document["cases"][0]["frame_table"]
        times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
        if (clock["width"], clock["height"]) == SIZES[0] and times == [Fraction(j, 25) for j in range(150)]:
            candidates.append((sum(case["bytes"] for case in document["cases"]), i, row))
    rtfs = []
    for _, i, row in sorted(candidates):
        if row["url"] in urls or row["donors"] & donors:
            continue
        rtfs.append(archive_io.source_pin(i, archive))
        urls.add(row["url"])
        donors.update(row["donors"])
        if len(rtfs) == 6:
            break
    require(len(rtfs) == 6, "Six complete disjoint native 720p25 source groups required")
    folder.mkdir(mode=0o700, exist_ok=False)
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-resolution-freeze/1", "bindings": bindings(),
        "rule": RULE, "gate": GATE, "dimensions": SIZES, "cadence_hz": face_return.CADENCE,
        "point": face_return.POINT, "maximum_length_exclusive_s": str(face_return.MAX_RANGE),
        "dedupe_iou": str(face_return.DEDUPE_IOU), "incumbent_point": trial.prior.POINT,
        "durations": list(map(str, DURATIONS)), "center": str(CENTER), "correct_iou": "1/2",
        "prior_freeze_sha256": [common.digest(path) for path in paths], "rtfs": rtfs,
        "cut_pairs": [[0, 1], [2, 3], [4, 5]],
        "selection": "Six smallest complete native 720p25 RTFS groups excluding confirmation, child-tagged, and known source/donor overlap with both previous panels. Three consecutive pairs supply both cut directions."})


def frozen(folder, archive):
    plan = common.read(folder / "freeze.json")
    require(plan["bindings"] == bindings() and plan["rule"] == RULE and plan["gate"] == GATE
            and plan["point"] == face_return.POINT and plan["incumbent_point"] == trial.prior.POINT
            and plan["prior_freeze_sha256"] == [common.digest(archive / name / "freeze.json") for name in PRIOR],
            "Frozen worker, method, or source exposure changed")
    return plan


def checked_frames(path, case, indices):
    require(common.digest(path) == case["sha256"], "Source file differs from qualification")
    expected = case["frame_table"]
    clock = common.probe_frames(path, timeout_s=30)
    require((clock.width, clock.height) == SIZES[0] == (expected["width"], expected["height"])
            and not (clock.truncated or clock.interpolated or clock.rotation)
            and [i * clock.tb for i in clock.pts] == [i * Fraction(expected["time_base"]) for i in expected["pts"]],
            "Source clock differs from qualification")

    def pictures():
        seen = set()
        for frame in common.decode_frames(path, clock, sorted(indices), timeout_s=110):
            require(frame.pts_verified and frame.index in indices and frame.index not in seen
                    and rgb_digest(frame.rgb) == case["expected_rgb_sha256"][frame.index],
                    "Source picture or clock differs from qualification")
            seen.add(frame.index)
            yield frame
        require(seen == set(indices), "Incomplete decode")

    return clock, pictures()


def rgb_digest(rgb):
    return hashlib.sha256(rgb).hexdigest()


def grid(rgb, size):
    return rgb if size == SIZES[0] else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def crop(rgb, box):
    x, y, width, height = box
    return rgb[y:y + height, x:x + width].copy()


def sample(path, case, indices, source_group, models):
    yunet, recognizer = models
    clock, frames = checked_frames(path, case, indices)
    rows = {size: {} for size in SIZES}
    for frame in frames:
        for size in SIZES:
            rgb = grid(frame.rgb, size)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = common.detect_faces(bgr, detector=yunet, min_size=48)
            face = faces[0] if len(faces) == 1 else None
            vector, evidence = None, None
            if face is not None:
                geometry = common.face_appearance.face_row(face.landmarks)
                geometry[0, :4], geometry[0, 14] = face.box(), face.score
                vector = common.face_appearance.unit(recognizer.feature(recognizer.alignCrop(bgr, geometry)))
                box = face.box()
                evidence = {"source_group": source_group, "condition": case["condition"], "box": box,
                            "source_rgb_sha256": case["expected_rgb_sha256"][frame.index],
                            "analysis_rgb_sha256": rgb_digest(rgb), "crop_rgb_sha256": rgb_digest(crop(rgb, box))}
            rows[size][frame.index] = (face_return.Sample(face, vector, face_return.scene(rgb, face)), len(faces), evidence)
    return clock, rows


def compare(clock, samples):
    schedule = face_return.schedule(replace(clock, width=1920, height=1080))
    require(schedule is not None, "Only resolution may differ from the existing schedule")
    times, dense = schedule
    coarse, expected, _ = trial.schedules(times)
    require(tuple(expected) == dense, "Cadence has changed")
    vectors = {i: row[0].vector for i, row in samples.items() if row[0].vector is not None}
    tracks = {"single-face": [SimpleNamespace(frame_index=i) for i in sorted(vectors)]}
    ranges = common.face_appearance.windows(set(coarse), tracks, vectors, clock)
    current, incumbent = None, None
    if ranges:
        first = ranges[0]
        start, end = first["centers"][2], first["centers"][-3]
        current = {"score": first["response"], "start": str(times[start]), "end": str(times[end])}
        if first["response"] > trial.prior.POINT:
            incumbent = {"start_pts": clock.pts[start], "end_pts": clock.pts[end]}
    multiple = any(samples[i][1] > 1 for i in dense)
    candidate = None if multiple else face_return.highest(times, dense, {i: samples[i][0] for i in dense})
    ranked = selected = None
    if candidate:
        ranked = {"score": candidate.response, "start": str(times[candidate.start]), "end": str(times[candidate.end])}
        if face_return.distinct(candidate, clock, incumbent):
            selected = {**ranked, "scene_correlations": candidate.scene_correlations,
                        "anchors": {role: {**samples[i][2], "frame": i, "time": str(times[i])}
                                    for role, i in candidate.anchors().items()}}
    return {"current": current, "ranked": ranked, "selected": selected, "multiple_faces": multiple,
            "single_face_samples": sum(samples[i][0].face is not None for i in dense), "dense_samples": len(dense)}


def matched(archive, pin, models):
    i = pin["index"]
    require(archive_io.source_pin(i, archive) == pin, "Source qualification changed")
    doc = common.read(common.DATA / f"group-{i:03d}-complete.json")
    cases = {case["condition"]: case for case in doc["cases"]}
    path = archive / "video-native-inserts-v95" / f"group-{i:03d}"
    base, times, spans = trial.prior.comparison.match_cases(path, cases, camera=False)
    intervals = [(duration, trial.prior.comparison.exact_span(times, CENTER, duration)) for duration in DURATIONS]
    lower, upper = spans["native_long"]
    require(all(lower <= a < b <= upper for _, (a, b) in intervals), "Trial intervention outside native qualification")
    _, _, wanted = trial.schedules(times)
    changed = {k for k in wanted if any(a <= k < b for _, (a, b) in intervals)}
    clock, original = sample(path / base["file"], base, wanted, i, models)

    def evaluate(condition, truth, duration, variant, bounds):
        arms = {}
        for size, source in original.items():
            mixed = source if variant is None else {
                k: variant[size][k] if bounds[0] <= k < bounds[1] else row for k, row in source.items()}
            arms[str(size[1])] = compare(clock, mixed)
        return {"condition": condition, "truth": truth, "duration": duration, "arms": arms}

    rows = [evaluate("original", None, None, None, None)]
    for name, case in cases.items():
        if name == "original":
            continue
        variant_clock, variant = sample(path / case["file"], case, changed, i, models)
        require(common.relative_times(variant_clock) == times, "The matched view has a different relative clock")
        for duration, (a, b) in intervals:
            truth = [str(times[a]), str(times[b])] if name in common.SWAP_CONDITIONS else None
            rows.append(evaluate(name, truth, str(duration), variant, (a, b)))
    return rows, len(wanted) + (len(cases) - 1) * len(changed), (clock, original)


def cuts(first, second):
    clock, left = first
    other, right = second
    times = common.relative_times(clock)
    require(times == common.relative_times(other), "Source cut needs a matched clock")
    rows = []
    for duration in DURATIONS:
        a, b = trial.prior.comparison.exact_span(times, CENTER, duration)
        for original, insert in ((left, right), (right, left)):
            arms = {str(size[1]): compare(clock, {i: insert[size][i] if a <= i < b else row
                                                 for i, row in original[size].items()}) for size in SIZES}
            rows.append({"condition": "source_cut", "truth": None, "duration": str(duration), "arms": arms})
    return rows


def run(folder, archive):
    plan = frozen(folder, archive)
    cv2.setNumThreads(2)
    models = (common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), ""))
    journal = folder / "scored.jsonl"
    start, frames, originals = time.monotonic(), 0, []
    with journal.open("x") as stream:
        def emit(family, number, rows):
            stream.write(json.dumps({"family": family, "group": number, "rows": rows}, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            print(family, number, len(rows), "views", flush=True)

        for number, pin in enumerate(plan["rtfs"], 1):
            rows, decoded, original = matched(archive, pin, models)
            frames += decoded
            originals.append(original)
            emit("matched", number, rows)
        for number, (left, right) in enumerate(plan["cut_pairs"], 1):
            emit("cut", number, cuts(originals[left], originals[right]))
    common.write(folder / "run.json", {"elapsed_s": time.monotonic() - start, "verified_source_frames": frames,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def journal_rows(folder, archive):
    plan = frozen(folder, archive)
    journal = folder / "scored.jsonl"
    runtime = common.read(folder / "run.json")
    require(runtime["journal_sha256"] == common.digest(journal), "Journal digest changed")
    groups = archive_io.boundary.journal(journal)
    require([len(group["rows"]) for group in groups] == [11] * len(plan["rtfs"]) + [4] * len(plan["cut_pairs"]),
            "Journal is missing planned cases")
    return plan, groups, runtime


def readout(folder, archive):
    _, groups, runtime = journal_rows(folder, archive)
    summary = {}
    for _, height in SIZES:
        totals, groups_gained = defaultdict(Counter), set()
        for group in groups:
            for row in group["rows"]:
                truth = row["truth"]
                result = row["arms"][str(height)]
                cohort = f"swap_{row['duration']}" if truth else group["family"]
                counts = totals[cohort]
                baseline = trial.classify(result["current"], trial.prior.POINT, truth)
                added = trial.classify(result["selected"], face_return.POINT, truth) if result["selected"] else None
                counts["views"] += 1
                counts[f"current_{baseline}"] += 1
                counts["rank_assessed"] += result["ranked"] is not None
                counts["multiple_face_skips"] += result["multiple_faces"]
                counts["dense_single_face_samples"] += result["single_face_samples"]
                counts["dense_samples"] += result["dense_samples"]
                counts["final_localized"] += baseline == "localized" or added == "localized"
                counts["final_false_flag"] += baseline == "false_flag" or added == "false_flag"
                if added:
                    counts[f"secondary_{added}"] += 1
                if added == "localized" and baseline != "localized":
                    counts["additional_correct"] += 1
                    if row["duration"] == "3/8":
                        groups_gained.add(group["group"])
        swaps = [value for key, value in totals.items() if key.startswith("swap")]
        controls = [totals["matched"], totals["cut"]]
        coverage = {name: str(Fraction(sum(c["rank_assessed"] for c in values), sum(c["views"] for c in values)))
                    for name, values in (("swap", swaps), ("control", controls))}
        passes = (totals["swap_3/8"]["additional_correct"] >= 6 and len(groups_gained) >= 3
                  and all(Fraction(rate) >= Fraction(4, 5) for rate in coverage.values())
                  and totals["matched"]["secondary_false_flag"] <= 2 and not totals["cut"]["secondary_false_flag"])
        summary[str(height)] = {"totals": totals, "shortest_gain_source_groups": len(groups_gained),
                                "assessed_coverage": coverage, "pilot_advances": passes}
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-short-return-resolution-transfer/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "rule": RULE, "gate": GATE, "by_height": summary, "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "readout": readout}[args.stage](args.folder, args.archive)


if __name__ == "__main__":
    main()
