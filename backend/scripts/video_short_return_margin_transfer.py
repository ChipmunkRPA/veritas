# SPDX-License-Identifier: Apache-2.0
"""Prospective 0.10-margin transfer for the shortest face-appearance changes."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import video_short_return_replication as previous

trial = previous.trial
common = previous.common
archive_io = previous.archive_io
require = previous.require
POINT = 0.10
CENTERS = (Fraction(3), Fraction(35, 12), Fraction(73, 24))
BASE_DURATION = Fraction(3, 8)
PHASES = [(CENTERS[0], duration) for duration in trial.SIZES] + [(center, BASE_DURATION) for center in CENTERS[1:]]
CAMERA_ROOTS = ("video-vcd2-matched-controls-v290", "video-vcd2-delivery-v206")
RULE = "Preserve every incumbent interval. Otherwise retain only the highest guarded rank interval, with raw margin >0.10 and half-open duration <0.50 seconds. Never rerank."
ADVANCE = "Recover at least six shortest intervals across at least three sources at each of three edit placements, with at most two additional ordinary false flags, zero additional source-cut flags, and every incumbent flag and endpoint preserved."
SELECTION = "Six distinct known RTFS source and donor groups at native 1080p and 30 Hz, smallest archived bytes first across prior project roles. Four distinct VCD2 recording hashes, one v290 and three v206, smallest archived bytes first after excluding the prior localization and coverage rosters."


def bindings():
    return {**previous.bindings(), "backend/scripts/video_short_return_margin_transfer.py": common.digest(Path(__file__))}


def camera_paths(value):
    if isinstance(value, dict):
        for name, element in value.items():
            if name in {"qualification", "relative"} and isinstance(element, str) and element.endswith("/qualification.json"):
                yield element
            else:
                yield from camera_paths(element)
    elif isinstance(value, list):
        for element in value:
            yield from camera_paths(element)


def camera_roster(archive, plans, texture_external):
    coverage = archive / CAMERA_ROOTS[0] / "slot-08-attempt-01/qualification.json"
    paths = {archive / name for document in (*plans, texture_external) for name in camera_paths(document)} | {coverage}
    excluded = {common.read(path)["source_sha256"] for path in paths}
    selected = []
    for dataset, target in zip(CAMERA_ROOTS, (1, 3), strict=True):
        options = []
        for path in sorted((archive / dataset).glob("slot-*/qualification.json")):
            row = common.read(path)
            if row["source_sha256"] in excluded:
                continue
            pin = archive_io.camera_pin(path, archive)
            if any(case.get("status", "prepared") != "prepared" for case in row["cases"].values()):
                continue
            size = sum((path.parent / case["file"]).stat().st_size for case in row["cases"].values())
            options.append((size, path, row["source_sha256"], pin))
        count = 0
        for _, _, identity, pin in sorted(options):
            if identity in excluded:
                continue
            selected.append(pin)
            excluded.add(identity)
            count += 1
            if count == target:
                break
        require(count == target, "Insufficient source-disjoint camera recordings")
    return selected


def rtfs_roster(archive, plans):
    excluded = set.union(*(trial.previous_rtf_sources(plan) for plan in plans))
    require(len(excluded) >= 46, "Incomplete previous RTFS localization roster")
    metadata = trial.prior.comparison.metadata()
    past = [metadata[common.read(common.DATA / f"group-{index:03d}-complete.json")["group"]] for index in excluded]
    urls, donors = {m["url"] for m in past}, set.union(*(m["donors"] for m in past))
    options = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        row = common.read(path)
        index, meta = row["index"], metadata[row["group"]]
        if index in excluded or "child" in meta["tags"] or meta["url"] in urls or donors & meta["donors"]:
            continue
        clock = row["cases"][0]["frame_table"]
        times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
        if (clock["width"], clock["height"]) != (1920, 1080) or times != [Fraction(i, 30) for i in range(180)]:
            continue
        options.append((sum(case["bytes"] for case in row["cases"]), index, meta))
    selected = []
    for _, index, meta in sorted(options):
        if meta["url"] in urls or donors & meta["donors"]:
            continue
        selected.append(archive_io.source_pin(index, archive))
        urls.add(meta["url"])
        donors.update(meta["donors"])
        if len(selected) == 6:
            break
    require(len(selected) == 6, "Insufficient source and donor disjoint RTFS groups")
    return selected


def freeze(folder, archive, replication, earlier, texture_external):
    old = previous.frozen(replication)
    result = common.read(replication / "aggregate.json")
    prior_paths = [p / "freeze.json" for p in earlier]
    hashes = [common.digest(p) for p in prior_paths]
    require(archive.resolve() == trial.prior.camera_saved.ARCHIVE
            and len(hashes) == len(set(hashes)) == 6 and set(hashes) == set(old["prior_roster_sha256"])
            and result["freeze_sha256"] == common.digest(replication / "freeze.json"), "Exact prior roster is required")
    plans = [old, *(common.read(path) for path in prior_paths)]
    external = common.read(texture_external)
    require(external["schema"] == "chipmunk.video-face-texture-external-freeze/1"
            and external["primary_freeze_sha256"] in hashes
            and len(list(camera_paths(external))) == 2, "Exact external camera study roster is required")
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-margin-transfer-freeze/1",
        "bindings": bindings(), "point": POINT, "incumbent_point": trial.prior.POINT, "rule": RULE,
        "advance": ADVANCE, "selection": SELECTION, "phases": [[str(c), str(d)] for c, d in PHASES],
        "rtfs": rtfs_roster(archive, plans), "camera": camera_roster(archive, plans, external),
        "cut_pairs": [[0, 1], [2, 3], [4, 5]], "cut_durations": list(map(str, previous.CUT_LENGTHS)),
        "maximum_length_exclusive_s": str(previous.MAX_LENGTH), "prior_roster_sha256": hashes,
        "texture_external_freeze_sha256": common.digest(texture_external),
        "replication_freeze_sha256": common.digest(replication / "freeze.json"),
        "replication_result_sha256": common.digest(replication / "aggregate.json")})


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == bindings() and lock["point"] == POINT and lock["incumbent_point"] == trial.prior.POINT
            and lock["phases"] == [[str(c), str(d)] for c, d in PHASES]
            and lock["maximum_length_exclusive_s"] == str(previous.MAX_LENGTH)
            and lock["cut_durations"] == list(map(str, previous.CUT_LENGTHS)), "Frozen method or settings differ")
    return lock


def route(arms):
    current = arms["current"]
    if current is not None and current["score"] > trial.prior.POINT:
        return current, "incumbent"
    candidate = arms["rank"]
    if candidate is not None and candidate["score"] > POINT:
        span = Fraction(candidate["end"]) - Fraction(candidate["start"])
        if 0 < span < previous.MAX_LENGTH:
            return candidate, "additive"
    return None, "no_flag" if current is not None or candidate is not None else "unassessed"


def phase_group(pin, archive, models):
    index = pin["index"]
    require(archive_io.source_pin(index, archive) == pin, "Source receipt or review changed")
    document = common.read(common.DATA / f"group-{index:03d}-complete.json")
    cases = {case["condition"]: case for case in document["cases"]}
    path = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    base, times, spans = trial.prior.comparison.match_cases(path, cases, camera=False)
    native_a, native_b = spans["native_long"]
    intervals = [(center, duration, trial.prior.comparison.exact_span(times, center, duration)) for center, duration in PHASES]
    _, _, wanted = trial.schedules(times)
    changed = {i for i in wanted if any(a <= i < b for _, _, (a, b) in intervals)}

    def load(case, indices):
        clock, rows = trial.sample(path / case["file"], {**case, "rgb_sha256": case["expected_rgb_sha256"]},
                                   case["frame_table"], indices, *models)
        require(common.relative_times(clock) == times, "Native input clocks differ")
        return clock, rows

    clock, original = load(base, wanted)
    rows = [{"condition": "original", "duration": None, "center": None, "truth": None, **trial.evaluate(clock, original)}]
    for name, case in cases.items():
        if name == "original":
            continue
        _, alternate = load(case, changed)
        for center, duration, (a, b) in intervals:
            require(native_a <= a < b <= native_b, "Edit extends beyond the verified native intervention")
            features = {i: alternate[i] if a <= i < b else row for i, row in original.items()}
            rows.append({"condition": name, "duration": str(duration), "center": str(center),
                         "truth": [str(times[a]), str(times[b])] if name in common.SWAP_CONDITIONS else None,
                         **trial.evaluate(clock, features)})
    return rows, len(wanted) + (len(cases) - 1) * len(changed), (clock, original)


def run(folder, archive):
    plan = frozen(folder)
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
                                     "verified_frames": frames}, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            print(family, number, "verified", frames, "frames", flush=True)

        for number, pin in enumerate(plan["rtfs"], 1):
            rows, frames, original = phase_group(pin, archive, models)
            originals.append(original)
            emit("rtfs", number, rows, frames)
        for number, pin in enumerate(plan["camera"], 1):
            emit("camera", number, *trial.camera_group(pin, archive, models))
        for number, (left, right) in enumerate(plan["cut_pairs"], 1):
            emit("cut", number, previous.cut_cases(originals[left], originals[right]), 0)
    common.write(folder / "run.json", {"elapsed_s": time.monotonic() - start, "verified_frames": verified,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def readout(folder):
    frozen(folder)
    journal = folder / "scored.jsonl"
    groups = archive_io.boundary.journal(journal)
    runtime = common.read(folder / "run.json")
    require(runtime["journal_sha256"] == common.digest(journal)
            and [len(group["rows"]) for group in groups] == [*([26] * 6), *([3] * 4), 4, 4, 4]
            and sum(group["verified_frames"] for group in groups) == runtime["verified_frames"], "Incomplete replay journal")
    totals = defaultdict(lambda: defaultdict(Counter))
    by_phase = defaultdict(Counter)
    gain_groups = defaultdict(set)
    added, preserved = Counter(), 0
    for group in groups:
        for row in group["rows"]:
            truth, arms = row["truth"], row["arms"]
            baseline = trial.classify(arms["current"], trial.prior.POINT, truth)
            result, selected = route(arms)
            merged = trial.classify(result, -float("inf"), truth) if result else (
                "unassessed" if selected == "unassessed" else "measured_no_flag")
            if selected == "incumbent":
                require(result == arms["current"] and merged == baseline, "An incumbent flag or endpoint changed")
                preserved += 1
            cohort = "swap" if truth else group["family"]
            totals[cohort]["current"][baseline] += 1
            totals[cohort]["additive"][merged] += 1
            if truth:
                phase = f"{row['center']}@{row['duration']}"
                by_phase[phase][f"current_{baseline}"] += 1
                by_phase[phase][f"additive_{merged}"] += 1
            if selected == "additive":
                added[cohort] += 1
                if merged == "localized":
                    by_phase[phase]["added_correct"] += 1
                    gain_groups[phase].add(group["group"])
    shortest = [f"{center}@{BASE_DURATION}" for center in CENTERS]
    passed = (all(by_phase[key]["added_correct"] >= 6 and len(gain_groups[key]) >= 3 for key in shortest)
              and added["rtfs"] + added["camera"] <= 2 and added["cut"] == 0)
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-short-return-margin-transfer/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "point": POINT, "rule": RULE,
        "advance": ADVANCE, "totals": totals, "swap_outcomes_by_center_and_duration": by_phase,
        "gain_group_counts_by_center_and_duration": {key: len(values) for key, values in gain_groups.items()},
        "added_flags_by_cohort": added, "incumbent_flags_preserved": preserved, "pilot_advances": passed, "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replication", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--texture-external", type=Path)
    parser.add_argument("--archive", type=Path, default=trial.prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.replication is None or args.texture_external is None:
            parser.error("freeze requires --replication and --texture-external")
        freeze(args.folder, args.archive, args.replication, args.prior, args.texture_external)
    elif args.stage == "run":
        run(args.folder, args.archive)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
