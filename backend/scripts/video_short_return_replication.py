# SPDX-License-Identifier: Apache-2.0
"""Replicate a positive, sub-half-second face cue without replacing incumbent flags."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import video_face_return_transfer as trial

common = trial.common
archive_io = trial.archive_io
require = trial.require
POINT = 0.
MAX_LENGTH = Fraction(1, 2)
CUT_LENGTHS = (Fraction(1, 2), Fraction(3, 4))
RULE = "Keep every incumbent flag and its endpoints. Otherwise use only the top guarded rank interval when its margin is strictly positive and its half-open duration is strictly below 0.50 seconds. Never rerank or infer originality."
ADVANCE = "At least six added correct 0.375-second localizations across at least three sources, no more than two added ordinary flags, zero added cut flags, and no incumbent flag or endpoint changed."


def bindings():
    return {**trial.bindings(), "backend/scripts/video_short_return_replication.py": common.digest(Path(__file__))}


def freeze(folder, archive, first, earlier):
    original = trial.frozen(first)
    original_result = common.read(first / "aggregate.json")
    require(original_result["freeze_sha256"] == common.digest(first / "freeze.json") and len(earlier) == 5
            and archive.resolve() == trial.prior.camera_saved.ARCHIVE, "Exact first result and five previous rosters required")
    earlier_docs = [common.read(run / "freeze.json") for run in earlier]
    used = set.union(*(trial.previous_rtf_sources(doc) for doc in (original, *earlier_docs)))
    metadata = trial.prior.comparison.metadata()
    prior_meta = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in used]
    links, donors = {m["url"] for m in prior_meta}, set.union(*(m["donors"] for m in prior_meta))
    eligible = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        meta = metadata[doc["group"]]
        clock = doc["cases"][0]["frame_table"]
        step = (clock["pts"][1] - clock["pts"][0]) * Fraction(clock["time_base"])
        if doc["role"] == "confidence_cal" and doc["index"] not in used and "child" not in meta["tags"] and (
            clock["width"], clock["height"], step) == (1920, 1080, Fraction(1, 30)):
            eligible.append((sum(c["bytes"] for c in doc["cases"]), doc["index"], meta))
    pins = []
    for _, index, meta in sorted(eligible):
        if meta["url"] in links or donors & meta["donors"]:
            continue
        pins.append(archive_io.source_pin(index, archive))
        links.add(meta["url"])
        donors.update(meta["donors"])
        if len(pins) == 6:
            break
    require(len(pins) == 6, "Six source and donor disjoint 30 Hz groups required")
    known = [pin for doc in (original, *earlier_docs) for phase in ("development", "held")
             for arm in [doc.get(phase)] if isinstance(arm, dict) for pin in arm.get("camera", [])]
    camera_hashes = {common.read(archive / p["qualification"])["source_sha256"] for p in known}
    for doc in earlier_docs:
        for item in doc.get("roster", []):
            if item.get("type") == "camera":
                camera_hashes.add(common.read(archive / item["relative"])["source_sha256"])
    covered = trial.prior.camera_native.PREPARATION / "slot-08-attempt-01/qualification.json"
    camera_hashes.add(common.read(covered)["source_sha256"])
    cameras = []
    for slot in (5, 6):
        path = trial.prior.camera_native.PREPARATION / f"slot-{slot:02d}-attempt-01/qualification.json"
        sha = common.read(path)["source_sha256"]
        require(sha not in camera_hashes, "Camera recording overlaps a development or previous transfer source")
        cameras.append(archive_io.camera_pin(path, archive))
        camera_hashes.add(sha)
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-replication-freeze/1",
        "bindings": bindings(), "first_result_sha256": common.digest(first / "aggregate.json"),
        "prior_roster_sha256": [common.digest(path / "freeze.json") for path in (first, *earlier)],
        "rtfs": pins, "camera": cameras, "cut_pairs": [[0, 1], [2, 3], [4, 5]],
        "cut_durations": list(map(str, CUT_LENGTHS)), "point": POINT, "maximum_length_exclusive_s": str(MAX_LENGTH),
        "incumbent_point": trial.prior.POINT, "rule": RULE, "advancement": ADVANCE})


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == bindings() and lock["point"] == POINT and lock["maximum_length_exclusive_s"] == str(MAX_LENGTH)
            and lock["incumbent_point"] == trial.prior.POINT, "Frozen worker or point differs")
    return lock


def route(arms):
    current = arms["current"]
    if current is not None and current["score"] > trial.prior.POINT:
        return current, "incumbent"
    candidate = arms["rank"]
    if candidate is not None and candidate["score"] > POINT:
        span = Fraction(candidate["end"]) - Fraction(candidate["start"])
        if 0 < span < MAX_LENGTH:
            return candidate, "additive"
    return None, "no_flag" if current is not None or candidate is not None else "unassessed"


def cut_cases(first, second):
    output = []
    for (base, original), (other, insert) in ((first, second), (second, first)):
        times = common.relative_times(base)
        require(times == common.relative_times(other) and (base.width, base.height) == (other.width, other.height)
                and original.keys() == insert.keys(), "Exact cut source pairing is required")
        for duration in CUT_LENGTHS:
            start, end = trial.prior.comparison.exact_span(times, Fraction(3), duration)
            mixed = {i: insert[i] if start <= i < end else row for i, row in original.items()}
            output.append({"condition": "ordinary_source_cut", "duration": str(duration), "truth": None,
                           "ordinary_cut_span": [str(times[start]), str(times[end])], **trial.evaluate(base, mixed)})
    return output


def run(folder, archive):
    plan = frozen(folder)
    cv2.setNumThreads(2)
    models = (common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), ""))
    originals = []
    started, verified = time.monotonic(), 0
    journal = folder / "scored.jsonl"
    with journal.open("x") as stream:
        def emit(family, number, rows, frames):
            nonlocal verified
            verified += frames
            stream.write(json.dumps({"family": family, "group": f"{family}_{number}", "rows": rows,
                                     "verified_frames": frames}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(family, number, "verified", frames, "frames", flush=True)

        for number, pin in enumerate(plan["camera"], 1):
            emit("camera", number, *trial.camera_group(pin, archive, models))
        for number, pin in enumerate(plan["rtfs"], 1):
            rows, frames, original = trial.rtfs_group(pin, archive, models, development=False)
            originals.append(original)
            emit("rtfs", number, rows, frames)
        for number, (left, right) in enumerate(plan["cut_pairs"], 1):
            emit("cut", number, cut_cases(originals[left], originals[right]), 0)
    common.write(folder / "run.json", {"elapsed_s": time.monotonic() - started, "verified_frames": verified,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def readout(folder):
    frozen(folder)
    journal = folder / "scored.jsonl"
    groups = archive_io.boundary.journal(journal)
    run_doc = common.read(folder / "run.json")
    require(run_doc["journal_sha256"] == common.digest(journal)
            and [len(group["rows"]) for group in groups] == [3, 3, *([16] * 6), 4, 4, 4]
            and sum(group["verified_frames"] for group in groups) == run_doc["verified_frames"], "Exact replay journal required")
    totals = defaultdict(lambda: defaultdict(Counter))
    gains, additional, by_duration = Counter(), Counter(), defaultdict(Counter)
    gained_groups = set()
    preserved = 0
    for group in groups:
        for row in group["rows"]:
            truth = row["truth"]
            arms = row["arms"]
            current = trial.classify(arms["current"], trial.prior.POINT, truth)
            candidate, route_name = route(arms)
            merged = trial.classify(candidate, -float("inf"), truth) if candidate else (
                "unassessed" if route_name == "unassessed" else "measured_no_flag")
            if route_name == "incumbent":
                require(candidate == arms["current"] and merged == current, "An incumbent flag was altered")
                preserved += 1
            label = "swap" if truth else group["family"]
            totals[label]["current"][current] += 1
            totals[label]["additive"][merged] += 1
            if truth:
                by_duration[row["duration"]][f"current_{current}"] += 1
                by_duration[row["duration"]][f"additive_{merged}"] += 1
            if route_name == "additive":
                additional[label] += 1
                if merged == "localized":
                    gains[row["duration"]] += 1
                    if row["duration"] == "3/8":
                        gained_groups.add(group["group"])
    passed = gains["3/8"] >= 6 and len(gained_groups) >= 3 and additional["rtfs"] + additional["camera"] <= 2 and additional["cut"] == 0
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-short-return-replication/1", "rule": RULE,
        "advancement": ADVANCE, "freeze_sha256": common.digest(folder / "freeze.json"), "totals": totals,
        "by_swap_duration": by_duration, "added_correct_by_duration": gains, "added_flags_by_cohort": additional,
        "groups_with_short_gain": len(gained_groups), "incumbent_flags_preserved": preserved,
        "pilot_advances": passed, "runtime": run_doc})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, default=trial.prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.first is None:
            parser.error("the first study path is required")
        freeze(args.folder, args.archive, args.first, args.prior)
    elif args.stage == "run":
        run(args.folder, args.archive)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
