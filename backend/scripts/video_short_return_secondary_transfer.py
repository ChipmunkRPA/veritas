# SPDX-License-Identifier: Apache-2.0
"""Prospectively validate a distinct secondary short face-change range."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import video_short_return_margin_transfer as margin

trial = margin.trial
common = margin.common
require = margin.require
DEDUPE_IOU = Fraction(1, 2)
RULE = "Preserve the incumbent interval. Separately display the highest guarded interval only for a margin >0.10, a native duration in (0,0.50s), and IoU <0.50 against any incumbent interval. Never rerank."
ADVANCE = "At least 18 additional correct shortest ranges across four source groups, including at least four in two groups where the incumbent is absent and four in two groups where it is wrong. At most two new ordinary review ranges, including on incumbent-flagged files, and no new source-cut review range. Preserve incumbent endpoints."
SELECTION = "Six complete native 1080p25 RTFS groups tagged head rotation or gesticulation, smallest archived bytes first after source and donor exclusion. Four distinct VCD2 camera groups from v206, smallest archived bytes first after excluding all prior localized and reserved source hashes."


def bindings():
    return {**margin.bindings(), "backend/scripts/video_short_return_secondary_transfer.py": common.digest(Path(__file__))}


def freeze(folder, archive, prior_margin, priors, texture_external):
    require(archive.resolve() == trial.prior.camera_saved.ARCHIVE, "The retained archive is required")
    old = margin.frozen(prior_margin)
    result = common.read(prior_margin / "aggregate.json")
    paths = [directory / "freeze.json" for directory in priors]
    hashes = [common.digest(path) for path in paths]
    require(len(hashes) == len(set(hashes)) == 7
            and set(hashes) == {old["replication_freeze_sha256"], *old["prior_roster_sha256"]}
            and result["freeze_sha256"] == common.digest(prior_margin / "freeze.json")
            and common.digest(texture_external) == old["texture_external_freeze_sha256"], "Exact prior exposure ledger is required")
    plans = [old, *(common.read(path) for path in paths)]
    excluded = set.union(*(trial.previous_rtf_sources(plan) for plan in plans))
    require(len(excluded) >= 52, "Incomplete previous RTFS source exclusions")
    metadata = trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    candidates = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        document = common.read(path)
        i, row = document["index"], metadata[document["group"]]
        if (i in excluded or row["url"] in urls or row["donors"] & donors or "child" in row["tags"]
                or not row["tags"] & {"head rotation", "gesticulation"}):
            continue
        clock = document["cases"][0]["frame_table"]
        times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
        if (clock["width"], clock["height"]) == (1920, 1080) and times == [Fraction(j, 25) for j in range(150)]:
            candidates.append((sum(case["bytes"] for case in document["cases"]), i, row))
    rtfs = []
    for _, i, row in sorted(candidates):
        if row["url"] in urls or row["donors"] & donors:
            continue
        rtfs.append(margin.archive_io.source_pin(i, archive))
        urls.add(row["url"])
        donors.update(row["donors"])
        if len(rtfs) == 6:
            break
    require(len(rtfs) == 6, "Six separate motion-tagged RTFS groups required")
    prior_camera = {archive / name for plan in (*plans, common.read(texture_external)) for name in margin.camera_paths(plan)}
    prior_camera.add(archive / margin.CAMERA_ROOTS[0] / "slot-08-attempt-01/qualification.json")
    camera_ids = {common.read(path)["source_sha256"] for path in prior_camera}
    camera_options = []
    for path in sorted((archive / margin.CAMERA_ROOTS[1]).glob("slot-*/qualification.json")):
        doc = common.read(path)
        if doc["source_sha256"] in camera_ids or any(case.get("status") != "prepared" for case in doc["cases"].values()):
            continue
        pin = margin.archive_io.camera_pin(path, archive)
        camera_options.append((sum(case["bytes"] for case in doc["cases"].values()), path, doc["source_sha256"], pin))
    camera = []
    for _, _, sha, pin in sorted(camera_options):
        if sha in camera_ids:
            continue
        camera.append(pin)
        camera_ids.add(sha)
        if len(camera) == 4:
            break
    require(len(camera) == 4, "Four separate camera groups required")
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-secondary-freeze/1", "bindings": bindings(),
        "rule": RULE, "advance": ADVANCE, "selection": SELECTION, "point": margin.POINT, "dedupe_iou": str(DEDUPE_IOU),
        "incumbent_point": trial.prior.POINT, "maximum_length_exclusive_s": str(margin.previous.MAX_LENGTH),
        "phases": [[str(center), str(duration)] for center, duration in margin.PHASES],
        "rtfs": rtfs, "camera": camera, "cut_pairs": [[0, 1], [2, 3], [4, 5]],
        "cut_durations": list(map(str, margin.previous.CUT_LENGTHS)), "prior_freeze_sha256": hashes,
        "prior_margin_freeze_sha256": common.digest(prior_margin / "freeze.json"),
        "prior_margin_result_sha256": common.digest(prior_margin / "aggregate.json"),
        "texture_external_freeze_sha256": common.digest(texture_external)})


def frozen(folder):
    plan = common.read(folder / "freeze.json")
    require(plan["bindings"] == bindings() and plan["point"] == margin.POINT and plan["incumbent_point"] == trial.prior.POINT
            and Fraction(plan["dedupe_iou"]) == DEDUPE_IOU and Fraction(plan["maximum_length_exclusive_s"]) == margin.previous.MAX_LENGTH
            and plan["phases"] == [[str(center), str(duration)] for center, duration in margin.PHASES]
            and plan["cut_durations"] == list(map(str, margin.previous.CUT_LENGTHS)), "Frozen method differs")
    return plan


def choose(arms):
    current = arms["current"]
    incumbent = current if current is not None and current["score"] > trial.prior.POINT else None
    rank = arms["rank"]
    if rank is None or rank["score"] <= margin.POINT or not 0 < Fraction(rank["end"]) - Fraction(rank["start"]) < margin.previous.MAX_LENGTH:
        return incumbent, None, False
    if incumbent is not None and common.overlap(rank, [incumbent["start"], incumbent["end"]]) >= DEDUPE_IOU:
        return incumbent, None, True
    return incumbent, rank, False


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
            rows, frames, original = margin.phase_group(pin, archive, models)
            originals.append(original)
            emit("rtfs", number, rows, frames)
        for number, pin in enumerate(plan["camera"], 1):
            emit("camera", number, *trial.camera_group(pin, archive, models))
        for number, (left, right) in enumerate(plan["cut_pairs"], 1):
            emit("cut", number, margin.previous.cut_cases(originals[left], originals[right]), 0)
    common.write(folder / "run.json", {"elapsed_s": time.monotonic() - start, "verified_frames": verified,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journal)})


def readout(folder):
    frozen(folder)
    journal = folder / "scored.jsonl"
    groups = margin.archive_io.boundary.journal(journal)
    runtime = common.read(folder / "run.json")
    require(runtime["journal_sha256"] == common.digest(journal)
            and [len(group["rows"]) for group in groups] == [*([26] * 6), *([3] * 4), 4, 4, 4]
            and sum(group["verified_frames"] for group in groups) == runtime["verified_frames"], "Incomplete score journal")
    cohorts, phases = defaultdict(Counter), defaultdict(Counter)
    gains, gain_groups = Counter(), defaultdict(set)
    false_groups = defaultdict(set)
    preserved = 0
    for group in groups:
        for row in group["rows"]:
            truth, arms = row["truth"], row["arms"]
            baseline = trial.classify(arms["current"], trial.prior.POINT, truth)
            incumbent, secondary, duplicate = choose(arms)
            require((incumbent is None) == (baseline in {"unassessed", "measured_no_flag"})
                    and (incumbent is None or incumbent is arms["current"]), "Incumbent range changed")
            preserved += incumbent is not None
            cohort = "swap" if truth else group["family"]
            counts = cohorts[cohort]
            counts[f"current_{baseline}"] += 1
            if duplicate:
                counts["suppressed_duplicate"] += 1
            if secondary is not None:
                status = trial.classify(secondary, margin.POINT, truth)
                counts[f"secondary_{status}"] += 1
                counts[f"secondary_{status}_with_current_{baseline}"] += 1
                if status == "false_flag":
                    false_groups[cohort].add(group["group"])
            else:
                status = None
            useful = baseline == "localized" or status == "localized"
            counts["at_least_one_correct"] += useful
            if truth:
                phase = phases[f"{row['center']}@{row['duration']}"]
                phase[f"current_{baseline}"] += 1
                phase["at_least_one_correct"] += useful
                if status == "localized" and baseline != "localized":
                    phase["added_correct"] += 1
                    if row["duration"] == "3/8":
                        category = "wrong" if baseline == "wrong_interval" else "absent"
                        gains[category] += 1
                        gain_groups[category].add(group["group"])
    ordinary = sum(cohorts[key]["secondary_false_flag"] for key in ("rtfs", "camera"))
    passes = (sum(gains.values()) >= 18 and len(gain_groups["absent"] | gain_groups["wrong"]) >= 4
              and all(gains[key] >= 4 and len(gain_groups[key]) >= 2 for key in ("absent", "wrong"))
              and ordinary <= 2 and cohorts["cut"]["secondary_false_flag"] == 0)
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-short-return-secondary-transfer/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "rule": RULE, "advance": ADVANCE, "totals": cohorts,
        "swaps_by_center_and_duration": phases, "shortest_gains_by_current_status": gains,
        "shortest_gain_source_groups_by_current_status": {key: len(value) for key, value in gain_groups.items()},
        "additional_false_source_groups_by_cohort": {key: len(value) for key, value in false_groups.items()},
        "incumbent_flags_preserved": preserved, "pilot_advances": passes, "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--margin-run", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--texture-external", type=Path)
    parser.add_argument("--archive", type=Path, default=trial.prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.margin_run is None or args.texture_external is None:
            parser.error("freeze requires --margin-run and --texture-external")
        freeze(args.folder, args.archive, args.margin_run, args.prior, args.texture_external)
    elif args.stage == "run":
        run(args.folder, args.archive)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
