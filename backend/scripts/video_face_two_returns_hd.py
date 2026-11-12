# SPDX-License-Identifier: Apache-2.0
"""Test transfer of a second brief face range to exact 720p30 deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from fractions import Fraction
from functools import cache
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker
from scripts import video_face_720p30_transfer as hd
from scripts import video_face_return_eligibility as ranges
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
EARLIER = ROOT / "backend/data/independent-evaluation/video-face-two-returns-2026-11-12/freeze.json"
ROLES = ("original", "tandem_original", "two_swaps", "tandem_two_swaps", "swap_then_jump", "light_then_swap",
         "same_source_cuts", "cross_recording_cuts")
SPEC = {
    "schema": "chipmunk.video-face-two-returns-hd/1",
    "scope": [1280, 720, 30, 120, "h264", "yuv420p", "mp4"],
    "placements": [[[62, 68], [90, 98]], [[67, 75], [94, 100]],
                   [[64, 70], [88, 96]], [[65, 73], [91, 97]]],
    "eligibility": "Use only the already measured 15 Hz grid after the actual Full 720p30 brief review publishes a first range. "
                   "Retain prior ranges, measured source frame and background rules, 80 ms separation and the existing time budget.",
    "source_cut_match": "Choose the other held original with the highest centered luminance border correlation on picture 60. "
                        "Use a 96 by 64 area downsample and exclude rows 8–53 and columns 24–71 in the center. No inference.",
    "compression": {"primary_crf": 23, "tandem_crf": 35},
    "native_clock": "Only exact original 30 fps sources. No duplicate, interpolated or retimed source pictures.",
    "maximum_media_bytes": 256 * 1024**2,
    "gate": {"added_useful": 4, "added_source_groups": 3, "new_complete_dual_files": 2, "added_under_tandem": 1,
             "additional_ordinary": 1, "additional_wrong_mixed": 1, "additional_wrong_dual": 1, "additional_cut": 0},
}


def plan():
    import cv2

    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance
    from scripts import video_face_six_picture_transfer as panel

    archive, common, earlier_hd, _, hd_history = hd.plan_sources()
    earlier = {row["index"] for row in read(EARLIER)["sources"]} | set(earlier_hd)
    docs = [read(path) for path in sorted(common.DATA.glob("group-*-complete.json"))]
    metadata = panel.trial.prior.comparison.metadata()
    known = [metadata[doc["group"]] for doc in docs if doc["index"] in earlier]
    links, donors = {item["url"] for item in known}, set().union(*(item["donors"] for item in known))
    selected = []
    for doc in docs:
        meta = metadata[doc["group"]]
        if doc["role"] != "confirmation" or "child" in meta["tags"] or meta["url"] in links or meta["donors"] & donors:
            continue
        cases = {row["condition"]: row for row in doc["cases"]}
        table = panel.FrameTable.from_json(cases["original"]["frame_table"])
        if (table.width, table.height) not in ((1280, 720), (1920, 1080)):
            continue
        if hd.relative(table) != [Fraction(i, 30) for i in range(len(table))] or len(table) < 120:
            continue
        placements = SPEC["placements"][len(selected) % len(SPEC["placements"])]
        if not all(set(range(*span)) <= set(cases[family]["actual_changed_frames"])
                   for span, family in zip(placements, ("swap_a", "swap_b"), strict=True)):
            continue
        selected.append({**panel.encoded.prior.archive_io.source_pin(doc["index"], archive), "placements": placements})
        links.add(meta["url"])
        donors.update(meta["donors"])
    require(len(selected) == 8, "Need all eight documented source-separated groups with two exact interventions")
    paths = [Path(__file__), Path(hd.__file__), Path(face_return.__file__), Path(face_appearance.__file__), Path(worker.__file__)]
    return {"spec": SPEC, "sources": selected, "prior_360_sha256": sha(EARLIER), "prior_hd_bindings": hd_history,
            "files": {path.name: sha(path) for path in paths}, "assets": common.bindings(), "opencv": cv2.__version__}


@cache
def archive_context():
    return hd.plan_sources()[:2]


def native(pin, role, indices):
    archive, common = archive_context()
    return hd.native(archive, common, pin["index"], role, indices)


def closest_background(pins):
    import cv2
    import numpy as np

    features = []
    for pin in pins:
        still = native(pin, "original", [60])[60]
        luma = cv2.cvtColor(cv2.resize(still, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        mask = np.ones(luma.shape, dtype=bool)
        mask[8:54, 24:72] = False
        border = luma[mask].astype(np.float32)
        border -= border.mean()
        features.append(border / max(1e-6, np.linalg.norm(border)))
    scores = np.stack(features) @ np.stack(features).T
    return [max((float(scores[i, j]), -j) for j in range(len(features)) if j != i) for i in range(len(features))]


def recipe(host, first, second, guest, spans, role):
    import numpy as np

    left, right = (range(*span) for span in spans)
    if role in {"two_swaps", "tandem_two_swaps"}:
        return {**first, **second}, spans
    if role == "swap_then_jump":
        return {**first, **{i: host[i-18] for i in right}}, [spans[0]]
    if role == "light_then_swap":
        return {**{i: np.clip(host[i].astype(np.float32)*1.15, 0, 255).astype("u1") for i in left}, **second}, [spans[1]]
    if role == "same_source_cuts":
        return {**{i: host[i-22] for i in left}, **{i: host[i-18] for i in right}}, []
    if role == "cross_recording_cuts":
        return guest, []
    return {}, []


def prepare(folder, freeze):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    roster = freeze["sources"]
    media = folder/"media"
    media.mkdir(mode=0o700)
    matches = closest_background(roster)
    rows, nbytes = [], 0
    for group, pin in enumerate(roster):
        a, b = pin["placements"]
        host = native(pin, "original", range(120))
        first, second = (native(pin, kind, range(*span)) for kind, span in (("swap_a", a), ("swap_b", b)))
        _, minus_guest = matches[group]
        guest = native(roster[-minus_guest], "original", tuple(range(*a))+tuple(range(*b)))
        for role in ROLES:
            path = media/f"g{group+1}-{role}.mp4"
            replacements, truth = recipe(host, first, second, guest, pin["placements"], role)
            if role.startswith("tandem_"):
                hd.authored(path, source=media/f"g{group+1}-{role.removeprefix('tandem_')}.mp4", crf=35)
            else:
                require(all(not np.array_equal(host[i], frame) for i, frame in replacements.items()),
                        "A known picture intervention left a native picture unchanged")
                hd.authored(path, pictures=(replacements.get(i, host[i]) for i in range(120)))
            receipt = hd.describe(path, group+1, role, truth)
            rows.append({**receipt, "authored_hashes": {str(i): hashlib.sha256(frame.tobytes()).hexdigest() for i, frame in replacements.items()}})
            nbytes += path.stat().st_size
            require(nbytes <= SPEC["maximum_media_bytes"], "The bounded local delivery folder reached its storage limit")
        print("prepared", group+1, "bytes", nbytes, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 64, "Complete distinct 64-file panel required")
    write(folder/"media.json", {"freeze_sha256": sha(folder/"freeze.json"), "rows": rows,
                                "cut_matches": [[-idx+1, corr] for corr, idx in matches], "media_bytes": nbytes})


def measure_trial(measure, path, row):
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance as adapter

    original = adapter._return_pass
    result = {"status": "not_run", "reason": "no_qualified_primary_range", "range": None}

    def capture(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, method, stop, **options):
        if method != adapter.GUARDED_720P30_METHOD:
            return original(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, method, stop, **options)
        context = []
        original(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, method, stop, context=context, **options)
        primary = details.get("face_return")
        if primary is None or not context:
            return
        features, _, scenes = context[0]
        table = inputs.table
        earlier = [value for value in (primary, incumbent, details.get("face_boundary")) if value]
        if time.monotonic() >= stop:
            result.update(status="unassessed", reason="time_budget")
            return
        candidate, quartiles = face_return.second(*planned, features, table, earlier)
        if time.monotonic() >= stop:
            result.update(status="unassessed", reason="time_budget")
            return
        result.update(status="measured", reason=None,
                      previous=[[table.pts.index(item["start_pts"]), table.pts.index(item["end_pts"])] for item in earlier])
        if candidate:
            packed = ranges._compact(candidate, features, row["rgb_sha256"], table)
            packed["scene_lower_quartiles"] = quartiles
            packed["scene_inputs"] = [scenes[index] for index in candidate.scene_frames]
            packed["clock_s"] = [str(table.pts[i]*table.tb) for i in packed["frames"]]
            result["range"] = packed

    with patch.object(adapter, "_return_pass", capture):
        baseline = measure(path, row)
    return {"baseline": baseline, "second": result}


def score(folder):
    state = worker.configure(folder)
    frozen, media = read(folder/"freeze.json"), read(folder/"media.json")
    require(frozen == plan() and media["freeze_sha256"] == sha(folder/"freeze.json"), "Native panel or method changed")
    require(len(media["rows"]) == 64, "Complete held panel required")
    scorer = worker.scorer(state)
    started = time.monotonic()
    with (folder/"held.jsonl").open("x") as output:
        for row in media["rows"]:
            result = measure_trial(scorer, folder/"media"/row["file"], row)
            output.write(json.dumps({key: row[key] for key in ("file", "sha256", "group", "condition", "truth")} | result,
                                    allow_nan=False)+"\n")
            output.flush()
            print("held", row["group"], row["condition"], result["second"]["status"], flush=True)
    write(folder/"runtime.json", {"held_sha256": sha(folder/"held.jsonl"), "elapsed_s": time.monotonic()-started,
                                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def report(folder):
    media, runtime = read(folder/"media.json"), read(folder/"runtime.json")
    require(runtime["held_sha256"] == sha(folder/"held.jsonl"), "Frozen held journal differs")
    strata, gains, additions, skipped = {role: Counter() for role in ROLES}, Counter(), set(), Counter()
    with (folder/"held.jsonl").open() as stream:
        for row, source in zip(map(json.loads, stream), media["rows"], strict=True):
            require(all(row[k] == source[k] for k in ("file", "sha256", "group", "condition", "truth")), "Held input differs")
            pts = source["clock"]["pts"]
            baseline, second = row["baseline"], row["second"]
            shown = [ranges.interval(baseline[k], pts) for k in ("coarse", "boundary_refinement", "short_range") if baseline[k]]
            found = {i for i, truth in enumerate(row["truth"]) if any(ranges.overlap(item, truth) >= Fraction(1,2) for item in shown)}
            extra = second["range"]
            novel = {i for i, truth in enumerate(row["truth"]) if extra and ranges.overlap(extra["frames"], truth) >= Fraction(1,2)} - found
            role = row["condition"]
            bad = bool(extra and not novel)
            if novel:
                additions.add(row["group"])
                gains["added_useful"] += len(novel)
                gains["added_under_tandem"] += role == "tandem_two_swaps"
                gains["new_complete_dual_files"] += len(row["truth"]) == len(found|novel) == 2
            if bad:
                key = ("additional_cut" if role.endswith("_cuts") else "additional_wrong_dual" if len(row["truth"]) == 2
                       else "additional_wrong_mixed" if row["truth"] else "additional_ordinary")
                gains[key] += 1
            if second["status"] != "measured":
                coverage = baseline["short_coverage"] or {}
                skipped[coverage.get("reason") or second["reason"]] += 1
            strata[role].update(files=1, true_ranges=len(row["truth"]), useful_before=len(found), useful_after=len(found|novel),
                                full_dual_before=len(found)==2, full_dual_after=len(found|novel)==2,
                                files_flagged_before=bool(shown), files_flagged_after=bool(shown or extra),
                                second_assessed=second["status"]=="measured", wrong_additions=bad)
    gains["added_source_groups"] = len(additions)
    criteria = {key: gains[key] <= limit if key.startswith("additional") else gains[key] >= limit for key, limit in SPEC["gate"].items()}
    write(folder/"aggregate.json", {"protocol": SPEC, "freeze_sha256": sha(folder/"freeze.json"), "runtime": runtime, "strata": strata,
                                    "advance": all(criteria.values()), "gate": criteria, "gains": {key:gains[key] for key in SPEC["gate"]},
                                    "cut_background_correlations": sorted(corr for _, corr in media["cut_matches"]),
                                    "not_assessed_reasons": skipped, "media_bytes": media["media_bytes"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "report"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        score(args.folder)
        return
    seal = plan()
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, parents=True)
        write(args.folder/"freeze.json", seal)
        return
    require(seal == read(args.folder/"freeze.json"), "The source roster or frozen method changed")
    if args.phase == "prepare":
        prepare(args.folder, seal)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
