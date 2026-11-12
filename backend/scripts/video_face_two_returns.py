# SPDX-License-Identifier: Apache-2.0
"""Measure a second disjoint short face-change range on exact paired deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker
from scripts import video_face_return_eligibility as ranks
from scripts.video_common_sources_v89 import read, require, sha, write

SPEC = {
    "schema": "chipmunk.video-face-two-returns/1",
    "profile": [640, 360, 25, 120, "h264", "yuv420p"],
    "native_intervals": [[52, 58], [78, 84]],
    "fit": "No fitting. Freeze eight confirmation sources by metadata order and verified changed source pixels.",
    "ranking": "Retain the actual Full face ranges. If there is at least one existing range and the native pass has "
               "measurable same-shot context, rank new disjoint candidates on the existing 12.5 Hz grid. Reuse measured "
               "pictures, or measure that grid within the existing budget if the sparse pass selected the first range.",
    "minimum_separation_frames": 2,
    "minimum_scene_lower_quartile": .60,
    "scope": "A second interval of observed local face change only, zero AI confidence weight.",
    "truth": "Half or greater temporal intersection over union on distinct authored intervals.",
    "gate": {"added_useful_ranges": 4, "added_source_groups": 3, "additional_ordinary_flags": 1,
             "additional_cut_flags": 0, "additional_wrong_single": 1, "additional_wrong_dual": 1},
}
ROLES = ("original", "recompressed", "single_early", "single_late", "dual_swap",
         "swap_then_jump", "light_then_swap", "dual_picture_cuts")
FIRST, SECOND = [range(*span) for span in SPEC["native_intervals"]]


def plan():
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance
    from app.evidence import face_return_binding
    from scripts import video_face_six_picture_transfer as panel

    metadata = panel.trial.prior.comparison.metadata()
    candidates = []
    for path in panel.common.DATA.glob("group-*-complete.json"):
        doc = read(path)
        meta = metadata[doc["group"]]
        if doc["role"] != "confirmation" or "child" in meta["tags"]:
            continue
        cases = {row["condition"]: row for row in doc["cases"]}
        table = panel.FrameTable.from_json(cases["original"]["frame_table"])
        size = table.width, table.height
        if size not in panel.encoded.PROFILES or panel.common.relative_times(table) != panel.encoded.CLOCK:
            continue
        if not set(FIRST) <= set(cases["swap_a"]["actual_changed_frames"]) or not set(SECOND) <= set(cases["swap_b"]["actual_changed_frames"]):
            continue
        key = hashlib.sha256(f"{SPEC['schema']}/{doc['group']}".encode()).hexdigest()
        candidates.append((key, doc["index"], size, meta))
    chosen, urls, donors = [], set(), set()
    for _, index, size, meta in sorted(candidates):
        if meta["url"] in urls or donors & meta["donors"]:
            continue
        chosen.append({**panel.encoded.prior.archive_io.source_pin(index, panel.ARCHIVE), "size": list(size)})
        urls.add(meta["url"])
        donors.update(meta["donors"])
        if len(chosen) == 8:
            break
    require(len(chosen) == 8, "Need eight separated host and donor groups")
    paths = [Path(module.__file__) for module in (face_return, face_appearance, face_return_binding, ranks, panel, worker)]
    return {"spec": SPEC, "sources": chosen, "sha256": {path.name: sha(path) for path in paths + [Path(__file__)]},
            "models": panel.common.bindings()}


def source(pin, condition, wanted):
    from scripts import video_face_six_picture_transfer as panel

    index = pin["index"]
    exact = panel.encoded.prior.archive_io.source_pin(index, panel.ARCHIVE)
    require({key: pin[key] for key in exact} == exact, "Qualified native source differs")
    doc = read(panel.common.DATA / f"group-{index:03d}-complete.json")
    case = next(item for item in doc["cases"] if item["condition"] == condition)
    path = panel.ARCHIVE / "video-native-inserts-v95" / f"group-{index:03d}" / case["file"]
    require(sha(path) == case["sha256"], "Qualified archive video differs")
    return panel.encoded.native_frames(path, case, tuple(pin["size"]), wanted)


def authored(host, swaps, guest, role):
    import numpy as np

    if role == "dual_picture_cuts":
        return guest, []
    if role in ("single_early", "swap_then_jump"):
        altered, truth = dict(swaps[0]), [list(SPEC["native_intervals"][0])]
        if role == "swap_then_jump":
            altered.update({i: host[i-20] for i in SECOND})
        return altered, truth
    if role in ("single_late", "light_then_swap"):
        altered, truth = dict(swaps[1]), [list(SPEC["native_intervals"][1])]
        if role == "light_then_swap":
            altered.update({i: np.clip(host[i].astype(np.float32)*1.15, 0, 255).astype(np.uint8) for i in FIRST})
        return altered, truth
    if role == "dual_swap":
        return {**swaps[0], **swaps[1]}, list(SPEC["native_intervals"])
    return {}, []


def prepare(folder, frozen):
    import cv2
    import numpy as np

    from scripts import video_face_six_picture_transfer as panel

    cv2.setNumThreads(1)
    media = folder / "media"
    media.mkdir(mode=0o700)
    rows = []
    pins = frozen["sources"]
    for group, pin in enumerate(pins):
        host = source(pin, "original", range(120))
        swapped = [source(pin, variant, wanted) for variant, wanted in (("swap_a", FIRST), ("swap_b", SECOND))]
        guest = source(pins[group ^ 1], "original", tuple(FIRST)+tuple(SECOND))
        for role in ROLES:
            path = media / f"g{group+1}-{role}.mp4"
            altered, truth = authored(host, swapped, guest, role)
            if role == "recompressed":
                panel.encoded.encode(path, source=media / f"g{group+1}-original.mp4", crf=36)
            else:
                require(all(not np.array_equal(frame, host[i]) for i, frame in altered.items()), "Intervention left a picture unchanged")
                panel.encoded.encode(path, frames=(altered.get(i, frame) for i, frame in host.items()), crf=23)
            receipt = panel.describe(path, group+1, "held", role, truth)
            rows.append({**receipt, "authored_rgb_sha256": {i: hashlib.sha256(frame.tobytes()).hexdigest() for i, frame in altered.items()}})
        print("prepared", group+1, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 64, "Incomplete 64-delivery roster")
    write(folder / "media.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": rows})


def candidate_for(observed, table, prior):
    from app.detectors.video import face_return as cue

    times, indices, features = observed
    pad = SPEC["minimum_separation_frames"]
    point = SPEC["minimum_scene_lower_quartile"]

    def distinct(candidate, frame_table, incumbent):
        if not cue_distinct(candidate, frame_table, incumbent):
            return False
        if any(not (candidate.end + pad <= a or b + pad <= candidate.start) for a, b in prior):
            return False
        for a, b in (candidate.scene_frames[:2], candidate.scene_frames[2:]):
            value = cue.scene_lower_quartile(features[a], features[b])
            if value is None or value < point:
                return False
        return True

    cue_distinct = cue.distinct
    with patch.object(cue, "distinct", distinct):
        _, selected = ranks.ranks(times, indices, features, table, None)
    return selected, features


def score(folder):
    state = worker.configure(folder)
    from app.detectors.video import face_return as cue
    from app.detectors.video.adapters import face_appearance as adapter

    frozen = read(folder / "freeze.json")
    require(frozen == plan(), "Method or source binding differs from freeze")
    roster = read(folder / "media.json")
    require(roster["freeze_sha256"] == sha(folder / "freeze.json") and len(roster["rows"]) == 64, "Frozen deliveries differ")
    measure = worker.scorer(state)
    inspect = adapter._inspect_return
    rank = cue.highest
    started = time.monotonic()
    with (folder / "held.jsonl").open("x") as output:
        for row in roster["rows"]:
            attempt = {"status": "not_run", "reason": "no_native_return_stage", "candidate": None}
            wanted = tuple(range(0, 120, 2))
            exact = {}

            def capture(times, indices, features, *, compare_scenes=None):
                if tuple(indices) == wanted and compare_scenes is None:
                    exact["context"] = times, indices, features
                return rank(times, indices, features, compare_scenes=compare_scenes)

            def observe(details, inputs, yunet, recognizer, deadline, incumbent):
                remaining = deadline.remaining() if deadline is not None else None
                budget = adapter.RETURN_BUDGET_S if remaining is None else min(adapter.RETURN_BUDGET_S, remaining-adapter.COARSE_RESULT_RESERVE_S)
                stop = time.monotonic() + budget
                with patch.object(cue, "highest", capture):
                    inspect(details, inputs, yunet, recognizer, deadline, incumbent)
                    table = inputs.table
                    before = [item for item in (incumbent, details.get("face_return")) if item]
                    coverage = details.get("face_return_check") or {}
                    if not before or coverage.get("status") not in {"localized", "no_distinct_range"}:
                        attempt["reason"] = "first_range_or_context_unavailable"
                        return
                    dense = cue.dense_schedule(table, inputs.encoding)
                    if dense is None:
                        attempt["reason"] = "outside_dense_profile"
                        return
                    if "context" not in exact:
                        # This adds native decoding only after the sparse pass has selected an existing range.
                        if not details.get("face_return") or details["face_return"]["method"] != cue.METHOD:
                            attempt["reason"] = "dense_context_unavailable"
                            return
                        status = {"measured_face_frames": 0}
                        adapter._return_pass({}, inputs, yunet, recognizer, deadline, incumbent, dense, status,
                                             adapter.DENSE_FRAME_PROFILES[len(table)], stop)
                        attempt["extra_native_pass"] = status
                if "context" not in exact or time.monotonic() >= stop:
                    attempt.update(status="unassessed", reason="time_or_native_measurement_unavailable")
                    return
                bounds = [(table.pts.index(item["start_pts"]), table.pts.index(item["end_pts"])) for item in before]
                picked, features = candidate_for(exact["context"], table, bounds)
                if time.monotonic() >= stop:
                    attempt.update(status="unassessed", reason="time_budget")
                    return
                attempt.update(status="measured", reason=None, excluded_frames=bounds)
                if picked:
                    packed = ranks._compact(picked, features, row["rgb_sha256"], table)
                    packed["clock_s"] = [str(table.pts[i]*table.tb) for i in packed["frames"]]
                    packed["scene_lower_quartiles"] = [cue.scene_lower_quartile(features[a], features[b]) for a, b in
                                                        (picked.scene_frames[:2], picked.scene_frames[2:])]
                    attempt["candidate"] = packed

            with patch.object(adapter, "_inspect_return", observe):
                result = measure(folder / "media" / row["file"], row)
            output.write(json.dumps({"file": row["file"], "sha256": row["sha256"], "truth": row["truth"], "group": row["group"],
                                     "condition": row["condition"], "before": result, "additional": attempt}, allow_nan=False) + "\n")
            output.flush()
            print("measured", row["group"], row["condition"], attempt["status"], bool(attempt["candidate"]), flush=True)
    write(folder / "runtime.json", {"held_sha256": sha(folder / "held.jsonl"), "elapsed_s": time.monotonic()-started,
                                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def report(folder):
    inputs = {row["file"]: row for row in read(folder / "media.json")["rows"]}
    runtime = read(folder / "runtime.json")
    require(runtime["held_sha256"] == sha(folder / "held.jsonl"), "Measured rows changed")
    held = [json.loads(line) for line in (folder / "held.jsonl").read_text().splitlines()]
    require(set(inputs) == {row["file"] for row in held} and len(held) == 64, "Incomplete held results")
    groups, strata, added = set(), {role: Counter() for role in ROLES}, Counter()
    for row in held:
        pin = inputs[row["file"]]
        require(row["sha256"] == pin["sha256"] and row["truth"] == pin["truth"], "Held input differs")
        status, after = row["additional"]["status"], row["additional"]["candidate"]
        truth = row["truth"]
        pts = pin["clock"]["pts"]
        before = [ranks.interval(row["before"][key], pts) for key in ("coarse", "boundary_refinement", "short_range")
                  if row["before"][key] is not None]
        found = {j for j, span in enumerate(truth) if any(ranks.overlap(item, span) >= Fraction(1,2) for item in before)}
        novel = {j for j, span in enumerate(truth) if after and ranks.overlap(after["frames"], span) >= Fraction(1,2)} - found
        errors = bool(after and not novel)
        condition = row["condition"]
        strata[condition].update({"files": 1, "truth_ranges": len(truth), "before_useful": len(found), "after_useful": len(found | novel),
                                  "before_files_flagged": bool(before), "after_files_flagged": bool(before or after),
                                  "second_measured": status == "measured", "added_wrong": errors})
        if novel:
            groups.add(row["group"])
            added["added_useful_ranges"] += len(novel)
        if errors:
            category = ("additional_cut_flags" if condition == "dual_picture_cuts" else "additional_wrong_dual" if len(truth) == 2
                        else "additional_wrong_single" if truth else "additional_ordinary_flags")
            added[category] += 1
    added["added_source_groups"] = len(groups)
    criteria = {key: added[key] <= bound if key.startswith("additional") else added[key] >= bound
                for key, bound in SPEC["gate"].items()}
    write(folder / "aggregate.json", {"schema": SPEC["schema"], "freeze_sha256": sha(folder / "freeze.json"),
          "spec": SPEC, "strata": strata, "gates": criteria, "advances": all(criteria.values()),
          "added": {key: added[key] for key in SPEC["gate"]}, "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "report"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    if args.phase == "score":
        score(folder)
        return
    if args.phase == "freeze":
        frozen = plan()
        folder.mkdir(mode=0o700, parents=True)
        write(folder / "freeze.json", frozen)
        return
    frozen = read(folder / "freeze.json")
    require(frozen == plan(), "Frozen video trial inputs differ")
    if args.phase == "prepare":
        prepare(folder, frozen)
    else:
        report(folder)


if __name__ == "__main__":
    main()
