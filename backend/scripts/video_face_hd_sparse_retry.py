# SPDX-License-Identifier: Apache-2.0
"""Test guarded 720p face review after the sparse check finds an unpublished range."""

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
from scripts import video_face_720p30_transfer as hd
from scripts import video_face_return_eligibility as spans
from scripts import video_face_two_returns_hd as earlier
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "backend/data/independent-evaluation/video-face-two-returns-hd-2026-11-12/freeze.json"
EARLY = (61, 69)
LATE = (83, 93)
ROLES = ("dual_swap", "tandem_dual_swap", "genuine_original", "tandem_original",
         "same_source_jump", "cross_recording_cuts")
SPEC = {
    "schema": "chipmunk.video-face-hd-sparse-retry/1",
    "scope": [1280, 720, 30, 120, "h264", "yuv420p", "mp4"],
    "candidate": "When the sparse pass has an otherwise qualifying range outside the 720p publication scope, "
                 "try the unchanged 15 Hz guarded pass within the original deadline. Reuse native inputs for "
                 "the existing second range if the dense pass qualifies.",
    "placements": [list(EARLY), list(LATE)],
    "sources": "Eight archived exact-30fps groups separated by original links and known donor keys from three preceding studies.",
    "codec": {"primary_crf": 23, "tandem_crf": 35},
    "gate": {"added_correct": 3, "sources_with_gain": 2, "added_ordinary_flags": 1, "added_cut_flags": 0,
             "added_misplaced_ranges": 1, "minimum_overlap": "1/2"},
}


def freeze_plan():
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance
    from scripts import video_face_six_picture_transfer as panel

    archive, common, published, _, history = hd.plan_sources()
    exclude = set(published) | {r["index"] for path in (earlier.EARLIER, PREVIOUS) for r in read(path)["sources"]}
    sources = [read(p) for p in common.DATA.glob("group-*-complete.json")]
    metadata = panel.trial.prior.comparison.metadata()
    known = [metadata[row["group"]] for row in sources if row["index"] in exclude]
    urls = {row["url"] for row in known}
    donors = set().union(*(row["donors"] for row in known))
    chosen = []
    for source in sorted(sources, key=lambda r: hashlib.sha256(f"{SPEC['schema']}/{r['group']}".encode()).hexdigest()):
        meta = metadata[source["group"]]
        if meta["url"] in urls or meta["donors"] & donors or "child" in meta["tags"]:
            continue
        cases = {row["condition"]: row for row in source["cases"]}
        table = panel.FrameTable.from_json(cases["original"]["frame_table"])
        if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or len(table) < 120:
            continue
        if hd.relative(table) != [Fraction(i, 30) for i in range(len(table))]:
            continue
        if not all(set(range(*interval)) <= set(cases[role]["actual_changed_frames"])
                   for interval, role in ((EARLY, "swap_a"), (LATE, "swap_b"))):
            continue
        chosen.append(panel.encoded.prior.archive_io.source_pin(source["index"], archive))
        urls.add(meta["url"])
        donors.update(meta["donors"])
        if len(chosen) == 8:
            break
    require(len(chosen) == 8, "Need eight source-separated exact-clock groups")
    bindings = [Path(module.__file__) for module in (face_appearance, face_return, worker, earlier, hd, spans)]
    return {"protocol": SPEC, "sources": chosen, "source_history": [*history, sha(PREVIOUS), sha(earlier.EARLIER)],
            "models": common.bindings(), "bindings": {p.name: sha(p) for p in [Path(__file__), *bindings]}}


def prepare(folder, lock):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    media = folder/"media"
    media.mkdir(mode=0o700)
    sources = lock["sources"]
    matches = earlier.closest_background(sources)
    rows = []
    for n, pin in enumerate(sources, 1):
        base = earlier.native(pin, "original", range(120))
        changed = {**earlier.native(pin, "swap_a", range(*EARLY)), **earlier.native(pin, "swap_b", range(*LATE))}
        _, guest = matches[n-1]
        cuts = earlier.native(sources[-guest], "original", (*range(*EARLY), *range(*LATE)))
        jump = {i: base[i-18] for i in (*range(*EARLY), *range(*LATE))}
        for role in ROLES:
            path = media/f"g{n}-{role}.mp4"
            altered = {"dual_swap": changed, "same_source_jump": jump, "cross_recording_cuts": cuts}.get(role, {})
            truth = SPEC["placements"] if role in ("dual_swap", "tandem_dual_swap") else []
            if role.startswith("tandem_"):
                base_role = "genuine_original" if role == "tandem_original" else "dual_swap"
                hd.authored(path, source=media/f"g{n}-{base_role}.mp4", crf=35)
            else:
                require(all(not np.array_equal(base[i], rgb) for i, rgb in altered.items()), "An authored picture did not change")
                hd.authored(path, pictures=(altered.get(i, base[i]) for i in range(120)))
            rows.append(hd.describe(path, n, role, truth))
        print("prepared", n, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 48, "Need all 48 distinct deliveries")
    write(folder/"media.json", {"freeze_sha256": sha(folder/"freeze.json"), "rows": rows,
                                "cut_border_correlations": sorted(corr for corr, _ in matches)})


def test_file(scorer, path, row):
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance as adapter

    original = adapter._return_pass
    attempts, incumbent_second = [], []

    def capture(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, method, stop, **options):
        original(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, method, stop, **options)
        if method != face_return.METHOD or coverage["status"] != "localized" or not face_return.guarded_720p30_schedule(
                inputs.table, inputs.encoding, path):
            return
        # This is the existing sparse-only disposition. The real adapter preserves it
        # after this isolated extra pass has measured a separate details record.
        plan = planned[0], tuple(range(0, len(inputs.table), 2))
        new, state, context = {}, {"measured_face_frames": 0}, []
        original(new, inputs, yunet, recognizer, deadline, incumbent, plan, state, adapter.GUARDED_720P30_METHOD, stop,
                 source_carrier="mp4", context=context)
        if context and (primary := new.get("face_return")) is not None:
            adapter._record_second_return(new, inputs, incumbent, primary, plan, stop, context[0], adapter.SECOND_HD_METHOD)
        attempts.append({"coverage": state, "ranges": [new[k] for k in ("face_return", "face_return_secondary") if k in new]})

    old_second = adapter._record_second_return

    def capture_second(details, *args):
        old_second(details, *args)
        if record := details.get("face_return_secondary"):
            incumbent_second.append(record)

    with patch.object(adapter, "_return_pass", capture), patch.object(adapter, "_record_second_return", capture_second):
        before = scorer(path, row)
    require(len(attempts) <= 1, "More than one optional retry was measured")
    return {"before": before, "incumbent_second": incumbent_second if not attempts else [],
            "retry": attempts[0] if attempts else None}


def score(folder):
    state = worker.configure(folder)
    import os

    os.environ.update(VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true")
    lock, media = read(folder/"freeze.json"), read(folder/"media.json")
    require(lock == freeze_plan() and media["freeze_sha256"] == sha(folder/"freeze.json"), "Frozen delivery inputs differ")
    run = worker.scorer(state)
    began = time.monotonic()
    with (folder/"held.jsonl").open("x") as output:
        for row in media["rows"]:
            answer = test_file(run, folder/"media"/row["file"], row)
            output.write(json.dumps({k: row[k] for k in ("file", "group", "condition", "truth", "sha256")} | answer, allow_nan=False)+"\n")
            output.flush()
            print("held", row["group"], row["condition"], "retry" if answer["retry"] else "original scope", flush=True)
    write(folder/"runtime.json", {"held_sha256": sha(folder/"held.jsonl"), "elapsed_s": time.monotonic()-began,
                                   "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def report(folder):
    runtime = read(folder/"runtime.json")
    require(runtime["held_sha256"] == sha(folder/"held.jsonl"), "Held journal differs")
    source = read(folder/"media.json")["rows"]
    strata, groups, gains = {role: Counter() for role in ROLES}, set(), Counter()
    with (folder/"held.jsonl").open() as stream:
        for row, pin in zip(map(json.loads, stream), source, strict=True):
            require(all(row[k] == pin[k] for k in ("file", "group", "condition", "truth", "sha256")), "Evaluated input differs")
            base, retry = row["before"], row["retry"]
            require((retry is not None) == ((base["short_coverage"] or {}).get("reason") == "sparse_range_outside_profile"),
                    "Retry routing differs from actual serving behavior")
            pts = pin["clock"]["pts"]
            records = [base[k] for k in ("coarse", "boundary_refinement", "short_range") if base[k]] + row["incumbent_second"]
            old = [spans.interval(r, pts) for r in records]
            after = [spans.interval(r, pts) for r in retry["ranges"]] if retry else []
            truth = row["truth"]
            found = {i for i, t in enumerate(truth) if any(spans.overlap(a, t) >= Fraction(1,2) for a in old)}
            new = {i for i, t in enumerate(truth) if any(spans.overlap(a, t) >= Fraction(1,2) for a in after)} - found
            wrong = sum(not any(spans.overlap(a, t) >= Fraction(1,2) for t in truth) for a in after)
            role = row["condition"]
            if new:
                gains["added_correct"] += len(new)
                groups.add(row["group"])
            gains["added_misplaced_ranges"] += wrong if truth else 0
            if not truth and after:
                gains["added_cut_flags" if role in ("same_source_jump", "cross_recording_cuts") else "added_ordinary_flags"] += 1
            strata[role].update(files=1, truth=len(truth), before=len(found), after=len(found|new), retry=retry is not None,
                                retry_measured=bool(retry and retry["coverage"]["status"] in ("localized", "no_distinct_range")),
                                added_wrong=wrong, before_flag=bool(old), after_flag=bool(old or after))
    gains["sources_with_gain"] = len(groups)
    gate = {key: gains[key] <= limit if key in ("added_ordinary_flags", "added_cut_flags", "added_misplaced_ranges")
            else gains[key] >= limit for key, limit in SPEC["gate"].items() if key != "minimum_overlap"}
    write(folder/"aggregate.json", {"protocol": SPEC, "freeze_sha256": sha(folder/"freeze.json"), "strata": strata,
                                     "gains": {k: gains[k] for k in gate}, "gate": gate, "advances": all(gate.values()), "runtime": runtime})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "report"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        score(args.folder)
        return
    plan = freeze_plan()
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, parents=True)
        write(args.folder/"freeze.json", plan)
        return
    require(plan == read(args.folder/"freeze.json"), "Frozen method or roster changed")
    if args.phase == "prepare":
        prepare(args.folder, plan)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
