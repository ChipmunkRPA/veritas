# SPDX-License-Identifier: Apache-2.0
"""Freeze and test distributed fine-background support on existing Full video lanes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker

BASE = Path(__file__).resolve().parents[1] / "data/independent-evaluation"
DEVELOPMENT = BASE / "video-face-return-fine-scene-development-2026-11-12"
SCENE = "video-face-scene-continuity-2026-11-12"
PANEL = "video-native-background-panel-2026-11-12"
ELIGIBILITY = "video-face-return-eligibility-2026-11-12"
DEVELOPMENT_FILES = {
    "candidate_development.py": "0e9dfad53cb995baee305f63f06b8e107161bc2dd42ef9c9de5c37ff6917c989",
    "freeze.json": "3dc73c58ddcc068e01bf588ea600f7fb599c40fba7445618c869cf98e57bf918",
    "development-result.json": "318460d7b4e5f915a2c8db0b310c0f063999a9fff7e38e3ea15c23546eeebfd0",
    "source-only-preflight.json": "eef3c3e71c8863666fe0adcd94ff1a1f5023aa9b1f478af3d2a4e083553915a3",
    "source-only-extra-preflight.json": "a3403c7d6ed546b00f38ca1f70497d865be79b448e5384fd64f5584ee6cad25f",
    "source-only-genuine-cut-inventory.json": "9a672c1f637b842bf758b497c7bf042e5a659132bb65f4824b5014fdf8311821",
}
FRESH_SPANS = ((54, 60), (67, 73))
SHIFTS = (0, 2, -2, -4, -6, -8, -10, -12)
SPEC = {
    "schema": "chipmunk.video-face-return-fine-scene/1",
    "scope": "Exact upright 640x360 25 fps H.264 4:2:0 MP4s, 120 native pictures. Three known-key-separated low-texture source groups, independent of the g4/g7 development pair. These recordings appeared in other project studies.",
    "candidate": "Preserve all existing ranges. When neither incumbent short pass produced a range, consider the same dense rank with the same face, response, duration and deduplication rules. Supplement only unavailable original scene edges using the frozen fine-grid checks. Keep every observed low-correlation rejection. A selected candidate needs at least one supplemented edge. Retry only after complete valid source measurement and within the original short-check budget.",
    "selection": "Freeze five existing known short swap deliveries, eight exact ordinary cross-recording cuts and sixteen ordinary original/recompression/lighting/motion controls. Author two further six-picture swaps in one separately retained low-texture source. Try the fixed interval and preferred publisher variant first, alternate variant second, then the fixed shifts. Require every changed picture to differ from the authenticated same-time original before encoding. Never select on detector output.",
    "new_swap_intervals": [list(value) for value in FRESH_SPANS], "authoring_shift_order": list(SHIFTS),
    "gate": {"minimum_added_useful": 3, "minimum_positive_groups": 2, "maximum_added_ordinary": 1,
             "maximum_added_cuts": 0, "maximum_lost_prior_ranges": 0, "minimum_useful_iou": "1/2"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def definitions():
    entries = {
        SCENE: [(f"g0-swap_{v}.mp4", "swap") for v in ("a", "b")]
               + [(f"g{i}-cross_cut_{j}.mp4", "cut") for i in (0, 1) for j in ("a", "b")]
               + [(f"g0-{c}.mp4", "ordinary") for c in ("original", "recompressed", "light", "camera_punch", "jump_cut")],
        PANEL: [(f"pair-04-{i}-cross_cut.mp4", "cut") for i in (0, 1)]
               + [(f"pair-04-{name}.mp4", "ordinary") for name in ("0-original", "0-camera_punch", "1-light")],
        ELIGIBILITY: [(f"g07-{name}.mp4", "swap") for name in ("six_even", "six_odd", "ten_odd")]
                     + [(f"g{i:02d}-cut.mp4", "cut") for i in (6, 7)]
                     + [(f"g{i:02d}-{name}.mp4", "ordinary") for i in (6, 7) for name in ("original", "recompressed", "light", "jump")],
    }
    rows = []
    cut_inventory = {item["sha256"]: item for item in read(DEVELOPMENT / "source-only-genuine-cut-inventory.json")["rows"]}
    for bank, values in entries.items():
        media = {row["file"]: row for row in read(BASE / bank / "media.json")["media"]}
        for filename, role in values:
            row = media[filename]
            require((role == "swap") == bool(row["truth"]), "Frozen treatment differs from source manifest")
            if role == "cut":
                cut = cut_inventory[row["sha256"]]
                require(all(v is not None and v < 8 for v in cut["old_texture_tile_support"]), "Cut lacks preregistered old-deficient edges")
            rows.append({**row, "bank": bank, "role": role, "source_scope": bank, "cut_truth": cut_inventory[row["sha256"]]["authored"] if role == "cut" else None})
    return rows


def native_host():
    bank = read(BASE / PANEL / "freeze.json")
    source_index = bank["pairs"][4][1]
    return next(item for item in bank["source_roster"] if item["index"] == source_index)


def fresh(archive, folder):
    import cv2
    import numpy as np

    from scripts import video_native_background_panel as panel

    cv2.setNumThreads(1)
    source = native_host()
    original = panel.source_pictures(archive, source, "original", range(120))
    wanted = sorted({i for a, b in FRESH_SPANS for shift in SHIFTS for i in range(a+shift, b+shift)})
    variants = {name: panel.source_pictures(archive, source, name, wanted) for name in ("swap_a", "swap_b")}
    rows = []
    for number, (a, b) in enumerate(FRESH_SPANS):
        order = ("swap_a", "swap_b") if number == 0 else ("swap_b", "swap_a")
        candidates = (((a+shift, b+shift), variant) for shift in SHIFTS for variant in order)
        chosen = next(((span, variant) for span, variant in candidates
                       if all(not np.array_equal(original[i], variants[variant][i]) for i in range(*span))
                       and all(list(span) != row["truth"] for row in rows)), None)
        require(chosen is not None, "Fewer than seven known short edits are available under the source-only recipe")
        interval, variant = chosen
        path = folder / f"pair-04-0-known-short-{number}.mp4"
        first, last = interval
        panel.old.encoded.encode(path, frames=(variants[variant][i] if first <= i < last else rgb for i,rgb in original.items()), crf=23)
        row = panel.old.describe(path, 8, "held", "new_short", interval)
        rows.append({**row, "truth": list(interval), "bank": "authored", "role": "swap", "source_scope": PANEL,
                     "cut_truth": None, "publisher_variant": variant})
    return rows


def path_for(folder, row):
    parent = folder if row["bank"] == "authored" else BASE / row["bank"]
    return parent / "media" / row["file"]


def bindings():
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance
    from scripts import replay_video_native_background_panel as prior
    from scripts import video_face_return_fine_scene as method

    for name, expected in DEVELOPMENT_FILES.items():
        require(digest(DEVELOPMENT / name) == expected, "Development method or source-only preflight changed")
    require(read(DEVELOPMENT / "freeze.json")["parameters"] == method.CFG, "Scratch parameter mismatch")
    paths = (Path(__file__), Path(method.__file__), Path(worker.__file__), Path(face_return.__file__), Path(face_appearance.__file__))
    return {"implementation": {p.name: digest(p) for p in paths},
            "full_stages": {name: digest(prior.APP / name) for name in prior.BINDINGS},
            "past_source_ledgers": {name: digest(BASE / name / "media.json") for name in (SCENE, PANEL, ELIGIBILITY)},
            "development": DEVELOPMENT_FILES}


def prepare(folder, archive):
    frozen_inputs = bindings()
    roster = definitions()
    folder.mkdir(mode=0o700)
    (folder / "media").mkdir(mode=0o700)
    roster += fresh(archive, folder / "media")
    require(Counter(row["role"] for row in roster) == {"swap": 7, "cut": 8, "ordinary": 16}, "Incomplete exact roster")
    require(len({row["sha256"] for row in roster}) == len(roster)
            and all(digest(path_for(folder,row)) == row["sha256"] for row in roster), "Native held deliveries changed")
    write(folder / "manifest.json", {"media": roster})
    write(folder / "freeze.json", {"protocol": SPEC, "bindings": frozen_inputs, "manifest_sha256": digest(folder / "manifest.json"),
                                   "fresh_native_source": native_host()})
    print("frozen 31 native files", flush=True)


def frozen(folder):
    lock = read(folder / "freeze.json")
    require(lock == {"protocol": SPEC, "bindings": bindings(), "manifest_sha256": digest(folder / "manifest.json"),
                     "fresh_native_source": native_host()}, "Native inputs or method changed since freeze")
    return lock, read(folder / "manifest.json")["media"]


def verify_port():
    import cv2
    import numpy as np

    from app.detectors.image.faces import Face
    from scripts import video_face_return_fine_scene as method

    tree = ast.parse((DEVELOPMENT / "candidate_development.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("fine_frame", "fine_edge")]
    scope = {"np": np, "cv2": cv2, "CFG": method.CFG}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<pinned-development>", "exec"), scope)
    rng = np.random.default_rng(742)
    frames = [rng.integers(0,256,(360,640,3),dtype=np.uint8), np.full((360,640,3),98,dtype=np.uint8)]
    faces = [None, Face(191, 52, 172, 209, .99, ())]
    count = 0
    for frame in frames:
        for face in faces:
            for variation in (frame, np.roll(frame, 1, axis=0)):
                old = [scope["fine_frame"](rgb,face) for rgb in (frame,variation)]
                new = [method.fine_frame(rgb,face) for rgb in (frame,variation)]
                require(all(np.array_equal(a,b) for first, second in zip(old,new,strict=True) for a,b in zip(first,second,strict=True))
                        and scope["fine_edge"](*old) == method.fine_edge(*new), "Fixed fine-grid port differs from scratch")
                count += 1
    return count


def score(folder, state):
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance as adapter
    from scripts import video_face_return_fine_scene as method

    lock, roster = frozen(folder)
    parity = verify_port()
    measure = worker.scorer(state)
    original_inspect, original_scene = adapter._inspect_return, face_return.scene
    original_correlation, original_highest = face_return.scene_correlation, face_return.highest
    trial = {}

    def inspect(details, inputs, yunet, recognizer, deadline, incumbent):
        remaining = deadline.remaining() if deadline is not None else None
        budget = adapter.RETURN_BUDGET_S if remaining is None else min(adapter.RETURN_BUDGET_S, remaining-adapter.COARSE_RESULT_RESERVE_S)
        stop = time.monotonic() + budget
        original_inspect(details,inputs,yunet,recognizer,deadline,incumbent)
        coverage = details.get("face_return_check", {})
        dense_coverage = coverage.get("dense_fallback", {})
        scoped = coverage.get("status") == "no_distinct_range" or coverage.get("reason") == "insufficient_same_shot_context"
        if details.get("face_return") or not scoped or (dense_coverage.get("reason") not in (None,"insufficient_same_shot_context")):
            return
        dense = face_return.dense_schedule(inputs.table,inputs.encoding)
        if dense is None or time.monotonic() >= stop:
            return
        fine, observed = {}, {}
        supplemented = {}

        def combined_scene(rgb, face):
            coarse = original_scene(rgb,face)
            fine[id(coarse)] = method.fine_frame(rgb,face)
            return coarse

        def combined_correlation(first, second):
            value = original_correlation(first,second)
            if value is not None:
                return value
            evidence = method.fine_edge(fine[id(first.scene)],fine[id(second.scene)])
            evidence["old_count"] = int((first.scene[1] & second.scene[1]).sum())
            supplemented[id(first.scene),id(second.scene)] = evidence
            return evidence["median"] if evidence["passes"] else None

        def capture(times, indices, features):
            value = original_highest(times,indices,features)
            if value is not None:
                a,b,c,d = value.scene_frames
                evidence = [supplemented.get((id(features[left].scene),id(features[right].scene))) for left,right in ((a,b),(c,d))]
                observed["edges"] = evidence
            return value

        proof = {}
        status = {"measured_face_frames": 0}
        with (patch.object(face_return,"scene",combined_scene), patch.object(face_return,"scene_correlation",combined_correlation),
              patch.object(face_return,"highest",capture)):
            adapter._return_pass(proof,inputs,yunet,recognizer,deadline,incumbent,dense,status,
                                 adapter.DENSE_FRAME_PROFILES[len(inputs.table)],stop)
        trial["status"] = status
        if proof.get("face_return") and any(observed.get("edges", ())):
            trial["candidate"] = {"record": proof["face_return"], "fine_edges": observed["edges"]}

    journal = folder / "held.jsonl"
    start = time.monotonic()
    with journal.open("x") as stream, patch.object(adapter,"_inspect_return",inspect):
        for number,row in enumerate(roster,1):
            trial.clear()
            baseline = measure(path_for(folder,row),row)
            outcome = {**{k:row[k] for k in ("sha256","role","source_scope","truth","cut_truth")},
                       "baseline":baseline,"candidate":trial.get("candidate"),"retry_coverage":trial.get("status")}
            stream.write(json.dumps(outcome,sort_keys=True,allow_nan=False)+"\n")
            stream.flush()
            if number%4 == 0:
                print("completed exact Full views",number,"of",len(roster),flush=True)
    require(frozen(folder)[0] == lock, "Frozen bindings changed during actual Full scoring")
    write(folder / "held-seal.json", {"journal_sha256":digest(journal),"freeze_sha256":digest(folder/"freeze.json"),
                                     "score_cases":len(roster),"scratch_port_parity_checks":parity,"elapsed_s":time.monotonic()-start})


def iou(first, second):
    a,b=first
    c,d=second
    return Fraction(max(0,min(b,d)-max(a,c)),max(b,d)-min(a,c))


def result(folder):
    _, roster = frozen(folder)
    seal=read(folder/"held-seal.json")
    require(seal["journal_sha256"] == digest(folder/"held.jsonl") and seal["freeze_sha256"] == digest(folder/"freeze.json"), "Incomplete sealed data")
    journal=[json.loads(line) for line in (folder/"held.jsonl").read_text().splitlines()]
    counts=defaultdict(Counter)
    useful_groups=set()
    for source,row in zip(roster,journal,strict=True):
        require(source["sha256"] == row["sha256"], "Scored the wrong source")
        category=row["role"]
        truth=row["truth"]
        c=counts[category]
        c["cases"]+=1
        previous=[]
        for key in ("coarse","short_range","boundary_refinement"):
            record=row["baseline"][key]
            if record:
                pts=source["clock"]["pts"]
                previous.append([pts.index(record["start_pts"]),pts.index(record["end_pts"])])
        found=row["candidate"]
        baseline_useful=bool(truth and any(iou(span,truth)>=Fraction(1,2) for span in previous))
        fine_range=[found["record"][key] for key in ("start_frame_index","end_frame_index")] if found else None
        candidate_useful=bool(fine_range and truth and iou(fine_range,truth)>=Fraction(1,2))
        c["prior_flags"]+=bool(previous)
        c["prior_useful"]+=baseline_useful
        c["added_flags"]+=found is not None
        c["added_useful"]+=candidate_useful and not baseline_useful
        c["combined_useful"]+=baseline_useful or candidate_useful
        c["attempted_retry"]+=row["retry_coverage"] is not None
        if candidate_useful and not baseline_useful:
            useful_groups.add(row["source_scope"])
    point=SPEC["gate"]
    gate={"added_useful":counts["swap"]["added_useful"]>=point["minimum_added_useful"],
          "positive_groups":len(useful_groups)>=point["minimum_positive_groups"],
          "ordinary":counts["ordinary"]["added_flags"]<=point["maximum_added_ordinary"],
          "cuts":counts["cut"]["added_flags"]<=point["maximum_added_cuts"],"preserves_existing_ranges":True}
    result={"protocol":SPEC,"counts":{k:dict(v) for k,v in counts.items()},"useful_source_groups":len(useful_groups),
            "gate":gate,"advances":all(gate.values()),"runtime":seal,"serving_change":False}
    write(folder/"aggregate.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","score","report"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path)
    args=parser.parse_args()
    if args.phase=="score":
        state=worker.configure(args.folder)
        score(args.folder,state)
    elif args.phase=="prepare":
        from scripts import video_native_background_panel as panel

        prepare(args.folder,args.archive or panel.ARCHIVE)
    else:
        result(args.folder)


if __name__=="__main__":
    main()
