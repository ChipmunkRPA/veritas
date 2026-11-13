# SPDX-License-Identifier: Apache-2.0
"""Run the frozen repeat-speech comparison on separate publisher recordings."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import video_lipsync_repeat_motion as base
from scripts import video_lipsync_repeat_picture_guard as picture
from scripts.prepare_video_lipsync_local_delay_transfer import acquire
from scripts.video_common_sources_v89 import CREMA_REV, read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels/video-lipsync-repeat-picture-guard-2026-11-13.json"
SCHEMA = "chipmunk.video-lipsync-repeat-fresh-actors/1"
SALT = "chipmunk-repeat-confirm-expressive-take/1"
PLANNED = 16
CHOICES = 4
PATTERN = re.compile(r"VideoFlash/(\d{4})_([A-Z]{3})_(?:HAP|SAD)_XX\.flv")


def order(value):
    return base.digest(f"{SALT}/{value}".encode())


def frozen_intake(tree_path, earlier):
    tree = read(tree_path)
    require(tree["sha"] == CREMA_REV and not tree["truncated"], "Pinned publisher tree required")
    previous = read(earlier)
    require(previous["schema"] == base.SCHEMA and previous["design"] == picture.DESIGN,
            "Use the earlier repeat picture-context freeze")
    require(read(PANEL)["freeze_sha256"] == sha(earlier), "Earlier actor roster does not bind the published experiment")
    used = {row["actor"] for row in previous["rows"]}
    require(len(used) == 31, "Unexpected prior actor count")
    candidates = {}
    for row in tree["tree"]:
        if row["type"] == "blob" and (match := PATTERN.fullmatch(row["path"])) and int(match[1]) not in used:
            candidates.setdefault(int(match[1]), []).append(row)
    actors = sorted((key for key, members in candidates.items() if len(members) >= CHOICES), key=order)[:PLANNED]
    require(len(actors) == PLANNED, "Incomplete independent actor intake")
    rows = []
    for group, actor in enumerate(actors, 201):
        # Pick different sentence codes before considering another take of the same sentence.
        choices = {}
        for row in sorted(candidates[actor], key=lambda row: order(row["path"])):
            choices.setdefault(PATTERN.fullmatch(row["path"])[2], row)
        require(len(choices) >= CHOICES, "Incomplete actor phrase roster")
        for choice, row in enumerate(list(choices.values())[:CHOICES]):
            ordinal = len(rows) + 1
            rows.append({"actor": actor, "group": group, "choice": choice, "ordinal": ordinal, "split": "confirmation",
                         "source_path": row["path"], "pointer_git_sha1": row["sha"], "pointer_bytes": row["size"],
                         "file": f"source-{ordinal:02d}.flv"})
    return {"schema": SCHEMA, "publisher_revision": CREMA_REV, "source_count": len(rows), "planned_groups": PLANNED,
            "previous_actors": len(used), "ordering_salt": SALT, "tree_sha256": sha(tree_path),
            "previous_freeze_sha256": sha(earlier), "sources": rows}


def freeze(run, tree_path, earlier):
    plan = frozen_intake(tree_path, earlier)
    run.mkdir(mode=0o700)
    write(run / "intake-freeze.json", plan)


def contract(run):
    intake = read(run / "intake-freeze.json")
    files = read(run / "intake-receipt.json")
    require(files["roster_sha256"] == sha(run / "intake-freeze.json"), "Source download does not bind the frozen roster")
    require([dict(r, sha256=None, bytes=None) for r in files["rows"]] ==
            [dict(r, sha256=None, bytes=None) for r in intake["sources"]], "Source roster differs")
    require(len({row["sha256"] for row in files["rows"]}) == PLANNED * CHOICES
            and all(sha(run / row["file"]) == row["sha256"] for row in files["rows"]), "Source bytes differ")
    previous = read(PANEL)
    require(previous["point"] == .65 and previous["gate"] == base.GATE and previous["grid"] == list(base.GRID)
            and previous["freeze_sha256"] == intake["previous_freeze_sha256"], "Fixed protocol differs")
    return {"schema": SCHEMA, "point": previous["point"], "gate": previous["gate"], "grid": previous["grid"],
            "planned_groups": PLANNED, "choices": CHOICES, "previous_actors": intake["previous_actors"],
            "parent_panel_sha256": sha(PANEL), "intake_sha256": sha(run / "intake-freeze.json"),
            "acquisition_sha256": sha(run / "intake-receipt.json"),
            "bindings": base.prior.binding() | {Path(p).name: sha(p) for p in (__file__, base.__file__, picture.__file__)},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "ffmpeg": base.prior.delivery.command("ffmpeg", "-version").decode().splitlines()[0]}}


def qualify(run, detector):
    dispositions = {}
    selected = []
    for row in read(run / "intake-receipt.json")["rows"]:
        group = row["group"]
        state = dispositions.setdefault(group, {"group": group, "choices": []})
        if "selected" in state:
            state["choices"].append({"sha256": row["sha256"], "outcome": "after_selection"})
            continue
        path = run / row["file"]
        require(sha(path) == row["sha256"], "Frozen source bytes differ")
        try:
            original = base.prior.original(path, row["sha256"])
        except ValueError:
            outcome = "outside_source_clock"
        else:
            outcome = "outside_shared_duration"
            if original is not None:
                sample = 14 + row["ordinal"] % 3
                frames = original[0]
                faces = [base.detect_faces(cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
                         for k in (sample, sample + 5, sample + 10)]
                outcome = "outside_source_geometry"
                if all(len(found) == 1 for found in faces):
                    state["selected"] = row["sha256"]
                    selected.append((row, original))
                    outcome = "selected"
        state["choices"].append({"sha256": row["sha256"], "outcome": outcome})
    return list(dispositions.values()), selected


def score(run):
    from scripts.review_video_lipsync_repeat_fresh_actors import summarize

    lock = contract(run)
    require(read(run / "freeze.json") == lock, "Protocol or bytes changed after freeze")
    cv2.setNumThreads(1)
    detector = base.load_yunet(base.prior.WEIGHT)
    qualification, selected = qualify(run, detector)
    require(len(selected) >= 2, "Too few qualified source groups for independent-sound controls")
    rows = []
    for i, (row, original) in enumerate(selected):
        donor, donor_original = selected[(i + 1) % len(selected)]
        measured = base.assess(run, row | {"bank": "expressive-confirmation"}, original, donor_original,
                               detector, picture.measurement)
        require(measured is not None, "Media could not reproduce the prequalified mouth geometry")
        measured["donor_group"] = donor["group"]
        rows.append(measured)
        print("measured group", row["group"], flush=True)
    receipt = {"freeze": lock, "freeze_sha256": sha(run / "freeze.json"), "qualification": qualification, "rows": rows}
    receipt["summary"] = summarize(receipt)
    write(run / "receipt.json", receipt)
    print(Counter(choice["outcome"] for row in qualification for choice in row["choices"]), flush=True)
    print(receipt["summary"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("intake", "acquire", "freeze", "score"))
    p.add_argument("run", type=Path)
    p.add_argument("--tree", type=Path)
    p.add_argument("--earlier", type=Path)
    args = p.parse_args()
    if args.phase == "intake":
        require(args.tree is not None and args.earlier is not None, "Supply pinned publisher tree and prior freeze")
        freeze(args.run, args.tree, args.earlier)
    elif args.phase == "acquire":
        acquire(args.run)
    elif args.phase == "freeze":
        write(args.run / "freeze.json", contract(args.run))
    else:
        score(args.run)


if __name__ == "__main__":
    main()
