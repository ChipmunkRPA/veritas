# SPDX-License-Identifier: Apache-2.0
"""Prospective stationary-background motion context on production face-review ranges."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2

from scripts import video_face_motion_context as context
from scripts import video_gros_range_intake as gros
from scripts import video_webcam_pose_transfer as pose
from scripts.video_appearance_corrected_v112 import scoped

common, require = pose.common, pose.require
GATE = ("For either independently assessed arm, contextualize at least four true incumbent webcam false ranges "
        "across two source groups, retain every correctly localized native swap, confirm at least four useful swaps "
        "across three groups, and complete at least three new licensed genuine app cases.")
MOTION_SLOTS = (1, 9)
POSITIVE_RUN = Path("/private/tmp/chipmunk-webcam-pose.7ms69sa6")
POSITIVE_FREEZE = "0e06508b9760f87c508e6b9e97d922474cf0e608db277ff0c40e13823151d789"
GROS_SOURCE = pose.raw.ARCHIVE / "video-gros-motion-held-2026-11-11"
POSITIVE_ROLES = ("swap_a", "swap_b") * 3


def bindings():
    files = (Path(__file__), Path(context.__file__), Path(gros.__file__),
             Path(__file__).with_name("video_webcam_motion_worker.py"))
    return {**pose.bindings(), **{str(path.relative_to(common.ROOT)): common.digest(path) for path in files}}


def freeze(folder):
    ordinary = scoped(pose.camera_cases, {"CAMERA_SLOTS": MOTION_SLOTS})()
    known = pose.frozen(POSITIVE_RUN)
    require(common.digest(POSITIVE_RUN / "freeze.json") == POSITIVE_FREEZE, "Earlier native positive roster changed")
    clips = common.read(folder / "gros-files.json")
    source = common.read(folder / "gros-roster.json")
    require(len(clips) == len(source["rows"]) == 4 and all((GROS_SOURCE / row["file"]).stat().st_size == row["bytes"] for row in clips),
            "Four licensed source files required")
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-webcam-motion-transfer-freeze/1",
        "rules": context.POLICY, "gate": GATE, "bindings": bindings(), "camera": ordinary,
        "earlier_v253_receipt_sha256": common.digest(pose.saved.WEB_RESULT),
        "gros_plan_sha256": common.digest(folder / "gros-roster.json"), "gros": clips,
        "primary_positive_freeze_sha256": POSITIVE_FREEZE, "primary_positive_pins": known["swaps"],
        "positive_roles": POSITIVE_ROLES,
        "development": "One unchanged previously inspected genuine camera range, excluded with its entire group.",
        "scope": "Previously reported full-app camera ranges and known full-app swap sources; fresh CC BY controlled recordings processed by the same app."})


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["rules"] == context.POLICY and lock["gate"] == GATE and lock["bindings"] == bindings()
            and lock["gros_plan_sha256"] == common.digest(folder / "gros-roster.json"), "Frozen study differs")
    return lock


def relative_extent(record, table):
    base = Fraction(table["time_base"])
    return [str((record[f"{side}_pts"] - table["pts"][0]) * base) for side in ("start", "end")]


def inspect(path, record, table, hashes=None):
    measured = context.native_context(path, record, table, hashes)
    return {"incumbent_extent": relative_extent(record, table), **measured}


def old_camera(pin):
    condition, row = pin["condition"], pin["case"]
    archive, receipt = pose.raw.archive_paths(condition)
    require(common.digest(receipt) == pin["receipt_sha256"]
            and common.digest(archive) == pin["archive"]["archive_sha256"], "Camera archive differs")
    with tarfile.open(archive, "r:") as stream:
        audit = pose.archived.document(stream, "stage/audit.json")
        select = scoped(pose.archived.select_record, {"sqlite3": SimpleNamespace(connect=pose.saved.memory_connect)})
        record, table, missing = select(stream, condition, row)
    if missing:
        return {"status": "incumbent_measured_no_flag" if audit["appearance_check_status"] == "ok" else "incumbent_unassessed",
                "incumbent_extent": None}
    expected = pose.raw.expected_case(row)
    source = pose.raw.PREPARATION / f"slot-{row['slot']:02d}-attempt-01" / f"{row['role']}.mp4"
    require(common.digest(source) == row["sha256"] and table["pts"] == expected["pts"]
            and Fraction(table["time_base"]) == Fraction(expected["time_base"]), "Archived source or native clock differs")
    return inspect(source, record, table, expected["rgb"])


def archived_v253():
    saved = pose.saved
    capture = scoped(saved.windows, {"support": SimpleNamespace(measure=lambda record: record)})
    reader = scoped(saved.webcam_cases, {"windows": capture})
    roster = common.read(saved.webcam.TRIAL / "private-plan.json")
    for ordinal, ((_, windows), condition) in enumerate(zip(reader(), saved.webcam.CONDITIONS, strict=True)):
        selected = [row for yes, row in windows if yes]
        require(len(selected) == 1, "Expected a single actual webcam appearance range")
        row = roster[condition]
        source = pose.raw.PREPARATION / f"slot-{row['slot']:02d}-attempt-01" / f"{row['role']}.mp4"
        expected = pose.raw.expected_case(row)
        require(common.digest(source) == row["sha256"], "Archived v253 source bytes differ")
        table = common.probe_frames(source, timeout_s=25).to_json()
        require(table["pts"] == expected["pts"] and Fraction(table["time_base"]) == Fraction(expected["time_base"]),
                "Archived v253 native picture clock differs")
        yield {"family": "archived_webcam", "group": 3 + ordinal // 3, "role": row["role"],
               **inspect(source, selected[0], table, expected["rgb"])}


def camera(folder):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(205)
    lock = frozen(folder)
    start = time.monotonic()
    with (folder / "camera.jsonl").open("x") as stream:
        for pin in lock["camera"]:
            record = {"family": "archived_webcam", "group": 1 + MOTION_SLOTS.index(pin["slot"]), "role": pin["role"],
                      **old_camera(pin)}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print("camera", record["group"], record["role"], record["status"], flush=True)
        for record in archived_v253():
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print("camera", record["group"], record["role"], record["status"], flush=True)
    print("camera_elapsed_s", round(time.monotonic() - start, 3), flush=True)


def live(folder, family):
    lock = frozen(folder)
    size = len(lock["gros"] if family == "gros" else lock["positive_roles"])
    with (folder / f"{family}.jsonl").open("x") as stream:
        for ordinal in range(1, size + 1):
            log = folder / f"{family}-worker-{ordinal}.log"
            with log.open("x") as capture:
                process = subprocess.run([sys.executable, "-B", "-m", "scripts.video_webcam_motion_worker",
                    str(folder), family, str(ordinal)], stdout=subprocess.PIPE, stderr=capture, text=True,
                    timeout=260, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                capture.write(process.stdout)
            require(process.returncode == 0, f"Operational failure in private worker {family} {ordinal}")
            result = json.loads(process.stdout.splitlines()[-1])
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(family, ordinal, result["status"], flush=True)


def readout(folder):
    lock = frozen(folder)
    cam, fresh, positive = (list(map(json.loads, (folder / f"{part}.jsonl").open()))
                            for part in ("camera", "gros", "positive"))
    require((len(cam), len(fresh), len(positive)) == (12, len(lock["gros"]), len(lock["positive_roles"])), "Incomplete journals")
    preserved = [r for r in positive if r["incumbent_extent"] and pose.iou(r["incumbent_extent"], r["truth"]) >= Fraction(1, 2)]
    outcome = {"schema": "chipmunk.video-webcam-motion-transfer-result/1",
        "freeze_sha256": common.digest(folder / "freeze.json"),
        "journal_sha256": {name: common.digest(folder / f"{name}.jsonl") for name in ("camera", "gros", "positive")},
        "views": {"camera": len(cam), "gros": len(fresh), "positive": len(positive)},
        "baseline_selected": {"camera": sum(bool(r["incumbent_extent"]) for r in cam),
                              "gros": sum(bool(r["incumbent_extent"]) for r in fresh),
                              "positive_correct": len(preserved)},
        "coverage": {name: dict(Counter(row["status"] for row in rows))
                     for name, rows in (("camera", cam), ("gros", fresh), ("positive", positive))},
        "arms": {}}
    for arm in ("translation_context", "translation_roll_context"):
        ordinary = [r for r in cam if r.get(arm)]
        new = [r for r in fresh if r.get(arm)]
        lost = [r for r in preserved if r.get(arm)]
        groups = len({r["group"] for r in ordinary})
        outcome["arms"][arm] = {"camera_contextualized": len(ordinary), "camera_groups_contextualized": groups,
                               "gros_contextualized": len(new), "useful_swap_ranges_contextualized": len(lost),
                               "advances": len(ordinary) >= 4 and groups >= 2 and not lost and len(preserved) >= 4
                                           and len({r["group"] for r in preserved}) >= 3
                                           and sum(r["status"] != "incumbent_unassessed" for r in fresh) >= 3}
    common.write(folder / "aggregate.json", outcome)
    print(json.dumps(outcome, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "camera", "gros", "positive", "readout"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.stage in {"gros", "positive"}:
        live(args.folder, args.stage)
    else:
        {"freeze": freeze, "camera": camera, "readout": readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
