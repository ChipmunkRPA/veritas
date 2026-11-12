# SPDX-License-Identifier: Apache-2.0
"""Test within-track face alignment context on independent production-scored clips."""

from __future__ import annotations

import argparse
import json
import math
import tarfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from app.evidence.face_interval_binding import validate_sequence_record
from scripts import video_chroma_boundary_transfer as native
from scripts import video_face_return_transfer as trial
from scripts import video_saved_landmark_support_v254 as saved
from scripts import video_vcd2_verify_cache_replay_v251 as prior
from scripts import video_webcam_confound_v224 as archived
from scripts.video_appearance_corrected_v112 import scoped

common, require = native.common, native.require
raw = prior.raw
CAMERA_SLOTS = (3, 4, 5, 7, 8, 10)
ROLES = ("before", "during", "after")
SWAP_ROLES = ("original", "swap_a", "swap_b")
RULE = {"weak_eye_ratio": 0.60, "flank_minimum_eye_box_width": 0.20, "candidate_eye_fraction": 0.70,
        "candidate_nose_displacement_face_width": 0.12, "minimum_middle_triplet_fraction": "2/3",
        "minimum_samples_per_triplet": 2, "minimum_flank_triplets": 2}
GATE = ("At least four incumbent genuine ranges contextualized across at least two held camera recordings, "
        "zero contextualized correctly localized swaps, at least four correctly localized incumbent swaps across "
        "at least three held native source groups, and at least 90% measurement coverage among selected ranges.")
METHOD = ("Preserve the original range. Compare aligned landmark orientation and bilateral eye support with both "
          "flanks. Mark strong geometric confounds for review only, with no overall confidence contribution.")


def bindings():
    files = (Path(__file__), Path(__file__).with_name("video_webcam_pose_native_worker.py"),
             common.ROOT / "backend/app/detectors/video/adapters/face_appearance.py",
             common.ROOT / "backend/app/detectors/video/adapters/tracks.py",
             common.ROOT / "backend/app/engine/stages/video.py")
    return {**common.bindings(), **{str(path.relative_to(common.ROOT)): common.digest(path) for path in files}}


def camera_cases():
    result, old = common.read(prior.OLD_RESULT), common.read(raw.PUBLIC)
    plan = common.read(raw.TRIAL / "private-plan.json")
    seals = {}
    for slot in prior.SLOTS:
        for role in raw.ROLES:
            condition = raw.case_name(slot, role)
            path, receipt = raw.archive_paths(condition)
            pin = common.read(receipt)
            require(pin["archive_path"] == str(path) and pin["receipt_path"] == str(receipt),
                    "Historical application archive path changed")
            seals[condition] = {"archive_sha256": pin["archive_sha256"], "receipt_sha256": common.digest(receipt)}
    require(common.digest(prior.OLD_RESULT) == prior.OLD_RESULT_SHA and common.digest(raw.PUBLIC) == prior.OLD_FREEZE_SHA
            and prior.archive_set_commitment(plan) == old["plan_sha256"]
            and result["archive_set_commitment_sha256"] == prior.archive_set_commitment(seals),
            "Exact historical application archive required")
    cases = []
    source_ids = set()
    for slot in CAMERA_SLOTS:
        qualification = raw.PREPARATION / f"slot-{slot:02d}-attempt-01" / "qualification.json"
        source = common.read(qualification)["source_sha256"]
        require(source not in source_ids, "Independent camera recording expected")
        source_ids.add(source)
        for role in raw.ROLES:
            condition = raw.case_name(slot, role)
            path, receipt = raw.archive_paths(condition)
            row = plan[condition]
            require(path.is_file() and common.digest(qualification) == row["qualification_sha256"],
                    "Expected exact sealed control source")
            cases.append({"condition": condition, "slot": slot, "role": role, "case": row,
                          "archive": seals[condition], "receipt_sha256": common.digest(receipt)})
    return cases


def swap_cases():
    metadata = trial.prior.comparison.metadata()
    options = []
    archive = raw.ARCHIVE
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        meta = metadata[doc["group"]]
        row = doc["cases"][0]["frame_table"]
        tb = Fraction(row["time_base"])
        if (row["width"], row["height"], len(row["pts"])) != (1920, 1080, 150) or "child" in meta["tags"]:
            continue
        if any((b - a) * tb != Fraction(1, 25) for a, b in zip(row["pts"], row["pts"][1:], strict=False)):
            continue
        if not meta["tags"] & {"gesticulation", "head rotation"}:
            continue
        if doc["role"] not in {"confidence_cal", "confirmation"}:
            continue
        cases = {item["condition"]: item for item in doc["cases"]}
        options.append((sum(cases[key]["bytes"] for key in SWAP_ROLES), doc["index"], meta))
    selected, urls, donors = [], set(), set()
    for _, index, meta in sorted(options):
        if meta["url"] in urls or meta["donors"] & donors:
            continue
        selected.append(native.source_pin(index, archive))
        urls.add(meta["url"])
        donors.update(meta["donors"])
        if len(selected) == 6:
            break
    require(len(selected) == 6, "Six native source groups with independent known URLs and donor keys are required")
    return selected


def freeze(folder):
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-webcam-pose-transfer-freeze/1",
        "rules": RULE, "method": METHOD, "gate": GATE, "bindings": bindings(), "camera": camera_cases(),
        "swaps": swap_cases(), "prior_development": common.digest(saved.WEB_RESULT),
        "comparator": "Existing top appearance selection, old v259 median 60% eye-width check, and the new corroborated rule.",
        "native_worker": "Original app's Full pipeline, the production shot and face tracking paths, face detection, and native appearance detector."})


def frozen(folder):
    doc = common.read(folder / "freeze.json")
    require(doc["bindings"] == bindings() and doc["rules"] == RULE and doc["method"] == METHOD and doc["gate"] == GATE,
            "Prospective rule or implementation changed")
    return doc


def geometry(row):
    a, b, nose, mouth_left, mouth_right = row["landmarks"]
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm, width = math.hypot(dx, dy), row["bbox"]["w"]
    require(norm > 0 and width > 0, "Missing finite landmark support")
    ex, ey = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    nx, ny = nose[0] - ex, nose[1] - ey
    mx, my = (mouth_left[0] + mouth_right[0]) / 2 - ex, (mouth_left[1] + mouth_right[1]) / 2 - ey
    return {"eye_fraction": norm / width, "nose_x": (dx * nx + dy * ny) / (norm * width),
            "nose_vertical": (dx * ny - dy * nx) / norm**2,
            "mouth_vertical": (dx * my - dy * mx) / norm**2}


def normal(frame):
    return frame["mouth_vertical"] > 0 and frame["nose_vertical"] > 0


def score(record):
    if any(item["landmarks"][0] == item["landmarks"][1] for item in record["frame_inputs"]):
        return {"status": "flank_geometry_unassessed", "old_eye_width_rule": False, "context": None}
    by_index = {item["frame_index"]: geometry(item) for item in record["frame_inputs"]}
    groups = {role: [[by_index[center + offset] for offset in (-1, 0, 1)]
                     for center in record["centers"][role]] for role in ROLES}
    phase = {role: [frame for triple in triples for frame in triple] for role, triples in groups.items()}
    eyes = {role: median(frame["eye_fraction"] for frame in frames) for role, frames in phase.items()}
    old = eyes["during"] < RULE["weak_eye_ratio"] * (eyes["before"] + eyes["after"]) / 2
    flanks = ("before", "after")
    usable = {role: [frame for frame in phase[role] if normal(frame)] for role in flanks}
    if any(len(usable[role]) < 4 or eyes[role] < RULE["flank_minimum_eye_box_width"] for role in flanks):
        return {"status": "flank_geometry_unassessed", "old_eye_width_rule": old, "context": None}
    ref = {role: median(frame["nose_x"] for frame in usable[role]) for role in flanks}
    floor = min(eyes[role] for role in flanks)
    def inversion(frame):
        return frame["mouth_vertical"] <= 0 and frame["nose_vertical"] <= 0
    def lateral(frame):
        changes = [frame["nose_x"] - ref[role] for role in flanks]
        delta = RULE["candidate_nose_displacement_face_width"]
        return normal(frame) and frame["eye_fraction"] <= floor * RULE["candidate_eye_fraction"] and (
            min(changes) >= delta or max(changes) <= -delta)
    def support(predicate):
        return sum(sum(predicate(frame) for frame in triple) >= RULE["minimum_samples_per_triplet"] for triple in groups["during"])
    counts = {name: support(predicate) for name, predicate in (("landmark_inversion", inversion), ("bilateral_support_change", lateral))}
    minimum = math.ceil(len(groups["during"]) * Fraction(RULE["minimum_middle_triplet_fraction"]))
    context = next((key for key, value in counts.items() if value >= minimum), None)
    return {"status": "measured", "old_eye_width_rule": old, "context": context, "triplet_counts": counts,
            "middle_triplets": len(groups["during"]), "minimum_context_triplets": minimum}


def case_result(record, table, expected, truth=None, status=None):
    if record is None:
        return {"status": status or "incumbent_unassessed", "truth": truth, "incumbent": None, "context": None}
    source = {**table, "frame_count": len(table["pts"])}
    require(not any(table[key] for key in ("rotation", "truncated", "interpolated"))
            and all(row["decoded_rgb_sha256"] == expected[row["frame_index"]] for row in record["frame_inputs"]),
            "Exact committed pixels and complete native picture clock required")
    validate_sequence_record(record, source, table_pts=table["pts"])
    tb = Fraction(table["time_base"])
    relative = [str((record[f"{side}_pts"] - table["pts"][0]) * tb) for side in ("start", "end")]
    return {"truth": truth, "incumbent": relative, **score(record), "samples": len(record["frame_inputs"])}


def camera_result(pin):
    condition, row = pin["condition"], pin["case"]
    tar, receipt_path = raw.archive_paths(condition)
    receipt = common.read(receipt_path)
    require(common.digest(receipt_path) == pin["receipt_sha256"]
            and receipt["archive_sha256"] == pin["archive"]["archive_sha256"],
            "Camera archive differs from the prospective roster")
    archived_sha = common.digest(tar)
    require(archived_sha == receipt["archive_sha256"], "Camera archived bytes changed")
    with tarfile.open(tar, "r:") as stream:
        audit = archived.document(stream, "stage/audit.json")
        select = scoped(archived.select_record, {"sqlite3": SimpleNamespace(connect=saved.memory_connect)})
        record, table, missing = select(stream, condition, row)
        if missing:
            status = ("incumbent_measured_no_flag" if audit["appearance_check_status"] == "ok"
                      and audit["appearance_measured_windows"] else "incumbent_unassessed")
            return case_result(None, None, None, status=status)
        expected = raw.expected_case(row)
        require(table["pts"] == expected["pts"] and Fraction(table["time_base"]) == Fraction(expected["time_base"]),
                "Saved camera clock differs")
        source = raw.PREPARATION / f"slot-{row['slot']:02d}-attempt-01" / f"{row['role']}.mp4"
        require(common.digest(source) == row["sha256"], "Retained source bytes differ from the actual upload")
        from app.detectors.video.decode import FrameTable
        from app.reports.verify_video_source import _verify_aligned_face

        frames = {item["frame_index"]: item for item in record["frame_inputs"]}
        observed = set()
        for frame in common.decode_frames(source, FrameTable.from_json(table), sorted(frames), timeout_s=60):
            item = frames[frame.index]
            require(frame.pts_verified and frame.pts == item["pts"]
                    and item["decoded_rgb_sha256"] == common.hashlib.sha256(frame.rgb).hexdigest(),
                    "Selected camera picture differs from the frozen source")
            _verify_aligned_face(frame.rgb, item)
            observed.add(frame.index)
        require(observed == set(frames), "Native camera replay omitted selected pictures")
        return case_result(record, table, expected["rgb"])


def camera(folder):
    lock = frozen(folder)
    start = time.monotonic()
    with (folder / "camera.jsonl").open("x") as stream:
        for row in lock["camera"]:
            outcome = camera_result(row)
            stream.write(json.dumps({"slot": row["slot"], "role": row["role"], **outcome}, allow_nan=False) + "\n")
            stream.flush()
            print("camera", row["slot"], row["role"], outcome["status"], flush=True)
    print("camera_elapsed_s", round(time.monotonic() - start, 3))


def native_run(folder):
    import os
    import subprocess
    import sys

    lock = frozen(folder)
    with (folder / "native.jsonl").open("x") as stream:
        for ordinal, pin in enumerate(lock["swaps"], 1):
            for role in SWAP_ROLES:
                proc = subprocess.run([sys.executable, "-B", "-m", "scripts.video_webcam_pose_native_worker",
                    str(folder), str(ordinal), role], check=True, capture_output=True, text=True, timeout=260,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
                result = json.loads(proc.stdout.splitlines()[-1])
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print("native", ordinal, role, result["status"], flush=True)


def iou(a, b):
    left, right = map(Fraction, a)
    start, end = map(Fraction, b)
    return max(0, min(right, end) - max(left, start)) / (max(right, end) - min(left, start))


def readout(folder):
    lock = frozen(folder)
    cams = list(map(json.loads, (folder / "camera.jsonl").open()))
    positives = list(map(json.loads, (folder / "native.jsonl").open()))
    require(len(cams) == len(lock["camera"]) and len(positives) == len(lock["swaps"]) * len(SWAP_ROLES), "Incomplete run")
    cameras = [row for row in cams if row["incumbent"]]
    true = [row for row in positives if row["truth"] and row["incumbent"] and iou(row["truth"], row["incumbent"]) >= Fraction(1, 2)]
    camera_contexts = [row for row in cameras if row["context"]]
    lost = [row for row in true if row["context"]]
    selected = cameras + [row for row in positives if row["incumbent"]]
    coverage = sum(row["status"] == "measured" for row in selected)
    groups = len({row["slot"] for row in camera_contexts})
    useful_groups = len({row["group"] for row in true})
    result = {"camera_statuses": dict(Counter(row["status"] for row in cams)),
              "native_statuses": dict(Counter(row["status"] for row in positives)),
              "camera_views": len(cams), "camera_groups": len(CAMERA_SLOTS), "camera_incumbent_flags": len(cameras),
              "camera_old_eye_width_rule": sum(row["old_eye_width_rule"] for row in cameras),
              "camera_context_flags": len(camera_contexts), "camera_context_groups": groups,
              "camera_context_types": dict(Counter(row["context"] for row in camera_contexts)),
              "native_known_swaps": sum(bool(row["truth"]) for row in positives), "native_source_groups": len(lock["swaps"]),
              "native_incumbent_correct_intervals": len(true), "native_incumbent_correct_groups": useful_groups,
              "correct_swap_contextualized": len(lost), "geometry_measured": coverage, "geometry_selected": len(selected),
              "pilot_advances": len(camera_contexts) >= 4 and groups >= 2 and not lost and len(true) >= 4
                                and useful_groups >= 3 and 10 * coverage >= 9 * len(selected)}
    common.write(folder / "aggregate.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "camera", "native", "readout"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "camera": camera, "native": native_run, "readout": readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
