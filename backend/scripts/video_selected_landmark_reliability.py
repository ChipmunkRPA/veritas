# SPDX-License-Identifier: Apache-2.0
"""Evaluate frozen v259 landmark support on the incumbent's selected video interval."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import tarfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2

from scripts import video_face_short_boundary_confirmation as comparison
from scripts import video_face_texture_return as common
from scripts import video_landmark_rerank_v259 as historical
from scripts import video_saved_landmark_support_v254 as saved
from scripts import video_vcd2_case_archive_v202 as archive_io
from scripts import video_vcd2_current_app_v294 as camera_native
from scripts import video_vcd2_full_app_v202 as camera_saved
from scripts import video_webcam_confound_v224 as webcam
from scripts.video_appearance_corrected_v112 import scoped

GROUPS = (0, 2, 3, 39, 66, 79)
POINT = common.face_appearance.REVIEW_POINT
PANEL = common.ROOT / "docs/evaluation-panels"
V259_FREEZE = PANEL / "video-landmark-rerank-v259-freeze.json"
V259_SHA = "9d154f5ba24c9c661e5be2a800d508bfd9cecb6331d56a1039940d6c156f4809"
V290_RESULT = PANEL / "video-vcd2-matched-controls-v290-complete.json"
V290_FREEZE = PANEL / "video-vcd2-matched-controls-v290-freeze.json"
V202_RESULT = PANEL / "video-vcd2-full-app-eight-triplets-v202.json"
V294_RESULT = PANEL / "video-vcd2-current-app-v294-twenty.json"
METHOD = "Apply frozen v259 60% ratio to the original top flag only; unavailable support retains the original flag with unavailable geometry."
ADVANCE = "Pilot only: withhold at least two of three historically false selected flags from the held saved camera group, lose no swap location at 0.50 IoU and measure at least 90% of all selected flags."
ROLES = saved.support.ROLES


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def freeze_sources(archive, previous):
    require(archive.resolve() == camera_saved.ARCHIVE and archive.is_dir(), "Exact existing source archive required")
    previous_pins = {label: common.digest(path / "freeze.json") for label, path in previous.items()}
    old_indexes = {row["index"] for path in previous.values() for row in common.read(path / "freeze.json")["roster"]
                   if "index" in row}
    require(not set(GROUPS) & old_indexes, "Held groups overlap a previous texture or timing study")
    metadata = comparison.metadata()
    prior = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in old_indexes]
    urls = {row["url"] for row in prior}
    donors = {name for row in prior for name in row["donors"]}
    rtfs = []
    for index in GROUPS:
        path = common.DATA / f"group-{index:03d}-complete.json"
        review_path = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
        source, review = common.read(path), common.read(review_path)
        meta = metadata[source["group"]]
        media = archive / "video-native-inserts-v95" / f"group-{index:03d}"
        require(source["index"] == index and source["role"] == "confirmation"
                and review["status"] == "complete" and review["private_group_receipt_sha256"] == common.digest(path)
                and set(row["condition"] for row in source["cases"]) == set(common.CONTROL_CONDITIONS + common.SWAP_CONDITIONS)
                and all((media / row["file"]).stat().st_size == row["bytes"] for row in source["cases"])
                and meta["url"] not in urls and not meta["donors"] & donors,
                "Exact source-disjoint confirmation group required")
        urls.add(meta["url"])
        donors.update(meta["donors"])
        rtfs.append({"index": index, "receipt_sha256": common.digest(path), "review_sha256": common.digest(review_path)})
    require(common.digest(V259_FREEZE) == V259_SHA
            and common.read(V259_FREEZE)["minimum_during_over_mean_flank_eye_box_ratio"] == historical.MINIMUM_FLANK_FRACTION == .60
            and common.read(V259_FREEZE)["appearance_review_point"] == POINT,
            "The exact existing v259 cutoff and current appearance point are required")
    require(common.read(V202_RESULT)["freeze_sha256"] == common.digest(camera_saved.PUBLIC)
            and common.read(V290_RESULT)["freeze_sha256"] == common.digest(V290_FREEZE)
            and common.read(V294_RESULT)["freeze_sha256"] == common.digest(camera_native.PUBLIC),
            "The exact camera freeze and published summaries are required")
    old_plan = camera_saved.TRIAL / "private-plan.json"
    prepared = camera_native.TRIAL / "private-plan.json"
    old, new = common.read(old_plan), common.read(prepared)
    require(common.read(camera_saved.PUBLIC)["plan_sha256"] == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
            and camera_native.later.replay.archive_set_commitment(new) == common.read(camera_native.PUBLIC)["plan_sha256"],
            "Camera preparation differs from published plan")
    slots = {}
    for slot, roster, module in ((9, old, camera_saved), (8, new, camera_native)):
        pinned = []
        for role in camera_saved.ROLES:
            condition = f"slot-{slot:02d}-{role}"
            row = roster[condition]
            source = module.PREPARATION / f"slot-{slot:02d}-attempt-01" / "qualification.json"
            require(row["slot"] == slot and row["role"] == role and row["status"] == "prepared"
                    and row["qualification_sha256"] == common.digest(source), "Camera preparation binding differs")
            tar, sidecar = module.archive_paths(condition)
            pin = {"role": role, "condition": condition, "source_sha256": row["sha256"],
                   "qualification_sha256": row["qualification_sha256"]}
            if slot == 9:
                receipt = common.read(sidecar)
                require(tar.is_file() and str(tar) == receipt["archive_path"]
                        and receipt["stage_path"] == str(module.authored(condition))
                        and receipt["work_path"] == str(module.work_dir(condition)),
                        "Historical saved archive is missing")
                pin.update(archive_sha256=receipt["archive_sha256"], receipt_sha256=common.digest(sidecar))
            else:
                require(not tar.exists() and not sidecar.exists() and not module.audit_path(condition).exists(),
                        "An earlier app score exists for the prepared control")
                source_case = common.read(source)["cases"][role]
                require(source_case["sha256"] == row["sha256"] and (source.parent / source_case["file"]).stat().st_size == row["bytes"],
                        "Prepared control file differs")
            pinned.append(pin)
        slots[str(slot)] = pinned
    require(len({common.read(camera_saved.PREPARATION / "slot-09-attempt-01/qualification.json")["source_sha256"],
                 common.read(camera_native.PREPARATION / "slot-08-attempt-01/qualification.json")["source_sha256"]}) == 2,
            "The camera source groups overlap")
    return rtfs, slots, previous_pins


def identity():
    paths = (Path(__file__), Path(saved.support.__file__), Path(historical.__file__), Path(saved.__file__),
             Path(comparison.__file__), *comparison.METADATA, V259_FREEZE, V202_RESULT, V290_RESULT, V294_RESULT)
    return {
        **common.bindings(),
        **{str(p.relative_to(common.ROOT)): common.digest(p) for p in paths},
    }


def freeze(folder, archive, previous):
    groups, slots, earlier = freeze_sources(archive, previous)
    bindings = identity()
    require(bindings["models/weights/face_detection_yunet_2023mar.onnx"] ==
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
            and bindings["models/weights/face_recognition_sface_2021dec.onnx"] ==
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
            "Exact commercial model weights required")
    common.write(folder / "freeze.json", {
        "schema": "chipmunk.video-selected-landmark-reliability-freeze/1",
        "policy": METHOD, "pilot_advancement": ADVANCE, "appearance_point": POINT,
        "ratio": historical.MINIMUM_FLANK_FRACTION, "bindings": bindings,
        "known_prior_studies": earlier, "rtfs": groups, "camera": slots,
        "interventions": "Same 4 Hz nearest-native schedule and 0.75s, 1.50s and full native RTFS interventions as the short timing study; camera uses complete prepared deliveries.",
        "release_scope": "Research pilot with source-group counts. No customer outcome, replacement window, or original-file inference.",
    })


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == identity() and lock["ratio"] == historical.MINIMUM_FLANK_FRACTION
            and lock["appearance_point"] == POINT, "Method or source bindings differ from freeze")
    return lock


def source_features(path, case, wanted, yunet, sface):
    require(common.digest(path) == case["sha256"], "Input differs from qualified source")
    table = common.probe_frames(path, timeout_s=25)
    original = case["frame_table"]
    tb = Fraction(original["time_base"])
    require(not table.truncated and not table.interpolated and table.rotation == 0
            and (table.width, table.height) == (original["width"], original["height"])
            and [p * table.tb for p in table.pts] == [p * tb for p in original["pts"]],
            "Exact native source picture clock and dimensions required")
    features, seen = {}, set()
    for frame in common.decode_frames(path, table, sorted(wanted), timeout_s=65):
        i = frame.index
        digest = hashlib.sha256(frame.rgb).hexdigest()
        require(frame.pts_verified and digest == case["expected_rgb_sha256"][i] and i not in seen,
                "Native source sample or clock differs")
        seen.add(i)
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        faces = common.detect_faces(bgr, detector=yunet, min_size=48)
        if len(faces) != 1:
            continue
        face = faces[0]
        aligned = common.face_appearance.face_row(face.landmarks)
        aligned[0, :4], aligned[0, 14] = face.box(), face.score
        vector = sface.feature(sface.alignCrop(bgr, aligned)).reshape(-1).copy()
        common.face_appearance.unit(vector)
        features[i] = {"appearance": vector, "face": face, "rgb_sha256": digest}
    require(seen == wanted, "Native sample roster incomplete")
    return table, features, len(seen)


def classify_measurement(measurement):
    eye = measurement.get("eye_box_width", {})
    quality = historical.quality(*(eye.get(role, {}).get("median") for role in ROLES))
    return {
        "status": {
            "supported": "flag_retained",
            "below_flank_fraction": "flag_withheld_low_support",
            "geometry_unavailable": "flag_retained_unavailable_geometry",
        }[quality],
        "eye_width_medians": [eye[role]["median"] for role in ROLES] if eye else None,
        "geometry_unavailable_reason": measurement.get("reason"),
    }


def selection(centers, table, times, features):
    vectors = {i: value["appearance"] for i, value in features.items()}
    tracks = {"single-face": [SimpleNamespace(frame_index=i) for i in sorted(vectors)]}
    windows = common.face_appearance.windows(set(centers), tracks, vectors, table)
    if not windows:
        return {"status": "baseline_unassessed", "original": None}
    best = windows[0]
    ids = best["centers"]
    original = {"score": best["response"], "start": str(times[ids[2]]), "end": str(times[ids[-3]])}
    if original["score"] <= POINT:
        return {"status": "no_baseline_flag", "original": original}
    by_role = dict(zip(ROLES, (ids[:2], ids[2:-2], ids[-2:]), strict=True))
    rows = []
    for role in ROLES:
        for index in (i + k for i in by_role[role] for k in (-1, 0, 1)):
            item = features[index]
            face = item["face"]
            rows.append({"frame_index": index, "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True)),
                         "landmarks": face.landmarks, "rgb_sha256": item["rgb_sha256"]})
    measurement = saved.support.measure({"centers": by_role, "frame_inputs": rows})
    return {"original": original, "selected_input_sha256": hashlib.sha256(
        "".join(row["rgb_sha256"] for row in rows).encode()).hexdigest(), **classify_measurement(measurement)}


def rtfs_group(archive, pin, yunet, sface):
    index = pin["index"]
    receipt = common.DATA / f"group-{index:03d}-complete.json"
    review = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
    require(common.digest(receipt) == pin["receipt_sha256"] and common.digest(review) == pin["review_sha256"],
            "RTFS source receipt differs from freeze")
    cases = {c["condition"]: c for c in common.read(receipt)["cases"]}
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    original, times, intervals = comparison.match_cases(folder, cases, camera=False)
    centers = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
    wanted = {i + offset for i in centers for offset in (-1, 0, 1)}
    table, plain, samples = source_features(folder / original["file"], original, wanted, yunet, sface)
    rows = [{"condition": "original", "duration": None, "truth": None, **selection(centers, table, times, plain)}]
    lo, hi = intervals["native_long"]
    within = {i for i in wanted if lo <= i < hi}
    for name, case in cases.items():
        if name == "original":
            continue
        require(set(range(lo, hi)).issubset(case["actual_changed_frames"]),
                "Missing intervention frame in the qualified interval")
        observed, changed, count = source_features(folder / case["file"], case, within, yunet, sface)
        require(common.relative_times(observed) == times, "Matched variant has a different source clock")
        samples += count
        for duration, (a, b) in intervals.items():
            values = {i: row for i, row in plain.items() if not a <= i < b}
            values.update({i: row for i, row in changed.items() if a <= i < b})
            truth = [str(times[a]), str(times[b])] if name in common.SWAP_CONDITIONS else None
            rows.append({"condition": name, "duration": duration, "truth": truth,
                         **selection(centers, table, times, values)})
    return rows, samples


def camera_saved_group(pins):
    roster = common.read(camera_saved.TRIAL / "private-plan.json")
    rows = []
    for pin in pins:
        condition = pin["condition"]
        source = roster[condition]
        tar, sidecar = camera_saved.archive_paths(condition)
        receipt = common.read(sidecar)
        require(common.digest(sidecar) == pin["receipt_sha256"] and receipt["archive_sha256"] == pin["archive_sha256"]
                and source["sha256"] == pin["source_sha256"] and source["qualification_sha256"] == pin["qualification_sha256"],
                "Historical camera does not match its frozen archive")
        archive_io.verify_archive(tar, receipt)
        with tarfile.open(tar, "r:") as stream:
            select = scoped(webcam.select_record, {"sqlite3": SimpleNamespace(connect=saved.memory_connect)})
            record, table, missing = select(stream, condition, source)
            if missing:
                rows.append({"condition": pin["role"], "duration": None, "truth": None, "original": None,
                             "status": "no_baseline_flag" if missing == "no_selected_appearance_window" else "baseline_unassessed"})
                continue
            expected = camera_saved.expected_case(source)
            require(table["pts"] == expected["pts"] and Fraction(table["time_base"]) == Fraction(expected["time_base"])
                    and (table["width"], table["height"]) == expected["size"] and table["rotation"] == 0
                    and not table["truncated"] and not table["interpolated"]
                    and all(expected["rgb"][item["frame_index"]] == item["decoded_rgb_sha256"] for item in record["frame_inputs"]),
                    "Saved selected interval differs from its exact source")
            saved.validate_sequence_record(record, {**table, "frame_count": len(table["pts"])}, table_pts=table["pts"])
            tb = Fraction(table["time_base"])
            original = {"score": float.fromhex(record["measurement_hex"]["response"]),
                        "start": str((record["start_pts"] - table["pts"][0]) * tb),
                        "end": str((record["end_pts"] - table["pts"][0]) * tb)}
            require(original["score"] > POINT, "Saved selected appearance response is below the incumbent point")
            measurement = saved.support.measure(record)
            rows.append({"condition": pin["role"], "duration": None, "truth": None, "original": original,
                         **classify_measurement(measurement)})
    return rows, 0


def camera_native_group(pins, yunet, sface):
    folder = camera_native.PREPARATION / "slot-08-attempt-01"
    path = folder / "qualification.json"
    receipt = common.read(path)
    table_meta = {"time_base": receipt["time_base"], "pts": receipt["relative_pts"],
                  "width": receipt["width"], "height": receipt["height"]}
    rows, samples = [], 0
    for pin in pins:
        name = pin["role"]
        case = receipt["cases"][name]
        require(common.digest(path) == pin["qualification_sha256"] and case["sha256"] == pin["source_sha256"],
                "Prepared camera delivery changed")
        tb = Fraction(table_meta["time_base"])
        times = [(p - table_meta["pts"][0]) * tb for p in table_meta["pts"]]
        centers = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
        wanted = {i + k for i in centers for k in (-1, 0, 1)}
        table, features, count = source_features(folder / case["file"],
            {**case, "frame_table": table_meta, "expected_rgb_sha256": case["rgb_sha256"]}, wanted, yunet, sface)
        samples += count
        rows.append({"condition": name, "duration": None, "truth": None, **selection(centers, table, times, features)})
    return rows, samples


def run(folder, archive):
    lock = frozen(folder)
    cv2.setNumThreads(2)
    yunet = common.load_yunet(common.FACE_FILE)
    sface = cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    start = time.monotonic()
    samples = 0
    with (folder / "scored.jsonl").open("x") as stream:
        def emit(group, cohort, outcome):
            nonlocal samples
            rows, count = outcome
            samples += count
            stream.write(json.dumps({"group": group, "cohort": cohort, "rows": rows, "native_samples": count},
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(group, len(rows), "views;", samples, "verified native samples", flush=True)

        emit("camera_saved", "genuine_camera_saved", camera_saved_group(lock["camera"]["9"]))
        for pin in lock["rtfs"]:
            emit(f"rtfs_{pin['index']:03d}", "matched_rtfs", rtfs_group(archive, pin, yunet, sface))
        emit("camera_native", "genuine_camera_native", camera_native_group(lock["camera"]["8"], yunet, sface))
    common.write(folder / "run.json", {
        "native_samples": samples, "elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scored_sha256": common.digest(folder / "scored.jsonl"),
    })


def summarize(groups):
    totals = {}
    for group in groups:
        for row in group["rows"]:
            name = "known_swap" if row["truth"] else group["cohort"]
            stats = totals.setdefault(name, {"views": Counter(), "source_groups": {}})
            c = stats["views"]
            gs = stats["source_groups"].setdefault(group["group"], Counter())
            status = row["status"]
            original = row["original"]
            flagged = original is not None and original["score"] > POINT
            require((status.startswith("flag_")) == flagged, "Result contradicts baseline score")
            for count in (c, gs):
                count["views"] += 1
                count["baseline_flags"] += flagged
                count[status] += 1
                if row["truth"] and flagged:
                    overlap = common.overlap(original, row["truth"])
                    good = overlap >= Fraction(1, 2)
                    count["baseline_iou_at_least_half"] += good
                    count["retained_iou_at_least_half"] += good and status != "flag_withheld_low_support"
    return {name: {"views": dict(data["views"]), "source_groups": len(data["source_groups"]),
                   "groups_with_withheld_flags": sum(row["flag_withheld_low_support"] > 0 for row in data["source_groups"].values()),
                   "groups_with_correct_range_loss": sum(row["retained_iou_at_least_half"] < row["baseline_iou_at_least_half"]
                                                        for row in data["source_groups"].values())}
            for name, data in totals.items()}


def readout(folder):
    lock = frozen(folder)
    groups = [json.loads(line) for line in (folder / "scored.jsonl").read_text().splitlines()]
    run = common.read(folder / "run.json")
    expected = ["camera_saved", *(f"rtfs_{pin['index']:03d}" for pin in lock["rtfs"]), "camera_native"]
    require([row["group"] for row in groups] == expected
            and [len(row["rows"]) for row in groups] == [3, *([16] * len(lock["rtfs"])), 3]
            and run["scored_sha256"] == common.digest(folder / "scored.jsonl")
            and run["native_samples"] == sum(row["native_samples"] for row in groups),
            "Incomplete or altered score journal")
    totals = summarize(groups)
    selected = sum(v["views"]["baseline_flags"] for v in totals.values())
    unavailable = sum(v["views"].get("flag_retained_unavailable_geometry", 0) for v in totals.values())
    swap = totals["known_swap"]["views"]
    passed = (totals["genuine_camera_saved"]["views"].get("flag_withheld_low_support", 0) >= 2
              and swap.get("retained_iou_at_least_half", 0) == swap.get("baseline_iou_at_least_half", 0)
              and selected and (selected - unavailable) / selected >= .9)
    common.write(folder / "aggregate.json", {
        "schema": "chipmunk.video-selected-landmark-reliability/1", "freeze_sha256": common.digest(folder / "freeze.json"),
        "cohorts": totals, "native_runtime": run, "pilot_advances": bool(passed),
        "selected_flags_measured": selected - unavailable, "selected_flags_total": selected,
        "policy": METHOD, "pilot_advancement": ADVANCE,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=camera_saved.ARCHIVE)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--short-boundary", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if not all((args.texture, args.boundary, args.short_boundary)):
            parser.error("the prior texture and both timing study directories are required")
        freeze(args.folder, args.archive, {"texture": args.texture, "boundary": args.boundary,
                                          "short_boundary": args.short_boundary})
    elif args.stage == "run":
        run(args.folder, args.archive)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
