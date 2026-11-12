# SPDX-License-Identifier: Apache-2.0
"""Prospective transfer of the six-picture fallback to two encoded durations."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import face_return
from app.detectors.video.decode import FrameTable
from scripts import video_face_six_picture_transfer as earlier

common, require, archive_default = earlier.common, earlier.require, earlier.ARCHIVE
DATA = earlier.DATA
PRIOR = DATA / "video-face-six-picture-2026-11-12-continuation"
SPANS = {80: ((54, 60), (61, 67)), 150: ((62, 68), (89, 95))}
ORDINARY = earlier.ORDINARY
CASE_NAMES = (*ORDINARY, "swap_a", "swap_b", "hard_cut")
SPEC = {
    "schema": "chipmunk.video-face-six-duration/1", "frames": list(SPANS), "shape": [640, 360],
    "codec": "h264/yuv420p", "fps": 25, "spans": {str(k): list(map(list, v)) for k, v in SPANS.items()},
    "threshold": face_return.POINT, "dedupe_iou": str(face_return.DEDUPE_IOU),
    "definition": "Preserve the coarse and 8 Hz checks. If 8 Hz measured a winner but emitted no distinct range, "
                  "evaluate the unchanged guarded rank on every second native picture. Only runtime duration eligibility varies.",
    "ordinary": list(ORDINARY) + ["hard_cut"], "primary_crf": 23, "tandem_crf": 36,
    "criteria_per_duration": {"additional_correct": 4, "additional_groups": 3, "additional_ordinary_max": 1,
                              "additional_cut_max": 0, "coverage_min": .90},
}


def selection(archive):
    binding, paths = earlier.bindings(archive)
    previous = common.read(PRIOR / "freeze.json")
    prior_audit = common.read(PRIOR / "independent-audit.json")
    require(prior_audit["status"] == "verified" and prior_audit["aggregate_sha256"] == common.digest(PRIOR / "aggregate.json"),
            "Prior six-picture transfer lacks its independent audit")
    excluded = set.union(*(earlier.trial.previous_rtf_sources(common.read(path)) for path in paths))
    excluded.update(row["index"] for row in previous["roster"])
    require(len(excluded) == 31, "Need the exact earlier localization exclusion roster")
    metadata = earlier.trial.prior.comparison.metadata()
    known = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    links, donors = {row["url"] for row in known}, set.union(*(row["donors"] for row in known))
    eligible = {"development": [], "held": []}
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        index, meta = doc["index"], metadata[doc["group"]]
        if index in excluded or meta["url"] in links or meta["donors"] & donors or "child" in meta["tags"] or "head rotation" not in meta["tags"]:
            continue
        original = next(row for row in doc["cases"] if row["condition"] == "original")
        table = FrameTable.from_json(original["frame_table"])
        shape = table.width, table.height
        if shape not in earlier.encoded.PROFILES or common.relative_times(table) != earlier.encoded.CLOCK:
            continue
        split = "held" if doc["role"] == "confirmation" else "development"
        eligible[split].append((sum(row["bytes"] for row in doc["cases"]), index, shape, meta))
    roster = []
    for role, n in (("development", 2), ("held", 4)):
        for _, index, shape, meta in sorted(eligible[role]):
            if meta["url"] in links or donors & meta["donors"]:
                continue
            roster.append({**earlier.encoded.prior.archive_io.source_pin(index, archive), "shape": list(shape), "role": role})
            links.add(meta["url"])
            donors.update(meta["donors"])
            if sum(pin["role"] == role for pin in roster) == n:
                break
        require(sum(pin["role"] == role for pin in roster) == n, f"Need {n} disjoint {role} sources")
    return {"spec": SPEC, "roster": roster, "preceding": binding,
            "prior_freeze": common.digest(PRIOR / "freeze.json"), "prior_audit": common.digest(PRIOR / "independent-audit.json"),
            "worker_sha256": common.digest(Path(__file__))}


def frozen(folder, archive):
    saved = common.read(folder / "freeze.json")
    require(saved == selection(archive), "Frozen study inputs or rules changed")
    return saved


def source_set(pin, archive):
    n, size = pin["index"], tuple(pin["shape"])
    require(earlier.encoded.prior.archive_io.source_pin(n, archive) == {k: pin[k] for k in ("index", "receipt_sha256", "review_sha256")},
            "Verified native archive source differs")
    cases = {row["condition"]: row for row in common.read(common.DATA / f"group-{n:03d}-complete.json")["cases"]}
    root = archive / "video-native-inserts-v95" / f"group-{n:03d}"
    return size, cases, root


def span(condition, group, length):
    phase = (group + (condition == "swap_b")) % 2 if condition.startswith("swap_") else int(condition == "ordinary_blur")
    return SPANS[length][phase]


def receipt(path, group, role, condition, n, truth):
    table = common.probe_frames(path, timeout_s=25)
    hashes = []
    for frame in earlier.exact_rgb(path, table, n):
        require(frame.index == len(hashes) and frame.pts_verified and frame.pts == table.pts[frame.index], "Native encoded clock missing")
        hashes.append(hashlib.sha256(frame.rgb.tobytes()).hexdigest())
    require(len(hashes) == n, "Complete native frame set required")
    return {"name": path.name, "sha256": common.digest(path), "group": group, "role": role, "condition": condition,
            "count": n, "truth": truth, "table": table.to_json(), "rgb_sha256": hashes}


def prepared(folder, archive):
    lock = frozen(folder, archive)
    media = folder / "media"
    media.mkdir(mode=0o700)
    result, pair = [], []
    for group, pin in enumerate(lock["roster"]):
        shape, cases, root = source_set(pin, archive)
        base = cases["original"]
        plain = earlier.encoded.native_frames(root / base["file"], base, shape, range(max(SPANS)))
        ordinary = ("ordinary_light", "ordinary_blur", "ordinary_time_step")
        changed = {}
        for condition in (*ordinary, "swap_a", "swap_b"):
            record = cases[condition]
            wanted = sorted({i for length in SPANS for i in range(*span(condition, group, length))})
            require(all(record["expected_rgb_sha256"][i] != base["expected_rgb_sha256"][i] for i in wanted),
                    "Native edits do not cover every authored picture")
            changed[condition] = earlier.encoded.native_frames(root / record["file"], record, shape, wanted)
            require(all(not np.array_equal(image, plain[i]) for i, image in changed[condition].items()),
                    "Authored edit lost its picture changes after resize")
        for length in SPANS:
            origin = media / f"g{group}-{length}-original.mp4"
            earlier.encoded.encode(origin, frames=(plain[i] for i in range(length)), crf=23)
            result.append(receipt(origin, group, pin["role"], "original", length, None))
            tandem = media / f"g{group}-{length}-recompressed.mp4"
            earlier.encoded.encode(tandem, source=origin, crf=36)
            result.append(receipt(tandem, group, pin["role"], "recompressed", length, None))
            for condition, replacement in changed.items():
                a, b = span(condition, group, length)
                path = media / f"g{group}-{length}-{condition}.mp4"
                earlier.encoded.encode(path, frames=(replacement[i] if a <= i < b else plain[i] for i in range(length)), crf=23)
                result.append(receipt(path, group, pin["role"], condition, length, [a,b] if condition.startswith("swap_") else None))
        pair.append(plain)
        if len(pair) == 2:
            for length in SPANS:
                for phase, (base_frames, donor_frames) in enumerate((pair, pair[::-1])):
                    a, b = SPANS[length][phase]
                    g = group-1+phase
                    path = media / f"g{g}-{length}-hard_cut.mp4"
                    earlier.encoded.encode(path, frames=(donor_frames[i] if a <= i < b else base_frames[i] for i in range(length)), crf=23)
                    result.append(receipt(path, g, pin["role"], "hard_cut", length, None))
            pair.clear()
        print("prepared", pin["role"], group, flush=True)
    require(len(result) == len({row["sha256"] for row in result}) == 96, "Incomplete encoded roster")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": result})


def measure(path, clip, detector, recognizer):
    table = common.probe_frames(path, timeout_s=25)
    require(common.digest(path) == clip["sha256"] and table.to_json() == clip["table"], "Encoded input changed")
    count = clip["count"]
    require(count in SPANS and len(table) == count, "Unmeasured delivery duration")
    times, base = face_return.schedule(table)
    dense = tuple(range(0, count, 2))
    centers = [i for i in common.schedule(times, 4) if 0 < i < count-1]
    indices = sorted(set(base) | set(dense) | {i + shift for i in centers for shift in (-1, 0, 1)})
    features, assets, multiple = {}, {}, set()
    for picture in common.decode_frames(path, table, indices, timeout_s=40):
        i = picture.index
        pixels = picture.rgb
        rgb = hashlib.sha256(pixels.tobytes()).hexdigest()
        require(picture.pts_verified and table.pts[i] == picture.pts and rgb == clip["rgb_sha256"][i], "Measured native picture differs")
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        faces = common.detect_faces(bgr, detector=detector, min_size=48)
        face, vector = None, None
        if len(faces) == 1:
            face = faces[0]
            aligned = common.face_appearance.face_row(face.landmarks)
            aligned[0,:4], aligned[0,14] = face.box(), face.score
            vector = common.face_appearance.unit(recognizer.feature(recognizer.alignCrop(bgr, aligned)))
            x, y, w, h = face.box()
            assets[i] = {"frame": i, "pts": picture.pts, "rgb_sha256": rgb, "xywh": [x,y,w,h],
                         "crop_sha256": hashlib.sha256(pixels[y:y+h, x:x+w].tobytes()).hexdigest()}
        elif len(faces) > 1:
            multiple.add(i)
        features[i] = face_return.Sample(face, vector, face_return.scene(pixels, face))
    require(list(features) == indices, "Observation roster differs")
    coarse = common.appearance(centers, table, times, {i:{"appearance": row.vector} for i,row in features.items()})
    incumbent = None
    if coarse and coarse["score"] > earlier.trial.prior.POINT:
        a, b = (int(Fraction(coarse[key])*25) for key in ("start", "end"))
        incumbent = {"start_pts": table.pts[a], "end_pts": table.pts[b]}
    current = None if multiple.intersection(base) else face_return.highest(times, base, features)

    def info(found):
        if found is None:
            return None
        return {"frames": [found.start, found.end], "margin": found.response, "selected": face_return.distinct(found, table, incumbent),
                "scene": found.scene_correlations, "support": found.support,
                "evidence": {role: assets[i] for role,i in found.anchors().items()}}

    legacy = info(current)
    fallback = None if legacy is None or legacy["selected"] or multiple.intersection(dense) else face_return.highest(times, dense, features)
    return {"coarse": coarse, "eight_hz": legacy, "fallback": info(fallback), "samples": len(indices), "observed_faces": len(assets)}


def run(folder, archive, role):
    frozen(folder, archive)
    manifest = common.read(folder / "media.json")
    require(manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Encoded manifest differs")
    if role == "held":
        journal(folder, "development")
    cv2.setNumThreads(1)
    model = common.load_yunet(common.FACE_FILE)
    face = cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    start = time.monotonic()
    completed = 0
    with (folder / f"{role}.jsonl").open("x") as stream:
        for row in manifest["media"]:
            if row["role"] != role:
                continue
            metrics = measure(folder / "media" / row["name"], row, model, face)
            stream.write(json.dumps({k: row[k] for k in ("name", "sha256", "group", "condition", "count", "truth")}
                                    | metrics, allow_nan=False) + "\n")
            stream.flush()
            completed += 1
            print(role, row["group"], row["count"], row["condition"], flush=True)
    require(completed == (32 if role == "development" else 64), "Incomplete scored source role")
    common.write(folder / f"{role}-receipt.json", {"journal_sha256": common.digest(folder / f"{role}.jsonl"),
          "media_sha256": common.digest(folder / "media.json"), "elapsed_s": time.monotonic()-start,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def journal(folder, role):
    path = folder / f"{role}.jsonl"
    seal = common.read(folder / f"{role}-receipt.json")
    require(seal["journal_sha256"] == common.digest(path) and seal["media_sha256"] == common.digest(folder / "media.json"),
            "Frozen scored receipt differs")
    return list(map(json.loads, path.read_text().splitlines()))


def outcomes(rows):
    result, groups = defaultdict(lambda: defaultdict(Counter)), defaultdict(set)
    for row in rows:
        truth = row["truth"]
        n = str(row["count"])
        kind = "positive" if truth else "cut" if row["condition"] == "hard_cut" else "ordinary"
        before, dense = [], row["fallback"]
        coarse = row["coarse"]
        if coarse and coarse["score"] > earlier.trial.prior.POINT:
            before.append([int(Fraction(coarse[k])*25) for k in ("start", "end")])
        if row["eight_hz"] and row["eight_hz"]["selected"]:
            before.append(row["eight_hz"]["frames"])
        old = bool(truth and any(earlier.iou(interval, truth) >= Fraction(1,2) for interval in before))
        flagged = bool(dense and dense["selected"])
        located = bool(flagged and truth and earlier.iou(dense["frames"], truth) >= Fraction(1,2))
        adds = located and not old
        if adds:
            groups[n].add(row["group"])
        result[n][kind].update({"cases":1, "current_located":old, "current_flagged":bool(before),
                                "measured_fallback":dense is not None, "fallback_flagged":flagged, "added_locations":adds,
                                "combined_located":old or located, "fallback_wrong_range":bool(flagged and truth and not located)})
    report = {}
    for n, values in result.items():
        rule = SPEC["criteria_per_duration"]
        criteria = {"located": values["positive"]["added_locations"] >= rule["additional_correct"],
                    "groups": len(groups[n]) >= rule["additional_groups"],
                    "ordinary": values["ordinary"]["fallback_flagged"] <= rule["additional_ordinary_max"],
                    "cut": values["cut"]["fallback_flagged"] <= rule["additional_cut_max"],
                    "coverage": all(values[k]["measured_fallback"] >= rule["coverage_min"] * values[k]["cases"]
                                    for k in ("positive", "ordinary"))}
        report[n] = {"counts": values, "criteria": criteria, "advance": all(criteria.values()), "added_groups": len(groups[n])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=archive_default)
    args = parser.parse_args()
    folder, archive = args.folder, args.archive
    if args.phase == "freeze":
        lock = selection(archive)
        folder.mkdir(mode=0o700, parents=True)
        common.write(folder / "freeze.json", lock)
    elif args.phase == "prepare":
        cv2.setNumThreads(1)
        prepared(folder, archive)
    elif args.phase == "report":
        frozen(folder, archive)
        held = journal(folder, "held")
        common.write(folder / "aggregate.json", {"protocol": SPEC, "development": outcomes(journal(folder, "development")),
             "held": outcomes(held), "runtime": common.read(folder / "held-receipt.json"),
             "scored_source_frames": sum(row["samples"] for row in held)})
    else:
        run(folder, archive, args.phase)


if __name__ == "__main__":
    main()
