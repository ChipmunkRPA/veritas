# SPDX-License-Identifier: Apache-2.0
"""Test short same-shot facial appearance separation with native source clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts import video_chroma_boundary_transfer as archive_io

prior = archive_io.prior
common = archive_io.common
require = archive_io.require
SPEC = {"cadence_hz": 8, "core_samples": [3, 4, 6], "flank_samples": 4, "minimum_flank": 3,
        "minimum_core_fraction": .66, "minimum_core": 3, "minimum_iou": .25, "maximum_sample_gap_s": .26,
        "scene_tile_minimum": 8, "scene_texture_minimum": 4., "scene_cut_correlation": .55}
ARMS = ("current", "dense_centroid", "rank_without_guard", "rank")
SIZES = (Fraction(3, 8), Fraction(1, 2), Fraction(3, 4))
ADVANCE = "Rank needs six more correct intervals than current, two more than dense centroid, gains in at least three held groups, at most two ordinary false flags, zero cut false flags, and at least 90% coverage separately on held swaps and ordinary controls."
POLICY = "Research evidence for a temporary face appearance transition only. No identity match, originality inference, or overall confidence contribution."


def bindings():
    return {**archive_io.bindings(), "backend/scripts/video_face_return_transfer.py": common.digest(Path(__file__))}


def previous_rtf_sources(doc):
    rows = list(doc.get("roster", doc.get("rtfs", [])))
    for phase in ("development", "held"):
        split = doc.get(phase)
        if isinstance(split, dict):
            rows.extend(split.get("rtfs", ()))
    return {row["index"] for row in rows if "index" in row}


def freeze(folder, archive, earlier):
    require(archive.resolve() == prior.camera_saved.ARCHIVE and len(earlier) == 5, "Exact retained archive and five prior freezes required")
    plans = [common.read(run / "freeze.json") for run in earlier]
    old = set.union(*(previous_rtf_sources(doc) for doc in plans))
    require(len(old) >= 34 and set(prior.GROUPS).issubset(old), "Incomplete prior exposure roster")
    metadata = prior.comparison.metadata()
    already = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in old]
    links = {m["url"] for m in already}
    donors = set.union(*(m["donors"] for m in already))
    candidates = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        meta = metadata[doc["group"]]
        clock = doc["cases"][0]["frame_table"]
        step = (clock["pts"][1] - clock["pts"][0]) * Fraction(clock["time_base"])
        if doc["role"] == "confidence_cal" and doc["index"] not in old and "child" not in meta["tags"] and (
            clock["width"], clock["height"], step) == (1920, 1080, Fraction(1, 25)):
            candidates.append((sum(c["bytes"] for c in doc["cases"]), doc["index"], meta))
    selected = []
    for _, index, meta in sorted(candidates):
        if meta["url"] in links or donors & meta["donors"]:
            continue
        selected.append(archive_io.source_pin(index, archive))
        links.add(meta["url"])
        donors.update(meta["donors"])
        if len(selected) == 6:
            break
    require(len(selected) == 6, "Six known source-disjoint 25 Hz groups are required for matched cut controls")
    dev = [archive_io.source_pin(i, archive) for i in (2, 39, 66, 0)]
    cameras = [prior.camera_saved.PREPARATION / "slot-09-attempt-01/qualification.json",
               *(prior.camera_native.PREPARATION / f"slot-{slot:02d}-attempt-01/qualification.json" for slot in (3, 4))]
    require(len({common.read(path)["source_sha256"] for path in cameras}) == 3, "Camera sources overlap")
    cdev, c1, c2 = (archive_io.camera_pin(path, archive) for path in cameras)
    common.write(folder / "freeze.json", {
        "schema": "chipmunk.video-face-return-transfer-freeze/1", "bindings": bindings(), "specification": SPEC,
        "prior_freeze_sha256": [common.digest(run / "freeze.json") for run in earlier],
        "development": {"rtfs": dev, "camera": [cdev], "cut_pairs": [[0, 1]]},
        "held": {"rtfs": selected, "camera": [c1, c2], "cut_pairs": [[0, 1], [2, 3], [4, 5]]},
        "methods": "Current 4 Hz triplet appearance and 8 Hz raw centroid, distribution separation with and without the non-face boundary shot-cut check on the same 8 Hz windows.",
        "cut_construction": "Both directions of exact-clock original A/B/A delivery swaps within the native half-open 0.75s interval. Case source links and known donor keys stay private.",
        "point": "Current unchanged. Dense centroid and rank each use their own maximum measured development genuine response including cut controls; rank without guard uses the guarded point to isolate the guard.",
        "advance": ADVANCE, "policy": POLICY,
    })


def frozen(folder):
    lock = common.read(folder / "freeze.json")
    require(lock["bindings"] == bindings() and lock["specification"] == SPEC, "Frozen settings or source changed")
    return lock


def scene(rgb, face):
    gray = cv2.cvtColor(cv2.resize(rgb, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    tiles = gray.reshape(4, 16, 8, 12).transpose(0, 2, 1, 3).reshape(32, 192).astype(np.float32)
    tiles -= tiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(tiles, axis=1)
    mask = norms >= SPEC["scene_texture_minimum"] * math.sqrt(192)
    if face is not None:
        h, w = rgb.shape[:2]
        x, y, fw, fh = face.box()
        for k in range(len(tiles)):
            row, col = divmod(k, 8)
            if ((col + 1) * w / 8 > x - .1 * fw and col * w / 8 < x + 1.1 * fw
                    and (row + 1) * h / 4 > y - .1 * fh and row * h / 4 < y + 1.1 * fh):
                mask[k] = False
    return tiles / np.maximum(norms[:, None], 1e-6), mask


def scene_correlation(a, b):
    first, valid_a = a
    second, valid_b = b
    valid = valid_a & valid_b
    if np.count_nonzero(valid) < SPEC["scene_tile_minimum"]:
        return None
    return float(np.median(np.sum(first[valid] * second[valid], axis=1)))


def distribution_margin(distances, before, core, after):
    """Require the same temporary separation in most individual face observations."""
    def cross(a, b):
        return distances[np.ix_(a, b)]

    lower = np.minimum(np.quantile(cross(core, before), .25, axis=1), np.quantile(cross(core, after), .25, axis=1))
    ref = [np.quantile(cross(before, after), .75)]
    for group in (before, core, after):
        samples = cross(group, group)
        ref.append(np.quantile(samples[np.triu_indices(len(group), k=1)], .75))
    return float(np.quantile(lower, .25) - max(ref))


def candidate_windows(times, schedule, features):
    visible = {i: row for i, row in features.items() if row["vector"] is not None}
    dense = [i for i in schedule if i in visible]
    if len(dense) < 9:
        return dict.fromkeys(ARMS[1:]), {"windows": 0, "guarded": 0, "scene_unavailable": 0}
    order = {i: k for k, i in enumerate(dense)}
    vectors = np.asarray([visible[i]["vector"] for i in dense])
    distances = 1. - np.clip(vectors @ vectors.T, -1., 1.)
    adjacent = {(a, b): scene_correlation(features[a]["scene"], features[b]["scene"])
                for a, b in zip(schedule, schedule[1:])}
    best = dict.fromkeys(ARMS[1:])
    counters = {"windows": 0, "guarded": 0, "scene_unavailable": 0}
    flank = SPEC["flank_samples"]
    for size in SPEC["core_samples"]:
        needed = max(SPEC["minimum_core"], math.ceil(size * SPEC["minimum_core_fraction"]))
        for start in range(flank, len(schedule) - size - flank + 1):
            stop = start + size
            groups = (schedule[start - flank:start], schedule[start:stop], schedule[stop:stop + flank])
            before, core, after = [[i for i in part if i in visible] for part in groups]
            if len(before) < SPEC["minimum_flank"] or len(after) < SPEC["minimum_flank"] or len(core) < needed:
                continue
            joined = before + core + after
            if any(times[b] - times[a] > SPEC["maximum_sample_gap_s"] or
                   common.iou(visible[a]["face"], visible[b]["face"]) < SPEC["minimum_iou"] for a, b in zip(joined, joined[1:])):
                continue
            counters["windows"] += 1
            edges = [adjacent[(schedule[start - 1], schedule[start])], adjacent[(schedule[stop - 1], schedule[stop])]]
            low_support = any(x is None for x in edges)
            cut = not low_support and min(edges) < SPEC["scene_cut_correlation"]
            counters["guarded"] += cut
            counters["scene_unavailable"] += low_support
            left, middle, right = [[order[i] for i in group] for group in (before, core, after)]
            a, b, c = (vectors[which].mean(axis=0) for which in (left, middle, right))
            distance = common.face_appearance.distance
            centroid = min(distance(a, b), distance(b, c)) - common.face_appearance.FLANK_PENALTY * distance(a, c)
            rank = distribution_margin(distances, left, middle, right)
            start_time, end_time = times[schedule[start]], times[schedule[stop]]
            anchors = [before[-1], core[len(core) // 2], after[0]]
            info = {"start": str(start_time), "end": str(end_time), "scene_correlations": edges,
                    "anchors": [{"index": i, "time": str(times[i]), "face_xywh": list(visible[i]["face"].box())} for i in anchors]}
            for arm, score in (("dense_centroid", centroid), ("rank_without_guard", rank), ("rank", rank)):
                if arm == "rank" and (low_support or cut):
                    continue
                current = best[arm]
                if current is None or score > current["score"]:
                    best[arm] = {**info, "score": score}
    return best, counters


def sample(path, case, expected_clock, wanted, yunet, recognizer):
    require(common.digest(path) == case["sha256"], "Source bytes differ from qualification")
    clock = common.probe_frames(path, timeout_s=30)
    tb = Fraction(expected_clock["time_base"])
    require(not clock.truncated and not clock.interpolated and clock.rotation == 0
            and (clock.width, clock.height) == (expected_clock["width"], expected_clock["height"])
            and [i * clock.tb for i in clock.pts] == [i * tb for i in expected_clock["pts"]], "Native source clock differs")
    rows = {}
    for frame in common.decode_frames(path, clock, sorted(wanted), timeout_s=110):
        i = frame.index
        require(frame.pts_verified and i not in rows and hashlib.sha256(frame.rgb).hexdigest() == case["rgb_sha256"][i],
                "Picture or decoder timestamp differs from qualification")
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        faces = common.detect_faces(bgr, detector=yunet, min_size=48)
        face = faces[0] if len(faces) == 1 else None
        vector = None
        if face is not None:
            geometry = common.face_appearance.face_row(face.landmarks)
            geometry[0, :4], geometry[0, 14] = face.box(), face.score
            vector = common.face_appearance.unit(recognizer.feature(recognizer.alignCrop(bgr, geometry)))
        rows[i] = {"vector": vector, "face": face, "scene": scene(frame.rgb, face)}
    require(rows.keys() == set(wanted), "Native sample roster incomplete")
    return clock, rows


def schedules(times):
    coarse = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
    dense = common.schedule(times, SPEC["cadence_hz"])
    wanted = {i + d for i in coarse for d in (-1, 0, 1)} | set(dense)
    return coarse, dense, wanted


def evaluate(clock, samples):
    times = common.relative_times(clock)
    coarse, dense, _ = schedules(times)
    vectors = {i: row["vector"] for i, row in samples.items() if row["vector"] is not None}
    tracks = {"single-face": [SimpleNamespace(frame_index=i) for i in sorted(vectors)]}
    ranges = common.face_appearance.windows(set(coarse), tracks, vectors, clock)
    baseline = None
    if ranges:
        highest = ranges[0]
        ids = highest["centers"]
        baseline = {"score": highest["response"], "start": str(times[ids[2]]), "end": str(times[ids[-3]])}
    candidates, coverage = candidate_windows(times, dense, samples)
    return {"arms": {"current": baseline, **candidates}, "window_coverage": coverage}


def rtfs_group(pin, archive, models, *, development):
    require(archive_io.source_pin(pin["index"], archive) == pin, "Source receipt or command review changed")
    index = pin["index"]
    document = common.read(common.DATA / f"group-{index:03d}-complete.json")
    cases = {case["condition"]: case for case in document["cases"]}
    path = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    base, times, spans = prior.comparison.match_cases(path, cases, camera=False)
    intervals = {str(size): prior.comparison.exact_span(times, Fraction(3), size) for size in SIZES}
    native_a, native_b = spans["native_long"]
    _, _, wanted = schedules(times)
    changed = {i for i in wanted if native_a <= i < native_b}

    def load(case, selected):
        clock, rows = sample(path / case["file"], {**case, "rgb_sha256": case["expected_rgb_sha256"]},
                             case["frame_table"], selected, *models)
        require(common.relative_times(clock) == times, "Matched delivery has a different relative native clock")
        return clock, rows

    clock, original = load(base, wanted)
    output = [{"condition": "original", "duration": None, "truth": None, **evaluate(clock, original)}]
    samples = len(wanted)
    for name, case in cases.items():
        if name == "original" or development and name in common.SWAP_CONDITIONS:
            continue
        _, alternative = load(case, changed)
        samples += len(changed)
        for duration, (a, b) in intervals.items():
            require(native_a <= a < b <= native_b, "Frozen cut is outside the qualified intervention")
            features = {i: alternative[i] if a <= i < b else value for i, value in original.items()}
            truth = [str(times[a]), str(times[b])] if name in common.SWAP_CONDITIONS else None
            output.append({"condition": name, "duration": duration, "truth": truth, **evaluate(clock, features)})
    return output, samples, (clock, original)


def camera_group(pin, archive, models):
    path = archive / pin["qualification"]
    require(archive_io.camera_pin(path, archive) == pin, "Camera qualification differs")
    document = common.read(path)
    tb = Fraction(document["time_base"])
    pts = document["relative_pts"]
    times = [(p - pts[0]) * tb for p in pts]
    _, _, wanted = schedules(times)
    expected_clock = {"width": document["width"], "height": document["height"], "pts": pts, "time_base": str(tb)}
    rows = []
    for name in archive_io.CAMERA_ROLES:
        case = document["cases"][name]
        clock, observed = sample(path.parent / case["file"], case, expected_clock, wanted, *models)
        rows.append({"condition": name, "duration": None, "truth": None, **evaluate(clock, observed)})
    return rows, len(wanted) * len(rows)


def cut_group(a, b):
    rows = []
    for (base_clock, original), (other_clock, insert) in ((a, b), (b, a)):
        times = common.relative_times(base_clock)
        require(times == common.relative_times(other_clock) and original.keys() == insert.keys()
                and (base_clock.width, base_clock.height) == (other_clock.width, other_clock.height),
                "Ordinary cut source pair lacks an exact shared picture clock")
        start, stop = prior.comparison.exact_span(times, Fraction(3), Fraction(3, 4))
        mixed = {i: insert[i] if start <= i < stop else value for i, value in original.items()}
        rows.append({"condition": "ordinary_source_cut", "duration": "3/4", "truth": None,
                     "ordinary_cut_span": [str(times[start]), str(times[stop])], **evaluate(base_clock, mixed)})
    return rows


def run(folder, archive, phase):
    plan = frozen(folder)[phase]
    if phase == "held":
        sealed_point(folder)
    cv2.setNumThreads(2)
    models = (common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), ""))
    journals = folder / f"{phase}.jsonl"
    start = time.monotonic()
    samples = 0
    originals = []
    with journals.open("x") as stream:
        def emit(family, number, rows, verified):
            nonlocal samples
            samples += verified
            stream.write(json.dumps({"family": family, "group": f"{family}_{number}", "rows": rows, "verified": verified},
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(phase, family, number, "verified", verified, "frames", flush=True)

        for number, pin in enumerate(plan["camera"], 1):
            emit("camera", number, *camera_group(pin, archive, models))
        for number, pin in enumerate(plan["rtfs"], 1):
            rows, count, original = rtfs_group(pin, archive, models, development=phase == "development")
            emit("rtfs", number, rows, count)
            originals.append(original)
        for number, (left, right) in enumerate(plan["cut_pairs"], 1):
            emit("cut", number, cut_group(originals[left], originals[right]), 0)
    common.write(folder / f"{phase}-run.json", {"verified_frames": samples, "elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "journal_sha256": common.digest(journals)})


def groups(folder, phase):
    path = folder / f"{phase}.jsonl"
    rows = archive_io.boundary.journal(path)
    receipt = common.read(folder / f"{phase}-run.json")
    sizes = [3, *([10] * 4), 2] if phase == "development" else [3, 3, *([16] * 6), 2, 2, 2]
    require(common.digest(path) == receipt["journal_sha256"] and [len(row["rows"]) for row in rows] == sizes
            and sum(row["verified"] for row in rows) == receipt["verified_frames"], "Score journal differs from receipt or roster")
    return rows, receipt


def seal(folder):
    frozen(folder)
    records, run_doc = groups(folder, "development")
    rows = [row for group in records for row in group["rows"]]
    require(all(row["truth"] is None for row in rows), "Only genuine controls may set the point")
    points = {"current": prior.POINT}
    coverage = {}
    for arm in ("dense_centroid", "rank"):
        controls = [row["arms"][arm]["score"] for row in rows if row["arms"][arm]]
        require(len(controls) >= 25, "Insufficient measured genuine development coverage")
        points[arm] = max(controls)
        coverage[arm] = len(controls)
    points["rank_without_guard"] = points["rank"]
    common.write(folder / "point.json", {"freeze_sha256": common.digest(folder / "freeze.json"),
        "development_journal_sha256": run_doc["journal_sha256"], "points": points, "measured_controls": coverage,
        "planned_controls": len(rows)})


def sealed_point(folder):
    doc = common.read(folder / "point.json")
    require(doc["freeze_sha256"] == common.digest(folder / "freeze.json")
            and doc["development_journal_sha256"] == common.digest(folder / "development.jsonl"), "Point differs from sealed development")
    return doc


def classify(selected, point, truth):
    if selected is None:
        return "unassessed"
    if selected["score"] <= point:
        return "measured_no_flag"
    if truth is None:
        return "false_flag"
    return "localized" if common.overlap(selected, truth) >= Fraction(1, 2) else "wrong_interval"


def readout(folder):
    frozen(folder)
    point = sealed_point(folder)
    development, d_run = groups(folder, "development")
    held, h_run = groups(folder, "held")
    for arm in ("dense_centroid", "rank"):
        scores = [row["arms"][arm]["score"] for group in development for row in group["rows"] if row["arms"][arm]]
        require(max(scores) == point["points"][arm], "Point failed replay")
    counts = {arm: defaultdict(Counter) for arm in ARMS}
    durations = {arm: defaultdict(Counter) for arm in ARMS}
    gains = Counter()
    coverage = Counter()
    for group in held:
        for row in group["rows"]:
            label = "swap" if row["truth"] else group["family"]
            outcomes = {}
            coverage.update(row["window_coverage"])
            for arm in ARMS:
                out = classify(row["arms"][arm], point["points"][arm], row["truth"])
                outcomes[arm] = out
                counts[arm][label][out] += 1
                if row["truth"]:
                    durations[arm][row["duration"]][out] += 1
            if label == "swap" and outcomes["rank"] == "localized" and outcomes["current"] != "localized":
                gains[group["group"]] += 1
    rank = counts["rank"]
    passed = (rank["swap"]["localized"] >= counts["current"]["swap"]["localized"] + 6
              and rank["swap"]["localized"] >= counts["dense_centroid"]["swap"]["localized"] + 2
              and len(gains) >= 3 and rank["rtfs"]["false_flag"] + rank["camera"]["false_flag"] <= 2
              and rank["cut"]["false_flag"] == 0 and rank["swap"]["unassessed"] <= 3
              and rank["rtfs"]["unassessed"] + rank["camera"]["unassessed"] <= 6)
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-face-return-transfer/1", "policy": POLICY,
        "advance": ADVANCE, "freeze_sha256": common.digest(folder / "freeze.json"), "point_sha256": common.digest(folder / "point.json"),
        "points": point, "held": counts, "swaps_by_duration": durations, "groups_with_rank_gain": len(gains),
        "window_coverage": coverage, "pilot_advances": passed, "runtime": {"development": d_run, "held": h_run}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "seal", "held", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, default=prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.archive, args.prior)
    elif args.stage in {"development", "held"}:
        run(args.folder, args.archive, args.stage)
    elif args.stage == "seal":
        seal(args.folder)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
