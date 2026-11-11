# SPDX-License-Identifier: Apache-2.0
"""Assess denser boundary placement for already-positive face appearance ranges."""
from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from app.detectors.image.faces import iou
from scripts import video_face_texture_return as common

CADENCE = 12
SEARCH = Fraction(3, 8)
PROTOTYPE_NEAR = Fraction(1, 4)
PROTOTYPE_FAR = Fraction(5, 8)
CORE_MARGIN = Fraction(1, 8)
MAX_GAP = Fraction(1, 8)


def freeze(output, source_run, media):
    old = common.read(source_run / "freeze.json")
    excluded = {row["index"] for row in old["roster"]}
    options = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        source = common.read(path)
        index = source["index"]
        if source["role"] != "confirmation" or index in excluded:
            continue
        review_path = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
        if not review_path.exists():
            continue
        review = common.read(review_path)
        receipt_sha = common.digest(path)
        if review["status"] != "complete" or review["private_group_receipt_sha256"] != receipt_sha:
            continue
        cases = source["cases"]
        table = cases[0]["frame_table"]
        if table["width"] < 1280 or table["height"] < 720 or len(table["pts"]) < 144:
            continue
        root = media / f"group-{index:03d}"
        if any((root / case["file"]).stat().st_size != case["bytes"] for case in cases):
            continue
        options.append((sum(case["bytes"] for case in cases), index, receipt_sha, common.digest(review_path)))
    if len(options) < 6:
        raise RuntimeError("Not enough source-disjoint native-clock groups")
    common.write(output / "freeze.json", {
        "schema": "chipmunk.video-face-native-boundary-freeze/1",
        "worker_sha256": common.digest(Path(__file__)), "bindings": common.bindings(),
        "excluded_prior_study_sha256": common.digest(source_run / "freeze.json"),
        "eligibility": "six smallest remaining prior confirmation groups; no detector-response selection",
        "roster": [{"index": i, "receipt_sha256": r, "review_sha256": v} for _, i, r, v in sorted(options)[:6]],
        "geometry": "same single observed face, complete neighboring samples and no cross-track split",
        "detection": "current 4Hz appearance review point unchanged. Refine only a flagged range",
        "placement": "12Hz native nearest, three phase prototypes, two supports on each side of both signed local splits",
        "window": "both edges within 3/8 second of current; require coherent movement and same-face normalized separation",
        "comparison": "both swap families and original/light/blur/time-step controls, nominal 0.75 and 1.50 second ranges; exact native time; no fitting",
    })


def prototype(pictures, vectors, times, low, high):
    candidates = [vectors[i] for i in pictures if low <= times[i] <= high and i in vectors]
    if len(candidates) < 3:
        return None
    return np.median(np.array(candidates), axis=0)


def state_switch(pictures, vectors, faces, times, near, before, after):
    if common.face_appearance.distance(before, after) < .025:
        return None
    local = [i for i in pictures if abs(times[i] - near) <= SEARCH + 2 * MAX_GAP]
    ranked = []
    for slot in range(2, len(local) - 1):
        split = local[slot]
        if abs(times[split] - near) > SEARCH:
            continue
        support = local[slot - 2:slot + 2]
        if any(i not in vectors for i in support):
            continue
        if any(times[b] - times[a] > MAX_GAP or iou(faces[a], faces[b]) < .35 for a, b in zip(support, support[1:])):
            continue
        signed = [common.face_appearance.distance(vectors[i], before) - common.face_appearance.distance(vectors[i], after)
                  for i in support]
        if max(signed[:2]) >= 0 or min(signed[2:]) <= 0:
            continue
        strength = min(abs(signed[1]), abs(signed[2]))
        ranked.append((-strength, abs(times[split] - near), split))
    return min(ranked)[-1] if ranked else None


def refined(baseline, pictures, times, features):
    if baseline is None or baseline["score"] <= common.face_appearance.REVIEW_POINT:
        return None
    start, end = Fraction(baseline["start"]), Fraction(baseline["end"])
    vectors = {i: common.face_appearance.unit(features[i]["appearance"]) for i in pictures
               if i in features and features[i]["appearance"] is not None}
    faces = {i: features[i]["face"] for i in vectors}
    before = prototype(pictures, vectors, times, start - PROTOTYPE_FAR, start - PROTOTYPE_NEAR)
    inside = prototype(pictures, vectors, times, start + CORE_MARGIN, end - CORE_MARGIN)
    after = prototype(pictures, vectors, times, end + PROTOTYPE_NEAR, end + PROTOTYPE_FAR)
    if any(v is None for v in (before, inside, after)):
        return None
    if (np.linalg.norm(before) <= 1e-12 or np.linalg.norm(inside) <= 1e-12 or np.linalg.norm(after) <= 1e-12):
        return None
    distance = common.face_appearance.distance
    if distance(before, after) >= min(distance(before, inside), distance(after, inside)):
        return None
    onset = state_switch(pictures, vectors, faces, times, start, before, inside)
    stop = state_switch(pictures, vectors, faces, times, end, inside, after)
    if onset is None or stop is None or times[stop] - times[onset] < Fraction(3, 8):
        return None
    onset_slot = pictures.index(onset)
    middle = [i for i in pictures if onset <= i < stop and i in vectors]
    if not middle:
        return None
    center = min(middle, key=lambda i: (abs(times[i] - (times[onset] + times[stop]) / 2), i))
    return {"score": baseline["score"], "start": str(times[onset]), "end": str(times[stop]),
            "anchors": [pictures[onset_slot - 1], center, stop]}


def group(media, pin, yunet, sface):
    index = pin["index"]
    receipt_path = common.DATA / f"group-{index:03d}-complete.json"
    review_path = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
    if common.digest(receipt_path) != pin["receipt_sha256"] or common.digest(review_path) != pin["review_sha256"]:
        raise RuntimeError("Native source commitments changed")
    receipt = common.read(receipt_path)
    cases = {case["condition"]: case for case in receipt["cases"]}
    original = cases["original"]
    raw = original["frame_table"]
    tb = Fraction(raw["time_base"])
    times = [(i - raw["pts"][0]) * tb for i in raw["pts"]]
    coarse = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
    dense = common.schedule(times, CADENCE)
    wanted = {i + delta for i in coarse for delta in (-1, 0, 1)} | set(dense)
    folder = media / f"group-{index:03d}"
    table, plain, count = common.source_features(folder / original["file"], original, wanted, wanted, yunet, sface)
    rows = []

    def compare(name, duration, truth, inputs):
        baseline = common.appearance(coarse, table, times, inputs)
        changed = refined(baseline, dense, times, inputs)
        rows.append({"condition": name, "duration": duration, "truth": truth,
                     "appearance": baseline, "boundary": changed})

    compare("original", None, None, plain)
    intervals = {str(d): common.intervention(times, d) for d in common.DURATIONS}
    lo, hi = intervals[str(max(common.DURATIONS))]
    inside = {i for i in wanted if lo <= i < hi}
    for name, case in cases.items():
        if name == "original":
            continue
        if not all(i in case["actual_changed_frames"] for i in range(lo, hi)):
            raise RuntimeError("Missing confirmed altered frame in experimental interval")
        other, values, n = common.source_features(folder / case["file"], case, inside, wanted, yunet, sface)
        if common.relative_times(other) != times:
            raise RuntimeError("Matched source clock changed")
        count += n
        for duration, (a, b) in intervals.items():
            combined = {i: v for i, v in plain.items() if not a <= i < b}
            combined.update({i: v for i, v in values.items() if a <= i < b})
            compare(name, duration, [str(times[a]), str(times[b])], combined)
    return {"group": index, "native_samples": count, "rows": rows}


def run(output, media):
    frozen = common.read(output / "freeze.json")
    if frozen["worker_sha256"] != common.digest(Path(__file__)) or frozen["bindings"] != common.bindings():
        raise RuntimeError("Frozen code, models or media tools changed")
    cv2.setNumThreads(2)
    yunet = common.load_yunet(common.FACE_FILE)
    sface = cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    elapsed = time.monotonic()
    decoded = 0
    with (output / "scored.jsonl").open("x") as file:
        for ordinal, pin in enumerate(frozen["roster"], 1):
            row = group(media, pin, yunet, sface)
            decoded += row["native_samples"]
            file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            print(ordinal, "groups", decoded, "verified native samples", flush=True)
    common.write(output / "run.json", {"native_samples": decoded, "seconds": time.monotonic() - elapsed,
                                       "max_self_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def readout(output):
    rows = [json.loads(line) for line in (output / "scored.jsonl").read_text().splitlines()]
    if len(rows) != len(common.read(output / "freeze.json")["roster"]):
        raise RuntimeError("Missing planned held group")
    controls = Counter()
    metrics = defaultdict(Counter)
    timing = defaultdict(list)
    groups = Counter()
    for group in rows:
        outcomes = Counter()
        for row in group["rows"]:
            baseline = row["appearance"]
            refined_range = row["boundary"]
            positive = row["condition"] in common.SWAP_CONDITIONS
            flagged = baseline is not None and baseline["score"] > common.face_appearance.REVIEW_POINT
            assert refined_range is None or (flagged and refined_range["score"] == baseline["score"])
            if not positive:
                controls["views"] += 1
                controls["unassessed"] += baseline is None
                controls["flagged_identically"] += flagged
                controls["refined_false_ranges"] += refined_range is not None
                continue
            duration = row["duration"]
            metrics[duration]["views"] += 1
            metrics[duration]["unassessed"] += baseline is None
            metrics[duration]["flagged_identically"] += flagged
            metrics[duration]["refined"] += refined_range is not None
            if not flagged:
                continue
            coarse_overlap = common.overlap(baseline, row["truth"])
            dense_overlap = common.overlap(refined_range or baseline, row["truth"])
            metrics[duration]["improved_overlap"] += dense_overlap > coarse_overlap
            metrics[duration]["worsened_overlap"] += dense_overlap < coarse_overlap
            outcomes["improved_overlap"] += dense_overlap > coarse_overlap
            outcomes["worsened_overlap"] += dense_overlap < coarse_overlap
            for arm, result in (("current", baseline), ("dense", refined_range or baseline)):
                overlap = common.overlap(result, row["truth"])
                for point in (Fraction(1, 5), Fraction(1, 2), Fraction(3, 4)):
                    metrics[duration][f"{arm}_iou_at_least_{point}"] += overlap >= point
                a, b = (Fraction(result[k]) for k in ("start", "end"))
                c, d = map(Fraction, row["truth"])
                timing[(duration, arm)].append((overlap, abs(a - c), abs(b - d)))
        groups.update(name for name, count in outcomes.items() if count)
    common.write(output / "aggregate.json", {
        "schema": "chipmunk.video-face-native-boundary-result/1",
        "held_source_groups": len(rows), "source_groups_with_change": groups, "controls": controls,
        "swap_views_by_duration": metrics,
        "timing": {"/".join(k): dict(zip(("median_iou", "median_onset_error_s", "median_end_error_s"),
                                          (float(median(values)) for values in zip(*v)))) for k, v in timing.items()},
        "run": common.read(output / "run.json"),
        "commitments": {name: common.digest(output / name) for name in ("freeze.json", "scored.jsonl", "run.json")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output, args.previous, args.media)
    elif args.stage == "run":
        run(args.output, args.media)
    else:
        readout(args.output)
