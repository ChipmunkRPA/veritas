# SPDX-License-Identifier: Apache-2.0
"""Test localized fixed-pixel residual continuity on native talking-head video."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_jaw_transfer as panel
from scripts import video_mouth_native_texture_transfer as baseline
from scripts import video_mouth_temporal_detail as comparator
from scripts.video_common_sources_v89 import read, require, sha, write

RESERVE = panel.DATA / "grid-reserve-neural-mouth-readout-2026-11-12/freeze.json"
RESERVE_PROOF = panel.DATA / "grid-reserve-neural-mouth-2026-11-12-admitted/independent-replay.json"
ARMS = ("fixed_pixel", "mouth_vs_face")
PRIMARY = "mouth_vs_face"
ROLES = ("before", "during", "after")
SPEC = {
    "schema": "chipmunk.video-mouth-fixed-residual/1",
    "signal": "Test loss of fixed native-pixel detail agreement in the mouth and nearby face separately. "
              "Average unwarped high-pass detail in two flanks and a middle interval. Subtract the average "
              "correlation obtained at four offset positions to discount broader features. Compare agreement "
              "between flanks with the larger of the two flank-to-middle comparisons. The primary arm "
              "subtracts the face-reference deficit from the mouth deficit. No device fingerprint is established.",
    "sample_step_s": str(panel.STEP),
    "core_samples": [4, 6, 8, 10],
    "flank_samples": 4,
    "native_blur_sigma": .85,
    "relative_residual_clip": 8.,
    "ellipses_aligned_px": [[17, 9], [22, 13], [28, 19]],
    "max_mouth_drift_eye_distance": .28,
    "max_relative_eye_scale_change": .18,
    "gradient_quantile": .60,
    "quiet_pixels": 48,
    "offset_pixels": [[0, -2], [0, 2], [-2, 0], [2, 0]],
    "min_flank_agreement": .06,
    "point": "Strict exceedance of the third largest ordinary development maximum or zero.",
    "temporal_iou": "1/2",
    "admission": {"grid_added": 4, "grid_components": 2, "grid_source_time_added": 2,
                  "crema_added": 2, "crema_groups": 2, "grid_extra_false": 3, "crema_extra_false": 3,
                  "reencode_extra_false": 1, "cut_extra_false": 0, "coverage_each": .75,
                  "reserve_source_time_correct": 1, "reserve_false": 0},
}


def lock(prior, reserve):
    replay = read(RESERVE_PROOF)
    sources = read(reserve)["sources"]
    counts = Counter(row["role"] for row in sources)
    require(replay["status"] == "verified" and counts == dict.fromkeys(
        ("original", "source_time_neural", "earlier_drive_neural", "earlier_pixels_ordinary"), 2),
        "Complete independently qualified reserve role roster required")
    frozen = read(prior / "freeze.json")
    require(not set(row["ordinal"] for row in sources) &
            {row["ordinal"] for phase in ("development", "held") for row in frozen["grid"][phase]},
            "Reserve clips overlap the development or held roster")
    return {"protocol": SPEC, "baseline": baseline.seal(prior),
            "reserve_manifest_sha256": sha(reserve), "reserve_proof_sha256": sha(RESERVE_PROOF),
            "bindings": {path.name: sha(path) for path in
                         (Path(__file__), Path(comparator.__file__), Path(comparator.review.__file__))}}


def native_detail(rgb, geometry):
    _, affine, mouth, _ = geometry
    inverse = cv2.invertAffineTransform(affine)
    center = inverse @ np.r_[mouth, 1.]
    eye_distance = float(np.linalg.norm(inverse[:, 0]) * 27)
    shape = np.asarray(rgb.shape[1::-1])
    low = np.maximum(0, np.floor(center - eye_distance * 1.8).astype(int))
    high = np.minimum(shape, np.ceil(center + eye_distance * 1.8).astype(int))
    gray = cv2.cvtColor(rgb[low[1]:high[1], low[0]:high[0]], cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), SPEC["native_blur_sigma"])
    residual = np.clip((gray - smooth) * 128 / np.maximum(32, smooth),
                       -SPEC["relative_residual_clip"], SPEC["relative_residual_clip"])
    gradient = np.abs(cv2.Sobel(smooth, -1, 1, 0)) + np.abs(cv2.Sobel(smooth, -1, 0, 1))
    box = panel.native_bounds(geometry)
    x0, y0, x1, y1 = box
    return center, eye_distance, affine, gradient, residual, low, high, box, panel.raw_digest(rgb[y0:y1, x0:x1])


def collect(path, source, detector, cache):
    require(sha(path) == source["file_sha256"], "Native file differs from its qualified receipt")
    table, frames = panel.inspect_frames(path, source["expected"])
    shots, observations, anchors = [], [], []
    cut = ShotDetector(table)
    for frame, digest in panel.exact_rgb(path, table, frames, source["expected"]):
        if digest not in cache:
            geometry = panel.prior.aligned_face(frame.rgb, detector)
            cache[digest] = None if geometry is None else native_detail(frame.rgb, geometry)
        observed = cache[digest]
        cut.update(frame.index, frame.rgb)
        shots.append(len(cut.boundaries))
        observations.append(observed)
        anchors.append(None if observed is None else {
            "frame": frame.index, "pts": frame.pts, "rgb_sha256": digest,
            "native_crop_xyxy": observed[-2], "crop_rgb_sha256": observed[-1]})
    return table, frames, shots, observations, anchors


def normalized(values):
    centered = values - values.mean()
    norm = np.linalg.norm(centered)
    return None if norm < .1 * np.sqrt(len(values)) else centered / norm


def phase_agreement(first, second, yy, xx):
    a = normalized(first[yy, xx])
    b = normalized(second[yy, xx])
    if a is None or b is None:
        return None
    offsets = []
    for dy, dx in SPEC["offset_pixels"]:
        before = normalized(first[yy+dy, xx+dx])
        after = normalized(second[yy+dy, xx+dx])
        if before is None or after is None:
            return None
        offsets.extend((np.dot(a, after), np.dot(before, b)))
    return float(np.dot(a, b) - np.mean(offsets))


def support(mean_affine, center, shape, low):
    yy, xx = np.indices(shape, dtype=np.float32)
    dx, dy = xx + low[0] - center[0], yy + low[1] - center[1]
    m = mean_affine[:, :2]
    u, v = m[0, 0]*dx + m[0, 1]*dy, m[1, 0]*dx + m[1, 1]*dy
    core, middle, outside = [(u/w)**2 + (v/h)**2 <= 1 for w, h in SPEC["ellipses_aligned_px"]]
    edge = (xx >= 2) & (yy >= 2) & (xx < shape[1]-2) & (yy < shape[0]-2)
    return core & edge, outside & ~middle & edge


def groups_detail(observations, segments, shape):
    values = [observations[i] for segment in segments for i in segment]
    if any(value is None for value in values):
        return None
    centers = np.asarray([value[0] for value in values])
    center = np.median(centers, axis=0)
    scales = np.asarray([value[1] for value in values])
    eye = float(np.median(scales))
    drift = float(np.max(np.linalg.norm(centers-center, axis=1)) / eye)
    scale_change = float(np.ptp(scales) / eye)
    if drift > SPEC["max_mouth_drift_eye_distance"] or scale_change > SPEC["max_relative_eye_scale_change"]:
        return None
    transform = np.median([value[2] for value in values], axis=0)
    low = np.maximum(0, np.floor(center-eye*1.26).astype(int))
    high = np.minimum(shape[::-1], np.ceil(center+eye*1.26).astype(int))
    parts = {}
    for segment in segments:
        for i in segment:
            origin, edge = observations[i][5:7]
            if np.any(low < np.where(origin == 0, origin, origin+4)) or np.any(
                    high > np.where(edge == shape[::-1], edge, edge-4)):
                return None
            left, right = low-origin, high-origin
            parts[i] = np.s_[left[1]:right[1], left[0]:right[0]]
    before, middle, after = [np.mean([observations[i][4][parts[i]] for i in segment], axis=0) for segment in segments]
    # Only the two flanks choose where ordinary scene edges are faint.
    gradient = np.mean([observations[i][3][parts[i]] for segment in (segments[0], segments[2]) for i in segment], axis=0)
    measures = []
    for region in support(transform, center, before.shape, low):
        if not region.any():
            return None
        yy, xx = np.where(region & (gradient <= np.quantile(gradient[region], SPEC["gradient_quantile"])))
        if len(yy) < SPEC["quiet_pixels"]:
            return None
        correlations = [phase_agreement(a, b, yy, xx) for a, b in ((before, after), (before, middle), (middle, after))]
        if any(value is None for value in correlations) or correlations[0] < SPEC["min_flank_agreement"]:
            return None
        measures.append({"pixels": len(yy), "agreement": correlations})
    return {"mouth": measures[0], "face_reference": measures[1], "mouth_drift_eye_distance": drift,
            "relative_eye_scale_change": scale_change}


def scores(measures):
    def deficit(region):
        flanks, left_middle, middle_right = measures[region]["agreement"]
        return flanks - max(left_middle, middle_right)
    mouth = deficit("mouth")
    return {"fixed_pixel": mouth, "mouth_vs_face": mouth - deficit("face_reference")}


def measure(path, source, detector, cache):
    table, frames, shots, observations, anchors = collect(path, source, detector, cache)
    times = panel.source_clock(table)
    measured = dict.fromkeys(ARMS)
    count = 0
    flank = SPEC["flank_samples"]
    for width in SPEC["core_samples"]:
        for a in range(flank, len(frames)-width-flank+1):
            b = a + width
            lo, hi = a-flank, b+flank
            if shots[lo] != shots[hi-1] or any(not Fraction(1,25) <= times[frames[j+1]]-times[frames[j]] <= Fraction(3,25)
                                               for j in range(lo, hi-1)):
                continue
            intervals = (range(lo, a), range(a, b), range(b, hi))
            observed = groups_detail(observations, intervals, (table.height, table.width))
            if observed is None:
                continue
            count += 1
            for arm, score in scores(observed).items():
                if measured[arm] is None or score > measured[arm]["score"]:
                    chosen = (a-1, (a+b-1)//2, b)
                    measured[arm] = {"score": score, "frames": [frames[a], frames[b]], "time_base": table.time_base,
                                     "source_clock_s": [str(times[frames[a]]), str(times[frames[b]])],
                                     "measurement": observed, "sample_support": [len(part) for part in intervals],
                                     "anchors": [{**anchors[i], "role": role} for i, role in zip(chosen, ROLES, strict=True)]}
    return measured, count


def development(folder, prior, detector):
    ordinary = []
    for index, component, rows in panel.grid_cases(read(prior / "freeze.json"), "development"):
        cache = {}
        for path, row, _ in rows:
            if row["condition"] in panel.prior.NEURAL:
                continue
            chosen, count = measure(path, row, detector, cache)
            ordinary.append({"index": index, "component": component, "condition": row["condition"],
                             "sha256": row["file_sha256"], "eligible_windows": count,
                             "scores": {arm: None if item is None else item["score"] for arm, item in chosen.items()}})
        print("development", index, flush=True)
    require(len(ordinary) == 56, "All 56 ordinary development views required")
    points = {}
    for arm in ARMS:
        values = sorted(row["scores"][arm] for row in ordinary if row["scores"][arm] is not None)
        require(len(values) >= 42, f"Measured {len(values)}/56 ordinary development views for {arm}; need 42")
        points[arm] = max(0., values[-3])
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"), "ordinary": ordinary, "points": points})


def held(folder, prior, detector):
    calibration = read(folder / "calibration.json")
    require(calibration["freeze_sha256"] == sha(folder / "freeze.json"), "Seal development calibration before held scoring")
    start = time.monotonic()
    direct = baseline.baselines(prior)
    seen = set()
    with (folder / "held.jsonl").open("x") as journal:
        for (bank, index), (component, rows) in baseline.roster(prior, read(prior / "freeze.json")).items():
            cache = {}
            for condition, (path, source) in rows.items():
                key = (bank, index, condition)
                old = direct[key]
                require(source["file_sha256"] == old["sha256"], "Matched direct comparator differs")
                found, count = measure(path, source, detector, cache)
                journal.write(json.dumps({"bank": bank, "index": index, "component": component, "condition": condition,
                    "source_sha256": source["file_sha256"], "truth": old["truth"], "compatible": old["source_time_compatible"],
                    "arms": {"direct": old["arms"]["direct"], **found}, "eligible_windows": count}, allow_nan=False) + "\n")
                seen.add(key)
            journal.flush()
            print("held", bank, index, len(rows), flush=True)
    require(seen == set(direct) and len(seen) == 160, "Exact 160 held inputs required")
    write(folder / "runtime.json", {"elapsed_s": time.monotonic()-start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json")})


def reserve_score(folder, reserve, detector):
    calibration = read(folder / "calibration.json")
    require(calibration["freeze_sha256"] == sha(folder / "freeze.json"), "Reserve uses the frozen ordinary calibration")
    rows = []
    caches = defaultdict(dict)
    for source in read(reserve)["sources"]:
        index, role = source["ordinal"], source["role"]
        path = panel.ROOT / source["path"]
        found, count = measure(path, {"file_sha256": source["sha256"], "expected": None}, detector, caches[index])
        neural = role in ("source_time_neural", "earlier_drive_neural")
        # The admission study separately verified these absolute source ticks, including the original PCM.
        truth = [str(Fraction(i, 25)) for i in source["picture_edit"]] if neural else None
        rows.append({"bank": "reserve", "index": index, "component": index, "condition": role,
                     "source_sha256": source["sha256"], "truth": truth, "compatible": role == "source_time_neural",
                     "arms": found, "eligible_windows": count})
        print("reserve", index, role, count, flush=True)
    write(folder / "reserve.json", {"calibration_sha256": sha(folder / "calibration.json"), "rows": rows})


def diagnostic(rows, arm):
    positives = [r for r in rows if r["truth"] and r["arms"][arm] is not None]
    ordinary = [r["arms"][arm]["score"] for r in rows if r["truth"] is None and r["arms"][arm] is not None]
    localized = [r["arms"][arm]["score"] for r in positives
                 if comparator.review.overlap(r["arms"][arm]["source_clock_s"], r["truth"]) >= Fraction(1, 2)]
    pair = [(r["arms"][arm]["score"], q) for r in positives for q in ordinary]
    return {"above_half_iou_before_point": len(localized), "measured_positives": len(positives),
            "measured_ordinary": len(ordinary), "top_score_auc": None if not pair else
            sum((a>b) + .5*(a==b) for a, b in pair) / len(pair)}


def report(folder, prior):
    calibration, runtime, reserve = (read(folder / name) for name in ("calibration.json", "runtime.json", "reserve.json"))
    require(runtime["held_sha256"] == sha(folder / "held.jsonl")
            and runtime["calibration_sha256"] == reserve["calibration_sha256"] == sha(folder / "calibration.json"),
            "Held or reserve scoring differs from its calibration")
    rows = panel.prior.jsonl(folder / "held.jsonl")
    require(len(rows) == 160 and len(reserve["rows"]) == 8, "Complete held and reserve inputs required")
    points = {"direct": read(prior / "calibration.json")["points"]["direct"], **calibration["points"]}
    banks = defaultdict(list)
    for row in [*rows, *reserve["rows"]]:
        bank = row["condition"] if row["bank"] == "delivery" else row["bank"]
        banks[bank].append(row)
    outcomes, additions, false_additions, groups, compatible, diagnostics = {}, Counter(), Counter(), defaultdict(set), {}, {}
    for bank, items in banks.items():
        arms = tuple(items[0]["arms"])
        outcomes[bank] = {}
        diagnostics[bank] = {arm: diagnostic(items, arm) for arm in arms}
        for arm in arms:
            outcomes[bank][arm] = Counter(comparator.decision(row["arms"][arm], points[arm], row["truth"]) for row in items)
        for row in items:
            decided = {arm: comparator.decision(row["arms"][arm], points[arm], row["truth"]) for arm in arms}
            if row["compatible"]:
                for arm, value in decided.items():
                    compatible.setdefault(bank, {}).setdefault(arm, Counter())[value] += 1
            if decided[PRIMARY] == "correct_range" and decided.get("direct") != "correct_range":
                additions[bank] += 1
                additions[bank+"_compatible"] += row["compatible"]
                groups[bank].add(row["component"])
            if decided[PRIMARY] == "false_range" and (decided.get("direct") != "false_range" or comparator.review.overlap(
                    row["arms"][PRIMARY]["source_clock_s"], row["arms"]["direct"]["source_clock_s"]) < Fraction(1, 2)):
                false_additions[bank] += 1
    admission = SPEC["admission"]
    criteria = {"grid": additions["grid"] >= admission["grid_added"] and len(groups["grid"]) >= admission["grid_components"],
                "grid_source_time": additions["grid_compatible"] >= admission["grid_source_time_added"],
                "crema": additions["native"] >= admission["crema_added"] and len(groups["native"]) >= admission["crema_groups"],
                "grid_ordinary": false_additions["grid"] <= admission["grid_extra_false"],
                "crema_ordinary": false_additions["native"] <= admission["crema_extra_false"],
                "reencode": false_additions["reencode"] <= admission["reencode_extra_false"],
                "cut": false_additions["hard_cut"] <= admission["cut_extra_false"],
                "reserve": additions["reserve_compatible"] >= admission["reserve_source_time_correct"]
                           and false_additions["reserve"] <= admission["reserve_false"],
                "coverage": all(sum(row["arms"][PRIMARY] is not None for row in items) >= admission["coverage_each"]*len(items)
                                for bank in ("grid", "native") for positive in (False, True)
                                if (items := [row for row in banks[bank] if bool(row["truth"]) == positive]))}
    write(folder / "aggregate.json", {"protocol": SPEC, "freeze_sha256": sha(folder / "freeze.json"),
          "outcomes": outcomes, "source_time_compatible": compatible, "diagnostics": diagnostics, "additions": additions,
          "new_false_ranges": false_additions, "addition_groups": {bank: len(parts) for bank, parts in groups.items()},
          "criteria": criteria, "advances": all(criteria.values()), "points": points, "runtime": runtime,
          "reserve_sha256": sha(folder / "reserve.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "reserve", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=baseline.PRIOR)
    parser.add_argument("--reserve", type=Path, default=RESERVE)
    args = parser.parse_args()
    frozen = lock(args.prior, args.reserve)
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(args.folder / "freeze.json", frozen)
        return
    require(read(args.folder / "freeze.json") == frozen, "Frozen source or method changed")
    if args.phase == "report":
        report(args.folder, args.prior)
        return
    cv2.setNumThreads(1)
    detector = load_yunet(panel.prior.WEIGHT)
    if args.phase == "reserve":
        reserve_score(args.folder, args.reserve, detector)
    else:
        action = development if args.phase == "development" else held
        action(args.folder, args.prior, detector)


if __name__ == "__main__":
    main()
