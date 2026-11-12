# SPDX-License-Identifier: Apache-2.0
"""Compare native texture and chroma residuals around short mouth redraws."""

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
from scripts import review_video_mouth_jaw_transfer as review
from scripts import video_mouth_jaw_transfer as panel
from scripts.video_common_sources_v89 import read, require, sha, write

PRIOR = panel.DATA / "video-mouth-jaw-transfer-2026-11-11"
SPEC = {
    "schema": "chipmunk.video-mouth-native-texture/1",
    "core": [17, 9],
    "perioral": [24, 14],
    "control": [29, 20],
    "min_quiet_pixels": 24,
    "quiet_quantile": .65,
    "scales_native_px": [.8, 1.65],
    "ridge": .20,
    "feature_scale_floor": .04,
    "calibration": "Third largest ordinary development maximum. Strict exceedance.",
    "training": "Within-clip authored window versus its before and after flanks, with matched positive and ordinary clips. "
                "Each GRID development source contributes equal positive and ordinary means.",
    "selection": "Same native time points, eligible window widths and both flank requirements as the direct comparator.",
    "advance": {"grid_added": 6, "grid_components": 3, "compatible_added": 2, "grid_extra_flags": 3,
                "native_added": 4, "native_components": 3, "native_extra_flags": 2,
                "reencode_extra_flags": 1, "hard_cut_extra_flags": 0, "min_assessed": .80},
    "scope": "Single supplied clip, visual texture only. No audio, clean counterpart or edit mask enters held scoring.",
}
OLD_FILES = ("freeze.json", "calibration.json", "controls.json", "scored.jsonl")
FIT_ORDINARY = ("original", "resample_musetalk", "resample_liveportrait",
                "brightness_musetalk", "brightness_liveportrait", "genuine_frame_retime")


def bindings():
    paths = [Path(__file__), Path(panel.__file__), Path(panel.prior.__file__),
             panel.ROOT / "backend/app/detectors/video/decode.py",
             panel.ROOT / "backend/app/detectors/video/shots.py",
             panel.ROOT / "backend/app/detectors/image/faces.py", panel.prior.WEIGHT]
    return {str(path.relative_to(panel.ROOT)): sha(path) for path in paths}


def seal(prior):
    study = read(prior / "freeze.json")
    runtime = read(prior / "run.json")
    require(runtime["journal_sha256"] == sha(prior / "scored.jsonl")
            and runtime["controls_sha256"] == sha(prior / "controls.json")
            and runtime["calibration_sha256"] == sha(prior / "calibration.json"), "Prior direct receipts differ")
    groups = {phase: {pin["component"] for pin in study["grid"][phase]} for phase in ("development", "held")}
    require(not (groups["development"] & groups["held"]) and all(len(v) == 4 for v in groups.values()),
            "Require separate frozen GRID source components")
    return {"protocol": SPEC, "prior": {name: sha(prior / name) for name in OLD_FILES},
            "bindings": bindings(), "versions": {"numpy": np.__version__, "opencv": cv2.__version__}}


def clipped_energy(values):
    return np.log1p(np.quantile(np.abs(values), (.50, .85), axis=0)).ravel()


def correlation(left, right):
    if len(left) < 10:
        return 0.
    left, right = (part - np.mean(part) for part in (left, right))
    return float(np.dot(left, right) / max(.1, float(np.linalg.norm(left) * np.linalg.norm(right))))


def local_vector(residual, support, edges):
    if np.count_nonzero(support) < SPEC["min_quiet_pixels"]:
        return None
    quiet = support & (edges <= np.quantile(edges[support], SPEC["quiet_quantile"]))
    if np.count_nonzero(quiet) < SPEC["min_quiet_pixels"]:
        return None
    values = residual[quiet]
    luma = residual[..., 0]
    x = quiet[:, 1:] & quiet[:, :-1]
    y = quiet[1:] & quiet[:-1]
    d = quiet[1:, 1:] & quiet[:-1, :-1]
    return np.r_[clipped_energy(values),
                 correlation(values[:, 0], values[:, 1]), correlation(values[:, 0], values[:, 2]),
                 correlation(values[:, 1], values[:, 2]),
                 correlation(luma[:, 1:][x], luma[:, :-1][x]),
                 correlation(luma[1:][y], luma[:-1][y]),
                 correlation(luma[1:, 1:][d], luma[:-1, :-1][d])]


def texture(rgb, geometry):
    """Describe native residuals. Eye registration places regions without resizing input pixels."""
    _, affine, mouth, _ = geometry
    x0, y0, x1, y1 = panel.native_bounds(geometry)
    patch = rgb[y0:y1, x0:x1]
    if min(patch.shape[:2]) < 24:
        return None
    y, x = np.mgrid[y0:y1, x0:x1]
    xx = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2] - mouth[0]
    yy = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2] - mouth[1]
    ellipses = [(xx / w) ** 2 + (yy / h) ** 2 <= 1 for w, h in
                (SPEC["core"], SPEC["perioral"], SPEC["control"])]
    core, inner, outer = ellipses
    bands = (core, inner & ~core, outer & ~inner)
    pixels = cv2.cvtColor(patch, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    coarse = cv2.GaussianBlur(pixels[..., 0], (0, 0), 1.65)
    edges = np.abs(cv2.Sobel(coarse, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(coarse, cv2.CV_64F, 0, 1))
    features = []
    for sigma in SPEC["scales_native_px"]:
        residual = pixels - cv2.GaussianBlur(pixels, (0, 0), sigma)
        values = [local_vector(residual, region, edges) for region in bands]
        if any(value is None for value in values):
            return None
        a, b, c = values
        features.extend((a-c, b-c))
    return np.concatenate(features), [x0, y0, x1, y1], panel.raw_digest(patch)


def measure(path, source, detector, cache):
    require(sha(path) == source["file_sha256"], "Qualified media commitment differs")
    table, indices = panel.inspect_frames(path, source["expected"])
    shots, anchors, features = [], [], []
    scene = ShotDetector(table)
    for frame, digest in panel.exact_rgb(path, table, indices, source["expected"]):
        if digest not in cache:
            geom = panel.prior.aligned_face(frame.rgb, detector)
            cache[digest] = None if geom is None else texture(frame.rgb, geom)
        scene.update(frame.index, frame.rgb)
        shots.append(len(scene.boundaries))
        observed = cache[digest]
        if observed is None:
            anchors.append(None)
            features.append(None)
            continue
        features.append(observed[0])
        anchors.append({"frame": frame.index, "pts": frame.pts, "rgb_sha256": digest,
                        "native_crop_xyxy": observed[1], "crop_rgb_sha256": observed[2]})
    return table, indices, shots, anchors, features


def design_at_known_range(observed):
    _, indices, shots, _, features = observed
    positions = [[j for j, frame in enumerate(indices) if a <= frame < b and features[j] is not None]
                 for a, b in ((20, 27), (27, 47), (47, 54))]
    if min(len(positions[0]), len(positions[2])) < 2 or len(positions[1]) < 6:
        return None
    if shots[positions[0][0]] != shots[positions[2][-1]]:
        return None
    left, core, right = (np.median([features[j] for j in selected], axis=0) for selected in positions)
    return core - (left + right) / 2


def fit(paired):
    ordinary = np.asarray([row for _, negative in paired for row in negative])
    center = np.median(ordinary, axis=0)
    scale = np.maximum(SPEC["feature_scale_floor"], 1.4826 * np.median(np.abs(ordinary-center), axis=0))
    n_features = len(center)
    cov = np.zeros((n_features, n_features), dtype=np.float64)
    direction = np.zeros(n_features)
    for positive, negative in paired:
        first, second = (np.clip((np.asarray(group)-center) / scale, -6, 6) for group in (positive, negative))
        cov += (first.T @ first / len(first) + second.T @ second / len(second)) / (2 * len(paired))
        direction += (first.mean(axis=0) - second.mean(axis=0)) / len(paired)
    weight = np.linalg.solve(cov + np.eye(n_features) * SPEC["ridge"], direction)
    require(np.isfinite(weight).all() and np.linalg.norm(weight) > 0, "A finite development direction is required")
    weight /= np.linalg.norm(weight)
    return {"center": center.tolist(), "scale": scale.tolist(), "weight": weight.tolist()}


def score(observed, model):
    table, indices, shots, anchors, features = observed
    center, scale, weight = (np.asarray(model[name]) for name in ("center", "scale", "weight"))
    values = [None if value is None else float(((value-center)/scale) @ weight) for value in features]
    return panel.best_window(values, table, indices, shots, anchors), values


def development(folder, prior, doc):
    detector = load_yunet(panel.prior.WEIGHT)
    paired, measured = [], []
    for index, _, rows in panel.grid_cases(doc, "development"):
        cache = {}
        observed = {entry["condition"]: measure(path, entry, detector, cache) for path, entry, _ in rows}
        positive, ordinary = ([value for kind in kinds if (value := design_at_known_range(observed[kind])) is not None]
                              for kinds in (panel.prior.NEURAL, FIT_ORDINARY))
        require(len(positive) >= 3 and len(ordinary) >= 4, "Incomplete development contrasts")
        paired.append((positive, ordinary))
        measured.extend((index, kind, value) for kind, value in observed.items() if kind not in panel.prior.NEURAL)
        print("development", index, len(positive), len(ordinary), flush=True)
    model = fit(paired)
    ordinary = [{"index": index, "condition": condition, "candidate": score(observed, model)[0]}
                for index, condition, observed in measured]
    maxima = sorted(row["candidate"]["score"] for row in ordinary if row["candidate"] is not None)
    require(len(maxima) >= 40, "Insufficient development maxima")
    write(folder / "calibration.json", {"model": model, "point": maxima[-3], "ordinary_development": ordinary,
          "fit_counts": [[len(pos), len(neg)] for pos, neg in paired], "freeze_sha256": sha(folder / "freeze.json"),
          "prior_direct_point": read(prior / "calibration.json")["points"]["direct"]})


def roster(prior, doc):
    return review.records(prior, doc, read(prior / "controls.json"))


def baselines(prior):
    return {(row["bank"], row["index"], case["condition"]): case for row in panel.prior.jsonl(prior / "scored.jsonl")
            for case in row["cases"]}


def held(folder, prior, doc):
    calibration = read(folder / "calibration.json")
    require(calibration["freeze_sha256"] == sha(folder / "freeze.json"), "Set the point before held scoring")
    detector = load_yunet(panel.prior.WEIGHT)
    baseline = baselines(prior)
    seen, start = set(), time.monotonic()
    with (folder / "held.jsonl").open("x") as stream:
        for (bank, index), (component, cases) in roster(prior, doc).items():
            cache = {}
            for condition, (path, source) in cases.items():
                key = bank, index, condition
                old = baseline[key]
                require(sha(path) == old["sha256"] == source["file_sha256"], "Direct comparison differs from matched input")
                observed = measure(path, source, detector, cache)
                candidate, scores = score(observed, calibration["model"])
                stream.write(json.dumps({"bank": bank, "index": index, "component": component, "condition": condition,
                    "source_sha256": source["file_sha256"], "truth": old["truth"],
                    "compatible": old["source_time_compatible"], "baseline": old["arms"]["direct"],
                    "candidate": candidate, "frames": observed[1], "shots": observed[2],
                    "scores": scores, "measured_centers": sum(x is not None for x in scores)}, allow_nan=False) + "\n")
                seen.add(key)
            stream.flush()
            print(bank, index, len(cases), flush=True)
    require(len(seen) == len(baseline) == 160, "Held comparisons are incomplete")
    write(folder / "run.json", {"elapsed_s": time.monotonic() - start,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json")})


def tally(rows, candidate_point, baseline_point):
    outcomes = defaultdict(lambda: {"direct": Counter(), "candidate": Counter()})
    added, source_groups, compatible = Counter(), defaultdict(set), Counter()
    assessment = defaultdict(Counter)
    for row in rows:
        cohort = row["condition"] if row["bank"] == "delivery" else row["bank"]
        truth = row["truth"]
        previous = review.outcome(row["baseline"], "direct", baseline_point, truth)
        current = review.outcome(row["candidate"], "direct", candidate_point, truth)
        outcomes[cohort]["direct"][previous] += 1
        outcomes[cohort]["candidate"][current] += 1
        label = "positive" if truth else "control"
        assessment[cohort][f"{label}_total"] += 1
        assessment[cohort][f"{label}_measured"] += current != "unassessed"
        if current == "correct_range" and previous != "correct_range":
            added[f"{cohort}_added"] += 1
            source_groups[cohort].add(row["component"])
            added["compatible_added"] += row["compatible"]
        if current == "false_range" and (previous != "false_range" or review.overlap(
                row["baseline"]["source_clock_s"], row["candidate"]["source_clock_s"]) < Fraction(1, 2)):
            added[f"{cohort}_extra_flags"] += 1
        if row["compatible"]:
            compatible[f"direct/{previous}"] += 1
            compatible[f"candidate/{current}"] += 1
    gate = SPEC["advance"]
    criteria = {key: added[key] >= gate[key] if key.endswith("_added") else added[key] <= gate[key]
                for key in gate if key.endswith(("_added", "_extra_flags"))}
    for bank in ("grid", "native"):
        criteria[f"{bank}_components"] = len(source_groups[bank]) >= gate[f"{bank}_components"]
        for label in ("positive", "control"):
            counts = assessment[bank]
            criteria[f"{bank}_{label}_coverage"] = counts[f"{label}_measured"] >= gate["min_assessed"] * counts[f"{label}_total"]
    return {"outcomes": dict(outcomes), "comparison": dict(added), "assessment": dict(assessment),
            "compatible": dict(compatible), "added_components": {k: len(v) for k, v in source_groups.items()},
            "criteria": criteria, "advances": all(criteria.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=PRIOR)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    folder, prior = args.folder, args.prior
    frozen = seal(prior)
    if args.stage == "freeze":
        folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(folder / "freeze.json", frozen)
        return
    require(read(folder / "freeze.json") == frozen, "Method, matched inputs or runtime changed")
    doc = read(prior / "freeze.json")
    if args.stage == "development":
        development(folder, prior, doc)
    elif args.stage == "held":
        held(folder, prior, doc)
    else:
        rows = panel.prior.jsonl(folder / "held.jsonl")
        calibration = read(folder / "calibration.json")
        runtime = read(folder / "run.json")
        require(len(rows) == 160 and runtime["held_sha256"] == sha(folder / "held.jsonl")
                and runtime["calibration_sha256"] == sha(folder / "calibration.json"), "Scored trial commitments differ")
        result = tally(rows, calibration["point"], calibration["prior_direct_point"])
        write(folder / "aggregate.json", {**result, "protocol": SPEC, "point": calibration["point"],
              "direct_point": calibration["prior_direct_point"], "fit_counts": calibration["fit_counts"],
              "run": runtime, "freeze_sha256": sha(folder / "freeze.json")})
        print(result, flush=True)


if __name__ == "__main__":
    main()
