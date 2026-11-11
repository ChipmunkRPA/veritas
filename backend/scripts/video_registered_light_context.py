# SPDX-License-Identifier: Apache-2.0
"""Replay fixed motion-compensated exposure context around saved appearance flags."""

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

from app.detectors.image.faces import detect_faces
from app.detectors.video import face_appearance
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_face_short_boundary_confirmation as prior
from scripts import video_face_texture_external as external
from scripts import video_face_texture_return as common

SCHEMA = "chipmunk.video-registered-light/1"
ROLES = ("before", "during", "after")
SCENE_WIDTH = 320
MARGIN = .50
TILES = (6, 4)
MIN_POINTS = 20
MIN_INLIER_SHARE = .55
MIN_TILES = 12
MIN_QUADRANTS = 3
MIN_TILE_PIXELS = 48
MIN_TEMPORARY_LOG_SHIFT = .06
MAX_FLANK_LOG_SHIFT = .04
MIN_SUPPORT = .60
FACE_RATIO = (.5, 2.)
POINT = face_appearance.REVIEW_POINT
PUBLIC = common.ROOT / "docs/evaluation-panels/video-face-short-boundary-confirmation-2026-11-11.json"
SELF = Path(__file__).resolve()
Y, X = np.ogrid[:112, :112]
FACE_INTERIOR = (((X - 56) / 26) ** 2 + ((Y - 72) / 25) ** 2 <= 1)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def contexts(frames):
    """Build three robust photometric roles from the nine measured pictures."""
    center = frames["during"][1]
    registered = {}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    reference = clahe.apply(center["scene"])
    h, w = reference.shape
    for role in ROLES:
        registered[role] = []
        for item in frames[role]:
            if item is center:
                registered[role].append((item["scene"], item["outside"]))
                continue
            matrix = scene_transform(reference, clahe.apply(item["scene"]), center["outside"], item["outside"])
            if matrix is None:
                return {"status": "unavailable", "reason": "scene_registration"}
            valid = cv2.warpAffine(item["outside"], matrix, (w, h), flags=cv2.INTER_NEAREST)
            # Exclude a one-pixel interpolation margin around faces and borders.
            valid = cv2.erode(valid, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            aligned = cv2.warpAffine(item["scene"], matrix, (w, h), flags=cv2.INTER_LINEAR)
            registered[role].append((aligned, valid))
    common_area = np.logical_and.reduce([valid for rows in registered.values() for _, valid in rows])
    images = [np.median([np.log(image.astype(np.float32) + 16) for image, _ in registered[role]], axis=0)
              for role in ROLES]
    tiles, quadrant = [], []
    for row, ys in enumerate(np.array_split(np.arange(h), TILES[1])):
        for col, xs in enumerate(np.array_split(np.arange(w), TILES[0])):
            area = np.ix_(ys, xs)
            good = common_area[area]
            if np.count_nonzero(good) < max(MIN_TILE_PIXELS, good.size // 2):
                continue
            tiles.append([float(np.median(image[area][good])) for image in images])
            quadrant.append((row >= TILES[1] // 2, col >= TILES[0] // 2))
    if len(tiles) < MIN_TILES or len(set(quadrant)) < MIN_QUADRANTS:
        return {"status": "unavailable", "reason": "scene_tile_coverage"}
    faces = np.stack([np.median([np.log(item["face"].astype(np.float32) + 16) for item in frames[role]], axis=0)
                      for role in ROLES])
    return illumination(np.asarray(tiles).T, quadrant, faces)


def scene_transform(reference, moving, reference_mask, moving_mask):
    points = cv2.goodFeaturesToTrack(reference, maxCorners=220, qualityLevel=.008, minDistance=6, mask=reference_mask)
    if points is None or len(points) < MIN_POINTS:
        return None
    opts = {"winSize": (17, 17), "maxLevel": 3}
    warped, forward, _ = cv2.calcOpticalFlowPyrLK(reference, moving, points, None, **opts)
    if warped is None or forward is None:
        return None
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(moving, reference, warped, None, **opts)
    if returned is None or backward is None:
        return None
    h, w = reference.shape
    p, q = points.reshape(-1, 2), warped.reshape(-1, 2)
    good = (forward.ravel() == 1) & (backward.ravel() == 1) & (np.linalg.norm(returned.reshape(-1, 2) - p, axis=1) <= 1.)
    good &= np.isfinite(q).all(axis=1) & (q[:, 0] >= 3) & (q[:, 1] >= 3) & (q[:, 0] < w - 3) & (q[:, 1] < h - 3)
    p, q = p[good], q[good]
    if len(p) < MIN_POINTS:
        return None
    x, y = np.rint(q).astype(int).T
    keep = moving_mask[y, x] > 0
    p, q = p[keep], q[keep]
    if len(p) < MIN_POINTS:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(q, p, method=cv2.RANSAC, ransacReprojThreshold=1.25,
                                                   maxIters=1000, confidence=.99, refineIters=10)
    if matrix is None or inliers is None:
        return None
    accepted = p[inliers.ravel() == 1]
    spread = set(zip(accepted[:, 0] >= w / 2, accepted[:, 1] >= h / 2, strict=True))
    scale = float(np.linalg.norm(matrix[:, 0]))
    if (len(accepted) < MIN_POINTS or len(accepted) / len(p) < MIN_INLIER_SHARE
            or len(spread) < MIN_QUADRANTS or not .9 <= scale <= 1.1):
        return None
    return matrix


def persistent(roles):
    before, during, after = roles
    a, b = during - before, during - after
    temporary = np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.)
    return temporary, np.abs(after - before)


def illumination(scene_roles, tile_quadrants, face_roles):
    scene, scene_flank = persistent(scene_roles)
    face, face_flank = persistent(face_roles)
    signed_scene = float(np.median(scene))
    signed_face = float(np.median(face))
    sign = np.sign(signed_scene)
    supporting = (sign * scene >= MIN_TEMPORARY_LOG_SHIFT) & (scene_flank <= MAX_FLANK_LOG_SHIFT)
    scene_support = float(np.mean(supporting))
    face_support = float(np.mean((sign * face >= MIN_TEMPORARY_LOG_SHIFT * FACE_RATIO[0]) &
                                 (face_flank <= MAX_FLANK_LOG_SHIFT)))
    quadrants = len({q for q, keep in zip(tile_quadrants, supporting, strict=True) if keep})
    ratio = abs(signed_face / signed_scene) if signed_scene else None
    possible = (
        scene_support >= MIN_SUPPORT and face_support >= MIN_SUPPORT and quadrants >= MIN_QUADRANTS
        and ratio is not None and FACE_RATIO[0] <= ratio <= FACE_RATIO[1]
        and sign * signed_face > 0
    )
    return {"status": "measured", "context_ambiguous": bool(possible), "tiles": len(tile_quadrants),
            "supporting_quadrants": quadrants, "scene_support_fraction": scene_support, "face_support_fraction": face_support,
            "median_scene_log_shift": signed_scene, "median_face_log_shift": signed_face}


def source_pixels(frame, yunet):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    faces = detect_faces(bgr, detector=yunet, min_size=48)
    if len(faces) != 1:
        return None
    face = faces[0]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fit = face_appearance.cpp_alignment(face.landmarks)
    crop = cv2.warpAffine(gray, fit, (112, 112), flags=cv2.INTER_LINEAR)
    valid_face = cv2.warpAffine(np.ones_like(gray), fit, (112, 112), flags=cv2.INTER_NEAREST)
    if not valid_face[FACE_INTERIOR].all():
        return None
    native_h, native_w = gray.shape
    scene_h = round(native_h * SCENE_WIDTH / native_w)
    scene = cv2.resize(gray, (SCENE_WIDTH, scene_h), interpolation=cv2.INTER_AREA)
    mask = np.full(scene.shape, 255, np.uint8)
    x, y, w, h = face.box()
    l = max(0, int((x - MARGIN * w) * SCENE_WIDTH / native_w))
    r = min(SCENE_WIDTH, 1 + int((x + (1 + MARGIN) * w) * SCENE_WIDTH / native_w))
    t = max(0, int((y - MARGIN * h) * scene_h / native_h))
    b = min(scene_h, 1 + int((y + (1 + MARGIN) * h) * scene_h / native_h))
    mask[t:b, l:r] = 0
    mask[:4] = mask[-4:] = 0
    mask[:, :4] = mask[:, -4:] = 0
    return {"scene": scene, "outside": mask, "face": crop[FACE_INTERIOR]}


def configuration(previous):
    names = ("freeze.json", "scored.jsonl", "run.json")
    committed = common.read(PUBLIC)["commitments"]
    require(all(common.digest(previous / name) == committed[name] for name in names), "Prior exact evaluated roster and response required")
    return {
        "schema": SCHEMA + "-freeze", "implementation_sha256": common.digest(SELF),
        "previous_result_sha256": common.digest(PUBLIC),
        "previous_private_commitments": {name: committed[name] for name in names},
        "runtime_bindings": common.bindings(),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "roles": "three previous before/during/after representative frame indices with their immediate native neighbors",
        "scene_registration": "320-pixel-width grayscale; CLAHE alignment only; face box expanded 0.5 dimensions; sparse forward/back KLT; partial affine RANSAC",
        "minimum_tracking_points": MIN_POINTS, "minimum_inlier_share": MIN_INLIER_SHARE,
        "minimum_quadrants": MIN_QUADRANTS, "scene_tiles_xy": list(TILES), "minimum_tile_count": MIN_TILES,
        "minimum_tile_pixels": MIN_TILE_PIXELS, "minimum_directional_support_fraction": MIN_SUPPORT,
        "log_transform": "log(Y + 16)", "minimum_temporary_scene_log_shift": MIN_TEMPORARY_LOG_SHIFT,
        "maximum_flank_log_shift": MAX_FLANK_LOG_SHIFT, "allowed_face_scene_shift_ratio": list(FACE_RATIO),
        "baseline_appearance_point": POINT,
        "advance_only_if": "at least four camera false flags withheld across both camera source groups, zero correctly localized swaps withheld, at least 90% of both selected cohorts measured",
        "policy": "only withhold a previously flagged range when measured exposure moves coherently in scene and aligned face; otherwise keep the range. Never rerank or conclude originality.",
    }


def inputs_for(pin, archive):
    if pin["type"] == "camera":
        folder, cases = external.unpack(pin, archive)
        return folder, {row["condition"]: {**row, "expected_rgb_sha256": row["rgb_sha256"]} for row in cases}
    i = pin["index"]
    receipt, review = (common.DATA / f"group-{i:03d}-complete.json",
                       common.DATA / f"group-{i:03d}-command-review-v96/result.json")
    require(common.digest(receipt) == pin["receipt_sha256"] and common.digest(review) == pin["review_sha256"],
            "Moving video receipts changed")
    return archive / "video-native-inserts-v95" / f"group-{i:03d}", {row["condition"]: row for row in common.read(receipt)["cases"]}


def source_measure(path, row, wanted, yunet):
    require(path.is_file() and common.digest(path) == row["sha256"], "Native video bytes differ")
    table = probe_frames(path, timeout_s=30)
    expected = row["frame_table"]
    base = Fraction(expected["time_base"])
    require((table.width, table.height) == (expected["width"], expected["height"])
            and not (table.interpolated or table.truncated or table.rotation)
            and [p * table.tb for p in table.pts] == [p * base for p in expected["pts"]], "Native video table differs")
    images = {}
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=65):
        require(frame.pts_verified and hashlib.sha256(frame.rgb).hexdigest() == row["expected_rgb_sha256"][frame.index],
                "Native picture time or pixels differ")
        images[frame.index] = source_pixels(frame.rgb, yunet)
    require(set(images) == wanted, "Native picture set incomplete")
    return images


def run(output, previous, archive):
    require(common.read(output / "freeze.json") == configuration(previous), "Current runtime differs from the prospective freeze")
    frozen = common.read(previous / "freeze.json")
    groups = [json.loads(line) for line in (previous / "scored.jsonl").read_text().splitlines()]
    cv2.setNumThreads(1)
    cv2.setRNGSeed(713)
    yunet = common.load_yunet(common.FACE_FILE)
    start = time.monotonic()
    pictures = 0
    with (output / "scored.jsonl").open("x") as journal:
        for ordinal, (pin, group) in enumerate(zip(frozen["roster"], groups, strict=True), 1):
            camera = pin["type"] == "camera"
            require(group["group"] == ordinal and group["cohort"] == pin["type"], "Previous score roster differs")
            folder, cases = inputs_for(pin, archive)
            original, times, intervals = prior.match_cases(folder, cases, camera=camera)
            saved = []
            queues = defaultdict(set)
            for row in group["rows"]:
                result = row["current"]
                flagged = bool(result and result["score"] > POINT)
                swap = row["condition"] in common.SWAP_CONDITIONS
                classified = {**{key: row[key] for key in ("condition", "duration", "truth")}, "flagged": flagged,
                              "baseline_assessed": result is not None,
                              "correct_swap": bool(swap and flagged and common.overlap(result, row["truth"]) >= Fraction(1, 2))}
                saved.append(classified)
                if not flagged:
                    continue
                anchors = result["anchors"]
                around = [[i - 1, i, i + 1] for i in anchors]
                flat = sum(around, [])
                if flat != sorted(set(flat)) or not 0 <= flat[0] <= flat[-1] < len(times):
                    classified["context"] = {"status": "unavailable", "reason": "native_role_coverage"}
                    continue
                variant = row["condition"] if row["condition"] != "original" else original["condition"]
                lo, hi = intervals[row["duration"]] if row["duration"] else (0, 0)
                selected = [[(variant if lo <= i < hi else original["condition"], i) for i in part] for part in around]
                classified["private_indices"] = selected
                for name, i in sum(selected, []):
                    queues[name].add(i)
            assets = {name: source_measure(folder / cases[name]["file"], cases[name], wanted, yunet) for name, wanted in queues.items()}
            pictures += sum(len(rows) for rows in assets.values())
            for row in saved:
                if "private_indices" not in row:
                    continue
                rows = [[assets[name][i] for name, i in part] for part in row.pop("private_indices")]
                row["context"] = ({"status": "unavailable", "reason": "single_face_missing"} if any(part is None for part in sum(rows, []))
                                  else contexts(dict(zip(ROLES, rows, strict=True))))
            journal.write(json.dumps({"group": ordinal, "cohort": pin["type"], "rows": saved},
                                     sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            print(f"{ordinal}/{len(groups)} source groups, {pictures} verified native pictures", flush=True)
    common.write(output / "run.json", {"native_pictures": pictures, "seconds": time.monotonic() - start,
                                      "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                      "new_model_weights": 0, "appearance_model_forwards": 0, "existing_geometry_forwards": pictures})


def readout(output, previous):
    require(common.read(output / "freeze.json") == configuration(previous), "Current runtime differs from the prospective freeze")
    groups = [json.loads(line) for line in (output / "scored.jsonl").read_text().splitlines()]
    require(len(groups) == len(common.read(previous / "freeze.json")["roster"]), "Run is incomplete")
    result = defaultdict(Counter)
    camera_groups = set()
    for group in groups:
        for row in group["rows"]:
            swap = row["condition"] in common.SWAP_CONDITIONS
            cohort = "known_swaps" if swap else "camera_controls" if group["cohort"] == "camera" else "matched_controls"
            counts = result[cohort]
            counts["views"] += 1
            counts["baseline_assessed"] += row["baseline_assessed"]
            counts["baseline_flags"] += row["flagged"]
            counts["baseline_correct_swap_ranges"] += row["correct_swap"]
            context = row.get("context")
            if row["flagged"]:
                require(context is not None, "Missing context outcome for baseline flag")
            measured = bool(context and context["status"] == "measured")
            ambiguous = bool(measured and context["context_ambiguous"])
            counts["context_measured"] += measured
            counts["withheld_flags"] += ambiguous
            counts["withheld_correct_swap_ranges"] += ambiguous and row["correct_swap"]
            if context and not measured:
                counts["unavailable_" + context["reason"]] += 1
            if ambiguous and cohort == "camera_controls":
                camera_groups.add(group["group"])
    camera, swaps = result["camera_controls"], result["known_swaps"]
    gate = {
        "at_least_four_camera_false_flags_withheld": camera["withheld_flags"] >= 4,
        "both_camera_source_groups_improved": len(camera_groups) == 2,
        "no_correct_swap_ranges_withheld": swaps["withheld_correct_swap_ranges"] == 0,
        "both_selected_cohorts_at_least_90_percent_measured": all(c["context_measured"] >= .90 * c["baseline_flags"] for c in (camera, swaps)),
    }
    common.write(output / "aggregate.json", {
        "schema": SCHEMA + "-result", "by_cohort": result, "camera_source_groups_with_withheld_flags": len(camera_groups),
        "passed_frozen_advancement_rule": all(gate.values()), "advancement_rule_components": gate,
        "source_groups": len(groups), "run": common.read(output / "run.json"),
        "commitments": {name: common.digest(output / name) for name in ("freeze.json", "scored.jsonl", "run.json")},
        "serving_changed": False,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "run", "readout"))
    cli.add_argument("output", type=Path)
    cli.add_argument("--previous", type=Path, required=True)
    cli.add_argument("--archive", type=Path)
    args = cli.parse_args()
    if args.stage == "freeze":
        common.write(args.output / "freeze.json", configuration(args.previous))
    elif args.stage == "run":
        if args.archive is None:
            cli.error("run requires --archive")
        run(args.output, args.previous, args.archive)
    else:
        readout(args.output, args.previous)
