# SPDX-License-Identifier: Apache-2.0
"""Measure a dense source-pixel guard for scaled third image views."""
from __future__ import annotations

import argparse
import io
import math
from collections import Counter
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import embedded_pair as native
from scripts import image_embedded_pair_canvas_transfer as prior

SCHEMA = "chipmunk.image-embedded-pair-dense-third/1"
SIZES = ((768, 768), (1024, 768), (1280, 720))
SCALES = (.62, .71, .87, 1.13, 1.21)
THIRDS = tuple(f"third_{round(scale * 100):03d}" for scale in SCALES)
CONTROLS = ("valid_pair", "unrelated_third", "duplicate_pair")
RULE = {"patch": 16, "grid_stride": 19, "minimum_sd": 8, "minimum_correlation": .97,
        "minimum_patches": 16, "minimum_cells": 6, "minimum_source_fraction": .55,
        "minimum_exact_masks_per_valid_role": 10,
        "maximum_new_withholds": 0, "minimum_missed_cases_per_size": 3,
        "minimum_rescued_fraction": .80, "maximum_candidate_seconds": 1.5}


def panels(data):
    old = list(prior.photo_sources(data))
    photos = prior.extra_sources(data, {row[1] for row in old}, {row[2] for row in old})
    assert len(photos) == 12 and len({row[1] for row in photos}) == len({row[2] for row in photos}) == 12
    for slot, (photo, _, _) in enumerate(photos):
        width, height = 264 + 4 * (slot % 4), 252 + 4 * (slot % 5)
        original = prior.center_pane(photo, width, height)
        changed = original.copy()
        x, y = 67 + 11 * (slot % 7), 80 + 9 * (slot % 5)
        changed[y:y+39, x:x+43] = original[20:59, 28:71]
        other = prior.center_pane(photos[(slot + 5) % len(photos)][0], width, height)
        yield slot, original, changed, other


def resized(image, scale, slot):
    height, width = image.shape[:2]
    method = Image.Resampling.BICUBIC if slot % 2 else Image.Resampling.LANCZOS
    result = Image.fromarray(image).resize((round(width * scale), round(height * scale)), method)
    if slot % 3 == 1:
        with io.BytesIO() as stream:
            result.save(stream, format="JPEG", quality=87)
            stream.seek(0)
            with Image.open(stream) as decoded:
                return np.asarray(decoded.convert("RGB")).copy()
    return np.asarray(result).copy()


def cases(data):
    inputs = list(panels(data))
    for width, height in SIZES:
        for slot, first, edited, unrelated in inputs:
            h, w = first.shape[:2]
            x, y = 24 + (slot * 7 + width) % 19, 25 + (slot * 13) % 23
            separation = w + 45 + (slot * 3 + height) % 19
            b_x, b_y = x + separation, y + (slot * 11) % 23 - 7
            for role, scale in zip(THIRDS, SCALES, strict=True):
                yield compose(width, height, slot, role, first, edited, resized(first, scale, slot), x, y, b_x, b_y)
            yield compose(width, height, slot, "valid_pair", first, edited, None, x, y, b_x, b_y)
            yield compose(width, height, slot, "unrelated_third", first, edited, resized(unrelated, .72, slot), x, y, b_x, b_y)
            yield compose(width, height, slot, "duplicate_pair", first, first, None, x, y, b_x, b_y)


def compose(width, height, slot, role, first, second, third, x, y, b_x, b_y):
    h, w = first.shape[:2]
    a, b = (x, y, x+w, y+h), (b_x, b_y, b_x+w, b_y+h)
    base = ((225, 235, 231), (44, 54, 66), (225, 216, 207))[slot % 3]
    canvas = np.full((height, width, 3), base, np.uint8)
    canvas[y:y+h, x:x+w], canvas[b_y:b_y+h, b_x:b_x+w] = first, second
    c = None
    if third is not None:
        th, tw = third.shape[:2]
        tx = 76 + (slot * 43) % 207
        ty = max(a[3], b[3]) + 22 + (slot * 17 + width) % 43
        if slot % 2 and width >= 1024 and b[2] + 24 + tw <= width - 7:
            tx = b[2] + 24 + (slot * 13) % min(21, width - 7 - tw - b[2] - 23)
            ty = 74 + (slot * 27) % 239
        assert tx + tw <= width and ty + th <= height
        c = (tx, ty, tx+tw, ty+th)
        assert native._separated(c, a, 8) and native._separated(c, b, 8)
        canvas[ty:ty+th, tx:tx+tw] = third
    return (width, height), slot, role, canvas, a, b, c, (first != second).any(axis=2)


def case_id(case):
    return {"size": list(case[0]), "slot": case[1], "role": case[2],
            "rgb_sha256": prior.sha(case[3].tobytes()), "native_areas": [list(box) if box else None for box in case[4:7]],
            "change_sha256": prior.sha(case[7].tobytes())}


def contract(data):
    return {"schema": SCHEMA, "rule": RULE, "code_sha256": prior.fs(Path(__file__)),
            "comparison_sha256": prior.fs(Path(native.__file__)), "source_code_sha256": prior.fs(Path(prior.__file__)),
            "cases": [case_id(case) for case in cases(data)]}


def alignment(gray, pair):
    height, width = gray.shape
    inside, outside = np.zeros_like(gray), np.full_like(gray, 255)
    x, y, right, bottom = pair.area_a
    inside[y+8:bottom-8, x+8:right-8] = 255
    for x, y, right, bottom in (pair.area_a, pair.area_b):
        outside[max(0, y-24):min(height, bottom+24), max(0, x-24):min(width, right+24)] = 0
    sift = cv2.SIFT_create(nfeatures=1200)
    first, desc_first = sift.detectAndCompute(gray, inside)
    other, desc_other = sift.detectAndCompute(gray, outside)
    if desc_first is None or desc_other is None or len(desc_other) < 2:
        return None
    matches = {}
    for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_first, desc_other, k=2):
        if m.distance < .72 * n.distance and (m.trainIdx not in matches or m.distance < matches[m.trainIdx].distance):
            matches[m.trainIdx] = m
    matched = sorted(matches.values(), key=lambda m: (m.queryIdx, m.trainIdx))
    if len(matched) < 12:
        return None
    src = np.float32([first[m.queryIdx].pt for m in matched])
    dst = np.float32([other[m.trainIdx].pt for m in matched])
    transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.,
                                                      maxIters=1000, confidence=.995, refineIters=10)
    if transform is None or inliers is None:
        return None
    count = int(inliers.sum())
    scale = math.hypot(transform[0, 0], transform[1, 0])
    angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    return transform if count >= max(12, .55 * len(matched)) and .60 <= scale <= 1.30 and abs(angle) <= 5 else None


def projected(box, transform):
    x, y, r, b = box
    points = np.float64([(x, y), (r, y), (x, b), (r, b)]) @ transform[:, :2].T + transform[:, 2]
    low, high = points.min(axis=0), points.max(axis=0)
    return math.floor(low[0]), math.floor(low[1]), math.ceil(high[0]), math.ceil(high[1])


def verify_dense(gray, pair, transform):
    height, width = gray.shape
    third = projected(pair.area_a, transform)
    if (third[0] < 0 or third[1] < 0 or third[2] > width or third[3] > height
            or not all(native._separated(third, box, 8) for box in (pair.area_a, pair.area_b))):
        return None
    x0, y0, right, bottom = pair.area_a
    w, h = right - x0, bottom - y0
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.)
    transform_local = transform.copy()
    transform_local[:, 2] += transform[:, :2] @ (x0, y0)
    target = cv2.warpAffine(smooth, transform_local, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    source = smooth[y0:bottom, x0:right]
    count, witnesses, occupied = 0, [], set()
    stride = RULE["grid_stride"]
    for y in range(9, h-8, stride):
        for x in range(9, w-8, stride):
            window = np.s_[y-8:y+8, x-8:x+8]
            p, q = (img[window].astype(np.float64).ravel() for img in (source, target))
            if p.std() < RULE["minimum_sd"] or pair.difference[window].any():
                continue
            count += 1
            if q.std() < RULE["minimum_sd"]:
                continue
            similarity = float(np.corrcoef(p, q)[0, 1])
            if similarity >= RULE["minimum_correlation"]:
                witnesses.append([x, y, similarity])
                occupied.add((3*x//w, 3*y//h))
    if (len(witnesses) < max(RULE["minimum_patches"], RULE["minimum_source_fraction"] * count)
            or len(occupied) < RULE["minimum_cells"]):
        return None
    return {"third": list(third), "transform": transform.tolist(), "source_candidates": count, "witnesses": witnesses}


def score(data, run):
    freeze = prior.read(run / "freeze.json")
    assert freeze == contract(data)
    rows = []
    for case, sealed in zip(cases(data), freeze["cases"], strict=True):
        assert case_id(case) == sealed
        size, slot, role, rgb, *_ = case
        status, _, pair = prior.inspect(rgb)
        started = monotonic()
        proof = None
        if pair is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            transform = alignment(gray, pair)
            if transform is not None:
                proof = verify_dense(gray, pair, transform)
        existing = None
        if pair is not None:
            existing = {"a": list(pair.area_a), "b": list(pair.area_b), "difference": list(pair.difference_bounds),
                        "mask_sha256": prior.sha(pair.difference.tobytes()), "pixels": int(pair.difference.sum()),
                        "witnesses": [list(point) for point in pair.witnesses]}
        seconds = monotonic() - started
        rows.append({"size": list(size), "slot": slot, "role": role, "current": status, "proof": proof,
                     "existing": existing, "seconds": seconds})
    prior.write(run / "score.json", {"freeze_sha256": prior.fs(run / "freeze.json"), "rows": rows})


def aggregate(rows):
    outcomes = {}
    for size in SIZES:
        key = "x".join(map(str, size))
        measured = [row for row in rows if row["size"] == list(size)]
        control_withholds = sum(row["role"] in CONTROLS and row["extra_withheld"] for row in measured)
        counts = {}
        for role in (*THIRDS, *CONTROLS):
            cases = [row for row in measured if row["role"] == role]
            assert len(cases) == 12
            counts[role] = {"current_reports": sum(row["current"] == "local_difference" for row in cases),
                            "extra_withholds": sum(row["extra_withheld"] for row in cases),
                            "exact_valid_masks": sum(row["exact_valid_mask"] for row in cases)}
        remaining = sum(counts[role]["current_reports"] - counts[role]["extra_withholds"] for role in THIRDS)
        challenges = sum(counts[role]["current_reports"] for role in THIRDS)
        highest = max(row["seconds"] for row in measured)
        passed = (control_withholds <= RULE["maximum_new_withholds"] and challenges >= RULE["minimum_missed_cases_per_size"]
                  and all(counts[role]["exact_valid_masks"] == counts[role]["current_reports"]
                          and counts[role]["exact_valid_masks"] >= RULE["minimum_exact_masks_per_valid_role"]
                          for role in ("valid_pair", "unrelated_third"))
                  and remaining <= (1 - RULE["minimum_rescued_fraction"]) * challenges
                  and highest <= RULE["maximum_candidate_seconds"])
        outcomes[key] = {"passed": passed, "roles": counts, "old_third_reports": challenges,
                         "remaining_third_reports": remaining, "max_seconds": highest}
    return outcomes


def replay(data, run):
    lock = prior.read(run / "freeze.json")
    assert lock == contract(data)
    measurements = prior.read(run / "score.json")
    assert measurements["freeze_sha256"] == prior.fs(run / "freeze.json")
    counts = Counter()
    exported = []
    for case, seal, row in zip(cases(data), lock["cases"], measurements["rows"], strict=True):
        assert case_id(case) == seal and (row["size"], row["slot"], row["role"]) == (list(case[0]), case[1], case[2])
        proof = row["proof"]
        existing = row["existing"]
        exact = False
        if existing and row["role"] in {"valid_pair", "unrelated_third"}:
            size, slot, _, rgb, a, b, third, truth = case
            exact = prior.audit_pair((size, slot, "ordinary_local", "ordinary_local", rgb, a, b, third, truth), existing)
            counts["base_exact_masks"] += exact
        if proof:
            assert row["current"] == "local_difference" and existing
            verify_proof(case, [existing["a"], existing["b"]], proof)
            counts["native_patch_witnesses"] += len(proof["witnesses"])
            counts["source_projections"] += 1
        exported.append({key: row[key] for key in ("size", "slot", "role", "current", "seconds")}
                        | {"extra_withheld": proof is not None, "exact_valid_mask": exact,
                           "native_witness_count": len(proof["witnesses"]) if proof else 0})
    result = {"schema": SCHEMA, "freeze_sha256": measurements["freeze_sha256"], "rule": RULE,
              "rows": exported, "audit": dict(counts), "outcomes": aggregate(exported)}
    prior.write(run / "public.json", result)
    print(result["audit"], result["outcomes"], flush=True)


def verify_proof(case, compared, proof):
    """Recompute recorded correlations from native patch coordinates and the saved affine map."""
    _, _, role, rgb, _, _, authored_third, _ = case
    a, b = (tuple(box) for box in compared)
    matrix = np.float64(proof["transform"])
    third = tuple(proof["third"])
    assert projected(a, matrix) == third and native._separated(a, third, 8) and native._separated(b, third, 8)
    x, y, right, bottom = a
    w, h = right - x, bottom - y
    if role in THIRDS:
        u, v, r, d = authored_third
        overlap = max(0, min(third[2], r) - max(third[0], u)) * max(0, min(third[3], d) - max(third[1], v))
        assert overlap >= .90 * (third[2] - third[0]) * (third[3] - third[1])
    source = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.)
    difference = (native.crop(rgb, a) != native.crop(rgb, b)).any(axis=2)
    all_cells = set()
    assert len(proof["witnesses"]) >= max(RULE["minimum_patches"], RULE["minimum_source_fraction"] * proof["source_candidates"])
    for cx, cy, reported in proof["witnesses"]:
        xx, yy = np.meshgrid(np.arange(x+cx-8, x+cx+8), np.arange(y+cy-8, y+cy+8))
        mapped = [np.asarray(matrix[i, 0]*xx + matrix[i, 1]*yy + matrix[i, 2], dtype=np.float32) for i in range(2)]
        p = source[yy, xx].astype(np.float64).ravel()
        q = cv2.remap(source, *mapped, cv2.INTER_LINEAR).astype(np.float64).ravel()
        correlation = float(np.corrcoef(p, q)[0, 1])
        assert 8 <= cx < w-8 and 8 <= cy < h-8 and not difference[cy-8:cy+8, cx-8:cx+8].any()
        assert min(p.std(), q.std()) >= RULE["minimum_sd"] and correlation >= RULE["minimum_correlation"] - .002
        assert abs(reported - correlation) < .002
        all_cells.add((3*cx//w, 3*cy//h))
    assert len(all_cells) >= RULE["minimum_cells"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score", "audit"))
    cli.add_argument("run", type=Path)
    cli.add_argument("--data", type=Path, default=Path("backend/data/independent-evaluation"))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        prior.write(args.run / "freeze.json", contract(args.data))
        print(prior.fs(args.run / "freeze.json"), flush=True)
    else:
        (score if args.stage == "score" else replay)(args.data, args.run)


if __name__ == "__main__":
    main()
