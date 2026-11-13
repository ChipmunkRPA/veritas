# SPDX-License-Identifier: Apache-2.0
"""Evaluate single-upload localization from loss of a recurring signed residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version
from scipy.spatial import cKDTree

from app.detectors.image import jpeg_dq
from scripts import image_reference_delivery_envelope_lama as publisher
from scripts import image_reference_monotone_jpeg_confirmation as source
from scripts import image_residual_encoder_transfer as control
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-residual-template-transfer/1"
CELL = 16
PERIODS = (2, 4, 8)
PEERS = 16
MAX_NEIGHBOURS = 144
MAX_DISTANCE = 1.35
MIN_TEMPLATE_NORM = .10
MIN_TEMPLATE_COSINE = .60
MIN_EXPECTED_PROJECTION = .075
MIN_DISJOINT_CELLS = 5
BOX_SIDES = (2, 3, 4, 5, 6, 8, 10)
ORDINARY = tuple(control.CONTROL)
POSITIVE = ("sharp", "feathered")
ARMS = ("aligned_luma", "template", "luma_or_template", "ungated_template")
DEVELOPMENT_COUNT = 8
HELD_MINIMUM = 16
DESIGN = {
    "inputs": "One decoded JPEG75, 4:2:0. The earlier copy, role and exact changed-pixel mask are never passed to the probe.",
    "cells": CELL,
    "periods": list(PERIODS),
    "denoise_sigma": 1.1,
    "peer_cells_per_fold": PEERS,
    "neighbour_limit": MAX_NEIGHBOURS,
    "content_distance_maximum": MAX_DISTANCE,
    "minimum_chebyshev_separation_in_cells": MIN_DISJOINT_CELLS,
    "minimum_peer_quadrants": 3,
    "minimum_template_norm": MIN_TEMPLATE_NORM,
    "minimum_template_cosine": MIN_TEMPLATE_COSINE,
    "minimum_expected_projection": MIN_EXPECTED_PROJECTION,
    "channels": "Luma, red-minus-luma and blue-minus-luma. Signed full-image phase means normalized by cell residual RMS.",
    "reference": "Fit two distant, alternating spatial folds matched by brightness, coarse texture and chroma. Measure shortfall against the lower cross-fold median projection, with a joint spread floor of 0.08.",
    "region": "At least 65% supported cells, 0.3–12% image area, side ratio at most 3, sides 2/3/4/5/6/8/10 cells. Maximum sum of cell shortfall above 0.35 divided by the square root of the number of cells; unsupported cells contribute zero.",
    "cutoff": "Strictly exceed the second-highest of eight development-source ordinary maxima, with a floor of 0.15. No edit scores used in cutoff selection.",
    "useful": "Native changed-pixel IoU at least 0.30 and recall at least 0.50.",
    "admission": "At least three additional useful deliveries in at least two held photo groups, at most two additional flagged ordinary files across at most one additional group. Further independent replication needed before use.",
    "confidence_weight": 0,
}


def cells(array):
    h, w, channels = array.shape
    return array.reshape(h // CELL, CELL, w // CELL, CELL, channels).transpose(0, 2, 1, 3, 4)


def matched_peers(features, y, x, valid):
    size = len(y)
    indices = np.flatnonzero(valid)
    empty = np.empty((0, PEERS), dtype=np.intp)
    if len(indices) < PEERS * 2:
        return np.array([], dtype=np.intp), empty, empty
    count = min(MAX_NEIGHBOURS, len(indices))
    distances, offsets = cKDTree(features[indices]).query(features, k=count, workers=1)
    nearby = indices[offsets]
    distant = np.maximum(np.abs(y[:, None] - y[nearby]), np.abs(x[:, None] - x[nearby])) >= MIN_DISJOINT_CELLS
    compatible = distant & (distances <= MAX_DISTANCE)
    fold = (y // 4 + x // 4) % 2
    quadrant = 2 * (y >= (1 + y.max()) // 2) + (x >= (1 + x.max()) // 2)
    accepted, first, second = [], [], []
    for index in np.flatnonzero(valid):
        candidates = nearby[index, compatible[index]]
        left = candidates[fold[candidates] == 0][:PEERS]
        right = candidates[fold[candidates] == 1][:PEERS]
        if min(len(left), len(right)) != PEERS:
            continue
        if len(set(quadrant[left]) | set(quadrant[right])) < 3:
            continue
        accepted.append(index)
        first.append(left)
        second.append(right)
    if not accepted:
        return np.array([], dtype=np.intp), empty, empty
    result = np.asarray(accepted), np.asarray(first), np.asarray(second)
    if np.any(result[0] >= size):
        raise ValueError("Invalid peer index")
    return result


def signed_profiles(residual, rms, period):
    rows, cols = rms.shape[:2]
    tiles = cells(residual).reshape(rows, cols, CELL // period, period, CELL // period, period, 3)
    phase = tiles.mean(axis=(2, 4)).reshape(rows * cols, period**2, 3).transpose(0, 2, 1)
    phase -= phase.mean(axis=2, keepdims=True)
    return phase / (np.maximum(rms.reshape(rows * cols, 3, 1), .65) * period)


def shortfall(profiles, accepted, left, right):
    a, b = profiles[left], profiles[right]
    template_a, template_b = a.mean(axis=1), b.mean(axis=1)
    norm_a, norm_b = np.linalg.norm(template_a, axis=2), np.linalg.norm(template_b, axis=2)
    unit_a = template_a / np.maximum(norm_a[..., None], 1e-6)
    unit_b = template_b / np.maximum(norm_b[..., None], 1e-6)
    cosine = np.sum(unit_a * unit_b, axis=2)
    projected_a = np.einsum("nkcf,ncf->nkc", a, unit_b)
    projected_b = np.einsum("nkcf,ncf->nkc", b, unit_a)
    median_a, median_b = np.median(projected_a, axis=1), np.median(projected_b, axis=1)
    spread_a = np.median(np.abs(projected_a - median_a[:, None]), axis=1)
    spread_b = np.median(np.abs(projected_b - median_b[:, None]), axis=1)
    observed = profiles[accepted]
    observed_a = np.einsum("ncf,ncf->nc", observed, unit_a)
    observed_b = np.einsum("ncf,ncf->nc", observed, unit_b)
    expected = np.minimum(median_a, median_b)
    reliable = (np.minimum(norm_a, norm_b) >= MIN_TEMPLATE_NORM) & (cosine >= MIN_TEMPLATE_COSINE)
    reliable &= expected >= MIN_EXPECTED_PROJECTION
    # Require the direction of the shortfall to agree when scoring against either fold.
    deficit = np.minimum(median_a - observed_a, median_b - observed_b)
    scores = deficit / np.maximum(.08, 1.4826 * np.maximum(spread_a, spread_b))
    return np.where(reliable, np.clip(scores, -4, 4), -np.inf)


def missing_phase(rgb):
    height, width = rgb.shape[:2]
    rows, cols = height // CELL, width // CELL
    values = rgb[: rows * CELL, : cols * CELL].astype(np.float32)
    luma = values @ np.array([.299, .587, .114], dtype=np.float32)
    planes = np.stack((luma, values[..., 0] - luma, values[..., 2] - luma), axis=-1)
    smooth = cv2.GaussianBlur(planes, (0, 0), 1.1, borderType=cv2.BORDER_REFLECT)
    residual = planes - smooth
    coarse = cv2.GaussianBlur(smooth[..., 0], (0, 0), 3.)
    gy, gx = np.gradient(coarse)
    stats = cells(planes).mean(axis=(2, 3))
    texture = np.sqrt(cells((gx**2 + gy**2)[..., None]).mean(axis=(2, 3))[..., 0])
    saturation = np.sqrt(np.sum(stats[..., 1:] ** 2, axis=-1))
    features = np.stack((stats[..., 0] / 38, np.log1p(texture) / .55, saturation / 28), axis=-1)
    clipped = cells((np.any((values <= 3) | (values >= 252), axis=-1))[..., None]).mean(axis=(2, 3, 4))
    valid = (clipped < .25) & (stats[..., 0] >= 18) & (stats[..., 0] <= 237)
    y, x = np.indices((rows, cols))
    accepted, left, right = matched_peers(features.reshape(-1, 3), y.ravel(), x.ravel(), valid.ravel())
    score = np.full(rows * cols, -np.inf, dtype=np.float32)
    if len(accepted):
        rms = np.sqrt(cells(residual**2).mean(axis=(2, 3)))
        readings = [shortfall(signed_profiles(residual, rms, p), accepted, left, right) for p in PERIODS]
        score[accepted] = np.max(np.stack(readings), axis=(0, 2))
    return score.reshape(rows, cols)


def window_sums(values, height, width):
    integral = cv2.integral(values, sdepth=cv2.CV_64F)
    return integral[height:, width:] - integral[:-height, width:] - integral[height:, :-width] + integral[:-height, :-width]


def proposal(scores, shape):
    supported = np.isfinite(scores)
    strength = np.where(supported, np.maximum(scores - .35, 0.), 0.)
    result = {"score": 0., "box": None, "supported_cells": int(supported.sum())}
    for height in BOX_SIDES:
        for width in BOX_SIDES:
            area = height * width * CELL**2 / (shape[0] * shape[1])
            if not .003 <= area <= .12 or max(height, width) > 3 * min(height, width):
                continue
            support = window_sums(supported.astype(np.uint8), height, width) >= .65 * height * width
            joint = window_sums(strength, height, width) / np.sqrt(height * width)
            eligible = np.where(support, joint, 0.)
            y, x = np.unravel_index(np.argmax(eligible), eligible.shape)
            if eligible[y, x] > result["score"]:
                result["score"] = float(eligible[y, x])
                result["box"] = [int(v) * CELL for v in (x, y, x + width, y + height)]
    return result


def rows_by_source(args):
    blocked = set()
    for path in args.excluded_plan:
        for item in source.read(path)["cases"]:
            blocked.update(source.identities(item))
    rows = []
    for folder in args.archive:
        path = folder / "qualified-private.json"
        source.require(source.file_sha(path) == source.read(folder / "summary.json")["qualified_private_sha256"],
                       "Qualified roster commitment changed")
        roster = source.read(path)
        rows.extend((row, Path(roster["cache"])) for row in roster["qualified"])
    rows.sort(key=lambda item: source.sha(f"{SCHEMA}/{item[0]['source']['flickr_photo_id']}".encode()))
    accepted = 0
    for row, cache in rows:
        item = row["source"]
        identity = source.identities(item)
        if blocked & identity or not 512 <= min(item["size"]) <= max(item["size"]) <= 768:
            continue
        source.require(item["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                       "Unexpected original photo license")
        original, edited, _ = publisher.source_pixels(cache, row)
        exact = np.any(original != edited, axis=2)
        if not .004 <= float(exact.mean()) <= .08:
            continue
        blocked.update(identity)
        split = "development" if accepted < DEVELOPMENT_COUNT else "held"
        accepted += 1
        yield split, source.sha(str(item["flickr_photo_id"]).encode()), item["recipient"], original, edited, exact
    source.require(accepted >= DEVELOPMENT_COUNT + HELD_MINIMUM, "Too few separate qualified photo groups")


def cases(args):
    for split, group, recipient, original, edited, exact in rows_by_source(args):
        ordinary, positives = control.authored({"original": original, "edited": edited}, exact, recipient)
        variants = ordinary if split == "development" else ordinary | positives
        for role, rgb in variants.items():
            truth = np.any(rgb != original, axis=2) if role in POSITIVE else np.zeros(original.shape[:2], bool)
            body = control.encode(rgb, 75)
            yield {
                "split": split,
                "group_sha256": group,
                "role": role,
                "delivery_sha256": source.sha(body),
                "truth_sha256": source.sha(truth.tobytes()),
                "truth_pixels": int(truth.sum()),
                "dimensions": list(truth.shape),
            }, body, truth


def box_metrics(box, truth):
    selected = np.zeros_like(truth)
    if box:
        x0, y0, x1, y1 = box
        selected[y0:y1, x0:x1] = True
    return mask_metrics(selected, truth)


def readout(body, truth):
    rgb = control.decode(body)
    result = proposal(missing_phase(rgb), rgb.shape)
    comparator = jpeg_dq.analyse(body)
    region = comparator.localised if comparator else None
    aligned_box = [8 * n for n in (region.x, region.y, region.x + region.w, region.y + region.h)] if region else None
    return result | {"aligned_box": aligned_box, "aligned_pixels": box_metrics(aligned_box, truth),
                     "proposal_pixels": box_metrics(result["box"], truth)}


def useful(metrics):
    return metrics["iou"] >= .30 and (metrics["recall"] or 0.) >= .50


def arms(row, point):
    aligned = row["aligned_pixels"]
    ungated = row["proposal_pixels"]
    selected = ungated if row["score"] > point else {"iou": 0., "recall": 0., "predicted_positive_pixels": 0}
    return dict(zip(ARMS, (aligned, selected, aligned if row["aligned_box"] else selected, ungated), strict=True))


def metrics(held, point):
    result = {}
    rows = [(row, arms(row, point)) for row in held]
    for arm in ARMS:
        counts = {}
        ordinary_groups = set()
        for role in ORDINARY + POSITIVE:
            selected = [(row, readings[arm]) for row, readings in rows if row["role"] == role]
            flagged = [(row, reading) for row, reading in selected if reading["predicted_positive_pixels"]]
            groups = {row["group_sha256"] for row, _ in flagged}
            if role in ORDINARY:
                ordinary_groups.update(groups)
            counts[role] = {"files": len(selected), "flagged": len(flagged), "flagged_groups": len(groups),
                            "useful": sum(useful(reading) for _, reading in flagged)}
        result[arm] = {"roles": counts, "ordinary_flagged": sum(counts[r]["flagged"] for r in ORDINARY),
                       "ordinary_groups": len(ordinary_groups)}
    gained = [row for row, reading in rows if row["role"] in POSITIVE
              and useful(reading["luma_or_template"]) and not useful(reading["aligned_luma"])]
    groups = len({row["group_sha256"] for row in gained})
    additional_controls = result["luma_or_template"]["ordinary_flagged"] - result["aligned_luma"]["ordinary_flagged"]
    additional_groups = result["luma_or_template"]["ordinary_groups"] - result["aligned_luma"]["ordinary_groups"]
    result["admission"] = {"additional_useful_files": len(gained), "additional_useful_groups": groups,
                           "additional_ordinary_flags": additional_controls, "additional_ordinary_groups": additional_groups,
                           "passed": len(gained) >= 3 and groups >= 2 and additional_controls <= 2 and additional_groups <= 1}
    return result


def smoke():
    rng = np.random.default_rng(42)
    y, x = np.indices((512, 512))
    underlying = np.rint(123 + rng.normal(size=(512, 512)) * 1.5).astype(np.float32)
    image = np.repeat((underlying + 5 * (1 - 2 * ((x + y) % 2)))[..., None], 3, axis=2)
    image[224:288, 224:288] = underlying[224:288, 224:288, None]
    scores = missing_phase(np.rint(image).astype(np.uint8))
    selection = proposal(scores, image.shape)
    source.require(selection["box"] is not None, "Known deleted phase should have a proposal")
    truth = (x >= 224) & (x < 288) & (y >= 224) & (y < 288)
    source.require(useful(box_metrics(selection["box"], truth)), "Known deleted phase did not localize")


def freeze(args):
    files = (Path(__file__), Path(jpeg_dq.__file__), Path(publisher.__file__), Path(source.__file__), Path(control.__file__))
    return {"schema": SCHEMA, "design": DESIGN, "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version},
            "code_sha256": {path.name: source.file_sha(path) for path in files},
            "qualified_roster_sha256": [source.file_sha(folder / "qualified-private.json") for folder in args.archive],
            "previously_exposed_plan_sha256": [source.file_sha(path) for path in args.excluded_plan],
            "rows": [row for row, _, _ in cases(args)]}


def read_rows(args, lock, split):
    expected = iter(lock["rows"])
    for row, body, truth in cases(args):
        source.require(row == next(expected), "Frozen case differs")
        if row["split"] == split:
            yield row | readout(body, truth)
    source.require(next(expected, None) is None, "Frozen roster no longer complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "freeze", "calibrate", "held", "review"))
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--excluded-plan", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    smoke()
    if args.phase == "smoke":
        return
    source.require(args.output and len(args.archive) == 2 and len(args.excluded_plan) >= 2, "Two qualified archives and earlier exposure ledgers are required")
    lock_path = args.output / "freeze.json"
    lock = freeze(args)
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        source.write(lock_path, lock)
        print(source.file_sha(lock_path), len(lock["rows"]), flush=True)
        return
    source.require(lock == source.read(lock_path), "Frozen inputs, masks or implementation changed")
    development_path = args.output / "development.json"
    held_path = args.output / "held.json"
    if args.phase in {"calibrate", "review"}:
        development = list(read_rows(args, lock, "development"))
        maxima = sorted((max(row["score"] for row in development if row["group_sha256"] == group)
                         for group in {row["group_sha256"] for row in development}), reverse=True)
        source.require(len(maxima) == DEVELOPMENT_COUNT, "Calibration needs eight independent groups")
        calibration = {"freeze_sha256": source.file_sha(lock_path), "point": max(.15, maxima[1]),
                       "ordinary_source_maxima": maxima, "rows": development}
        if args.phase == "calibrate":
            source.write(development_path, calibration)
            print("Point", calibration["point"], flush=True)
            return
        source.require(calibration == source.read(development_path), "Calibration replay differs")
    calibration = source.read(development_path)
    source.require(calibration["freeze_sha256"] == source.file_sha(lock_path), "Calibration belongs to a different freeze")
    held = list(read_rows(args, lock, "held"))
    if args.phase == "held":
        source.write(held_path, {"development_sha256": source.file_sha(development_path), "rows": held})
    else:
        source.require({"development_sha256": source.file_sha(development_path), "rows": held} == source.read(held_path),
                       "Held replay differs")
    summary = metrics(held, calibration["point"])
    if args.phase == "review" and args.receipt:
        source.write(args.receipt, {key: val for key, val in lock.items() if key != "rows"} | {
            "freeze_sha256": source.file_sha(lock_path), "development_sha256": source.file_sha(development_path),
            "held_sha256": source.file_sha(held_path), "exact_replays": len(held) + len(calibration["rows"]),
            "development_groups": DEVELOPMENT_COUNT, "held_groups": len({row["group_sha256"] for row in held}),
            "scoring_point": calibration["point"],
            "held_template_supported_files": sum(row["supported_cells"] > 0 for row in held),
            "held_metrics": summary,
        })
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
