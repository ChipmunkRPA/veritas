# SPDX-License-Identifier: Apache-2.0
"""Test structure-supported earlier-copy JPEG differences on separate photo groups."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts import image_reference_monotone_jpeg_confirmation as prior
from scripts import image_reference_monotone_jpeg_transfer as tone

SCHEMA = "chipmunk.image-reference-structure-jpeg-transfer/1"
ARMS = ("current", "monotone_fallback", "sharpness_fallback", "structure_fallback")
SIGMA = 2.5
WINDOW_SIGMA = 4.5
MIN_GRADIENT = 2.5
MAX_GRADIENT_COSINE = .78
MAX_GRADIENT_RATIO = .35
SUPPORT_RADIUS = 9
MAX_AREA = .10
DESIGN = {
    "point": [SIGMA, WINDOW_SIGMA, MIN_GRADIENT, MAX_GRADIENT_COSINE, MAX_GRADIENT_RATIO, SUPPORT_RADIUS, MAX_AREA],
    "eligibility": "Use only when the current comparison is unassessed and the preceding sharpness gate accepts the pair.",
    "structure": "Require both tone-fold predictions to show localized disagreement in pooled RGB gradient direction or energy. Grow 9 native pixels inside the surviving color mask. Withhold if selected regions exceed 10% of the picture.",
    "admission": "Add two useful sharp and two feathered removals, at most two ordinary delivery highlights, at most one additional ordinary source group, and zero incompatible comparisons.",
    "confidence_weight": 0,
}


def gradients(rgb):
    smooth = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), SIGMA)
    return np.stack([cv2.Sobel(smooth, cv2.CV_32F, dx, dy, ksize=3, scale=.125)
                     for dx, dy in ((1, 0), (0, 1))], axis=3)


def structural_disagreement(predicted_gradients, submitted_gradients, submitted_energy):
    def smooth(matrix):
        return cv2.GaussianBlur(matrix, (0, 0), WINDOW_SIGMA)

    predicted_energy = smooth(np.sum(predicted_gradients**2, axis=(2, 3)))
    dot = smooth(np.sum(predicted_gradients * submitted_gradients, axis=(2, 3)))
    largest_energy = np.maximum(predicted_energy, submitted_energy)
    smallest_energy = np.minimum(predicted_energy, submitted_energy)
    angle_disagrees = dot < MAX_GRADIENT_COSINE * np.sqrt(predicted_energy * submitted_energy)
    strength_disagrees = smallest_energy < MAX_GRADIENT_RATIO**2 * largest_energy
    return (largest_energy >= MIN_GRADIENT**2) & (angle_disagrees | strength_disagrees)


def structure(rgb_a, rgb_b, submitted, curves, color_mask):
    if not color_mask.any():
        return {"status": "comparable", "regions": [], "additional_regions": 0}, color_mask
    observed = gradients(rgb_b)
    energy = cv2.GaussianBlur(np.sum(observed**2, axis=(2, 3)), (0, 0), WINDOW_SIGMA)
    seeds = color_mask.copy()
    for fold in curves:
        corrected = np.stack([np.rint(np.interp(rgb_a[:, :, channel], *zip(*knots, strict=True))).astype(np.uint8)
                              for channel, knots in enumerate(fold)], axis=-1)
        predicted, _ = tone.reference_compare._codec(corrected, submitted)
        seeds &= structural_disagreement(gradients(predicted), observed, energy)
    opened = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    span = 2 * SUPPORT_RADIUS + 1
    expanded = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span))) > 0
    regions, mask, omitted = tone.reference_compare._regions(expanded & color_mask, max(48, round(color_mask.size * .0008)))
    if float(mask.mean()) > MAX_AREA:
        return {"status": "unassessed_broad_structure_change", "regions": []}, None
    return {"status": "comparable", "regions": regions, "additional_regions": omitted}, mask


def source_groups(args, stage):
    development = prior.sources(args)
    if stage == "development":
        return development
    excluded = {photo for photo, *_ in development}
    documents = [args.source_roster, args.backup_roster]
    for directory in (args.later_f5, args.external):
        excluded.update(tone.sha(str(item["flickr_photo_id"]).encode()) for item in tone.read(directory / "plan.json")["cases"])
    keys = set()
    rows = []
    for directory in documents:
        path = directory / "qualified-private.json"
        require_digest = tone.read(directory / "summary.json")["qualified_private_sha256"]
        tone.require(tone.file_sha(path) == require_digest, "Qualified source roster changed")
        roster = tone.read(path)
        for row in roster["qualified"]:
            photo = tone.sha(str(row["source"]["flickr_photo_id"]).encode())
            if photo in excluded:
                keys.update(prior.identities(row["source"]))
        rows.extend((row, Path(roster["cache"])) for row in roster["qualified"])
    for directory in (args.later_f5, args.external):
        for item in tone.read(directory / "plan.json")["cases"]:
            keys.update(prior.identities(item))
    selected = []
    rows.sort(key=lambda pair: tone.sha(f"{SCHEMA}/{pair[0]['source']['flickr_photo_id']}".encode()))
    for row, cache in rows:
        item = row["source"]
        identity = prior.identities(item)
        if keys & identity or not 512 <= min(item["size"]) <= max(item["size"]) <= 768:
            continue
        tone.require(item["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                     "Unexpected photo licence")
        original, edited, _ = prior.intake.source_pixels(cache, row)
        exact = np.any(original != edited, axis=2)
        if not .004 <= float(exact.mean()) <= .08:
            continue
        ordinary, edits = tone.panel.source_panel.common.authored({"original": original, "edited": edited}, exact, item["recipient"])
        selected.append((tone.sha(str(item["flickr_photo_id"]).encode()), ordinary, edits))
        keys.update(identity)
    tone.require(len(selected) >= 16, "Too few separate confirmation photos")
    return selected


def cases(args, stage):
    if stage == "development":
        yield from prior.cases(args)
        return
    bank = source_groups(args, stage)
    encode, decode = tone.panel.source_panel.common.encode, tone.panel.source_panel.common.decode
    for index, (source, ordinary, edited) in enumerate(bank):
        original = ordinary["original"]
        quality = [tone.SOURCE_QUALITIES[index % 3], tone.TARGET_QUALITIES[index % 2]]
        grade_index = (index + 1) % 3
        graded = tone.grade(original, grade_index)
        before = encode(original, quality[0])
        reduced = cv2.resize(original, None, fx=.81, fy=.81, interpolation=cv2.INTER_AREA)
        resized = cv2.resize(reduced, original.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        native_image = Image.fromarray(original)
        lanczos = native_image.resize(tuple(round(side * .74) for side in native_image.size), Image.Resampling.LANCZOS)
        lanczos = np.asarray(lanczos.resize(native_image.size, Image.Resampling.LANCZOS))
        other = cv2.resize(bank[(index + 1) % len(bank)][1]["original"], original.shape[1::-1])
        variants = {"no_tone": original, "ordinary_roundtrip": decode(encode(original, 80)), "ordinary_graded": graded,
                    "resize_roundtrip": tone.grade(resized, grade_index), "resize_lanczos": tone.grade(lanczos, grade_index),
                    "global_blur": cv2.GaussianBlur(graded, (0, 0), .95), "fractional": tone.motion(graded, 1.65),
                    "translated": tone.motion(graded, 8.), "unrelated": tone.grade(other, grade_index)}
        variants.update({role: tone.grade(rgb, grade_index) for role, rgb in (ordinary | edited).items() if role in tone.LOCAL})
        for role, rgb in variants.items():
            truth = np.any(rgb != graded, axis=2) if role in tone.LOCAL else np.zeros(original.shape[:2], bool)
            after = encode(rgb, quality[1])
            yield {"source": source, "role": role, "dimensions": list(truth.shape), "quality": quality,
                   "before_sha256": tone.sha(before), "after_sha256": tone.sha(after), "truth_sha256": tone.sha(truth.tobytes())}, before, after, truth


def assess(inputs, expected):
    for (row, before, after, truth), witness in zip(inputs, expected, strict=True):
        tone.require(row == witness, "Frozen delivery or exact mask differs")
        with Image.open(io.BytesIO(before)) as earlier, Image.open(io.BytesIO(after)) as submitted:
            rgb_a, rgb_b = np.asarray(earlier), np.asarray(submitted)
            current = tone.reference_alignment.compare(rgb_a, rgb_b, submitted)
            fallback = guarded = candidate = current
            if current[0]["status"] != "comparable":
                fallback = guarded = candidate = tone.compare(before, after, earlier, submitted, rgb_a, rgb_b)
                record, mask = fallback
                if record["status"] == "comparable":
                    scope, _ = prior.scope_check(rgb_a, rgb_b, submitted, record["tone_knots"])
                    if scope != "comparable":
                        guarded = candidate = ({"status": scope}, None)
                    else:
                        candidate = structure(rgb_a, rgb_b, submitted, record["tone_knots"], mask)
        results = {}
        for arm, (record, selection) in zip(ARMS, (current, fallback, guarded, candidate), strict=True):
            mask = selection if selection is not None else np.zeros_like(truth)
            results[arm] = {"status": record["status"], "mask_sha256": tone.sha(mask.tobytes()),
                            "pixels": tone.mask_metrics(mask, truth)}
        yield row | {"arms": results}


def summary(rows):
    data = []
    for row in rows:
        current, fallback, guarded, candidate = (row["arms"][arm] for arm in ARMS)
        data.append(row | {"arms": dict(zip(prior.ARMS, (current, fallback, guarded, candidate), strict=True))})
    result = prior.summary(data)
    return {key: dict(zip(ARMS, (section[arm] for arm in prior.ARMS), strict=True)) for key, section in result.items()}


def freeze(args):
    paths = [args.source_roster / "qualified-private.json", args.backup_roster / "qualified-private.json",
             args.later_f5 / "plan.json", args.external / "plan.json"]
    return {"schema": SCHEMA, "design": DESIGN, "stage": args.stage,
            "code": {Path(module.__file__).name: tone.file_sha(Path(module.__file__)) for module in (prior, tone, tone.reference_alignment, tone.reference_compare)},
            "runner": tone.file_sha(Path(__file__)), "inputs": [tone.file_sha(path) for path in paths],
            "rows": [row for row, *_ in cases(args, args.stage)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("stage", choices=("development", "held"))
    for name in ("source-roster", "backup-roster", "later-f5", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    profile = freeze(args)
    lock = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        tone.write(lock, profile)
        print(tone.file_sha(lock), len(profile["rows"]), flush=True)
        return
    tone.require(tone.read(lock) == profile, "Frozen sources or comparison code changed")
    rows = list(assess(cases(args, args.stage), profile["rows"]))
    result = summary(rows)
    if args.phase == "run":
        tone.write(args.output / "rows.json", rows)
    else:
        tone.require(rows == tone.read(args.output / "rows.json"), "Independent exact JPEG and mask replay differs")
        if args.publication:
            tone.write(args.publication, {key: value for key, value in profile.items() if key != "rows"}
                       | {"comparisons": len(rows), "sources": len({row["source"] for row in rows}),
                          "freeze_sha256": tone.file_sha(lock), "rows_sha256": tone.file_sha(args.output / "rows.json"),
                          "exact_replays": len(rows), **result})
    print(json.dumps(result["paired"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
