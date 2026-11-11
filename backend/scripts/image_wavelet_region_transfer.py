"""Compare local multi-scale detail contrast on source-matched image edits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-wavelet-region-transfer/1"
CELL = 16
RING = 3
SEPARATION = 1
SCALES = (1 / 6, 1 / 4, 1 / 3)
FEATURES = ("luma_fine_mid", "luma_mid_coarse", "luma_fine_diagonal", "luma_mid_diagonal",
            "chroma_fine_mid", "chroma_mid_coarse")
ARMS = ("luma_detail_deficit", "fitted_wavelet")
ROLES = ("edited", "feathered_edit")
sha, write = shared.file_digest, shared.write


def block_features(rgb: np.ndarray) -> np.ndarray:
    height, width = (size // CELL * CELL for size in rgb.shape[:2])
    data = cv2.cvtColor(rgb[:height, :width].astype(np.float32) / 255, cv2.COLOR_RGB2YCrCb) * 255
    details = []
    diagonals = []
    for level in (1, 2, 3):
        top_left, top_right = data[::2, ::2], data[::2, 1::2]
        bottom_left, bottom_right = data[1::2, ::2], data[1::2, 1::2]
        horizontal = np.abs((top_left - top_right) + (bottom_left - bottom_right)) * .25
        vertical = np.abs((top_left - bottom_left) + (top_right - bottom_right)) * .25
        diagonal = np.abs((top_left - top_right) - (bottom_left - bottom_right)) * .25
        side = CELL // (2 ** level)
        layout = (height // CELL, side, width // CELL, side, 3)
        oriented = ((horizontal + vertical) * .5).reshape(layout).mean(axis=(1, 3))
        corner = diagonal.reshape(layout).mean(axis=(1, 3))
        details.append(np.log(.5 + (oriented + corner) * .5))
        diagonals.append(np.log(.5 + corner[:, :, 0]) - np.log(.5 + oriented[:, :, 0]))
        data = (top_left + top_right + bottom_left + bottom_right) * .25
    fine_mid, mid_coarse = details[0] - details[1], details[1] - details[2]
    return np.stack((fine_mid[:, :, 0], mid_coarse[:, :, 0], diagonals[0], diagonals[1],
                     fine_mid[:, :, 1:].mean(axis=2), mid_coarse[:, :, 1:].mean(axis=2)), axis=2)


def regions(shape: tuple[int, int]) -> np.ndarray:
    rows, columns = shape
    candidates = []
    for width in sorted({max(3, round(columns * scale)) for scale in SCALES}):
        for height in sorted({max(3, round(rows * scale)) for scale in SCALES}):
            left, top = np.meshgrid(np.arange(RING, columns - RING - width + 1),
                                    np.arange(RING, rows - RING - height + 1))
            candidates.append(np.stack((left, top, left + width, top + height), axis=-1).reshape(-1, 4))
    return np.concatenate(candidates)


def sums(integral: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = boxes.T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def contrast(pixels: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    features = block_features(pixels)
    integral = np.pad(features.astype(np.float64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
    middle = boxes + np.array([1, 1, -1, -1])
    separated = boxes + np.array([-SEPARATION, -SEPARATION, SEPARATION, SEPARATION])
    outer = boxes + np.array([-RING, -RING, RING, RING])
    extent = boxes[:, 2:] - boxes[:, :2]
    inner_area = np.prod(extent - 2, axis=1, keepdims=True)
    ring_area = (np.prod(extent + 2 * RING, axis=1, keepdims=True)
                 - np.prod(extent + 2 * SEPARATION, axis=1, keepdims=True))
    background = (sums(integral, outer) - sums(integral, separated)) / ring_area
    return sums(integral, middle) / inner_area - background


def check_case(truth: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    integral = np.pad(truth.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    pixels = boxes * CELL
    overlap = sums(integral, pixels)
    union = np.prod(pixels[:, 2:] - pixels[:, :2], axis=1) + truth.sum() - overlap
    return boxes[[int((overlap / union).argmax())]]


def view_controls(pixels, processed, box):
    photo = Image.fromarray(pixels)
    return (pixels, processed, blend.local(pixels, processed, box, feather=True),
            blend.local(pixels, np.asarray(photo.filter(ImageFilter.GaussianBlur(1.25))), box, feather=True),
            blend.local(pixels, np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), box, feather=True),
            blend.local(pixels, np.asarray(ImageEnhance.Sharpness(photo).enhance(1.5)), box, feather=True))


def fit(folder, fmt):
    _, plan, attempts = scoring.checked_inputs(folder)
    differences = []
    for _, _, views, truth in scoring.panels(folder, plan, attempts):
        source = views["original"]
        eligible = regions(tuple(size // CELL for size in truth.shape))
        box = check_case(truth, eligible)
        bounds = tuple(int(value) for value in box[0] * CELL)
        controls = view_controls(source, views["processed"], bounds)
        positive = (views["edited"], blend.local(source, views["edited"], bounds, feather=True))
        values = [contrast(np.asarray(paired.encoded(image, fmt)[1]), box)[0] for image in (*positive, *controls)]
        differences.append((np.array(values[:2])[:, None] - np.array(values[2:])).reshape(-1, len(FEATURES)))
    assert len(differences) == 8
    differences = np.concatenate(differences)
    scale = np.maximum(np.sqrt(np.mean(differences ** 2, axis=0)), .01)
    standardized = differences / scale
    covariance = standardized.T @ standardized / len(standardized)
    weights = np.linalg.solve(covariance + np.eye(len(FEATURES)), standardized.mean(axis=0)) / scale
    return {"weights": weights.tolist(), "pairs": len(differences), "source_groups": 8}


def rank(image, weights):
    pixels = np.asarray(image)
    boxes = regions(tuple(size // CELL for size in pixels.shape[:2]))
    signals = contrast(pixels, boxes)
    values = (-signals[:, 0], signals @ weights)
    output = {}
    for arm, response in zip(ARMS, values, strict=True):
        index = int(response.argmax())
        output[arm] = {"score": float(response[index]), "box": (boxes[index] * CELL).tolist()}
    return output


def control_points(folder, fmt, weights):
    _, plan, attempts = scoring.checked_inputs(folder)
    maxima = {arm: [] for arm in ARMS}
    for _, _, views, _ in scoring.panels(folder, plan, attempts):
        source = views["original"]
        h, w = source.shape[:2]
        controls = view_controls(source, views["processed"], (w // 3, h // 3, w * 7 // 12, h * 7 // 12))
        measurements = [rank(paired.encoded(image, fmt)[1], weights) for image in controls]
        for arm in ARMS:
            maxima[arm].append(max(m[arm]["score"] for m in measurements))
    assert all(len(group) == 8 for group in maxima.values())
    return {arm: {"point": sorted(values)[-2], "group_maxima": values} for arm, values in maxima.items()}


def confirm(intake, frozen_rows, fitted):
    binding = {(row["slot"], row["format"], row["role"]): row for row in frozen_rows}
    for row, native, _ in paired.inputs(intake):
        source, box = native["original"], row["box"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(source)).enhance(1.2))
        views = native | {"feathered_edit": blend.local(source, native["edited"], box, feather=True),
                          "feathered_processing": blend.local(source, native["local_processing"], box, feather=True),
                          "local_contrast": blend.local(source, tone, box)}
        for role, pixels in views.items():
            truth = np.any(pixels != source, axis=2) if role in ROLES else np.zeros(source.shape[:2], dtype=bool)
            for fmt in paired.FORMATS:
                body, image = paired.encoded(pixels, fmt)
                receipt = binding.pop((row["slot"], fmt, role))
                assert shared.digest(body) == receipt["input_sha256"] and row["generator"] == receipt["generator"]
                assert int(truth.sum()) == receipt["measurements"]["boundary_prominence"]["target"]
                selected = rank(image, np.array(fitted[fmt]["fit"]["weights"]))
                measurements = {arm: scoring.measure(paired.boxed(result["box"], truth.shape,
                                result["score"] > fitted[fmt]["controls"][arm]["point"]), truth)
                                for arm, result in selected.items()}
                yield {**{key: receipt[key] for key in ("slot", "generator", "format", "role", "input_sha256")},
                       "measurements": measurements, "proposals": selected}
    assert not binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("fit", "controls", "intake", "paired_receipt", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    sources = [{case["flickr_photo_id"] for case in json.loads((folder / "plan.json").read_bytes())["cases"]}
               for folder in (args.fit, args.controls, args.intake)]
    assert [len(items) for items in sources] == [8, 8, 12] and len(set.union(*sources)) == 28
    paired_file = args.paired_receipt / "rows.json"
    assert sha(paired_file) == json.loads((args.paired_receipt / "result.json").read_bytes())["journal_sha256"]
    frozen = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)), "fit_sha256": sha(args.fit / "result.json"),
              "controls_sha256": sha(args.controls / "result.json"), "intake_sha256": sha(args.intake / "result.json"),
              "paired_rows_sha256": sha(paired_file), "composition_sha256": sha(Path(blend.__file__)),
              "cell": CELL, "surrounding_ring_extent": RING, "separation": SEPARATION, "scales": SCALES,
              "features": FEATURES, "arms": ARMS, "ridge": 1.0, "point": "strictly above second largest group maximum"}
    frozen = json.loads(json.dumps(frozen))
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", frozen)
        return
    assert frozen == json.loads((args.output / "freeze.json").read_bytes())
    cv2.setNumThreads(1)
    started = time.monotonic()
    fitted = {}
    for fmt in paired.FORMATS:
        head = fit(args.fit, fmt)
        fitted[fmt] = {"fit": head, "controls": control_points(args.controls, fmt, np.array(head["weights"]))}
    write(args.output / "points.json", fitted)
    print("Both format points sealed before paired scoring", flush=True)
    rows = list(confirm(args.intake, json.loads(paired_file.read_bytes()), fitted))
    assert len(rows) == 192
    write(args.output / "rows.json", rows)
    feathered = [{**row, "delivery": row["format"], "role": "edited"} for row in rows if row["role"] == "feathered_edit"]
    write(args.output / "result.json", {"schema": SCHEMA, "paired_views": len(rows), "elapsed_s": round(time.monotonic() - started, 3),
          "freeze_sha256": sha(args.output / "freeze.json"), "points_sha256": sha(args.output / "points.json"),
          "journal_sha256": sha(args.output / "rows.json"), "sharp_and_controls": paired.summarise([r for r in rows if r["role"] != "feathered_edit"]),
          "feathered": scoring.summarize(feathered)})


if __name__ == "__main__":
    main()
