# SPDX-License-Identifier: Apache-2.0
"""Measure whether a standalone local-rotation observation transfers across cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image import native_cfa
from scripts import image_native_generated_grid_transfer as grid
from scripts import image_native_rotated_stencil_transfer as rotation
from scripts import image_native_stencil_interpolation_transfer as source
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.native-rotation-location/1"
SUPPORTED = (1536, 2048, 3)
POSITIVE = {"bilinear_rotation", "bicubic_rotation", "feathered_rotation"}
ROLES = POSITIVE | {"untouched", "full_rotation", "translation", "quantized_translation", "blur", "tone",
                    "generated_sharp", "generated_feathered", "resized", "eight_bit"}
PLAN = {
    "sources": "All 13 RAW camera models in the prior stencil interpolation transfer. These cameras were not in the earlier rotated-carrier development or confirmation cohorts. Exact generated controls reuse existing outputs and positions.",
    "positives": "Two new signed small local rotations and one feathered small rotation per camera. An exact change mask comes from differences against the original after rounding.",
    "controls": "Untouched, full-image rotation, conventional and 8-bit-quantized translations, blur, local tone, two prior generated replacements, whole-image resize, and an 8-bit view. No new model or synthesis.",
    "method": "Four frozen grid positions for texture-supported stencil loss. Reuse the original parity check and report a local rotation clue only when the locked spectral rule supports the same region.",
    "gate": "At least 10 useful positions across the 26 sharp rotations and at least four useful across 13 feathered rotations. At most one flag across all ordinary controls, none on 26 generated alternatives or 13 originals. All unsupported 8-bit files decline. The output is geometric evidence only.",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def inspect(rgb):
    if rgb.dtype != np.uint16 or rgb.shape != SUPPORTED:
        return {"status": "unsupported", "box": None, "readout": None}
    h, w = rgb.shape[:2]
    choices = []
    for dy, dx in grid.GRIDS:
        body = rgb[dy:h, dx:w]
        rates = native_cfa.stencil(body[:body.shape[0] // 32 * 32, :body.shape[1] // 32 * 32])
        learned = native_cfa.learn(rates)
        if learned is None:
            continue
        phase, expected, eligible = learned
        body = body[:rates.shape[0] * 32, :rates.shape[1] * 32]
        luma = body.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
        fine = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), .85))
        energy = (native_cfa.cell_sums(fine) / 1024).astype(np.float32)
        nearby = cv2.GaussianBlur(cv2.medianBlur(energy, 3), (9, 9), 2.)
        proposal = native_cfa.best_region(np.where(energy >= np.maximum(nearby * .50, 1.),
                                                  expected - rates[..., phase] - .12, 0.), eligible)
        if proposal["box"]:
            x, y, x1, y1 = proposal["box"]
            choices.append((proposal["score"], [x + dx, y + dy, x1 + dx, y1 + dy], phase))
    if not choices:
        return {"status": "no_supported_region", "box": None, "readout": None}
    strength, box, phase = max(choices, key=lambda item: item[0])
    if strength <= grid.prior.POINT:
        return {"status": "no_supported_region", "box": None, "readout": None}
    x, y, x1, y1 = box
    if max(map(min, grid.prior.interpolation_fit(rgb[y+8:y1-8, x+8:x1-8], phase))) > grid.prior.VETO:
        return {"status": "ordinary_axis_interpolation", "box": None, "readout": None}
    detail = rotation.rotated_pattern(rgb[y+6:y1-6, x+6:x1-6])
    return {"status": "local_rotation" if detail["selected"] else "no_rotation_support",
            "box": box if detail["selected"] else None, "readout": detail}


def fixtures(rgb, binding, generated, index):
    for role, pixels, truth in source.variants(rgb, binding, generated):
        if role in {"untouched", "generated_sharp", "generated_feathered"}:
            yield role, pixels, truth
    h, w = rgb.shape[:2]
    x, y = (coordinate + (source.PATCH - source.WIDTH) // 2 for coordinate in binding["crop"][:2])
    n = source.WIDTH
    roi = np.s_[y:y+n, x:x+n]
    patch = rgb[roi]
    empty = np.zeros((h, w), bool)
    angle = (1 if index % 2 else -1) * (2.1 + .29 * (index % 12))
    smooth = cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1.)
    other = cv2.getRotationMatrix2D((n / 2, n / 2), -angle - .27, 1.)
    shift = np.float32([[1, 0, .41], [0, 1, -.28]])

    def warp(body, matrix, mode):
        return cv2.warpAffine(body, matrix, (n, n), flags=mode, borderMode=cv2.BORDER_REFLECT_101)

    bilinear = warp(patch, smooth, cv2.INTER_LINEAR)
    bicubic = warp(patch, other, cv2.INTER_CUBIC)
    yy, xx = np.indices((n, n))
    edge = np.minimum.reduce((yy + 1, xx + 1, n - yy, n - xx))
    alpha = ((1 - np.cos(np.pi * np.minimum(1., edge / 9.))) / 2.)[..., None]
    fields = {
        "bilinear_rotation": bilinear,
        "bicubic_rotation": bicubic,
        "feathered_rotation": np.rint(patch * (1 - alpha) + bicubic * alpha).astype(np.uint16),
        "translation": warp(patch, shift, cv2.INTER_CUBIC),
        "quantized_translation": warp((patch >> 8).astype(np.uint8), shift, cv2.INTER_LINEAR).astype(np.uint16) * 257,
        "blur": cv2.GaussianBlur(patch, (7, 7), 1.3),
        "tone": np.rint(patch * .87 + 1100).astype(np.uint16),
    }
    for role, changed in fields.items():
        output = rgb.copy()
        output[roi] = changed
        truth = empty.copy()
        truth[roi] = np.any(changed != patch, axis=2)
        yield role, output, truth
    whole = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.)
    yield "full_rotation", cv2.warpAffine(rgb, whole, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101), empty
    down = cv2.resize(rgb, (w * 7 // 8, h * 7 // 8), interpolation=cv2.INTER_AREA)
    yield "resized", cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC), empty
    yield "eight_bit", (rgb >> 8).astype(np.uint8), empty


def score(cached, archive, previous, *, inspector=inspect):
    from PIL import Image

    frozen = source.lock(previous)
    output = []
    for i, ((binding, rgb), witness) in enumerate(zip(source.sources(cached, archive), frozen["rows"], strict=True)):
        if source.digest(rgb) != witness["source_rgb_sha256"] or binding != {k: witness[k] for k in binding}:
            raise ValueError("Native pixels differ from the pinned capture")
        file = previous / "generated" / f"{binding['key']}.png"
        generated = source.earlier.read(file.with_suffix(".json"))
        if (generated["generated_sha256"] != sha(file) or generated["reference_sha256"] != witness["reference_png_sha256"]
                or generated["freeze_sha256"] != sha(previous / "freeze.json")):
            raise ValueError("Generated control differs from the retained source")
        with Image.open(file, formats=["PNG"]) as image:
            pictures = np.asarray(image.convert("RGB"))
        for role, pixels, truth in fixtures(rgb, binding, pictures, i):
            record = inspector(pixels)
            chosen = np.zeros_like(truth)
            if record["box"]:
                x, y, x1, y1 = record["box"]
                chosen[y:y1, x:x1] = True
            output.append({"camera": binding["key"], "role": role, "input_sha256": source.digest(pixels),
                           "truth_sha256": source.digest(truth), "overlap": mask_metrics(chosen, truth), **record})
        print("camera", i + 1, len(frozen["rows"]), flush=True)
    return output


def summarize(rows):
    counts = {role: Counter(files=0, flagged=0, useful=0, assessed=0) for role in sorted(ROLES)}
    for row in rows:
        overlap = row["overlap"]
        positive = row["role"] in POSITIVE
        flag = row["status"] == "local_rotation"
        counts[row["role"]].update(files=1, flagged=flag,
                                   useful=positive and flag and overlap["iou"] >= .3 and overlap["recall"] >= .5,
                                   assessed=row["status"] not in ("unsupported", "no_supported_region"))
    sharp = sum(counts[k]["useful"] for k in POSITIVE - {"feathered_rotation"})
    advances = (sharp >= 10 and counts["feathered_rotation"]["useful"] >= 4
                and sum(counts[r]["flagged"] for r in ROLES - POSITIVE) <= 1
                and not any(counts[r]["flagged"] for r in ("untouched", "generated_sharp", "generated_feathered"))
                and counts["eight_bit"]["assessed"] == 0)
    return {"counts": counts, "passes": advances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "serving-parity"))
    parser.add_argument("run", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("cached", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "serving-parity":
        from app.detectors.image import native_cfa_rotation

        previous = json.loads((args.run / "result.json").read_bytes())
        rows = score(args.cached, args.archive, args.previous, inspector=native_cfa_rotation.inspect)
        if previous["rows"] != rows:
            raise ValueError("Serving detector differs from the frozen rotation trial")
        print(json.dumps({"serving_parity_cases": len(rows), **summarize(rows)}, sort_keys=True))
        return
    binding = {"schema": SCHEMA, "plan": PLAN, "roster_sha256": sha(source.ROSTER),
               "previous_sha256": sha(args.previous / "freeze.json"),
               "code": {Path(p).name: sha(Path(p)) for p in (
                   __file__, grid.__file__, source.__file__, rotation.__file__, native_cfa.__file__,
               )}}
    lock = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(lock, binding)
        return
    if json.loads(lock.read_bytes()) != binding:
        raise ValueError("Frozen inputs or code differ")
    rows = score(args.cached, args.archive, args.previous)
    result = {"freeze": binding, "rows": rows, **summarize(rows)}
    target = args.run / "result.json"
    if args.phase == "score":
        write(target, result)
    elif json.loads(target.read_bytes()) != result:
        raise ValueError("Native replay differs")
    print(json.dumps(summarize(rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
