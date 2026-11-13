"""Recompute the source-bound quantized image observation on a third camera panel."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import native_cfa, native_cfa_quantized, native_cfa_rotation
from scripts import image_native_stencil_interpolation_transfer as source
from scripts.image_quantized_phase_transfer import read, sha, write

ROLES = {"untouched", "quantized_bilinear", "rotated", "generated_sharp", "generated_feathered", "bilinear_diagonal",
         "bicubic_diagonal", "local_tone", "blur"}


def run(cached, archive, local, reference):
    sealed = source.lock(local)
    witness = {(row["key"], row["role"]): row for row in reference["rows"] if row["role"] in ROLES}
    rows = []
    for (case, rgb), pinned in zip(source.sources(cached, archive), sealed["rows"], strict=True):
        if (any(pinned[key] != value for key, value in case.items()) or source.digest(rgb) != pinned["source_rgb_sha256"]):
            raise ValueError("Camera RAW render differs")
        path = local / "generated" / f"{case['key']}.png"
        generated = read(path.with_suffix(".json"))
        if (generated["generated_sha256"], generated["reference_sha256"], generated["freeze_sha256"]) != (
                sha(path), pinned["reference_png_sha256"], reference["freeze_sha256"]):
            raise ValueError("Generated patch or retained commitment differs")
        with Image.open(path, formats=["PNG"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        for role, body, mask in source.variants(rgb, case, pixels):
            if role not in ROLES:
                continue
            previous = witness.pop((case["key"], role))
            target = int(mask.sum())
            if (source.digest(body), source.digest(mask), target) != (
                    previous["input_sha256"], previous["truth_sha256"], previous["truth_pixels"]):
                raise ValueError("Input RGB bytes or exact edit mask differs")
            initial = native_cfa.measure(body)
            if initial.region is not None:
                status, box, reading = "higher_priority", None, None
            elif initial.status != "assessed":
                raise ValueError("Camera render is outside the intended serving scope")
            else:
                status, box, reading = native_cfa_quantized.inspect(body)
            overlap = 0
            if box:
                x0, y0, x1, y1 = box
                overlap = int(mask[y0:y1, x0:x1].sum())
            rows.append({"key": case["key"], "role": role, "input_sha256": previous["input_sha256"],
                         "truth_sha256": previous["truth_sha256"], "truth_pixels": target, "intersection": overlap,
                         "status": status, "box": box, "reading": list(reading) if reading else None})
    if witness:
        raise ValueError("Some expected serving inputs were not assessed")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cached", "archive", "retained", "comparison", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--expect", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    native_lock = args.retained / "freeze.json"
    reference = source.ROSTER.with_name("image-native-stencil-interpolation-measurements-2026-11-13.json")
    prior = read(reference)
    if sha(native_lock) != prior["freeze_sha256"]:
        raise ValueError("Retained source freeze differs")
    rows = run(args.cached, args.archive, args.retained, prior)
    result = {"schema": "chipmunk.quantized-serving-transfer/1", "rows": rows, "base_sha256": sha(args.comparison),
              "source_roster_sha256": sha(source.ROSTER), "source_measurements_sha256": sha(reference),
              "serving_sha256": {module.__name__: sha(Path(module.__file__))
                                 for module in (native_cfa, native_cfa_rotation, native_cfa_quantized)},
              "counts": {role: Counter(row["status"] for row in rows if row["role"] == role) for role in sorted(ROLES)}}
    if args.expect is not None:
        earlier = read(args.expect)
        if result != {key: value for key, value in earlier.items() if key != "native_replay"}:
            raise ValueError("The native serving replay changed")
        result["native_replay"] = True
    write(args.output, result)
    print(len(rows), result["counts"])


if __name__ == "__main__":
    main()
