# SPDX-License-Identifier: Apache-2.0
"""Prospective transfer of the frozen native PNG phase-shift detector across geometry."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image import native_cfa
from scripts import image_native_cfa_png_transfer as earlier
from scripts.localization_benchmark import mask_metrics

ROOT, prior = earlier.prior.ROOT, earlier.prior
ROSTER = ROOT / "docs/evaluation-panels/image-native-cfa-dimensions-cc0-roster-2026-11-13.json"
SCHEMA = "chipmunk.image-native-cfa-dimensions-transfer/1"
ANCHOR = (1536, 2048)
GEOMETRIES = {
    "current_landscape": (1536, 2048),
    "portrait": (2048, 1536),
    "large_landscape": (1728, 2304),
    "wide": (1344, 2368),
    "square": (1792, 1792),
}
ROLES = ("original", "even_copy", "global_crop", "local_fractional", "resized", "phase_shift", "phase_shift_feather")
POSITIVE = ("phase_shift", "phase_shift_feather")
DESIGN = {
    "scope": "A bare PNG in one of five stated geometries, each side divisible by 32, 16-bit truecolor RGB. No metadata, rotation tags, ancillary chunks or resampling. Portrait pixels in fixtures use an exact quarter-turn, so native lattice remains intact and submitted image is already upright.",
    "frozen_method": "The same 32-pixel cells, parity support, scoring point, rectangle shapes and phase-shift-only candidate as the existing app. No rescale, no padding and no geometry-dependent point.",
    "sources": "Ten new checksum-bound CC0 camera models outside all prior native transfer banks. One deterministic crop per input geometry; no measurements used to choose sources or destinations.",
    "controls": "Untouched, ordinary even-phase copy, odd whole-image crop, local fractional interpolation and down/up-resampled views.",
    "admission": "Per geometry at least 7/10 useful sharp and 7/10 useful feathered shifts. Zero highlights in ordinary views and all ten resampled files decline. Advance each geometry separately.",
    "useful": "Native RGB mask IoU at least 0.30, recall at least 0.50.",
    "confidence_weight": 0,
}


def identity(path):
    return prior.sha(path)


def native_crop(decoder, path, scope, *, odd=False):
    height, width = scope
    with decoder.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        prior.raw_source.require(pattern is not None and pattern.shape == (2, 2)
                                 and sorted(raw.color_desc[index] for index in pattern.flat) == sorted(b"RGGB"),
                                 "Expected genuine Bayer capture")
        source_h, source_w = raw.sizes.height, raw.sizes.width
        rgb = raw.postprocess(demosaic_algorithm=decoder.DemosaicAlgorithm.LINEAR, use_camera_wb=False, use_auto_wb=False,
                              user_wb=[1., 1., 1., 1.], output_color=decoder.ColorSpace.raw, output_bps=16, gamma=(1., 1.),
                              no_auto_bright=True, user_flip=0, fbdd_noise_reduction=decoder.FBDDNoiseReductionMode.Off)
    prior.raw_source.require(rgb.dtype == np.uint16 and rgb.shape == (source_h, source_w, 3), "Native dimensions changed")
    if height > width and source_h < source_w:
        rgb = np.rot90(rgb)
    y = (rgb.shape[0] - height) // 4 * 2 + odd
    x = (rgb.shape[1] - width) // 4 * 2 + odd
    prior.raw_source.require(y >= 0 and x >= 0 and y + height <= rgb.shape[0] and x + width <= rgb.shape[1],
                             "Native source too small for geometry")
    return np.ascontiguousarray(rgb[y:y + height, x:x + width])


def _allowed(body, shape):
    """Apply the product PNG limits and chunk checks with only the IHDR geometry generalized."""
    if not 33 <= len(body) <= native_cfa.MAX_BODY_BYTES:
        return False
    h, w = shape
    if body[:24] != native_cfa.PNG + struct.pack(">I4sII", 13, b"IHDR", w, h):
        return False
    # Substitute the dimensions and IHDR checksum, only for identical metadata/CRC parser.
    # The real encoded data is always decoded from its original unmodified body below.
    if earlier.zlib.crc32(body[12:29]) != struct.unpack_from(">I", body, 29)[0]:
        return False
    expected_h, expected_w = ANCHOR
    header = body[12:16] + struct.pack(">II", expected_w, expected_h) + body[24:29]
    aligned = body[:12] + header + struct.pack(">I", earlier.zlib.crc32(header)) + body[33:]
    return native_cfa.allowed_png(aligned)


def decode(body, shape):
    if not _allowed(body, shape):
        return None
    data = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None or data.dtype != np.uint16 or data.shape != (*shape, 3):
        return None
    return np.ascontiguousarray(data[..., ::-1])


def measure(pixels):
    rates = native_cfa.stencil(pixels)
    learned = native_cfa.learn(rates)
    if learned is None:
        return {"status": "no_repeatable_stencil", "box": None, "score": 0.}
    phase, _, usable = learned
    strongest_other = np.max(rates[..., [p for p in range(4) if p != phase]], axis=-1)
    proposal = native_cfa.best_region(strongest_other - rates[..., phase] - .20, usable & (strongest_other >= .55))
    return {"status": "assessed", "phase": phase, **proposal}


def variants(original, odd_crop, ordinal):
    height, width = original.shape[:2]
    y = (12 + (ordinal * 7) % (height // 32 - 19)) * 32 + (16 if ordinal % 2 else 0)
    x = (15 + (ordinal * 9) % (width // 32 - 23)) * 32 + (16 if ordinal % 2 else 0)
    target = np.s_[y:y+128, x:x+128]
    source = original[target]
    empty = np.zeros((height, width), bool)
    yield "original", original, empty
    yield "global_crop", odd_crop, empty
    scaled = cv2.resize(original, (width * 3 // 4, height * 3 // 4), interpolation=cv2.INTER_AREA)
    yield "resized", cv2.resize(scaled, (width, height), interpolation=cv2.INTER_LINEAR), empty
    yy, xx = np.indices((128, 128))
    edge = np.minimum.reduce((yy+1, xx+1, 128-yy, 128-xx))
    alpha = ((1 - np.cos(np.pi * np.minimum(1., edge / 8.))) / 2.)[..., None]
    dx, dy = ((1, 0), (0, 1), (1, 1))[ordinal % 3]
    odd = original[160+dy:288+dy, 128+dx:256+dx]
    interpolated = cv2.warpAffine(source, np.array([[1, 0, .42], [0, 1, -.31]], np.float32),
                                  (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    for role, replacement in {
        "even_copy": original[160:288, 128:256],
        "local_fractional": interpolated,
        "phase_shift": odd,
        "phase_shift_feather": np.rint(alpha * odd + (1-alpha) * source).astype(np.uint16),
    }.items():
        final = original.copy()
        final[target] = replacement
        truth = empty.copy()
        truth[target] = np.any(replacement != source, axis=-1)
        prior.raw_source.require(truth.any(), "Frozen local intervention did not change any pixels")
        yield role, final, truth


def source_cases(archive):
    occupied = set()
    for glob in ("image-native-cfa-cc0*roster-2026-11-13.json", "image-native-stencil-interpolation-cc0-roster-2026-11-13.json"):
        for path in sorted(ROSTER.parent.glob(glob)):
            occupied.update(row["model"].casefold() for row in prior.read(path))
    roster = prior.read(ROSTER)
    models = {row["model"].casefold() for row in roster}
    prior.raw_source.require(len(models) == len(roster) == 10 and not occupied & models, "Camera roster not source-disjoint")
    for item in roster:
        path = archive / f"sample-{item['id']}{Path(item['url']).suffix.lower()}"
        prior.raw_source.require(item["license"] == "CC0-1.0" and identity(path) == item["sha256"], "Camera license or hash differs")
        yield item, path


def lock(archive):
    return {"schema": SCHEMA, "design": DESIGN, "geometries": {k: list(v) for k, v in GEOMETRIES.items()},
            "roster_sha256": identity(ROSTER), "runner_sha256": identity(Path(__file__)),
            "detector_sha256": identity(Path(native_cfa.__file__)), "point": native_cfa.POINT,
            "source_sha256": [item["sha256"] for item, _ in source_cases(archive)]}


def case(row, geometry, scope, role, pixels, truth):
    body = earlier.png_body(pixels)
    decoded = decode(body, scope)
    prior.raw_source.require(decoded is not None and np.array_equal(decoded, pixels), "Native PNG changed during round trip")
    result = measure(decoded)
    if scope == ANCHOR:
        current = native_cfa.inspect_png(body)
        prior.raw_source.require((current.status, current.score, current.proposal) == (
            result["status"], result["score"], tuple(result["box"]) if result["box"] else None), "Existing-geometry detector changed")
    selected = np.zeros_like(truth)
    if result["box"]:
        x, y, x1, y1 = result["box"]
        selected[y:y1, x:x1] = True
    return {"camera_id": row["id"], "geometry": geometry, "role": role,
            "png_sha256": hashlib.sha256(body).hexdigest(), "truth_sha256": hashlib.sha256(truth).hexdigest(),
            **result, "pixels": mask_metrics(selected, truth)}


def run(cached, archive):
    wheel = prior.raw_source.verified_wheel(cached / prior.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-cfa-dimensions-") as directory:
        prior.raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            prior.raw_source.require("rawpy" not in sys.modules, "Fresh local decoder required")
            decoder = importlib.import_module("rawpy")
            prior.raw_source.require(Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()), "Wrong RAW decoder")
            for ordinal, (row, path) in enumerate(source_cases(archive)):
                for geometry, scope in GEOMETRIES.items():
                    original = native_crop(decoder, path, scope)
                    odd_crop = native_crop(decoder, path, scope, odd=True)
                    for role, pixels, truth in variants(original, odd_crop, ordinal):
                        yield case(row, geometry, scope, role, pixels, truth)
                print("verified camera", row["id"], flush=True)
        finally:
            sys.path.pop(0)


def summary(rows):
    observed = {}
    for row in rows:
        bucket = observed.setdefault(row["geometry"], {}).setdefault(row["role"], Counter(files=0, assessed=0, flags=0, useful=0))
        bucket["files"] += 1
        bucket["assessed"] += row["status"] == "assessed"
        flagged = row["score"] > native_cfa.POINT
        pixels = row["pixels"]
        bucket["flags"] += flagged
        bucket["useful"] += flagged and pixels["iou"] >= .30 and (pixels["recall"] or 0.) >= .50
    return {shape: {"roles": counts, "passed": all(counts[r]["useful"] >= 7 for r in POSITIVE)
                    and not any(counts[r]["flags"] for r in ROLES if r not in POSITIVE)
                    and counts["resized"]["assessed"] == 0} for shape, counts in observed.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "held", "review"))
    parser.add_argument("cached", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frozen = lock(args.archive)
    lock_path = args.output / "freeze.json"
    held = args.output / "held.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(lock_path, frozen)
        return
    prior.raw_source.require(prior.read(lock_path) == frozen, "Frozen code, sources or scope changed")
    rows = list(run(args.cached, args.archive))
    ledger = {"freeze_sha256": identity(lock_path), "rows": rows}
    if args.phase == "held":
        prior.write(held, ledger)
    else:
        prior.raw_source.require(prior.read(held) == ledger, "Exact native geometry replay differs")
        if args.receipt:
            prior.write(args.receipt, frozen | {"freeze_sha256": ledger["freeze_sha256"], "held_sha256": identity(held),
                                               "exact_replay_rows": len(rows), "held": summary(rows)})
    print(json.dumps(summary(rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
