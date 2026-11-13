# SPDX-License-Identifier: Apache-2.0
"""Transfer the frozen Bayer phase-shift check to native 16-bit PNG bytes."""
from __future__ import annotations

import argparse
import importlib
import json
import struct
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import image_native_cfa_closure_transfer as prior
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-native-cfa-png-transfer/1"
ROSTER = prior.ROOT / "docs/evaluation-panels/image-native-cfa-cc0-png-roster-2026-11-13.json"
PRIOR_RESULT = prior.ROOT / "docs/evaluation-panels/image-native-cfa-closure-transfer-2026-11-13.json"
POINT = .25
MAX_BODY_BYTES = 25 * 1024 * 1024
MAX_CHUNKS = 4096
PROFILE = b"\x10\x02\x00\x00\x00"
PNG = b"\x89PNG\r\n\x1a\n"
SIZES = (32, 64, 128)
AXES = {"even": (0, 0), "x_odd": (1, 0), "y_odd": (0, 1), "both_odd": (1, 1)}
NUISANCES = ("original", "crop_odd", "global_gain", "local_fractional", "resize", "eight_bit", "alpha")
DESIGN = {
    "eligible": "Static noninterlaced 2048x1536 RGB PNG, 16 bits per channel, without rotation, color transform or transparency metadata. Decode unchanged and measure repeating bilinear red/blue closure from two independent folds of submitted pixels.",
    "fixed_prior": "Unchanged 32-pixel cells, proposal sizes, phase-shift measure and 0.25 operating point from the earlier RAW experiment. No sizing or cutoff changes after freezing. Score exact transmitted PNG bytes.",
    "inputs": "Ten distinct held camera models, CC0 files verified by publisher SHA-256, excluding previous models. Edited panel includes exact 32-, 64- and 128-pixel square support, three odd phase shifts and even-aligned copy, for both hard and feathered boundaries.",
    "controls": "Untouched, globally odd-offset crops, global affine gain, local fractional interpolation, image-wide resize and re-encode, 8-bit and alpha PNG. Keep original-size camera pixels until lossless PNG transport.",
    "useful": "Changed-pixel IoU at least 0.30 and recall at least 0.50.",
    "admission": "At least eight exact-useful sharp 128-pixel regions for each odd phase on ten sources; no even-copy or ordinary highlights; every resized, 8-bit or alpha view declines. Report all smaller and feathered strata separately.",
    "confidence_weight": 0,
}


def png_body(rgb):
    order = (2, 1, 0, 3) if rgb.shape[2] == 4 else (2, 1, 0)
    success, payload = cv2.imencode(".png", np.ascontiguousarray(rgb[..., order]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    prior.raw_source.require(success, "Lossless test PNG encode failed")
    return bytes(payload)


def allowed_png(body):
    if not 33 <= len(body) <= MAX_BODY_BYTES or not body.startswith(PNG):
        return False
    height, width = prior.SCOPE
    if body[8:29] != struct.pack(">I4sII", 13, b"IHDR", width, height) + PROFILE:
        return False
    view = memoryview(body)
    offset = 8
    idat = False
    for index in range(MAX_CHUNKS):
        if offset + 12 > len(body):
            return False
        size, kind = struct.unpack_from(">I4s", body, offset)
        end = offset + 8 + size
        if end + 4 > len(body) or zlib.crc32(view[offset+4:end]) != struct.unpack_from(">I", body, end)[0]:
            return False
        if kind == b"IHDR":
            if index != 0:
                return False
        elif kind == b"IDAT":
            idat = True
        elif kind == b"IEND":
            return idat and size == 0 and end + 4 == len(body)
        else:
            # The tested transport is deliberately a bare PNG. A broader chunk scope needs its own qualification.
            return False
        offset = end + 4
    return False


def decoded_rgb(body):
    if not allowed_png(body):
        return None
    bgr = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.shape != (*prior.SCOPE, 3):
        return None
    return np.ascontiguousarray(bgr[..., ::-1])


def composite(original, ordinal, width, name, blend):
    y = (13 + (ordinal * 7) % 18) * prior.CELL
    x = (15 + (ordinal * 9) % 28) * prior.CELL
    # Test the smallest widths at both cell alignment and half-cell displacement.
    if ordinal % 2:
        y, x = y + 16, x + 16
    tile = np.s_[y:y+width, x:x+width]
    old = original[tile]
    dx, dy = AXES[name]
    donor = original[5*prior.CELL+dy:5*prior.CELL+dy+width, 4*prior.CELL+dx:4*prior.CELL+dx+width]
    if blend == "feather":
        yy, xx = np.indices(old.shape[:2])
        distance = np.minimum.reduce((yy+1, xx+1, width-yy, width-xx))
        alpha = ((1 - np.cos(np.pi * np.minimum(1., distance / 8.))) / 2.)[..., None]
        donor = np.rint(donor * alpha + old * (1 - alpha)).astype(np.uint16)
    modified = original.copy()
    modified[tile] = donor
    truth = np.zeros(prior.SCOPE, bool)
    truth[tile] = np.any(donor != old, axis=-1)
    prior.raw_source.require(truth.any(), "Known paste has no changed pixels")
    return modified, truth


def nuisances(original, shifted):
    height, width = prior.SCOPE
    empty = np.zeros(prior.SCOPE, bool)
    yield "original", original, empty
    yield "crop_odd", shifted, empty
    yield "global_gain", np.rint(original * .82 + 1200).astype(np.uint16), empty
    n = 128
    altered = original.copy()
    tile = original[400:400+n, 640:640+n]
    new = cv2.warpAffine(tile, np.array([[1, 0, .5], [0, 1, .375]], dtype=np.float32),
                         (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    altered[400:400+n, 640:640+n] = new
    yield "local_fractional", altered, empty
    resized = cv2.resize(original, (width * 3 // 4, height * 3 // 4), interpolation=cv2.INTER_AREA)
    yield "resize", cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR), empty
    yield "eight_bit", (original >> 8).astype(np.uint8), empty
    yield "alpha", np.concatenate([original, np.full((height, width, 1), 65535, dtype=np.uint16)], axis=2), empty


def source_rows(archive):
    source = prior.read(ROSTER)
    excluded = prior.read(prior.ROSTER)
    prior.raw_source.require(len(source) == len({r["model"].casefold() for r in source}) == 10
                             and not {r["model"].casefold() for r in source} & {r["model"].casefold() for r in excluded},
                             "Ten prospectively distinct held camera models required")
    for row in source:
        path = archive / f"sample-{row['id']}{Path(row['url']).suffix.lower()}"
        prior.raw_source.require(row["license"] == "CC0-1.0" and prior.sha(path) == row["sha256"],
                                 "Unqualified camera RAW source")
        yield row, path


def freeze(archive):
    receipt = prior.read(PRIOR_RESULT)
    prior.raw_source.require(receipt["runner_sha256"] == prior.sha(Path(prior.__file__))
                             and receipt["points"]["phase_shift"] == POINT, "Earlier frozen phase check differs")
    return {"schema": SCHEMA, "design": DESIGN, "point": POINT, "runner_sha256": prior.sha(Path(__file__)),
            "phase_implementation_sha256": prior.sha(Path(prior.__file__)), "prior_receipt_sha256": prior.sha(PRIOR_RESULT),
            "roster_sha256": prior.sha(ROSTER), "wheel_sha256": prior.raw_source.WHEEL_SHA,
            "sources": [row["sha256"] for row, _ in source_rows(archive)],
            "opencv": cv2.__version__, "numpy": np.__version__}


def readout(case, role, rgb, truth):
    body = png_body(rgb)
    decoded = decoded_rgb(body)
    row = {"camera_id": case, "role": role, "png_sha256": prior.raw_source.sha256(body),
           "truth_sha256": prior.digest(truth), "truth_pixels": int(truth.sum())}
    if decoded is None:
        return row | {"status": "unsupported_png"}
    prior.raw_source.require(np.array_equal(rgb, decoded), "PNG decoder changed exact native pixel samples")
    measured = prior.measure(decoded)
    if measured["status"] != "assessed":
        return row | {"status": measured["status"]}
    arm = measured["arms"]["phase_shift"]
    box = arm["box"]
    proposal = np.zeros(prior.SCOPE, bool)
    if box:
        x, y, x1, y1 = box
        proposal[y:y1, x:x1] = True
    return row | {"status": "assessed", "phase": measured["phase"], **arm, "overlap": mask_metrics(proposal, truth)}


def observe(cached, archive):
    wheel = prior.raw_source.verified_wheel(cached / prior.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-native-cfa-png-") as directory:
        prior.raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            prior.raw_source.require("rawpy" not in sys.modules, "Fresh isolated decoder process required")
            decoder = importlib.import_module("rawpy")
            prior.raw_source.require(Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()),
                                     "Pinned RAW decoder required")
            for ordinal, (row, path) in enumerate(source_rows(archive)):
                rgb, _ = prior.raw_render(decoder, path)
                # Shift by one in both axes at native resolution, with reflection only at the crop's far boundaries.
                y, x = np.indices(prior.SCOPE)
                shifted = rgb[np.minimum(y+1, rgb.shape[0]-1), np.minimum(x+1, rgb.shape[1]-1)]
                for role, values, truth in nuisances(rgb, shifted):
                    yield readout(row["id"], role, values, truth)
                for size in SIZES:
                    for name in AXES:
                        for blend in ("sharp", "feather"):
                            values, truth = composite(rgb, ordinal, size, name, blend)
                            yield readout(row["id"], f"{size}_{name}_{blend}", values, truth)
                print("measured", row["id"], flush=True)
        finally:
            sys.path.pop(0)


def summarize(rows):
    status = {}
    for row in rows:
        counts = status.setdefault(row["role"], Counter())
        counts["files"] += 1
        counts[row["status"]] += 1
        if row["status"] != "assessed":
            continue
        flag = row["score"] > POINT
        overlap = row["overlap"]
        useful = overlap["iou"] >= .30 and (overlap["recall"] or 0.) >= .50
        counts["flags"] += flag
        counts["ungated_useful"] += useful
        counts["useful"] += useful and flag
    sharp = all(status[f"128_{p}_sharp"]["useful"] >= 8 for p in AXES if p != "even")
    ordinary = sum(counts["flags"] for role, counts in status.items() if role in NUISANCES or "_even_" in role)
    declines = all(status[p]["assessed"] == 0 for p in ("resize", "eight_bit", "alpha"))
    return {"roles": {name: dict(counts) for name, counts in status.items()}, "ordinary_highlights": ordinary,
            "admission": sharp and ordinary == 0 and declines}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "held", "review"))
    parser.add_argument("cached", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.archive)
    frozen = args.output / "freeze.json"
    measured = args.output / "held.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(frozen, lock)
        return
    prior.raw_source.require(prior.read(frozen) == lock, "Frozen method or capture set changed")
    rows = list(observe(args.cached.resolve(), args.archive))
    result = {"freeze_sha256": prior.sha(frozen), "rows": rows}
    if args.phase == "held":
        prior.write(measured, result)
    else:
        prior.raw_source.require(prior.read(measured) == result, "Exact native PNG, mask or geometry replay differs")
        if args.receipt:
            prior.write(args.receipt, {key: val for key, val in lock.items() if key != "sources"}
                       | {"freeze_sha256": prior.sha(frozen), "held_sha256": prior.sha(measured), "held_sources": len(lock["sources"]),
                          "exact_held_replays": len(rows), "held": summarize(rows)})
    print(json.dumps(summarize(rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
