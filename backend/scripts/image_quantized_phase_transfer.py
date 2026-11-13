"""Test residual camera phase inside candidate 16-bit image edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import native_cfa_rotation
from scripts import image_native_generated_grid_transfer as shifted
from scripts import image_native_rotated_stencil_transfer as rotated

PANELS = shifted.ROSTER.parent
SCHEMA = "chipmunk.image-quantized-phase-transfer/1"
POSITIVE = frozenset(shifted.POSITIVES)
POINT = shifted.prior.POINT
PARITY_VETO = shifted.prior.VETO
MATERIALS = {
    "development": (shifted, PANELS / "image-native-generated-grid-measurements-2026-11-13.json"),
    "held": (rotated, PANELS / "image-native-rotated-stencil-measurements-2026-11-13.json"),
}
DESIGN = {
    "cohorts": "Eight development and eight different held CC0 camera models and prior generated composites. Prior project exposure in both groups and same editor; native 128-pixel local edits with distinct prompts and source geometry.",
    "candidate": "Measure only above-point four-grid boxes that survive the earlier signed-parity veto and rotation explanation. Match a modulo 256 or 257 code lattice occupying at least 95% of each color channel. Divide by the modulus and compare local two-pixel horizontal and vertical curvature with the intact nearby Bayer phase. A phase explanation needs a consistent red/blue reversal in all four quadrants. Report whether the frozen phase point separates quantized translations from generated cases including linearly promoted generation.",
    "fit": "Seal the score at the maximum admitted generated development response; use strict greater than. Report promotion and phase geometry, never authorship.",
    "gate": "Keep all useful generated boxes retained by the rotation-explained comparator; explain at least two admitted ordinary quantized translations on at least two cameras. Never explain an untouched or whole-image format control.",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, record):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def data_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def lattice(tile):
    result = None
    for step in (256, 257):
        levels = [np.bincount(channel.ravel() % step, minlength=step) for channel in tile.transpose(2, 0, 1)]
        fraction = min(float(channel.max() / tile.shape[0] / tile.shape[1]) for channel in levels)
        if fraction >= .95 and (result is None or fraction > result[0]):
            result = fraction, step
    return result


def phase_strength(tile):
    """Return the strongest 2D checker-aligned contrast shared by both colors."""
    source = tile.astype(np.float64)
    planes = []
    for axis in (0, 1):
        view = np.moveaxis(source, axis, 0)
        detail = np.abs(view[2:, 1:-1] + view[:-2, 1:-1] - 2 * view[1:-1, 1:-1])
        detail = np.clip(np.log1p(detail), 0, 3)
        planes.append(np.moveaxis(detail, 0, axis))
    result = []
    for axis, detail in enumerate(planes):
        h, w = detail.shape[:2]
        mid_y, mid_x = h // 4 * 2, w // 4 * 2
        for channel in (0, 2):
            folds = []
            for rows in (slice(0, mid_y), slice(mid_y, h)):
                for cols in (slice(0, mid_x), slice(mid_x, w)):
                    body = detail[rows, cols, channel]
                    if len(body) < 8 or body.shape[1] < 8:
                        return 0.
                    # Project curvature onto both possible axis-aligned phases.
                    along = body.T if axis == 1 else body
                    folds.append(float(along[::2].mean() - along[1::2].mean()))
            result.append(folds)
    axes = []
    for red, blue in zip(result[::2], result[1::2], strict=True):
        opposition = [*red, *(-value for value in blue)]
        axes.append(max(0., min(opposition), -max(opposition)))
    return min(axes)


def measure(body, reading):
    if not reading or not reading["box"] or reading["score"] <= POINT or max(map(min, reading["parity_fit"])) > PARITY_VETO:
        return {"status": "no_candidate", "lattice": None, "score": None}
    x, y, x1, y1 = reading["box"]
    if native_cfa_rotation.rotated_pattern(body[y+6:y1-6, x+6:x1-6])["selected"]:
        return {"status": "rotation", "lattice": None, "score": None}
    tile = body[y+10:y1-10, x+10:x1-10]
    bits = lattice(tile)
    if bits is None:
        return {"status": "no_lattice", "lattice": None, "score": None}
    fraction, step = bits
    return {"status": "measured", "lattice": [step, fraction], "score": phase_strength(tile / step)}


def freeze(run, development, held):
    run.mkdir(mode=0o700)
    record = {"schema": SCHEMA, "design": DESIGN, "runner_sha256": sha(Path(__file__)),
              "rotation_sha256": sha(Path(native_cfa_rotation.__file__)), "sources": {}}
    for (split, (method, prior)), local in zip(MATERIALS.items(), (development, held), strict=True):
        record["sources"][split] = {"roster": sha(method.ROSTER), "prior": sha(prior), "prior_lock": sha(local / "freeze.json"),
                                     "renderer": sha(Path(method.__file__))}
    write(run / "freeze.json", record)


def run_rows(cached, archive, local, split, lock):
    method, prior = MATERIALS[split]
    actual = {"roster": sha(method.ROSTER), "prior": sha(prior), "prior_lock": sha(local / "freeze.json"),
              "renderer": sha(Path(method.__file__))}
    if lock["sources"][split] != actual:
        raise ValueError("Source or renderer changed")
    witness = read(prior)
    if witness["freeze_sha256"] != actual["prior_lock"]:
        raise ValueError("Prior measurements do not bind the retained source")
    planned = {(row["key"], row["role"]): row for row in witness["rows"]}
    source = method.lock(local)
    rows = []
    for (case, original), bound in zip(method.sources(cached, archive), source["rows"], strict=True):
        if any(bound[k] != v for k, v in case.items()) or data_sha(original) != bound["source_rgb_sha256"]:
            raise ValueError("Pinned camera bytes or source geometry changed")
        path = local / "generated" / f"{case['key']}.png"
        generator = read(path.with_suffix(".json"))
        if (generator["generated_sha256"], generator["freeze_sha256"], generator["reference_sha256"]) != (
                sha(path), actual["prior_lock"], bound["reference_png_sha256"]):
            raise ValueError("Generated source bytes changed")
        with Image.open(path, formats=["PNG"]) as image:
            generated = np.asarray(image.convert("RGB"))
        for role, body, mask in method.variants(original, case, generated):
            old = planned.pop((case["key"], role))
            if (data_sha(body), data_sha(mask), int(mask.sum())) != (old["input_sha256"], old["truth_sha256"], old["truth_pixels"]):
                raise ValueError("Exact delivery or edit mask differs")
            reading = old.get("shifted", old.get("proposal"))
            rows.append({"camera": case["key"], "role": role, "input_sha256": old["input_sha256"],
                         "truth_sha256": old["truth_sha256"], **measure(body, reading)})
    if planned:
        raise ValueError("Missing planned deliveries")
    return rows


def point(rows):
    return max(row["score"] for row in rows if row["role"] in POSITIVE and row["score"] is not None)


def summarize(rows, cutoff):
    totals = {}
    groups = set()
    for row in rows:
        role = row["role"]
        flag = row["score"] is not None and row["score"] > cutoff
        totals.setdefault(role, Counter()).update(files=1, eligible=row["status"] in {"measured", "no_lattice"},
                                                   measured=row["status"] == "measured", explained=flag)
        if role == "quantized_shift" and flag:
            groups.add(row["camera"])
    advances = (len(groups) >= 2 and not any(totals[role]["explained"] for role in POSITIVE | {"original", "resized", "eight_bit"}))
    return {"counts": totals, "explained_camera_groups": len(groups), "advances": advances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "replay-development", "replay-held"))
    for name in ("run", "cached", "archive", "development", "held"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.run, args.development, args.held)
        return
    lock_path = args.run / "freeze.json"
    lock = read(lock_path)
    if (lock["schema"], lock["design"], lock["runner_sha256"], lock["rotation_sha256"]) != (
            SCHEMA, DESIGN, sha(Path(__file__)), sha(Path(native_cfa_rotation.__file__))):
        raise ValueError("Frozen implementation changed")
    split = args.phase.removeprefix("replay-")
    rows = run_rows(args.cached, args.archive, getattr(args, split), split, lock)
    dev_path = args.run / "development.json"
    if split == "development":
        result = {"freeze_sha256": sha(lock_path), "rows": rows, "point": point(rows)}
    else:
        dev = read(dev_path)
        cutoff = point(dev["rows"])
        if (dev["freeze_sha256"], dev["point"]) != (sha(lock_path), cutoff):
            raise ValueError("Frozen operating point differs")
        result = {"freeze": lock, "development": dev, "held": rows, "point": cutoff, **summarize(rows, cutoff)}
    output = args.run / f"{split}.json"
    if args.phase.startswith("replay-"):
        if result != read(output):
            raise ValueError("Native replay differs")
    else:
        write(output, result)
    print(split, len(rows), "point", result["point"], "advances", result.get("advances"))


if __name__ == "__main__":
    main()
