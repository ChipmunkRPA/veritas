# SPDX-License-Identifier: Apache-2.0
"""Test a shifted grid and code-lattice explanation on localized generated edits."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import image_native_stencil_interpolation_transfer as prior

ROOT, flux, base = prior.ROOT, prior.flux, prior.earlier
ROSTER = ROOT / "docs/evaluation-panels/image-native-generated-grid-cc0-roster-2026-11-13.json"
SCHEMA = "chipmunk.image-native-generated-grid-transfer/1"
PROMPT = ("Add a small bright yellow star sticker in the center of the source photo. "
          "Keep the framing and all other objects unchanged.")
WIDTH = 128
PATCH = base.PATCH
GRIDS = ((0, 0), (0, 16), (16, 0), (16, 16))
POSITIVES = ("generated_sharp", "generated_feathered", "generated_linear_promote")
OUT_OF_SCOPE = ("resized", "eight_bit")
ROLES = ("original", "even_copy", "blur", "local_tone", "diagonal_cubic", "rotation_bilinear",
         "quantized_256", "quantized_257", *POSITIVES, *OUT_OF_SCOPE)
DESIGN = {
    "scope": "Eight checksum-bound new CC0 cameras, controlled native linear 16-bit RGB sensor grid, exact masks. One local editor. 8-bit and resized views are separately checked as unsupported.",
    "generated": {**base.DESIGN["generator"], "prompt": PROMPT,
                  "delivery": "Sharp and 12-pixel feathered quadratic 8-to-16-bit transfer, plus sharp linear 8-to-16-bit promotion"},
    "baseline": "Fixed 32-pixel grid, prior texture-supported stencil-loss threshold, then frozen global parity veto. No source pair at inference.",
    "grid_extension": "Add three half-cell offsets, recover absolute coordinates, and select the strongest proposal at the original point before the original parity veto.",
    "quantization_note": "Declare a separate promotion clue only if every RGB channel's strongest modulo-256 or modulo-257 lattice accounts for at least 0.95 of interior codes, while it accounts for at most 0.70 of the surrounding-ring codes; both regions need at least 24 distinct codes per channel. Audit conventional and generated promotions separately. A promotion clue has zero independent authorship value.",
    "gate": "Add at least two useful generated sharp and two feathered locations; add at most one ordinary file highlight. Never use the promotion note as a veto if it removes a known promoted generated case. Untouched must stay clear and all format challenges decline.",
    "confidence_weight": 0,
}


def sources(cached, archive):
    from scripts import image_native_cfa_closure_transfer as native

    used = set()
    for pattern in ("image-native-cfa-*roster-2026-11-13.json", "image-native-stencil-interpolation-cc0-roster-2026-11-13.json"):
        for path in sorted(ROSTER.parent.glob(pattern)):
            used.update(row["model"].casefold() for row in base.read(path))
    rows = base.read(ROSTER)
    models = {r["model"].casefold() for r in rows}
    flux.require(len(rows) == len(models) == 8 and not used & models, "Held camera models overlap earlier studies")
    wheel = native.raw_source.verified_wheel(cached.resolve() / native.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-generated-grid-") as folder:
        native.raw_source.unpack_decoder(wheel, folder)
        sys.path.insert(0, folder)
        try:
            flux.require("rawpy" not in sys.modules, "Fresh RAW decoder process required")
            raw = importlib.import_module("rawpy")
            flux.require(Path(raw.__file__).resolve().is_relative_to(Path(folder).resolve()), "Wrong pinned RAW decoder")
            for i, row in enumerate(rows):
                key = row["id"]
                path = archive / f"sample-{key}{Path(row['url']).suffix.lower()}"
                flux.require(row["license"] == "CC0-1.0" and base.sha(path) == row["sha256"], "Camera license or hash differs")
                x, y = (9 + i * 11 % 34) * 32 + 16, (9 + i * 7 % 27) * 32 + 16
                case = {"key": key, "source_sha256": row["sha256"], "camera": row["model"],
                        "crop": [x, y, x + PATCH, y + PATCH], "seed": 490_100 + i}
                yield case, native.raw_render(raw, path)[0]
        finally:
            sys.path.pop(0)


def freeze(cached, archive, output):
    from PIL import Image

    output.mkdir(mode=0o700)
    (output / "references").mkdir(mode=0o700)
    (output / "generated").mkdir(mode=0o700)
    rows = []
    for case, rgb in sources(cached, archive):
        x, y, x1, y1 = case["crop"]
        path = output / "references" / f"{case['key']}.png"
        Image.fromarray(base.preview(rgb[y:y1, x:x1])).save(path)
        rows.append(case | {"reference_png_sha256": base.sha(path), "source_rgb_sha256": prior.digest(rgb)})
    base.write(output / "freeze.json", {"schema": SCHEMA, "design": DESIGN, "runner_sha256": base.sha(Path(__file__)),
               "roster_sha256": base.sha(ROSTER), "prior_sha256": base.sha(Path(prior.__file__)), "rows": rows})


def lock(output):
    frozen = base.read(output / "freeze.json")
    flux.require((frozen["design"], frozen["runner_sha256"], frozen["roster_sha256"], frozen["prior_sha256"])
                 == (DESIGN, base.sha(Path(__file__)), base.sha(ROSTER), base.sha(Path(prior.__file__))),
                 "Frozen methods or sources differ")
    return frozen


def variants(rgb, row, generated):
    import cv2
    import numpy as np

    center = (PATCH - WIDTH) // 2
    x, y = (v + center for v in row["crop"][:2])
    target = np.s_[y:y+WIDTH, x:x+WIDTH]
    native = rgb[target]
    truth = np.zeros(rgb.shape[:2], bool)
    yield "original", rgb, truth
    yield "eight_bit", (rgb >> 8).astype(np.uint8), truth
    h, w = rgb.shape[:2]
    reduced = cv2.resize(rgb, (w * 4 // 5, h * 4 // 5), interpolation=cv2.INTER_AREA)
    yield "resized", cv2.resize(reduced, (w, h), interpolation=cv2.INTER_CUBIC), truth
    gy, gx = np.indices((WIDTH, WIDTH))
    distance = np.minimum.reduce((gy+1, gx+1, WIDTH-gy, WIDTH-gx))
    alpha = ((1 - np.cos(np.pi * np.minimum(1., distance / 12.))) / 2.)[..., None]
    new = generated[center:center+WIDTH, center:center+WIDTH]
    quad = np.rint((new.astype(np.float32) / 255.) ** 2 * 65535.).astype(np.uint16)
    matrix = np.array([[1, 0, .438], [0, 1, -.281]], dtype=np.float32)
    size = WIDTH, WIDTH
    shift_8 = cv2.warpAffine((native >> 8).astype(np.uint8), matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101).astype(np.uint16)
    controls = {
        "even_copy": rgb[256:384, 128:256],
        "blur": cv2.GaussianBlur(native, (9, 9), 1.8),
        "local_tone": np.rint(native * .89 + 1900).astype(np.uint16),
        "diagonal_cubic": cv2.warpAffine(native, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101),
        "rotation_bilinear": cv2.warpAffine(native, cv2.getRotationMatrix2D((64, 64), -3.4, 1.), size,
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
        "quantized_256": shift_8 << 8,
        "quantized_257": shift_8 * 257,
        "generated_sharp": quad,
        "generated_feathered": np.rint(quad * alpha + native * (1 - alpha)).astype(np.uint16),
        "generated_linear_promote": new.astype(np.uint16) * 257,
    }
    for role, patch_rgb in controls.items():
        value = rgb.copy()
        value[target] = patch_rgb
        exact = truth.copy()
        exact[target] = np.any(native != patch_rgb, axis=-1)
        flux.require(exact.any() and np.array_equal(exact, np.any(value != rgb, axis=-1)), "Exact mask differs")
        yield role, value, exact


def lattice_fraction(body, modulus):
    import numpy as np

    values = body.reshape(-1, 3)
    if not len(values) or min(len(np.unique(column)) for column in values.T) < 24:
        return None
    return [float(np.bincount(column % modulus).max() / len(values)) for column in values.T]


def promotion(rgb, box):
    import numpy as np

    x0, y0, x1, y1 = box
    core = rgb[y0+12:y1-12, x0+12:x1-12]
    pieces = [rgb[max(0,y0-24):y0, x0:x1], rgb[y1:y1+24, x0:x1],
              rgb[y0:y1, max(0,x0-24):x0], rgb[y0:y1, x1:x1+24]]
    ring = np.concatenate([p.reshape(-1, 3) for p in pieces], axis=0)
    readings = []
    for modulus in (256, 257):
        inner = lattice_fraction(core, modulus)
        surrounding = lattice_fraction(ring, modulus)
        readings.append({"modulus": modulus, "core": inner, "ring": surrounding})
    return readings


def proposal(rgb, truth, dy, dx):
    # Feed an unchanged model an exact native crop. Its output is shifted back into full-source coordinates.
    h, w = rgb.shape[:2]
    stop_y, stop_x = h - dy, w - dx
    result = base.measure(rgb[dy:stop_y, dx:stop_x], truth[dy:stop_y, dx:stop_x])
    if result["status"] != "assessed":
        return None
    item = result["arms"]["textured_phase_loss"]
    if item["box"]:
        x0, y0, x1, y1 = item["box"]
        item["box"] = [x0+dx, y0+dy, x1+dx, y1+dy]
    item["phase"] = result["phase"]
    item["grid"] = [dx, dy]
    return item


def finish(rgb, truth, choice):
    import numpy as np

    if choice is None:
        return None
    from scripts.localization_benchmark import mask_metrics

    box = choice["box"]
    selected = np.zeros_like(truth)
    if box:
        x, y, x1, y1 = box
        selected[y:y1, x:x1] = True
        choice["parity_fit"] = prior.interpolation_fit(rgb[y+8:y1-8, x+8:x1-8], choice["phase"])
        choice["promotion"] = promotion(rgb, box)
    else:
        choice["parity_fit"] = choice["promotion"] = None
    return choice | {"pixels": mask_metrics(selected, truth)}


def measured(rgb, truth):
    import numpy as np

    if rgb.dtype != np.uint16:
        return {"status": "unsupported_format", "base": None, "shifted": None}
    proposals = [proposal(rgb, truth, *pair) for pair in GRIDS]
    allowed = [p for p in proposals if p is not None]
    return {"status": "assessed" if allowed else "no_repeatable_stencil",
            "base": finish(rgb, truth, proposals[0]),
            "shifted": finish(rgb, truth, max(allowed, key=lambda x: x["score"]).copy()) if allowed else None}


def measure(cached, archive, output):
    import numpy as np
    from PIL import Image

    frozen = lock(output)
    freeze_hash = base.sha(output / "freeze.json")
    rows = []
    for (case, rgb), witness in zip(sources(cached, archive), frozen["rows"], strict=True):
        flux.require(witness["source_rgb_sha256"] == prior.digest(rgb) and all(witness[k] == v for k,v in case.items()),
                     "Frozen source pixels differ")
        png = output / "generated" / f"{case['key']}.png"
        journal = base.read(png.with_suffix(".json"))
        flux.require((journal["generated_sha256"], journal["freeze_sha256"], journal["reference_sha256"])
                     == (base.sha(png), freeze_hash, witness["reference_png_sha256"]), "Generated binding differs")
        with Image.open(png, formats=["PNG"]) as handle:
            generated = np.asarray(handle.convert("RGB"))
        for role, pixels, truth in variants(rgb, case, generated):
            rows.append({"key": case["key"], "role": role, "input_sha256": prior.digest(pixels),
                         "truth_sha256": prior.digest(truth), "truth_pixels": int(truth.sum()), **measured(pixels, truth)})
        print("scored", case["key"], flush=True)
    return {"freeze_sha256": freeze_hash, "rows": rows}


def summary(rows):
    summary_ = {role: Counter(files=0, unsupported_format=0, no_repeatable_stencil=0, assessed=0,
                             base_flags=0, base_useful=0, shifted_flags=0, shifted_useful=0,
                             shifted_promotion=0) for role in ROLES}
    for row in rows:
        result = summary_[row["role"]]
        result["files"] += 1
        result[row["status"]] += 1
        for arm in ("base", "shifted"):
            read = row[arm]
            if read is None or read["score"] <= prior.POINT or max(map(min, read["parity_fit"])) > prior.VETO:
                continue
            result[arm+"_flags"] += 1
            pixels = read["pixels"]
            result[arm+"_useful"] += pixels["iou"] >= .30 and (pixels["recall"] or 0.) >= .50
            if arm == "shifted":
                result["shifted_promotion"] += any(m["core"] is not None and m["ring"] is not None
                                                  and min(m["core"]) >= .95 and max(m["ring"]) <= .70 for m in read["promotion"])
    ordinary = [r for r in ROLES if r not in (*POSITIVES, *OUT_OF_SCOPE)]
    prior_flags = sum(summary_[r]["base_flags"] for r in ordinary)
    newer_flags = sum(summary_[r]["shifted_flags"] for r in ordinary)
    passed = all(summary_[r]["shifted_useful"] >= summary_[r]["base_useful"] + 2 for r in POSITIVES[:2])
    passed &= newer_flags <= prior_flags + 1 and not summary_["original"]["shifted_flags"]
    passed &= all(summary_[r]["assessed"] == 0 for r in OUT_OF_SCOPE)
    return {"by_role": summary_, "ordinary_base_flags": prior_flags, "ordinary_shifted_flags": newer_flags,
            "shifted_grid_passed": bool(passed),
            "promotion_safe_as_veto": not any(summary_[role]["shifted_promotion"] for role in POSITIVES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "generate", "score", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cached", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "generate":
        with patch.multiple(prior, lock=lock, PROMPT=PROMPT):
            prior.generate(args.output, args.runtime, args.overlay)
        return
    import cv2

    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.cached, args.archive, args.output)
        return
    result = measure(args.cached, args.archive, args.output)
    ledger = args.output / "scored.json"
    if args.phase == "score":
        base.write(ledger, result)
    else:
        flux.require(base.read(ledger) == result, "Native pixels, truth or regions changed on replay")
        if args.receipt:
            frozen = lock(args.output)
            base.write(args.receipt, {key: value for key,value in frozen.items() if key != "rows"} | {
                "freeze_sha256": result["freeze_sha256"], "scored_sha256": base.sha(ledger),
                "sources": len(frozen["rows"]), "exact_native_replays": len(result["rows"]),
                "generated_sha256": [base.sha(args.output / "generated" / f"{row['key']}.png") for row in frozen["rows"]],
                "measurements": summary(result["rows"]),
            })
    print(json.dumps(summary(result["rows"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
