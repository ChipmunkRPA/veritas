# SPDX-License-Identifier: Apache-2.0
"""Test a rotated CFA spectral explanation after four-grid phase loss."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import image_native_generated_grid_transfer as grid

ROOT, base, prior, flux = grid.ROOT, grid.base, grid.prior, grid.flux
ROSTER = ROOT / "docs/evaluation-panels/image-native-rotated-stencil-cc0-roster-2026-11-13.json"
SCHEMA = "chipmunk.image-native-rotated-stencil-transfer/1"
PROMPT = ("Add a small matte purple paper hexagon in the center of the photo. "
          "Preserve the existing lighting, framing and background.")
ANGLES = tuple(step / 4 for step in range(7, 25))
POINT = .08
MARGIN = 1.35
ROTATIONS = ("rotate_bilinear", "rotate_bicubic")
POSITIVES = ("generated_sharp", "generated_feathered", "generated_linear_promote")
OUT_OF_SCOPE = ("resized", "eight_bit")
ROLES = ("original", "blur", "tone", "copy_even", "diagonal_shift", "quantized_shift", *ROTATIONS, *POSITIVES, *OUT_OF_SCOPE)
DESIGN = {
    "generator": {**base.DESIGN["generator"], "prompt": PROMPT},
    "scope": "Eight new CC0 camera models with exact native-grid edits. All sixteenth-bit developed inputs retain native RGB samples; ordinary controls are authored at the same coordinates.",
    "prior": "Four-grid texture-supported loss with its frozen proposal, point and previous global signed-parity veto.",
    "rotation": "Log horizontal plus vertical absolute curvature, Gaussian-detrend and clip. Compare opposite signed red and blue phase at both orthogonal Nyquist axes rotated through ±1.75–6 degrees in 0.25-degree steps. Require agreement across two horizontal spatial folds. Veto only if the strongest off-axis support is at least 0.08 and exceeds 1.35 times zero-angle support.",
    "admission": "Recover at least eight useful combined sharp/feathered generation locations, preserve every preselected useful generated location including promoted outputs, eliminate at least four ordinary rotation flags with at least half of earlier rotations suppressed, keep all untouched clear and separate unsupported images.",
    "confidence_weight": 0,
}


def sources(cached, archive):
    from scripts import image_native_cfa_closure_transfer as native

    used = set()
    for path in sorted(ROSTER.parent.glob("image-native-*-roster-2026-11-13.json")):
        if path != ROSTER:
            used.update(row["model"].casefold() for row in base.read(path) if "model" in row)
    roster = base.read(ROSTER)
    models = {r["model"].casefold() for r in roster}
    flux.require(len(roster) == len(models) == 8 and not used & models, "Camera roster overlaps previous studies")
    wheel = native.raw_source.verified_wheel(cached.resolve() / native.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-rotated-stencil-") as directory:
        native.raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            flux.require("rawpy" not in sys.modules, "Isolated RAW decoder required")
            decoder = importlib.import_module("rawpy")
            flux.require(Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()), "Wrong decoder")
            for index, row in enumerate(roster):
                asset = archive / f"sample-{row['id']}{Path(row['url']).suffix.lower()}"
                flux.require(row["license"] == "CC0-1.0" and base.sha(asset) == row["sha256"], "Invalid source commitment")
                x, y = (10 + (index * 9) % 33) * 32 + 16, (10 + (index * 7) % 23) * 32 + 16
                binding = {"key": row["id"], "source_sha256": row["sha256"], "camera": row["model"],
                           "crop": [x, y, x + grid.PATCH, y + grid.PATCH], "seed": 609_300 + index, "index": index}
                yield binding, native.raw_render(decoder, asset)[0]
        finally:
            sys.path.pop(0)


def freeze(cached, archive, output):
    with patch.multiple(grid, sources=sources, DESIGN=DESIGN, ROSTER=ROSTER):
        grid.freeze(cached, archive, output)
    frozen = base.read(output / "freeze.json")
    frozen.update(schema=SCHEMA, runner_sha256=base.sha(Path(__file__)), grid_sha256=base.sha(Path(grid.__file__)))
    base.write(output / "freeze.json", frozen)


def lock(output):
    frozen = base.read(output / "freeze.json")
    flux.require((frozen["design"], frozen["runner_sha256"], frozen["roster_sha256"], frozen["grid_sha256"])
                 == (DESIGN, base.sha(Path(__file__)), base.sha(ROSTER), base.sha(Path(grid.__file__))),
                 "Frozen method or sources differ")
    return frozen


def variants(original, binding, generated):
    import cv2
    import numpy as np

    n = grid.WIDTH
    center = (grid.PATCH - n) // 2
    x, y = (p + center for p in binding["crop"][:2])
    destination = np.s_[y:y+n, x:x+n]
    patch_rgb = original[destination]
    empty = np.zeros(original.shape[:2], bool)
    yield "original", original, empty
    yield "eight_bit", (original >> 8).astype(np.uint8), empty
    h, w = empty.shape
    resized = cv2.resize(original, (w * 7 // 8, h * 7 // 8), interpolation=cv2.INTER_AREA)
    yield "resized", cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR), empty
    angles = (-4.1, 3.25, -2.4, 4.75, 3.65, -3.85, 2.9, -4.45)
    angle = angles[binding["index"]]
    rotated_linear = cv2.getRotationMatrix2D((64, 64), angle, 1.)
    rotated_cubic = cv2.getRotationMatrix2D((64, 64), -angle + .31, 1.)
    translated = np.array([[1, 0, -.344], [0, 1, .219]], dtype=np.float32)

    def transform(body, matrix, interpolation):
        return cv2.warpAffine(body, matrix, (n,n), flags=interpolation, borderMode=cv2.BORDER_REFLECT_101)

    rendered = generated[center:center+n, center:center+n]
    color = np.rint((rendered.astype(np.float32) / 255.) ** 2 * 65535.).astype(np.uint16)
    yy, xx = np.indices((n,n))
    edge = np.minimum.reduce((yy + 1, xx + 1, n - yy, n - xx))
    feather = ((1 - np.cos(np.pi * np.minimum(1., edge / 11.))) / 2.)[..., None]
    interventions = {
        "blur": cv2.GaussianBlur(patch_rgb, (7, 7), 1.75),
        "tone": np.rint(patch_rgb * .81 + 2600).astype(np.uint16),
        "copy_even": original[192:192+n, 160:160+n],
        "diagonal_shift": transform(patch_rgb, translated, cv2.INTER_CUBIC),
        "quantized_shift": transform((patch_rgb >> 8).astype(np.uint8), translated, cv2.INTER_LINEAR).astype(np.uint16) * 257,
        "rotate_bilinear": transform(patch_rgb, rotated_linear, cv2.INTER_LINEAR),
        "rotate_bicubic": transform(patch_rgb, rotated_cubic, cv2.INTER_CUBIC),
        "generated_sharp": color,
        "generated_feathered": np.rint(color * feather + patch_rgb * (1 - feather)).astype(np.uint16),
        "generated_linear_promote": rendered.astype(np.uint16) * 257,
    }
    for role, replacement in interventions.items():
        modified = original.copy()
        modified[destination] = replacement
        exact = empty.copy()
        exact[destination] = np.any(replacement != patch_rgb, axis=-1)
        flux.require(exact.any() and np.array_equal(exact, np.any(modified != original, axis=2)), "Wrong exact edit region")
        yield role, modified, exact


def rotated_pattern(rgb):
    import cv2
    import numpy as np

    region = rgb.astype(np.float32)
    vertical = region[2:, 1:-1] - 2 * region[1:-1, 1:-1] + region[:-2, 1:-1]
    horizontal = region[1:-1, 2:] - 2 * region[1:-1, 1:-1] + region[1:-1, :-2]
    logged = np.log1p(np.abs(vertical) + np.abs(horizontal))
    detail = np.clip(logged - cv2.GaussianBlur(logged, (0, 0), 3.), -2., 2.)
    yy, xx = np.indices(detail.shape[:2], dtype=np.float64)
    halves = xx < xx.max() / 2
    red, blue = detail[..., 0], detail[..., 2]

    def at_angle(angle):
        rad = np.deg2rad(angle)
        a, b = np.cos(rad) * np.pi, np.sin(rad) * np.pi
        amplitudes = []
        for fx, fy in ((a,b),(-b,a)):
            carrier = np.exp(1j * (fx * xx + fy * yy))
            folds = []
            for selection in (halves, ~halves):
                ar = np.mean((red * carrier)[selection])
                ab = np.mean((blue * carrier)[selection])
                contra = ar - ab if np.real(ar * np.conj(-ab)) > 0 else 0j
                folds.append((contra, abs(ar) + abs(ab)))
            (left, strength_left), (right, strength_right) = folds
            agreement = np.real(left * np.conj(right)) / max(1e-7, abs(left) * abs(right))
            amplitudes.append(float(min(strength_left, strength_right) * max(0., agreement)))
        return min(amplitudes)

    zero = at_angle(0.)
    peak, angle = max((at_angle(a), a) for a in (*ANGLES, *(-a for a in ANGLES)))
    return {"zero": zero, "peak": peak, "angle": angle, "selected": peak >= POINT and peak > MARGIN * zero}


def measure(cached, archive, output):
    import numpy as np
    from PIL import Image

    frozen = lock(output)
    rows = []
    frozen_sha = base.sha(output / "freeze.json")
    for (binding, original), witness in zip(sources(cached, archive), frozen["rows"], strict=True):
        flux.require(witness["source_rgb_sha256"] == prior.digest(original) and all(witness[k] == v for k,v in binding.items()),
                     "Frozen source differs")
        png = output / "generated" / f"{binding['key']}.png"
        record = base.read(png.with_suffix(".json"))
        flux.require((record["generated_sha256"], record["freeze_sha256"], record["reference_sha256"])
                     == (base.sha(png), frozen_sha, witness["reference_png_sha256"]), "Generated output differs")
        with Image.open(png, formats=["PNG"]) as image:
            rendered = np.asarray(image.convert("RGB"))
        for role, rgb, truth in variants(original, binding, rendered):
            old = grid.measured(rgb, truth)
            proposal = old["shifted"]
            found = bool(proposal and proposal["score"] > prior.POINT and max(map(min, proposal["parity_fit"])) <= prior.VETO)
            readout = None
            if found:
                x, y, x1, y1 = proposal["box"]
                readout = rotated_pattern(rgb[y+6:y1-6, x+6:x1-6])
            rows.append({"key": binding["key"], "role": role, "input_sha256": prior.digest(rgb),
                         "truth_sha256": prior.digest(truth), "truth_pixels": int(truth.sum()),
                         "status": old["status"], "proposal": proposal, "preselected": found, "rotated_pattern": readout})
        print("scored", binding["key"], flush=True)
    return {"freeze_sha256": frozen_sha, "rows": rows}


def summarize(rows):
    strata = {role: Counter(files=0, assessed=0, unsupported_format=0, no_repeatable_stencil=0, old_flags=0,
                            new_flags=0, old_useful=0, new_useful=0, withheld=0) for role in ROLES}
    for row in rows:
        result = strata[row["role"]]
        result["files"] += 1
        result[row["status"]] += 1
        if not row["preselected"]:
            continue
        result["old_flags"] += 1
        metrics = row["proposal"]["pixels"]
        useful = metrics["iou"] >= .30 and (metrics["recall"] or 0.) >= .50
        result["old_useful"] += useful
        withheld = row["rotated_pattern"]["selected"]
        result["withheld"] += withheld
        result["new_flags"] += not withheld
        result["new_useful"] += useful and not withheld
    rotations = sum(strata[role]["old_flags"] for role in ROTATIONS)
    removed = sum(strata[role]["withheld"] for role in ROTATIONS)
    promoted = (sum(strata[p]["new_useful"] for p in POSITIVES[:2]) >= 8
                and all(strata[p]["old_useful"] == strata[p]["new_useful"] for p in POSITIVES)
                and removed >= 4 and removed * 2 >= rotations and strata["original"]["new_flags"] == 0
                and not any(strata[r]["assessed"] for r in OUT_OF_SCOPE))
    return {"roles": strata, "preselected_rotations": rotations, "withheld_rotations": removed,
            "advancement": promoted}


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
    ledger = args.output / "scored.json"
    result = measure(args.cached, args.archive, args.output)
    if args.phase == "score":
        base.write(ledger, result)
    else:
        flux.require(base.read(ledger) == result, "Independent native replay differs")
        frozen = lock(args.output)
        if args.receipt:
            base.write(args.receipt, {k: v for k,v in frozen.items() if k != "rows"} | {
                "freeze_sha256": result["freeze_sha256"], "scored_sha256": base.sha(ledger),
                "exact_native_replays": len(result["rows"]), "source_groups": len(frozen["rows"]),
                "generated_sha256": [base.sha(args.output / "generated" / f"{row['key']}.png") for row in frozen["rows"]],
                "measurements": summarize(result["rows"]),
            })
    print(json.dumps(summarize(result["rows"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
