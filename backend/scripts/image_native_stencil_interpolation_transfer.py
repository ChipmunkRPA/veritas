# SPDX-License-Identifier: Apache-2.0
"""Compare camera-stencil loss with and without a local interpolation explanation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import image_native_generated_stencil_transfer as earlier

ROOT, flux = earlier.ROOT, earlier.flux
ROSTER = ROOT / "docs/evaluation-panels/image-native-stencil-interpolation-cc0-roster-2026-11-13.json"
PRIOR = ROOT / "docs/evaluation-panels/image-native-generated-stencil-transfer-2026-11-13.json"
SCHEMA = "chipmunk.image-native-stencil-interpolation-transfer/1"
WIDTH = 128
PATCH = earlier.PATCH
PROMPT = ("Place a small cobalt-blue ceramic diamond tile in the center of the scene. "
          "Keep the existing surfaces, colors, exposure and composition outside the new tile.")
POINT = earlier.DESIGN["arms"]["textured_phase_loss"]
VETO = .70
ROLES = ("untouched", "bilinear_horizontal", "bilinear_diagonal", "bicubic_diagonal", "rotated",
         "blur", "quantized_bilinear", "local_tone", "generated_sharp", "generated_feathered")
POSITIVE = ("generated_sharp", "generated_feathered")
DESIGN = {
    "generator": {**earlier.DESIGN["generator"], "prompt": PROMPT},
    "scope": "Native 16-bit linear developed crops from new camera models. Source-conditioned generated composites use only exact local changed pixels; one model, prompt and fixed compositing workflow.",
    "footprint": "A new source-coordinate 128-pixel square per source, with locations alternating on-grid and half-cell offset. Sharp and 10-pixel feathered alternatives share one generated render.",
    "preselection": "Unchanged texture-supported stencil loss, at its earlier frozen point.",
    "additional_cue": "In the interior of the selected box, fit each signed second-difference parity from two same-parity neighbors on the other lattice. Use nonnegative coefficients, both directions, offsets 0 and 1, red and blue separately. Take the stronger axis's weaker channel norm recovery. Veto strictly above 0.70. Prior pilot consumed as development.",
    "ordinary": "Independent positive/negative controls include one-axis shift, unequal diagonal shift, cubic interpolation, bilinear rotation, blur, a bilinear operation after 8-bit quantization and local tone; same native position and support as the generated replacements.",
    "admission": "Keep at least 80% of prior-arm useful sharp and feathered results separately, retain at least seven useful generated results in each delivery, halve ordinary flags with at most one additional flagged untouched delivery (and no more than one total). Inference support is evaluated separately for each ordinary transformation.",
    "useful": "Native changed-pixel IoU at least 0.30 and changed-pixel recall at least 0.50.",
    "confidence_weight": 0,
}


def source_cases(archive):
    used = set()
    for path in sorted(ROSTER.parent.glob("image-native-cfa-cc0*roster-2026-11-13.json")):
        used.update(row["model"].casefold() for row in earlier.read(path))
    source = earlier.read(ROSTER)
    models = {row["model"].casefold() for row in source}
    flux.require(len(source) == len(models) == 13 and not used & models, "Frozen source groups overlap earlier camera banks")
    for index, row in enumerate(source):
        key = row["id"]
        file = archive / f"sample-{key}{Path(row['url']).suffix.lower()}"
        flux.require(row["license"] == "CC0-1.0" and earlier.sha(file) == row["sha256"], "Unqualified source file")
        x = (9 + (11 * index) % 35) * 32 + (16 if index % 2 else 0)
        y = (8 + (13 * index) % 28) * 32 + (16 if index % 2 else 0)
        yield {"key": key, "camera": row["model"], "source_sha256": row["sha256"], "crop": [x, y, x + PATCH, y + PATCH],
               "seed": 381_021 + index}, row, file


def sources(cached, archive):
    from scripts import image_native_cfa_closure_transfer as native

    wheel = native.raw_source.verified_wheel(cached.resolve() / native.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-stencil-transfer-") as folder:
        native.raw_source.unpack_decoder(wheel, folder)
        sys.path.insert(0, folder)
        try:
            flux.require("rawpy" not in sys.modules, "Isolated RAW decoder required")
            decoder = importlib.import_module("rawpy")
            flux.require(Path(decoder.__file__).resolve().is_relative_to(Path(folder).resolve()), "Wrong decoder")
            for binding, case, path in source_cases(archive):
                yield binding, earlier.render_original(case, path, decoder, native)
        finally:
            sys.path.pop(0)


def freeze(cached, archive, output):
    from PIL import Image

    output.mkdir(mode=0o700)
    for name in ("references", "generated"):
        (output / name).mkdir(mode=0o700)
    rows = []
    for binding, rgb in sources(cached, archive):
        x, y, x1, y1 = binding["crop"]
        reference = output / "references" / f"{binding['key']}.png"
        Image.fromarray(earlier.preview(rgb[y:y1, x:x1])).save(reference)
        rows.append(binding | {"reference_png_sha256": earlier.sha(reference), "source_rgb_sha256": digest(rgb)})
    earlier.write(output / "freeze.json", {"schema": SCHEMA, "design": DESIGN, "runner_sha256": earlier.sha(Path(__file__)),
                  "roster_sha256": earlier.sha(ROSTER), "prior_receipt_sha256": earlier.sha(PRIOR),
                  "prior_runner_sha256": earlier.sha(Path(earlier.__file__)), "rows": rows})


def lock(output):
    frozen = earlier.read(output / "freeze.json")
    flux.require((frozen["design"], frozen["runner_sha256"], frozen["roster_sha256"], frozen["prior_receipt_sha256"],
                  frozen["prior_runner_sha256"]) == (DESIGN, earlier.sha(Path(__file__)), earlier.sha(ROSTER), earlier.sha(PRIOR),
                                                     earlier.sha(Path(earlier.__file__))), "Frozen code or sources differ")
    return frozen


def generate(output, runtime, overlay):
    if overlay:
        site = runtime.resolve() / "lib/python3.12/site-packages"
        sys.path[:0] = [str(overlay.resolve()), str(site)]

        def offline(event, _):
            if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"}:
                raise RuntimeError("Offline generation refused network access")

        sys.addaudithook(offline)
        transformers = importlib.import_module("transformers")
        flux.require(Path(transformers.__file__).is_relative_to(overlay.resolve()), "Expected clean archived dependency copy")
    with patch.multiple(earlier, verified_lock=lock, PROMPT=PROMPT):
        earlier.generate(output, runtime)


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def variants(original, binding, generated):
    import cv2
    import numpy as np

    center = (PATCH - WIDTH) // 2
    x0, y0 = (value + center for value in binding["crop"][:2])
    target = np.s_[y0:y0 + WIDTH, x0:x0 + WIDTH]
    native = original[target]
    empty = np.zeros(original.shape[:2], bool)
    yield "untouched", original, empty
    paint = np.rint((generated[center:center + WIDTH, center:center + WIDTH].astype(np.float32) / 255.) ** 2 * 65535.).astype(np.uint16)
    yy, xx = np.indices((WIDTH, WIDTH))
    edge = np.minimum.reduce((yy + 1, xx + 1, WIDTH - yy, WIDTH - xx))
    alpha = ((1 - np.cos(np.pi * np.minimum(edge / 10., 1))) / 2.)[..., None]

    def shift(body, x, y, method=cv2.INTER_LINEAR):
        return cv2.warpAffine(body, np.array([[1, 0, x], [0, 1, y]], np.float32), (WIDTH, WIDTH), flags=method,
                              borderMode=cv2.BORDER_REFLECT_101)

    controls = {
        "bilinear_horizontal": shift(native, -.3125, 0),
        "bilinear_diagonal": shift(native, .1875, -.625),
        "bicubic_diagonal": shift(native, -.37, .58, cv2.INTER_CUBIC),
        "rotated": cv2.warpAffine(native, cv2.getRotationMatrix2D((WIDTH / 2, WIDTH / 2), 2.75, 1.),
                                 (WIDTH, WIDTH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
        "blur": cv2.GaussianBlur(native, (7, 7), 1.55),
        "quantized_bilinear": ((shift((native >> 8).astype(np.uint8), .68, .22).astype(np.uint16)) * 257),
        "local_tone": np.rint(native * .77 + 4500).astype(np.uint16),
        "generated_sharp": paint,
        "generated_feathered": np.rint(alpha * paint + (1 - alpha) * native).astype(np.uint16),
    }
    for role, replacement in controls.items():
        changed = original.copy()
        changed[target] = replacement
        truth = empty.copy()
        truth[target] = np.any(replacement != native, axis=2)
        flux.require(truth.any() and np.array_equal(np.any(changed != original, axis=2), truth), "Wrong exact edit support")
        yield role, changed, truth


def interpolation_fit(tile, phase):
    import numpy as np
    from scipy.optimize import nnls

    axes = []
    for axis in (0, 1):
        channels = []
        for channel in (0, 2):
            plane = tile[..., channel].astype(np.float64)
            if axis:
                plane = plane.T
            base = (phase % 2 if axis else phase // 2) ^ (channel == 2)
            second = plane[2:] - 2 * plane[1:-1] + plane[:-2]
            first, second = second[base::2], second[1-base::2]
            n = min(len(first), len(second)) - 2
            best = 0.
            if n > 10:
                for anchor, other in ((first, second), (second, first)):
                    for offset in (0, 1):
                        left, right, target = (array[2:-2, 2:-2].ravel() for array in (
                            anchor[:n], anchor[1:n+1], other[offset:offset+n]))
                        norm = np.linalg.norm(target)
                        if norm < 150:
                            continue
                        model = np.stack((left, right), axis=1)
                        coefficients, _ = nnls(model, target)
                        fit = 1. - np.linalg.norm(target - model @ coefficients) / norm
                        best = max(best, float(fit))
            channels.append(best)
        axes.append(channels)
    return axes


def score(cached, archive, output):
    import numpy as np
    from PIL import Image

    frozen = lock(output)
    freeze_hash = earlier.sha(output / "freeze.json")
    rows = []
    for (binding, original), expected in zip(sources(cached, archive), frozen["rows"], strict=True):
        flux.require(all(expected[key] == val for key, val in binding.items()) and expected["source_rgb_sha256"] == digest(original),
                     "Frozen source binding differs")
        image = output / "generated" / f"{binding['key']}.png"
        record = earlier.read(image.with_suffix(".json"))
        flux.require(record["generated_sha256"] == earlier.sha(image) and record["freeze_sha256"] == freeze_hash
                     and record["reference_sha256"] == expected["reference_png_sha256"], "Frozen generator outcome changed")
        with Image.open(image, formats=["PNG"]) as payload:
            generated = np.array(payload.convert("RGB"))
        for role, rgb, truth in variants(original, binding, generated):
            measured = earlier.measure(rgb, truth)
            row = {"key": binding["key"], "role": role, "input_sha256": digest(rgb), "truth_sha256": digest(truth),
                   "truth_pixels": int(truth.sum()), "status": measured["status"], "base": None, "parity_fit": None}
            if measured["status"] == "assessed":
                candidate = measured["arms"]["textured_phase_loss"]
                row["base"] = candidate
                if candidate["box"]:
                    x, y, x1, y1 = candidate["box"]
                    row["parity_fit"] = interpolation_fit(rgb[y+8:y1-8, x+8:x1-8], measured["phase"])
            rows.append(row)
        print("scored", binding["key"], flush=True)
    return {"freeze_sha256": freeze_hash, "rows": rows}


def summary(rows):
    result = {role: Counter(files=0, assessed=0, base_flags=0, base_useful=0, vetoed=0, kept_flags=0, kept_useful=0)
              for role in ROLES}
    for row in rows:
        counts = result[row["role"]]
        counts["files"] += 1
        if row["base"] is None:
            continue
        counts["assessed"] += 1
        candidate = row["base"]
        if candidate["score"] <= POINT:
            continue
        counts["base_flags"] += 1
        measured = candidate["pixels"]
        useful = measured["iou"] >= .30 and (measured["recall"] or 0.) >= .50
        counts["base_useful"] += useful
        veto = max(map(min, row["parity_fit"])) > VETO
        counts["vetoed"] += veto
        counts["kept_flags"] += not veto
        counts["kept_useful"] += useful and not veto
    ordinary = [result[role] for role in ROLES if role not in POSITIVE]
    old = sum(c["base_flags"] for c in ordinary)
    kept = sum(c["kept_flags"] for c in ordinary)
    enough = all(result[role]["kept_useful"] >= 7 and 5 * result[role]["kept_useful"] >= 4 * result[role]["base_useful"]
                 for role in POSITIVE)
    return {"by_role": result, "ordinary_base_flags": old, "ordinary_kept_flags": kept,
            "gate_passed": enough and old > 0 and 2 * kept <= old and result["untouched"]["kept_flags"] <= 1}


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
        generate(args.output, args.runtime, args.overlay)
        return
    import cv2

    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.cached, args.archive, args.output)
        return
    result = score(args.cached, args.archive, args.output)
    ledger = args.output / "scored.json"
    if args.phase == "score":
        earlier.write(ledger, result)
    else:
        flux.require(earlier.read(ledger) == result, "Exact native replay differs")
        if args.receipt:
            frozen = lock(args.output)
            earlier.write(args.receipt, {k: v for k, v in frozen.items() if k != "rows"}
                          | {"freeze_sha256": result["freeze_sha256"], "ledger_sha256": earlier.sha(ledger),
                             "exact_native_replays": len(result["rows"]), "source_groups": len(frozen["rows"]),
                             "generated_output_sha256": [earlier.sha(args.output / "generated" / f"{r['key']}.png") for r in frozen["rows"]],
                             "measurements": summary(result["rows"])})
    print(json.dumps(summary(result["rows"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
