# SPDX-License-Identifier: Apache-2.0
"""Build exact small generated edits on native camera renders and test a stencil cue."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts import image_flux_reference_edit_v271 as flux

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "docs/evaluation-panels/image-native-cfa-cc0-png-roster-2026-11-13.json"
SCHEMA = "chipmunk.image-native-generated-stencil-transfer/1"
CASES = (1079, 1114, 1121, 1325, 1694, 2088)
WIDTH = 128
PATCH = 256
PROMPT = "Add a small matte red paper disk at the exact center of this photograph, with a slight natural shadow. Preserve the framing and the rest of the photo."
ROLES = ("original", "blur", "tone", "even_copy", "fractional_interpolation", "classical_fill", "generated_hard",
         "generated_feathered")
DESIGN = {
    "edit": "Generate a 256×256 RGB reference edit, lift code values to 16-bit by the fixed square transfer, and copy only a centered 128×128 source-coordinate area. Also deliver a feathered exact variant with an 8-pixel cosine ramp.",
    "generator": {"source_revision": flux.REVISION, "size": PATCH, "steps": flux.STEPS, "scheduler": flux.SCHEDULER,
                  "guidance": 1.0, "use_kv_cache": False, "prompt": PROMPT},
    "scope": "CC0, checksum-matched RAW sources from six camera models. These cameras previously contributed conventional edits; these generated outputs and cases are new. Test losses on fixed linear 16-bit RGB, not a common JPEG workflow.",
    "metrics": "A location is useful with 0.30 exact changed-pixel IoU and 0.50 recall. Record ordinary flags by role and all top proposals before the cutoff. Keep a separate prompt-semantic audit.",
    "arms": {"phase_shift": .25, "phase_loss": 2.4346873354224834, "textured_phase_loss": 2.4346873354224834},
    "texture_gate": "For the third arm require fine luma edge activity at least half the median in the surrounding nine-cell neighborhood; this tests whether blur suppression preserves textured synthesized replacements.",
    "admission": "At least four useful generated sharp and four useful generated feathered regions with zero flags on untouched and at most one of 30 ordinary edit controls. Study only; any passing check needs separate generator and camera validation.",
    "confidence_weight": 0,
}


def sha(path):
    return flux.sha(path)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def crop_bounds(index):
    x = (18 + (index * 7) % 24) * 32
    y = (12 + (index * 5) % 18) * 32
    return x, y, x + PATCH, y + PATCH


def render_original(case, path, decoder, prior):
    expected = f"sample-{case['id']}{Path(case['url']).suffix.lower()}"
    flux.require(path.name == expected and sha(path) == case["sha256"], "CC0 RAW bytes differ")
    return prior.raw_render(decoder, path)[0]


def qualified_cases(archive):
    candidates = {row["id"]: row for row in read(ROSTER)}
    for index, key in enumerate(CASES):
        case = candidates[key]
        path = archive / f"sample-{key}{Path(case['url']).suffix.lower()}"
        flux.require(case["license"] == "CC0-1.0" and sha(path) == case["sha256"], "CC0 or SHA-256 failed")
        yield {"key": key, "source_sha256": case["sha256"], "camera": case["model"],
               "crop": list(crop_bounds(index)), "seed": 271_004 + index}, case, path


def raw_views(cached, archive):
    from scripts import image_native_cfa_closure_transfer as prior

    wheel = prior.raw_source.verified_wheel(cached.resolve() / prior.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-generated-cfa-") as directory:
        prior.raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            flux.require("rawpy" not in sys.modules, "Fresh isolated RAW decoder process required")
            raw = importlib.import_module("rawpy")
            flux.require(Path(raw.__file__).resolve().is_relative_to(Path(directory).resolve()), "Wrong RAW decoder")
            for binding, case, path in qualified_cases(archive):
                yield binding, render_original(case, path, raw, prior)
        finally:
            sys.path.pop(0)


def preview(rgb):
    import numpy as np

    return np.rint(np.sqrt(rgb.astype(np.float32) / 65535.) * 255.).astype(np.uint8)


def freeze(cached, archive, output):
    import numpy as np
    from PIL import Image

    output.mkdir(mode=0o700)
    for name in ("references", "generated"):
        (output / name).mkdir(mode=0o700)
    sources = []
    for case, rgb in raw_views(cached, archive):
        x0, y0, x1, y1 = case["crop"]
        patch = preview(rgb[y0:y1, x0:x1])
        reference = output / "references" / f"{case['key']}.png"
        Image.fromarray(patch).save(reference)
        sources.append(case | {"reference_png_sha256": sha(reference),
                               "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest()})
    lock = {"schema": SCHEMA, "design": DESIGN, "runner_sha256": sha(Path(__file__)), "roster_sha256": sha(ROSTER),
            "generator_inventory_sha256": flux.ACQUISITION_SHA,
            "prior_measure_sha256": sha(ROOT / "backend/scripts/image_native_cfa_closure_transfer.py"), "rows": sources}
    write(output / "freeze.json", lock)


def verified_lock(output):
    lock = read(output / "freeze.json")
    flux.require(lock["design"] == DESIGN and lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(ROSTER),
                 "Frozen method, configuration or source roster changed")
    return lock


def generate(output, runtime):
    import platform
    import re
    import zipfile
    from importlib import metadata

    lock = verified_lock(output)
    site = runtime.resolve() / "lib/python3.12/site-packages"
    flux.require(platform.python_version() == "3.12.6" and platform.machine() == "arm64", "Expected reviewed Mac Python")
    expected = dict(re.findall(r"^([a-z0-9_.-]+)==([^\s]+)", flux.LOCK.read_text(), re.M))
    distributions = (metadata.PathDistribution(path) for path in site.glob("*.dist-info") if path.is_dir())
    installed = {row.metadata["Name"].lower().replace("_", "-"): row.version for row in distributions}
    flux.require(installed == expected and len(installed) == 56, "Restored runtime distributions differ")
    flux.require(sha(flux.LOCK) == flux.LOCK_SHA and sha(flux.WHEEL) == flux.WHEEL_SHA
                 and sha(runtime / "THIRD_PARTY_NOTICES/MFLUX-MIT.txt") == flux.NOTICE_SHA, "Runtime source or license changed")
    with zipfile.ZipFile(flux.WHEEL) as wheel:
        names = {name for name in wheel.namelist() if name.startswith("mflux/") and not name.endswith("/")}
        observed = {str(path.relative_to(site)) for path in (site / "mflux").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._")}
        flux.require(len(names) == 1059 and names == observed and all(wheel.read(name) == (site / name).read_bytes() for name in names),
                     "MFlux source bytes differ from reviewed wheel")
    flux.verify_bundle(flux.BUNDLE, flux.inventory())
    sys.path.insert(0, str(site))

    def offline(event, _):
        if event.startswith("socket.") and event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}:
            raise RuntimeError("Offline generation refused network access")

    sys.addaudithook(offline)
    import mlx.core as mx
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

    for name in ("mflux", "numpy", "PIL", "mlx.core"):
        flux.require(Path(sys.modules[name].__file__).resolve().is_relative_to(site), f"Unexpected dependency origin: {name}")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024**2)
    model = Flux2KleinEdit(model_path=str(flux.BUNDLE), small_text_encoder=False, small_decoder=False)
    flux.require(model.bits == 4, "Expected pinned 4-bit model")
    for row in lock["rows"]:
        key = row["key"]
        reference = output / "references" / f"{key}.png"
        image = output / "generated" / f"{key}.png"
        record = output / "generated" / f"{key}.json"
        flux.require(sha(reference) == row["reference_png_sha256"], "Pinned source patch changed")
        if record.is_file():
            flux.require(read(record)["generated_sha256"] == sha(image), "Existing generated output changed")
            continue
        flux.require(not image.exists(), "Unjournaled generated output requires review")
        result = model.generate_image(seed=row["seed"], prompt=PROMPT, num_inference_steps=flux.STEPS,
                                      width=PATCH, height=PATCH, guidance=1.0, scheduler=flux.SCHEDULER,
                                      use_kv_cache=False, image_paths=[reference]).image
        flux.require(result.mode == "RGB" and result.size == (PATCH, PATCH), "Expected exact generated RGB patch")
        result.save(image)
        write(record, {"key": key, "seed": row["seed"], "reference_sha256": sha(reference), "generated_sha256": sha(image),
                       "freeze_sha256": sha(output / "freeze.json")})
        print("generated", key, flush=True)
        mx.clear_cache()


def variants(original, case, generated):
    import cv2
    import numpy as np

    x0, y0, _, _ = case["crop"]
    x0, y0 = x0 + (PATCH - WIDTH) // 2, y0 + (PATCH - WIDTH) // 2
    target = np.s_[y0:y0+WIDTH, x0:x0+WIDTH]
    source = original[target]
    empty = np.zeros(original.shape[:2], bool)
    yield "original", original, empty
    center = (PATCH - WIDTH) // 2
    paint = np.rint((generated[center:center+WIDTH, center:center+WIDTH].astype(np.float32) / 255.) ** 2 * 65535.).astype(np.uint16)
    y, x = np.indices((WIDTH, WIDTH))
    edge = np.minimum.reduce((y+1, x+1, WIDTH-y, WIDTH-x))
    alpha = ((1 - np.cos(np.pi * np.minimum(edge / 8., 1))) / 2.)[..., None]
    mask = np.zeros((PATCH, PATCH), np.uint8)
    mask[center:center+WIDTH, center:center+WIDTH] = 255
    context = original[y0-center:y0+WIDTH+center, x0-center:x0+WIDTH+center]
    fill = np.stack([cv2.inpaint(channel.astype(np.float32), mask, 4, cv2.INPAINT_TELEA)
                     for channel in context.transpose(2, 0, 1)], axis=-1)[center:center+WIDTH, center:center+WIDTH]
    controls = {
        "blur": cv2.GaussianBlur(source, (5, 5), 1.1),
        "tone": np.rint(source * .88 + 2050).astype(np.uint16),
        "even_copy": original[160:160+WIDTH, 128:128+WIDTH],
        "fractional_interpolation": cv2.warpAffine(source, np.array([[1, 0, .5], [0, 1, .375]], np.float32),
                                                  (WIDTH, WIDTH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
        "classical_fill": np.clip(np.rint(fill), 0, 65535).astype(np.uint16),
        "generated_hard": paint,
        "generated_feathered": np.rint(alpha * paint + (1-alpha) * source).astype(np.uint16),
    }
    for role, replacement in controls.items():
        final = original.copy()
        final[target] = replacement
        truth = empty.copy()
        truth[target] = np.any(replacement != source, axis=2)
        flux.require(truth.any() and np.array_equal(np.any(final != original, axis=2), truth), "Exact edit mask differs")
        yield role, final, truth


def measure(rgb, truth):
    import cv2
    import numpy as np

    from scripts import image_native_cfa_closure_transfer as prior

    rates = prior.stencil(rgb)
    reference = prior.learn(rates)
    if reference is None:
        return {"status": "no_repeatable_stencil", "arms": {}}
    phase, expected, eligible = reference
    observed = rates[..., phase]
    strongest_other = np.max(rates[..., [p for p in range(4) if p != phase]], axis=-1)
    luma = rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    fine = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), .85))
    energy = (prior.cell_sums(fine) / prior.CELL ** 2).astype(np.float32)
    # An odd 3×3 local median followed by 9×9 Gaussian is defined for float32 on OpenCV.
    nearby = cv2.GaussianBlur(cv2.medianBlur(energy, 3), (9, 9), 2.)
    phase_loss = expected - observed - .12
    candidates = {
        "phase_shift": prior.best_region(strongest_other - observed - .20, eligible & (strongest_other >= .55)),
        "phase_loss": prior.best_region(phase_loss, eligible),
        "textured_phase_loss": prior.best_region(np.where(energy >= np.maximum(nearby * .50, 1.), phase_loss, 0.), eligible),
    }
    for report in candidates.values():
        selected = np.zeros_like(truth)
        if report["box"]:
            x0, y0, x1, y1 = report["box"]
            selected[y0:y1, x0:x1] = True
        report["pixels"] = prior.mask_metrics(selected, truth)
    return {"status": "assessed", "phase": phase, "arms": candidates}


def score(cached, archive, output, lock):
    import numpy as np
    from PIL import Image

    result = []
    for (case, original), witness in zip(raw_views(cached, archive), lock["rows"], strict=True):
        flux.require({k: witness[k] for k in case} == case
                     and witness["source_rgb_sha256"] == hashlib.sha256(np.ascontiguousarray(original)).hexdigest(), "Frozen pixels differ")
        image = output / "generated" / f"{case['key']}.png"
        record = read(output / "generated" / f"{case['key']}.json")
        flux.require(record["generated_sha256"] == sha(image) and record["freeze_sha256"] == sha(output / "freeze.json")
                     and record["reference_sha256"] == witness["reference_png_sha256"], "Generated output binding differs")
        with Image.open(image, formats=["PNG"]) as handle:
            generated = np.array(handle.convert("RGB"))
        for role, rgb, truth in variants(original, case, generated):
            result.append({"case": case["key"], "role": role, "input_sha256": hashlib.sha256(rgb).hexdigest(),
                           "truth_sha256": hashlib.sha256(truth).hexdigest(), "truth_pixels": int(truth.sum()),
                           **measure(rgb, truth)})
        print("scored", case["key"], flush=True)
    return {"freeze_sha256": sha(output / "freeze.json"), "rows": result}


def summary(rows):
    results = {}
    for arm, point in DESIGN["arms"].items():
        grouped = {role: Counter(files=0, supported=0, useful=0, top_useful=0, flags=0) for role in ROLES}
        for row in rows:
            result = grouped[row["role"]]
            result["files"] += 1
            if arm not in row["arms"]:
                continue
            result["supported"] += 1
            reading = row["arms"][arm]
            pixels = reading["pixels"]
            useful = pixels["iou"] >= .30 and (pixels["recall"] or 0.) >= .50
            flag = reading["score"] > point
            result.update(useful=int(useful and flag), top_useful=int(useful), flags=int(flag))
        results[arm] = grouped
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "generate", "score", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cached", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "generate":
        generate(args.output, args.runtime)
        return
    import cv2

    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.cached, args.archive, args.output)
        return
    lock = verified_lock(args.output)
    result = score(args.cached, args.archive, args.output, lock)
    ledger = args.output / "scored.json"
    if args.phase == "score":
        write(ledger, result)
    else:
        flux.require(read(ledger) == result, "Independent native source or score replay differs")
        if args.receipt:
            write(args.receipt, {k: v for k, v in lock.items() if k != "rows"}
                  | {"freeze_sha256": sha(args.output / "freeze.json"), "scored_sha256": sha(ledger),
                     "exact_replay_rows": len(result["rows"]), "sources": len(lock["rows"]),
                     "bound_generated_outputs": [read(args.output / "generated" / f"{row['key']}.json")
                                                 for row in lock["rows"]],
                     "measurements": summary(result["rows"])})
    print(json.dumps(summary(result["rows"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
