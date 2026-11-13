# SPDX-License-Identifier: Apache-2.0
"""Compare full Bayer reconstruction residuals on held local image edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import native_cfa, native_cfa_rotation
from scripts import image_native_generated_grid_transfer as grid
from scripts import image_native_stencil_interpolation_transfer as stencil

ROOT = grid.ROOT
PANELS = ROOT / "docs/evaluation-panels"
STEM = "image-bayer-prediction-transfer-2026-11-13"
SOURCES = {"development": (stencil, "image-native-stencil-interpolation-measurements-2026-11-13.json"),
           "confirmation": (grid, "image-native-generated-grid-measurements-2026-11-13.json")}
SCHEMA = "chipmunk.image-bayer-prediction-transfer/1"
POSITIVE = frozenset(grid.POSITIVES)
UNSUPPORTED = frozenset(grid.OUT_OF_SCOPE)
CELL = 16
PLAN = {
    "development": "13 prior CC0 camera models, cobalt object composites, ordinary processing and untouched controls.",
    "confirmation": "Eight other prior CC0 camera models, yellow object composites, native and promoted deliveries, untouched and matched processing controls. Same source-conditioned generator in both panels. No new training or inference weights.",
    "candidate": "Learn the dominant two-pixel grid from elsewhere in the upload using the incumbent admission. Predict red and blue at nonacquired sites using the actual observed red or blue lattice; use horizontal, vertical and diagonal neighbors. Score only interior, nonclipped, nonflat comparisons. Normalize each channel by the corresponding acquired-sample range. On a native 16-pixel grid, take the weaker local shift of the two channel error rates relative to the nearby median. Report one rectangular proposal at the existing area ranking. No original, crop coordinates or known role at inference.",
    "cutoff": "For each development camera use the highest ordinary response. The strict point is the second-highest source maximum. Count no repeatable grid as unassessed, never as an original-file verdict.",
    "gate": "Against prior four-grid texture-supported stencil-loss with the frozen interpolation veto, gain at least two useful feathered regions and two additional useful regions from the other generated deliveries, preserve at least two thirds of its useful locations per delivery, and flag at most two additional ordinary deliveries with no untouched or unsupported flag. An eligible result advances to another editor, not to AI confidence.",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def cells(array):
    h, w = array.shape
    return array.reshape(h // CELL, CELL, w // CELL, CELL).sum(axis=(1, 3))


def channel_reading(rgb, index, base):
    """Compare all three predicted parities with adjacent native acquisitions."""
    h, w = rgb.shape[:2]
    channel = rgb[..., index].astype(np.int32)
    offset_y, offset_x = divmod(base, 2)
    good = np.zeros((h, w), dtype=bool)
    wrong = np.zeros_like(good)
    for shift_y, shift_x in ((0, 1), (1, 0), (1, 1)):
        y, x = offset_y ^ shift_y, offset_x ^ shift_x
        rows, cols = slice(y + 2, h - 2, 2), slice(x + 2, w - 2, 2)
        samples = [channel[rows.start + dy:rows.stop + dy:2, cols.start + dx:cols.stop + dx:2]
                   for dy in ((-1, 1) if shift_y else (0,)) for dx in ((-1, 1) if shift_x else (0,))]
        low, high = np.minimum.reduce(samples), np.maximum.reduce(samples)
        span = high - low
        error = np.abs(channel[rows, cols] * len(samples) - np.add.reduce(samples))
        good[rows, cols] = (span >= 48) & (low > 1024) & (high < 64511)
        wrong[rows, cols] = good[rows, cols] & (error > len(samples) * np.maximum(12., span * .10))
    count = cells(good)
    return cells(wrong) / np.maximum(count, 1), count >= 72


def measure(rgb):
    if rgb.shape != native_cfa_rotation.SUPPORTED or rgb.dtype != np.uint16:
        return {"status": "unsupported", "score": None, "box": None}
    learned = native_cfa.learn(native_cfa.stencil(rgb))
    if learned is None:
        return {"status": "no_repeatable_stencil", "score": None, "box": None}
    phase = learned[0]
    ratios = []
    good = []
    for channel, parity in ((0, phase), (2, phase ^ 3)):
        rate, valid = channel_reading(rgb, channel, parity)
        local_median = cv2.medianBlur(np.where(valid, rate, np.median(rate[valid]) if valid.any() else 0.).astype(np.float32), 5)
        ratios.append(np.maximum(rate - local_median, 0.))
        good.append(valid)
    score = np.minimum(*ratios)
    usable = good[0] & good[1]
    best = {"score": 0., "box": None}
    # The incumbent integral ranking takes a size list in its own fixed grid.
    energy = cv2.integral(np.where(usable, score, 0.))
    support = cv2.integral(usable.astype(np.uint8))
    for h in (4, 6, 8, 10, 12):
        for w in (4, 6, 8, 10, 12):
            if max(h, w) > 2 * min(h, w):
                continue
            summed = energy[h:, w:] + energy[:-h, :-w] - energy[h:, :-w] - energy[:-h, w:]
            enough = support[h:, w:] + support[:-h, :-w] - support[h:, :-w] - support[:-h, w:]
            ranking = np.where(enough >= .65 * h * w, summed / np.sqrt(h * w), 0.)
            y, x = np.unravel_index(np.argmax(ranking), ranking.shape)
            if ranking[y, x] > best["score"]:
                best = {"score": float(ranking[y, x]), "box": [int(n) * CELL for n in (x, y, x + w, y + h)]}
    return {"status": "assessed", **best}


def freeze(run, earlier, later):
    modules = (stencil, grid, stencil.earlier, native_cfa, native_cfa_rotation)
    config = {"schema": SCHEMA, "plan": PLAN, "code_sha256": sha(Path(__file__)),
              "dependencies": {mod.__name__: sha(Path(mod.__file__)) for mod in modules},
              "sources": {split: {"roster_sha256": sha(mod.ROSTER), "measurements_sha256": sha(PANELS / measurements),
                                  "source_freeze_sha256": sha(local / "freeze.json")}
                          for (split, (mod, measurements)), local in zip(SOURCES.items(), (earlier, later), strict=True)}}
    run.mkdir(mode=0o700)
    write(run / "freeze.json", config)
    return config


def native_rows(cached, archive, local, split, lock):
    module, measurements = SOURCES[split]
    binding = lock["sources"][split]
    if (binding != {"roster_sha256": sha(module.ROSTER), "measurements_sha256": sha(PANELS / measurements),
                    "source_freeze_sha256": sha(local / "freeze.json")}):
        raise ValueError("Frozen camera roster or retained delivery commitment changed")
    source = module.lock(local)
    original_rows = {(row["key"], row["role"]): row for row in read(PANELS / measurements)["rows"]}
    for (camera, rgb), witness in zip(module.sources(cached, archive), source["rows"], strict=True):
        key = camera["key"]
        image = local / "generated" / f"{key}.png"
        generated = read(image.with_suffix(".json"))
        if (data_sha(rgb) != witness["source_rgb_sha256"] or any(witness[k] != v for k, v in camera.items())
                or generated["generated_sha256"] != sha(image) or generated["freeze_sha256"] != binding["source_freeze_sha256"]
                or generated["reference_sha256"] != witness["reference_png_sha256"]):
            raise ValueError("Native RAW or generated pixels do not match the frozen input")
        with Image.open(image, formats=["PNG"]) as handle:
            generated_rgb = np.asarray(handle.convert("RGB"))
        for role, body, truth in module.variants(rgb, camera, generated_rgb):
            prior = original_rows.pop((key, role))
            input_sha, truth_sha = data_sha(body), data_sha(truth)
            target = int(truth.sum())
            if (input_sha, truth_sha, target) != (prior["input_sha256"], prior["truth_sha256"], prior["truth_pixels"]):
                raise ValueError("Native delivery or exact truth differs")
            row = measure(body)
            hit = 0
            if row["box"]:
                x, y, x1, y1 = row["box"]
                hit = int(truth[y:y1, x:x1].sum())
            yield {"camera": key, "role": role, "input_sha256": input_sha, "truth_sha256": truth_sha,
                   "truth_pixels": target, "intersection": hit, **row}
        print(split, key, flush=True)
    if original_rows:
        raise ValueError("Missing frozen source deliveries")


def point(rows):
    maxima = dict.fromkeys({row["camera"] for row in rows}, 0.)
    for row in rows:
        if row["role"] not in stencil.POSITIVE and row["score"] is not None:
            maxima[row["camera"]] = max(maxima[row["camera"]], row["score"])
    return sorted(maxima.values())[-2]


def useful(box, target, hit):
    if not box or not target:
        return False
    x, y, x1, y1 = box
    return 2 * hit >= target and 10 * hit >= 3 * ((x1 - x) * (y1 - y) + target - hit)


def results(rows, cutoff):
    baselines = {(row["key"], row["role"]): row["shifted"] for row in read(PANELS / SOURCES["confirmation"][1])["rows"]}
    counts = {role: Counter(files=0, assessed=0, baseline=0, candidate=0, baseline_useful=0, candidate_useful=0,
                            kept=0, gained=0, lost=0, top_useful=0) for role in grid.ROLES}
    for row in rows:
        role, box, target, hit = (row[key] for key in ("role", "box", "truth_pixels", "intersection"))
        earlier = baselines[row["camera"], role]
        old = bool(earlier and earlier["box"] and earlier["score"] > stencil.POINT
                   and max(map(min, earlier["parity_fit"])) <= stencil.VETO)
        new = bool(box and row["score"] > cutoff)
        old_good = bool(role in POSITIVE and old and useful(earlier["box"], target, earlier["pixels"]["intersection"]))
        top_good = role in POSITIVE and useful(box, target, hit)
        new_good = new and top_good
        counts[role].update(files=1, assessed=row["status"] == "assessed", baseline=old, candidate=new,
                            baseline_useful=old_good, candidate_useful=new_good, kept=old_good and new_good,
                            gained=new_good and not old_good, lost=old_good and not new_good, top_useful=top_good)
    passed = (counts["generated_feathered"]["candidate_useful"] >= counts["generated_feathered"]["baseline_useful"] + 2
              and sum(counts[r]["candidate_useful"] - counts[r]["baseline_useful"] for r in POSITIVE - {"generated_feathered"}) >= 2
              and all(3 * counts[r]["kept"] >= 2 * counts[r]["baseline_useful"] for r in POSITIVE)
              and sum(counts[r]["candidate"] - counts[r]["baseline"] for r in counts.keys() - POSITIVE) <= 2
              and not any(counts[r]["candidate"] for r in UNSUPPORTED | {"original"}))
    return dict(counts), passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "calibrate", "confirm", "replay-development", "replay"))
    for name in ("run", "cached", "archive", "development", "confirmation"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        print(json.dumps(freeze(args.run, args.development, args.confirmation)["sources"], sort_keys=True))
        return
    lock_path = args.run / "freeze.json"
    lock = read(lock_path)
    modules = (stencil, grid, stencil.earlier, native_cfa, native_cfa_rotation)
    if (lock["code_sha256"] != sha(Path(__file__)) or lock["plan"] != PLAN
            or lock["dependencies"] != {mod.__name__: sha(Path(mod.__file__)) for mod in modules}):
        raise ValueError("Frozen check changed")
    if args.phase == "calibrate":
        rows = list(native_rows(args.cached, args.archive, args.development, "development", lock))
        write(args.run / "development.json", {"freeze_sha256": sha(lock_path), "rows": rows, "point": point(rows)})
        return
    dev_path = args.run / "development.json"
    development = read(dev_path)
    if development["freeze_sha256"] != sha(lock_path) or development["point"] != point(development["rows"]):
        raise ValueError("Operating point or freeze differs")
    if args.phase == "replay-development":
        if development["rows"] != list(native_rows(args.cached, args.archive, args.development, "development", lock)):
            raise ValueError("Native development replay changed")
        return
    rows = list(native_rows(args.cached, args.archive, args.confirmation, "confirmation", lock))
    counts, passed = results(rows, development["point"])
    record = {"freeze": lock, "development_sha256": sha(dev_path), "development": development, "confirmation": rows,
              "counts": counts, "advances": passed}
    if args.phase == "confirm":
        write(args.run / "evaluation.json", record)
    elif record != read(args.run / "evaluation.json"):
        raise ValueError("Native confirmation replay changed")
    print(json.dumps({"counts": counts, "advances": passed, "point": development["point"]}, sort_keys=True))


if __name__ == "__main__":
    main()
