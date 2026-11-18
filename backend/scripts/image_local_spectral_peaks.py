# SPDX-License-Identifier: Apache-2.0
"""Test whether compact off-grid spectral peaks locate single-JPEG edits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from app.bench.image_jpeg_boundary import POSITIVE, QUALITIES, incumbent_flag, incumbent_hit, useful
from scripts import image_region_rank_inputs as files

SCHEMA = "chipmunk.image-local-spectral-peaks/1"
SIDES = (40, 64, 96, 128)
STEP = 16
MIN_ANGLE = 35
MIN_RADIUS, MAX_RADIUS = .10, .32
ORDINARY_GROUP_ALLOWANCE = 1


def score_windows(gray, side):
    import numpy as np
    from scipy.fft import rfft2
    from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

    stride = np.lib.stride_tricks.sliding_window_view(gray, (side, side))[::STEP, ::STEP]
    rows, columns = stride.shape[:2]
    taper = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    fy, fx = np.meshgrid(np.fft.fftfreq(side), np.fft.rfftfreq(side), indexing="ij")
    radius = np.hypot(fx, fy)
    jpeg_x = np.abs(fx * 8 - np.rint(fx * 8)) <= 4 / side
    jpeg_y = np.abs(fy * 8 - np.rint(fy * 8)) <= 4 / side
    eligible = ((radius >= MIN_RADIUS) & (radius <= MAX_RADIUS) & (fx >= 1 / side)
                & (np.abs(fy) >= 1 / side) & ~(jpeg_x & jpeg_y))
    minimum_cross = math.sin(math.radians(MIN_ANGLE))
    scores = np.empty((rows, columns), dtype=np.float32)
    for start in range(0, rows, 2):
        windows = stride[start:start + 2].reshape(-1, side, side) * taper
        variance = np.sqrt((windows * windows).sum(axis=(1, 2)))
        spectrum = np.log1p(np.abs(rfft2(windows, axes=(1, 2))) / np.maximum(1., variance[:, None, None]))
        local = gaussian_filter(spectrum, (0, 1.1, 1.1), mode=("nearest", "wrap", "nearest"))
        peak = (spectrum == maximum_filter(spectrum, size=(1, 3, 3), mode=("nearest", "wrap", "nearest"))) & eligible
        excess = np.where(peak, np.maximum(0., spectrum - local), 0.)
        primary = excess.reshape(len(windows), -1).argmax(axis=1)
        a, b = np.unravel_index(primary, eligible.shape)
        separated = np.abs(fx[None] * fy[a, b, None, None] - fy[None] * fx[a, b, None, None]) >= (
            minimum_cross * radius[None] * radius[a, b, None, None])
        second = np.where(separated, excess, 0.).max(axis=(1, 2))
        scores[start:start + 2] = second.reshape(-1, columns)
    immediate = median_filter(scores, size=3, mode="nearest")
    neighborhood = median_filter(scores, size=7, mode="nearest")
    return np.maximum(0., np.minimum(scores, immediate) - .8 * neighborhood)


def candidate(data):
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
        if not 256 <= min(image.size) <= max(image.size) <= 768:
            raise ValueError("A JPEG is outside the specified input profile")
        image.draft("L", image.size)
        gray = np.asarray(image, dtype=np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.25)
    proposals = []
    for side in SIDES:
        scores = score_windows(residual, side)
        y, x = np.unravel_index(scores.argmax(), scores.shape)
        proposals.append((float(scores[y, x]), [int(x)*STEP, int(y)*STEP, side, side]))
    return max(proposals, key=lambda item: item[0])


def plan(manifest):
    data = files.read(manifest)
    groups = {row["tag"]: row for row in data["groups"]}
    identities = {key: sorted({row[key] for row in groups.values()}) for key in ("source", "rgb_sha256")}
    return data, groups, {"manifest_sha256": files.sha(manifest), "identities": identities}


def measure(bank, reference, manifest, groups):
    import cv2

    from app.detectors.image import jpeg_dq, regional
    from app.detectors.image.adapters.regional import repeat_components
    from app.detectors.image.regional_support import readout
    from scripts import image_region_rank_reference
    from scripts.image_cross_quality_fit import load_mask

    cv2.setNumThreads(1)
    with reference.open() as stream:
        headers = json.loads(next(stream))
        current = {row["file"]: row for row in map(json.loads, stream)}
    expected = {"manifest_sha256": files.sha(bank / "manifest.json"),
                "code_sha256": files.sha(Path(image_region_rank_reference.__file__)),
                "regional_sha256": regional.CHECKPOINT_SHA256, "jpeg_sha256": files.sha(Path(jpeg_dq.__file__)),
                "regional_rule_sha256": files.sha(Path(readout.__code__.co_filename)), "repeat_jpeg90": True,
                "repeat_rule_sha256": files.sha(Path(repeat_components.__code__.co_filename))}
    if expected != headers or set(current) != {row["file"] for row in manifest["images"]}:
        raise ValueError("Existing checks were not measured on these exact files with the current rules")
    rows, locations = [], {}
    for index, entry in enumerate(manifest["images"], 1):
        data = (bank / entry["file"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        truth = load_mask(bank, entry)
        height, width = truth.shape
        if sha != entry["sha256"] or sha != current[entry["file"]]["sha256"] or list(truth.shape) != groups[entry["group"]]["shape"]:
            raise ValueError("The JPEG or truth delivered to the current checks changed")
        if sha not in locations:
            locations[sha] = candidate(data)
        score, box = locations[sha]
        x, y, w, h = box
        if x + w > width or y + h > height:
            raise ValueError("The candidate is outside submitted coordinates")
        rows.append({key: entry[key] for key in ("file", "role", "quality", "sha256", "mask_sha256")} | {
            "source": groups[entry["group"]]["source"], "family": groups[entry["group"]]["family"],
            "shape": [height, width], "current": {key: current[entry["file"]][key] for key in ("jpeg", "regional")},
            "candidate": {"score": score, "box": box, "selected": w*h,
                          "intersection": int(truth[y:y + h, x:x + w].sum()), "truth": int(truth.sum())},
        })
        if index % 48 == 0 or index == len(manifest["images"]):
            print(f"Measured {index}/{len(manifest['images'])} JPEGs.", flush=True)
    return rows


def cutoffs(rows):
    points = {}
    for quality in QUALITIES:
        maxima = {}
        for row in rows:
            if row["quality"] == quality and row["role"] not in POSITIVE and not incumbent_flag(row):
                source = row["source"]
                maxima[source] = max(row["candidate"]["score"], maxima.get(source, 0.))
        if len(maxima) <= ORDINARY_GROUP_ALLOWANCE:
            raise ValueError("Insufficient source groups for a fixed cutoff")
        points[str(quality)] = sorted(maxima.values(), reverse=True)[ORDINARY_GROUP_ALLOWANCE]
    return points


def summarize(rows, points):
    summary = {}
    for quality in QUALITIES:
        counts = Counter()
        ordinary_sources = set()
        for row in rows:
            if row["quality"] != quality:
                continue
            positive = row["role"] in POSITIVE
            current_hit, current_flag = incumbent_hit(row), incumbent_flag(row)
            proposal = row["candidate"]
            selected, good = proposal["score"] > points[str(quality)], useful(proposal)
            counts.update(edited=positive, ordinary=not positive,
                          current_correct=positive and current_hit, current_ordinary=not positive and current_flag,
                          best_useful=positive and good, additional_correct=positive and selected and good and not current_hit,
                          misplaced=positive and selected and not good, additional_ordinary=not positive and selected and not current_flag,
                          extra_unchanged=row["role"] == "ordinary" and selected and not current_flag)
            if not positive and selected and not current_flag:
                ordinary_sources.add(row["source"])
        summary[str(quality)] = dict(counts) | {"additional_ordinary_sources": len(ordinary_sources)}
    return summary


def freeze(bank, reference, output, held):
    manifest, groups, development = plan(bank / "manifest.json")
    holdout = [plan(path / "manifest.json")[2] for path in held]
    for record in holdout:
        if any(set(identities) & set(record["identities"][key]) for key, identities in development["identities"].items()):
            raise ValueError("A held source overlaps the cutoff split")
    rows = measure(bank, reference, manifest, groups)
    points = cutoffs(rows)
    report = summarize(rows, points)
    files.save(output, {"schema": SCHEMA, "code_sha256": files.sha(Path(__file__)), "development": development,
                        "holdout": holdout, "reference_sha256": files.sha(reference), "points": points,
                        "summary": report, "rows": rows})
    print(report, flush=True)


def validate(frozen, output, banks):
    lock = files.read(frozen)
    manifest_records = [(bank, baseline, *plan(bank / "manifest.json")) for bank, baseline in banks]
    if (lock["schema"] != SCHEMA or lock["points"] != cutoffs(lock["rows"]) or lock["code_sha256"] != files.sha(Path(__file__))
            or any(record not in lock["holdout"] for _, _, _, _, record in manifest_records)):
        raise ValueError("The comparison has new held files, code or cutoffs")
    rows = []
    for bank, baseline, manifest, groups, _ in manifest_records:
        rows.extend(measure(bank, baseline, manifest, groups))
    report = summarize(rows, lock["points"])
    files.save(output, {"schema": SCHEMA, "freeze_sha256": files.sha(frozen), "summary": report, "rows": rows,
                        "banks": [{"manifest_sha256": record["manifest_sha256"], "reference_sha256": files.sha(baseline)}
                                  for _, baseline, _, _, record in manifest_records]})
    print(report, flush=True)


def review(frozen, report):
    lock, held = (files.read(path) for path in (frozen, report))
    if (lock["schema"] != SCHEMA or held["schema"] != SCHEMA or held["freeze_sha256"] != files.sha(frozen)
            or lock["points"] != cutoffs(lock["rows"]) or lock["summary"] != summarize(lock["rows"], lock["points"])
            or held["summary"] != summarize(held["rows"], lock["points"])):
        raise ValueError("The reported cutoff or counts do not match the ledger")
    print(held["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    calibrate = actions.add_parser("freeze")
    for field in ("bank", "reference", "output"):
        calibrate.add_argument(field, type=Path)
    calibrate.add_argument("--held", type=Path, action="append", required=True)
    compare = actions.add_parser("validate")
    compare.add_argument("frozen", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--banks", type=Path, nargs=2, action="append", required=True)
    recount = actions.add_parser("review")
    recount.add_argument("frozen", type=Path)
    recount.add_argument("report", type=Path)
    parsed = vars(parser.parse_args())
    action = parsed.pop("action")
    {"freeze": freeze, "validate": validate, "review": review}[action](**parsed)
