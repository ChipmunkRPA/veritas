# SPDX-License-Identifier: Apache-2.0
"""Compare PSCC's whole-image maps and native overlapping tile maps on JPEG75."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import image_jpeg_boundary as metrics
from app.bench import image_pscc_localization as detector
from scripts.image_local_spectral_peaks import plan
from scripts.image_pscc_localization_panel import _area
from scripts.image_safe_dense_transfer import baseline
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.pscc-native-tiles/1"
QUALITY = 75
TILE = 256
STEP = 192
FADE = 24
BATCH = 4
ARMS = ("whole", "overlap", "repeat")
DEVELOPMENT_GROUPS = 8
CAMERA_GROUPS = 6


def axes(length):
    import numpy as np

    endpoint = length - TILE
    if endpoint < 0:
        raise ValueError("Tile evidence requires at least 256 submitted pixels on each side")
    count = (endpoint + STEP - 1) // STEP + 1
    return list(map(int, np.rint(np.linspace(0, endpoint, count))))


def tile_maps(model, rgb):
    import numpy as np
    import torch

    height, width = rgb.shape[:2]
    windows = [(y, x) for y in axes(height) for x in axes(width)]
    distance = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE))
    taper = np.minimum(distance / FADE, 1.).astype(np.float32)
    weight = np.outer(taper, taper)
    trusted = weight == 1
    summed, weights, maximum, second = (np.zeros((height, width), np.float32) for _ in range(4))
    views = np.zeros((height, width), np.uint8)
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH):
            batch = windows[start:start+BATCH]
            pixels = np.stack([rgb[y:y+TILE, x:x+TILE] for y, x in batch]).astype(np.float32) / 255
            tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous().to(model.device)
            maps = model.head(model.backbone(tensor))[0][:, 0].cpu().numpy()
            if maps.shape != (len(batch), TILE, TILE) or not np.isfinite(maps).all():
                raise ValueError("The model returned an invalid native tile map")
            for (y, x), native in zip(batch, maps, strict=True):
                area = np.s_[y:y+TILE, x:x+TILE]
                weights[area] += weight
                summed[area] += native * weight
                votes = np.where(trusted, native, 0.)
                second[area] = np.maximum(second[area], np.minimum(maximum[area], votes))
                maximum[area] = np.maximum(maximum[area], votes)
                views[area] += trusted
    if not np.all(weights > 0.):
        raise ValueError("The tile grid omitted submitted pixels")
    overlap = summed / weights
    recurrent = np.where(views >= 2, np.minimum(overlap, second), 0.)
    return overlap, recurrent, len(windows), int(np.count_nonzero(views >= 2))


def shape_candidate(field):
    import numpy as np

    from app.detectors.image.jpeg_boundary_localizer import proposals

    side = detector.CELL
    rows, cols = (length // side for length in field.shape)
    cells = field[:rows*side, :cols*side].reshape(rows, side, cols, side).mean(axis=(1, 3))
    score, area = proposals(cells)
    mask = np.zeros(field.shape, dtype=bool)
    mask[:rows*side, :cols*side] = area.repeat(side, axis=0).repeat(side, axis=1)
    return score, mask


def chosen_sources(manifest, count):
    fingerprints = {group["rgb_sha256"] for group in manifest["groups"]}
    return sorted(fingerprints, key=lambda fingerprint: hashlib.sha256(f"{SCHEMA}/{fingerprint}".encode()).digest())[:count]


def bindings(inputs, limit=None):
    assembled = []
    for bank, reference in inputs:
        manifest, groups, identity = plan(bank / "manifest.json")
        selected = chosen_sources(manifest, limit) if limit else sorted(identity["identities"]["rgb_sha256"])
        assembled.append((bank, reference, manifest, groups, identity, selected))
    return assembled


def rules(rows):
    points, preferences = {}, []
    for arm in ARMS:
        maxima = defaultdict(float)
        for row in rows:
            if row["role"] not in metrics.POSITIVE and not metrics.incumbent_flag(row):
                source = row["source_group"]
                maxima[source] = max(maxima[source], row["checks"][arm]["score"] or 0.)
        if len(maxima) < 2:
            raise ValueError("Too few independent control sources")
        points[arm] = sorted(maxima.values(), reverse=True)[1]
        if arm != "whole":
            counts = metrics.measure(rows, arm, points[arm])
            preference = (counts["added_correct"], counts["added_feather"], -counts["added_control_groups"],
                          -counts["candidate_wrong_location"], -counts["added_control"], arm)
            preferences.append(preference)
    return points, max(preferences)[-1]


def summary(rows, points):
    return {arm: metrics.measure(rows, arm, points[arm]) | {
        "same_site_exceeds_controls": sum(row["role"] in metrics.POSITIVE and metrics.useful(row["checks"][arm])
                                          and row["checks"][arm]["ordinary_margin"] is not None
                                          and row["checks"][arm]["ordinary_margin"] > 0 for row in rows),
    } for arm in ARMS}


def evaluate(models, plans):
    import numpy as np
    from PIL import Image

    from scripts.image_cross_quality_fit import load_mask

    rows = []
    for bank, reference, manifest, groups, _, selected in plans:
        current = baseline(reference, bank / "manifest.json")
        grouped = defaultdict(list)
        for item in manifest["images"]:
            if item["quality"] == QUALITY and groups[item["group"]]["rgb_sha256"] in selected:
                grouped[item["group"]].append(item)
        view_counts = Counter()
        for index, (group, deliveries) in enumerate(grouped.items(), 1):
            measured = []
            for item in deliveries:
                body = (bank / item["file"]).read_bytes()
                truth = load_mask(bank, item)
                checks = current[item["file"]]
                digest = hashlib.sha256(body).hexdigest()
                if digest != item["sha256"] or digest != checks["sha256"] or list(truth.shape) != groups[group]["shape"]:
                    raise ValueError("The earlier comparator used different JPEGs or masks")
                if any(result and result["truth"] != item["changed_pixels"] for key, result in checks.items()
                       if key in {"jpeg", "regional"}):
                    raise ValueError("The existing comparison used a different truth extent")
                with Image.open(io.BytesIO(body), formats=["JPEG"]) as source:
                    rgb = np.array(source.convert("RGB"))
                if rgb.shape[:2] != truth.shape:
                    raise ValueError("Source raster and truth coordinates differ")
                whole = models.maps(body)[0]
                native, recurrent, count, covered = tile_maps(models, rgb)
                reading = dict(zip(ARMS, (whole, native, recurrent), strict=True))
                matched = {name: shape_candidate(field) for name, field in reading.items()}
                view_counts.update(files=1, tile_forwards=count)
                measured.append((item, truth, checks, reading, matched, count, covered))
            controls = [scores for item, _, _, scores, _, _, _ in measured if item["role"] not in metrics.POSITIVE]
            for item, truth, checks, reading, matches, count, covered in measured:
                result = {}
                for arm, (score, mask) in matches.items():
                    margin = None
                    if item["role"] in metrics.POSITIVE and controls and mask.any():
                        best_ordinary = max(float(values[arm][mask].mean()) for values in controls)
                        margin = float(reading[arm][mask].mean()) - best_ordinary
                    result[arm] = {"score": score, "ordinary_margin": margin, **_area(mask, truth.nonzero())}
                rows.append({key: item[key] for key in ("file", "sha256", "quality", "role", "mask_sha256")} | {
                    "source_group": groups[group]["rgb_sha256"], "family": groups[group]["family"],
                    "shape": groups[group]["shape"], "current": {key: checks[key] for key in ("jpeg", "regional")},
                    "checks": result, "tile_count": count, "covered_twice_pixels": covered,
                })
            if index % 3 == 0 or index == len(grouped):
                print(f"{bank.name}: {index}/{len(grouped)} groups, {view_counts['files']} JPEGs and "
                      f"{view_counts['tile_forwards']} tiles measured.", flush=True)
    return rows


def method(device):
    import cv2
    import numpy as np
    import torch
    from PIL import __version__ as pillow

    return {"code_sha256": sha(Path(__file__)), "pscc_runner_sha256": sha(Path(detector.__file__)),
            "pscc_revision": detector.REVISION, "asset_sha256": detector.SOURCE_FILES,
            "device": device, "torch": torch.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "pillow": pillow}


def freeze(assets, device, bank, reference, output, held):
    calibration = bindings(((bank, reference),), DEVELOPMENT_GROUPS)
    development_sources = set(calibration[0][-1])
    reserved = [plan(path / "manifest.json")[2] for path in held]
    if any(development_sources.intersection(entry["identities"]["rgb_sha256"]) for entry in reserved):
        raise ValueError("Calibration source overlaps a held photo")
    rows = evaluate(detector.Localizer(assets, device=device, threads=2), calibration)
    points, preferred = rules(rows)
    measured = summary(rows, points)
    save(output, {"schema": SCHEMA, "method": method(device), "calibration": calibration[0][-2],
                  "reference_sha256": sha(reference), "reserved": reserved, "points": points,
                  "preferred": preferred, "summary": measured, "rows": rows})
    print(json.dumps(measured, sort_keys=True), flush=True)


def validate(assets, device, frozen, output, banks, camera):
    lock = read(frozen)
    plans = bindings(banks, CAMERA_GROUPS if camera else None)
    if (lock["schema"] != SCHEMA or lock["method"] != method(device) or rules(lock["rows"]) != (lock["points"], lock["preferred"])
            or any(identity not in lock["reserved"] for *_, identity, _ in plans)):
        raise ValueError("The held run must use the frozen source roster, settings and cutoffs")
    rows = evaluate(detector.Localizer(assets, device=device, threads=2), plans)
    measured = summary(rows, lock["points"])
    save(output, {"schema": SCHEMA, "freeze_sha256": sha(frozen), "camera": camera, "summary": measured, "rows": rows,
                  "manifests": [identity for *_, identity, _ in plans],
                  "references": [sha(reference) for _, reference, *_ in plans]})
    print(json.dumps(measured, sort_keys=True), flush=True)


def review(frozen, held):
    lock, study = read(frozen), read(held)
    if (lock["schema"] != SCHEMA or study["schema"] != SCHEMA or study["freeze_sha256"] != sha(frozen)
            or rules(lock["rows"]) != (lock["points"], lock["preferred"])
            or any(summary(item["rows"], lock["points"]) != item["summary"] for item in (lock, study))):
        raise ValueError("The released rules or counts disagree with the measurement rows")
    print(json.dumps(study["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    calibration = commands.add_parser("freeze")
    for field in ("assets", "bank", "reference", "output"):
        calibration.add_argument(field, type=Path)
    calibration.add_argument("--held", type=Path, action="append", required=True)
    holdout = commands.add_parser("validate")
    for field in ("assets", "frozen", "output"):
        holdout.add_argument(field, type=Path)
    holdout.add_argument("--banks", type=Path, nargs=2, action="append", required=True)
    holdout.add_argument("--camera", action="store_true")
    for command in (calibration, holdout):
        command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    recount = commands.add_parser("review")
    recount.add_argument("frozen", type=Path)
    recount.add_argument("held", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    {"freeze": freeze, "validate": validate, "review": review}[action](**args)
