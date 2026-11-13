# SPDX-License-Identifier: Apache-2.0
"""Compare normalized local patch spectra on paired native image edits."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from PIL import __version__ as pillow
from scipy.spatial.distance import cdist

from app.detectors.image import sharp_boundary
from scripts import image_matched_detail_transfer as common
from scripts import image_reference_delivery_envelope_lama as lama
from scripts import image_reference_png_jpeg_stability as flux

SCHEMA = "chipmunk.image-local-residual-covariance/1"
CELL, PATCH, PATCH_STRIDE = 16, 4, 2
SIGMAS = (.9, 2.4)
FEATURE_SLICES = ((0, 1), (1, 4), (4, 8), (8, 16))
CONTEXT_FLOORS = np.array([12., .25, .25, .18])
ARM_NAMES = ("luma_spectrum", "color_spectrum", "agreement")
MIN_MATCHES, MAX_MATCHES, EXCLUDE_RADIUS, MAX_CONTEXT_DISTANCE = 8, 18, 6, 2.5
FORMATS = ("png", "jpeg75")
ROLES = ("ordinary", "sharp", "feather", "local_blur", "local_tone")
CONTROL_COUNT = 10
EDIT_COUNTS = {"lama": 12, "flux": 6}
GATE = {"minimum_added_lama": 2, "minimum_added_flux": 1, "maximum_added_control_groups": 2,
        "maximum_wrong_edit_views": 2}
DESIGN = {
    "cell": CELL, "patch": PATCH, "stride": PATCH_STRIDE, "luma_lowpass_sigma": SIGMAS[0],
    "coarse_lowpass_sigma": SIGMAS[1], "spectrum_bins": list(map(list, FEATURE_SLICES)),
    "coarse_context_floors": CONTEXT_FLOORS.tolist(), "comparison": "trace-normalized patch covariance spectra",
    "reference": "distant cells matched on coarse luma, variation, and edge direction",
    "minimum_matches": MIN_MATCHES, "maximum_matches": MAX_MATCHES, "exclusion_cells": EXCLUDE_RADIUS,
    "maximum_context_distance": MAX_CONTEXT_DISTANCE, "minimum_per_feature_mad": .18,
    "rectangle_sides_cells": list(common.SIDES), "minimum_usable_inside": .8,
    "minimum_usable_surround": 8, "minimum_supporting_cell_fraction": .6, "supporting_cell_point": .75,
    "surround_penalty": .65, "area_power": .18, "feather_width_px": 12,
    "calibration": "Separate per encoding and arm, strictly above second-highest group maximum.",
    "useful": {"iou": .30, "recall": .50}, "gate": GATE, "confidence_weight": 0,
}
need, read, write, sha, pin = common.require, common.read, common.write, common.sha, common.file_sha


def descriptor(tile):
    patches = np.lib.stride_tricks.sliding_window_view(tile, (PATCH, PATCH), axis=(-2, -1))
    patches = patches[..., ::PATCH_STRIDE, ::PATCH_STRIDE, :, :]
    vectors = patches.reshape(*tile.shape[:2], -1, PATCH ** 2)
    vectors = vectors - vectors.mean(axis=-2, keepdims=True)
    covariance = np.einsum("...ki,...kj->...ij", vectors, vectors, optimize=True)
    energy = covariance.trace(axis1=-2, axis2=-1)
    regularized = covariance / np.maximum(energy[..., None, None], 1e-6)
    eigen = np.maximum(np.linalg.eigvalsh(regularized)[..., ::-1], 0.)
    return np.stack([np.log(eigen[..., begin:end].sum(axis=-1) + .008) for begin, end in FEATURE_SLICES], axis=-1)


def context_and_spectra(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    y = .299 * r + .587 * g + .114 * b

    def smooth(values, sigma):
        return cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)

    coarse = common.windows(smooth(y, SIGMAS[1]))
    sy, sx = (np.abs(np.diff(coarse, axis=axis)).mean(axis=(-2, -1)) for axis in (-2, -1))
    context = np.stack((coarse.mean(axis=(-2, -1)), np.log(coarse.std(axis=(-2, -1)) + .75),
                        np.log(sx + sy + .2), (sx - sy) / (sx + sy + .2)), axis=-1)
    spectra = []
    for plane in (y, r - g, b - g):
        residual = common.windows(plane - smooth(plane, SIGMAS[0]))
        spectra.append(descriptor(residual))
    luma = common.windows(y)
    valid = ((luma < 3) | (luma > 252)).mean(axis=(-2, -1)) < .15
    valid &= luma.std(axis=(-2, -1)) >= 1.5
    return context, np.concatenate(spectra, axis=-1), valid


def deviations(context, spectra, valid):
    active = np.flatnonzero(valid.ravel())
    maps = np.full((*valid.shape, len(ARM_NAMES)), np.nan)
    if len(active) <= MAX_MATCHES:
        return maps
    rowcol = np.column_stack(np.unravel_index(active, valid.shape))
    distances = cdist(rowcol, rowcol, "chebyshev")
    context = context.reshape(-1, context.shape[-1])[active]
    scale = np.maximum(np.percentile(context, 75, axis=0) - np.percentile(context, 25, axis=0), CONTEXT_FLOORS)
    match = cdist(context / scale, context / scale)
    match[(distances <= EXCLUDE_RADIUS) | (match > MAX_CONTEXT_DISTANCE)] = np.inf
    # Explicit tie-break for equal context distances.
    match += np.arange(len(active)) * (1e-9 / len(active))
    nearest = np.argpartition(match, MAX_MATCHES - 1, axis=1)[:, :MAX_MATCHES]
    support = np.isfinite(np.take_along_axis(match, nearest, axis=1))
    observed = spectra.reshape(-1, spectra.shape[-1])[active]
    flat = maps.reshape(-1, len(ARM_NAMES))
    for index in np.flatnonzero(support.sum(axis=1) >= MIN_MATCHES):
        refs = observed[nearest[index, support[index]]]
        median = np.median(refs, axis=0)
        spread = np.maximum(1.4826 * np.median(np.abs(refs - median), axis=0), .18)
        score = np.minimum(np.abs(observed[index] - median) / spread, 5.)
        luma = score[:len(FEATURE_SLICES)].mean()
        color = score[len(FEATURE_SLICES):].mean()
        flat[active[index]] = luma, color, min(luma, color)
    return maps


def proposals(maps):
    height, width, channels = maps.shape
    observed = np.isfinite(maps[..., 0])
    values = np.nan_to_num(maps)
    table = common.integral(values)
    support = common.integral((values > .75).astype(np.int64))
    count = common.integral(observed.astype(np.int64))
    best = [{"score": None, "box": None, "inside": None, "surround": None, "assessed_cells": int(observed.sum())}
            for _ in ARM_NAMES]
    rect = common.rectangle
    for top, left, bottom, right, area in common.shapes(height, width):
        n = rect(count, top, left, bottom, right)
        ring_n = rect(count, top - 1, left - 1, bottom + 1, right + 1) - n
        total = rect(table, top, left, bottom, right)
        ring = rect(table, top - 1, left - 1, bottom + 1, right + 1) - total
        inside = total / np.maximum(n[..., None], 1)
        surround = ring / np.maximum(ring_n[..., None], 1)
        agreeing = rect(support, top, left, bottom, right)
        usable = (5 * n[..., None] >= 4 * area) & (ring_n[..., None] >= 8) & (5 * agreeing >= 3 * n[..., None])
        ranks = np.where(usable, (inside - .65 * surround) * (area / 9) ** .18, -np.inf)
        for channel in range(channels):
            index = np.unravel_index(ranks[..., channel].argmax(), ranks.shape[:2])
            value = float(ranks[(*index, channel)])
            if np.isfinite(value) and (best[channel]["score"] is None or value > best[channel]["score"]):
                best[channel].update(score=value, box=[int(item[index] * CELL) for item in (left, top, right, bottom)],
                                     inside=float(inside[(*index, channel)]), surround=float(surround[(*index, channel)]))
    return best


def sources(args, part):
    if part == "calibration":
        # Public roster and exact original JPEGs previously used for structural checks.
        replay = flux.public
        from unittest.mock import patch

        slots = (0, 2, 5, 6, 8, 10, 11, 29, 34, 37)
        with patch.object(replay, "SLOTS", slots):
            rows = replay.roster()
        for row in rows:
            _, image = replay._source(args.public / f"source-{row['slot']:02d}.jpg", row)
            yield "calibration", row["source_sha256"], row["author"].casefold(), np.asarray(image), None
        return
    path = args.lama / "qualified-private.json"
    roster = read(path)
    need(pin(path) == read(args.lama / "summary-v225.json")["qualified_private_sha256"], "LaMa qualification receipt changed")
    groups = sorted(roster["qualified"],
                    key=lambda row: sha(f"{SCHEMA}/{row['source']['flickr_photo_id']}".encode()))
    owners = set()
    for row in groups[:EDIT_COUNTS["lama"]]:
        source = row["source"]
        author = source["photo_attribution"]["Author"].casefold()
        need(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/"
             and author not in owners, "LaMa source contributor or licence differs")
        owners.update((source["flickr_owner"].casefold(), author))
        original, edited, _ = lama.source_pixels(Path(roster["cache"]), row)
        yield "lama", row["measurement"]["files"]["original"]["sha256"], author, original, edited
    earlier = read(args.flux_earlier)["cases"]
    held, _ = flux._licensed_flux({"cases": earlier})
    cache = read(args.flux_cache)
    by_digest = {row.get("source_jpeg_sha256"): row for row in cache["sources"]}
    for source, original, edited, _ in held:
        author = by_digest[source]["author"].casefold()
        need(author not in owners, "FLUX source contributor overlaps LaMa roster")
        owners.add(author)
        yield "flux", source, author, original, edited


def variants(original, edited, slot):
    blurred = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(1.5)))
    toned = np.clip(original.astype(np.int16) + [-16, 11, 18], 0, 255).astype(np.uint8)
    if edited is None:
        height, width = original.shape[:2]
        y, x = height // 4 + slot, width // 4 + slot
        mask = np.zeros((height, width), dtype=bool)
        mask[y:y + height // 4, x:x + width // 4] = True
    else:
        mask = (original != edited).any(axis=2)
    feather = common.panel.BASE.feather
    yield "ordinary", original, False
    if edited is not None:
        yield "sharp", edited, True
        yield "feather", feather(original, edited, mask), True
    yield "local_blur", feather(original, blurred, mask), False
    yield "local_tone", feather(original, toned, mask), False


def views(args, part):
    seen = set()
    for slot, (family, source, author, original, edited) in enumerate(sources(args, part)):
        need(author not in seen, "A contributor appears in multiple groups")
        seen.add(author)
        for role, rgb, positive in variants(original, edited, slot):
            changed = np.any(rgb != original, axis=-1)
            mask = changed if positive else np.zeros_like(changed)
            for encoding in FORMATS:
                if encoding == "png":
                    body = common.panel.BASE.png(rgb)
                else:
                    body = flux.public.original.encode(Image.fromarray(rgb), 75)
                with Image.open(io.BytesIO(body)) as source_image:
                    decoded = np.asarray(source_image.convert("RGB"))
                need(384 <= min(mask.shape) <= max(mask.shape) <= 768 and decoded.shape == rgb.shape,
                     "Native case outside the frozen geometry")
                yield {"family": family, "source": source, "author": author, "role": role, "format": encoding,
                       "input_sha256": sha(body), "truth_sha256": sha(mask.tobytes()),
                       "ordinary_processing_sha256": sha(changed.tobytes()) if not positive else None,
                       "shape": list(mask.shape), "target": int(mask.sum())}, decoded, mask


def measure(args, part, lock, points=None):
    for (row, rgb, mask), expected in zip(views(args, part), lock["views"][part], strict=True):
        need(row == expected, "Native source input or truth differs from the freeze")
        field = deviations(*context_and_spectra(rgb))
        responses = {}
        for arm, item in zip(ARM_NAMES, proposals(field), strict=True):
            item["pixels"] = common.pixels(item["box"], mask)
            item["flag"] = bool(points and item["score"] is not None and item["score"] > points[row["format"]][arm])
            responses[arm] = item
        if points is not None:
            h, w = mask.shape
            item = sharp_boundary.measure(rgb) if row["format"] == "png" and sharp_boundary.supported_size(w, h) else None
            box = list(item.box) if item else None
            responses["current_sharp"] = {"score": item.score if item else None, "box": box,
                "flag": bool(item and item.score > sharp_boundary.REVIEW_POINT), "pixels": common.pixels(box, mask)}
        yield {**row, "responses": responses}


def operating_points(rows):
    sealed = {}
    for encoding in FORMATS:
        sealed[encoding] = {}
        for arm in ARM_NAMES:
            maxima = defaultdict(float)
            for row in rows:
                if row["format"] == encoding:
                    key = row["source"]
                    maxima[key] = max(maxima[key], row["responses"][arm]["score"] or 0.)
            need(len(maxima) == CONTROL_COUNT, "Calibration source roster changed")
            sealed[encoding][arm] = sorted(maxima.values())[-2]
    return sealed


def summarize(rows):
    strata = defaultdict(Counter)
    added = {arm: defaultdict(set) for arm in ARM_NAMES}
    wrong = Counter()
    for row in rows:
        for arm, item in row["responses"].items():
            useful = common.useful(item["pixels"])
            key = f"{row['family']}/{row['format']}/{row['role']}/{arm}"
            strata[key].update(views=1, assessed=item["score"] is not None,
                               flagged=item["flag"], useful_before_point=useful, useful=useful and item["flag"])
            base = row["responses"]["current_sharp"]
            if arm == "current_sharp" or not item["flag"] or base["flag"]:
                continue
            if not row["target"]:
                added[arm]["control_groups"].add(row["source"])
            elif not useful:
                wrong[arm] += 1
            elif row["role"] == "feather":
                added[arm][row["family"]].add(row["source"])
    outcomes = {}
    for arm, groups in added.items():
        actual = {"minimum_added_lama": len(groups["lama"]), "minimum_added_flux": len(groups["flux"]),
                  "maximum_added_control_groups": len(groups["control_groups"]), "maximum_wrong_edit_views": wrong[arm]}
        outcomes[arm] = {"actual": actual, "passed": all(
            actual[key] >= limit if key.startswith("minimum") else actual[key] <= limit for key, limit in GATE.items())}
    return {"strata": {key: dict(value) for key, value in strata.items()}, "outcomes": outcomes}


def freeze(args):
    modules = (common, common.panel.BASE, flux, flux.public, flux.public.original, lama, sharp_boundary)
    paths = (Path(__file__), *(Path(module.__file__) for module in modules))
    rows = {part: [row for row, *_ in views(args, part)] for part in ("calibration", "held")}
    counts = Counter((row["family"], row["format"], row["role"]) for group in rows.values() for row in group)
    need(all(count == (CONTROL_COUNT if family == "calibration" else EDIT_COUNTS[family])
             for (family, _, _), count in counts.items()) and len(counts) == 26, "Incomplete frozen view strata")
    calibration = {row["author"] for row in rows["calibration"]}
    held = {row["author"] for row in rows["held"]}
    need(calibration.isdisjoint(held), "Calibration and held authors overlap")
    return {"schema": SCHEMA, "design": DESIGN, "views": rows, "code": {path.name: pin(path) for path in paths},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held", "review"))
    parser.add_argument("run", type=Path)
    data = Path("backend/data/independent-evaluation")
    for name, path in (("public", "image-affine-native-scale-2026-11-11"),
                       ("lama", "image-bg-real-acquisition-v225"),
                       ("flux-earlier", "image-residual-patch-held-2026-11-11/plan.json"),
                       ("flux-cache", "image-fresh-source-cache-v39/private-groups.json")):
        parser.add_argument("--" + name, type=Path, default=data / path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    plan = args.run / "freeze.json"
    computed = freeze(args)
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(plan, computed)
        print("Frozen", pin(plan), {key: len(rows) for key, rows in computed["views"].items()}, flush=True)
        return
    need(read(plan) == computed, "Native source inputs or method changed")
    calibration = args.run / "calibration.json"
    if args.stage == "calibrate":
        rows = list(measure(args, "calibration", computed))
        write(calibration, {"freeze_sha256": pin(plan), "points": operating_points(rows), "rows": rows})
        print("Calibration sealed", pin(calibration), flush=True)
        return
    cal = read(calibration)
    need(cal["freeze_sha256"] == pin(plan) and cal["points"] == operating_points(cal["rows"]), "Calibration changed")
    if args.stage == "review":
        need(cal["rows"] == list(measure(args, "calibration", computed)), "Calibration replay differs")
    rows = []
    for row in measure(args, "held", computed, cal["points"]):
        rows.append(row)
        if len(rows) % 30 == 0:
            print(f"Scored {len(rows)}/{len(computed['views']['held'])}", flush=True)
    result = {"freeze_sha256": pin(plan), "calibration_sha256": pin(calibration), "points": cal["points"],
              "summary": summarize(rows), "rows": rows}
    held = args.run / "held.json"
    if args.stage == "review":
        need(result == read(held), "Held native replay differs")
    else:
        write(held, result)
    print(json.dumps(result["summary"]["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
