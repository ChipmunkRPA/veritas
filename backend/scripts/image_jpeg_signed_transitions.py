# SPDX-License-Identifier: Apache-2.0
"""Measure local signed JPEG coefficient transitions on exact generated edit pairs."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import uniform_filter

from app.detectors.image import jpeg_dq
from scripts import image_matched_detail_transfer as geometry
from scripts import prepare_tgif_localization as publisher

SCHEMA = "chipmunk.jpeg-signed-local-transitions/1"
QUALITIES = (75, 90)
FAMILIES = ("firefly", "sd2", "flux-fill")
PARTS = {"fit": (("tgif-pairs-v2", 8),), "calibration": (("tgif-confirmation-v3", 8),),
         "held": (("tgif-confirmation-v4", 8), ("tgif-confirmation-v5", 8))}
FREQUENCIES = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3), (3, 0))
WINDOW, CELL = 6, 8
BOX_SIDES = (4, 6, 8, 10, 12, 16, 20, 24, 30, 36)
ARMS = ("within", "between", "combined")
RIDGE = .1
MAX_CELLS = 48
DESIGN = {"formats": list(QUALITIES), "families": list(FAMILIES), "source_roles": PARTS, "quality_subsampling": "4:2:0",
          "minimum_mask_share": .001, "maximum_mask_share": .08, "mask_box_sides": [32, 360],
          "frequencies": list(map(list, FREQUENCIES)), "window_blocks": WINDOW, "cell_px": CELL,
          "coefficient_states": [-2, -1, 0, 1, 2], "ridge": RIDGE, "maximum_cells_per_group_class": MAX_CELLS,
          "proposal_sizes_in_blocks": list(BOX_SIDES), "support": {"minimum_positive_pixel_share": .50, "min_iou": .30},
          "operating_point": "Strictly above the second-highest ordinary per-source maximum; fixed by delivery and arm.",
          "advance": {"added_feathered_sources": 6, "added_families": 2, "extra_control_groups": 3, "wrong_edit_views": 4},
          "confidence_weight": 0}
require, read, write, digest = geometry.require, geometry.read, geometry.write, geometry.sha


def qualified(root, name, count):
    folder = root / name
    manifest = read(folder / "manifest.json")
    require(manifest["dataset"] == "TGIF and TGIF2" and manifest["revision"] == publisher.REVISION,
            "Source manifest not from pinned publisher release")
    by_id = {(r["family"], r["source_id"], r["label"]): r for r in manifest["items"]}
    selected = []
    for family in FAMILIES:
        options = []
        for (kind, source, label), entry in by_id.items():
            if family != kind or label != 1:
                continue
            path = folder / entry["mask"]
            require(digest(path.read_bytes()) == entry["mask_sha256"], "Exact changed mask differs from prior intake")
            with Image.open(path) as mask:
                share = 1 - mask.convert("L").histogram()[0] / (mask.width * mask.height)
                x, y, right, bottom = mask.getbbox()
                if .001 <= share <= .08 and 32 <= min(right - x, bottom - y) <= max(right - x, bottom - y) <= 360:
                    options.append((source, entry))
        ordered = sorted(options, key=lambda item: digest(f"{SCHEMA}/{name}/{family}/{item[0]}".encode()))
        require(len(ordered) >= count, f"Insufficient exact small masks for {name}/{family}")
        selected.extend((folder, by_id[family, source, 0], row) for source, row in ordered[:count])
    return selected


def sources(args):
    groups = {part: [row for name, count in plans for row in qualified(args.sources, name, count)] for part, plans in PARTS.items()}
    rows = [row for group in groups.values() for row in group]
    for key in ("source_id", "image_sha256"):
        require(len({row[1][key] for row in rows}) == len(rows), "Fit, calibration and held source identities overlap")
    return groups


def load(folder, row):
    data = (folder / row["image"]).read_bytes()
    require(digest(data) == row["image_sha256"], "Exact paired source image differs from its archived digest")
    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        require(image.getexif().get(274, 1) == 1, "Prior source is not upright")
        return np.array(image.convert("RGB"))


def archive_pairs(groups, part):
    for folder, negative, positive in groups[part]:
        original = load(folder, negative)
        edited = load(folder, positive)
        mask_data = (folder / positive["mask"]).read_bytes()
        require(digest(mask_data) == positive["mask_sha256"], "Saved mask differs")
        with Image.open(io.BytesIO(mask_data)) as mask:
            exact = np.asarray(mask.convert("L")) > 0
        require(np.array_equal(exact, np.any(original != edited, axis=2)), "Known native edit differs from exact mask")
        yield negative["image_sha256"], positive["family"], original, edited, exact


def views(groups, part):
    yield from views_from_pairs(archive_pairs(groups, part), part)


def views_from_pairs(pairs, part):
    for source, family, original, edited, exact in pairs:
        image = Image.fromarray(original)
        blur = np.asarray(image.filter(ImageFilter.GaussianBlur(1.4)))
        tone = np.asarray(ImageEnhance.Color(image).enhance(.72)).astype(np.int16)
        tone = np.clip(tone + [12, -6, 7], 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(exact.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = ((1 - np.cos(np.clip(dist / 8, 0, 1) * np.pi)) / 2)[..., None]

        variants = {name: np.rint(original * (1 - alpha) + other * alpha).astype(np.uint8)
                    for name, other in (("local_blur", blur), ("local_tone", tone), ("feather", edited))}
        items = [("ordinary", original), ("local_blur", variants["local_blur"]), ("local_tone", variants["local_tone"])]
        if part == "calibration":
            items.append(("global_contrast", np.asarray(ImageEnhance.Contrast(image).enhance(1.12))))
        else:
            items.extend((("sharp", edited), ("feather", variants["feather"])))
        for role, rgb in items:
            changed = np.any(original != rgb, axis=2)
            for quality in QUALITIES:
                with io.BytesIO() as output:
                    Image.fromarray(rgb).save(output, "JPEG", quality=quality, subsampling=2)
                    body = output.getvalue()
                yield {"source": source, "family": family, "role": role, "quality": quality,
                       "input_sha256": digest(body), "native_mask_sha256": digest(changed.tobytes()),
                       "changed_pixels": int(changed.sum()), "shape": list(changed.shape)}, body, changed


def descriptor(body):
    coeff = jpeg_dq.read_coefficients(body)
    require(coeff is not None, "Frozen JPEG delivery has no coefficients")
    y, x = zip(*FREQUENCIES, strict=True)
    states = np.clip(coeff.y[..., y, x], -2, 2).astype(np.int8) + 2
    # Keep zero/small signs distinct and pool low frequencies to stabilize small windows.
    def table(first, second):
        encoded = 5 * first + second
        return np.stack([np.mean(encoded == value, axis=2) for value in range(25)], axis=2).astype(np.float32)

    within = table(states[..., :-1], states[..., 1:])
    between = (table(states, np.roll(states, -1, axis=0)) + table(states, np.roll(states, -1, axis=1))) / 2
    return {key: np.sqrt(uniform_filter(field, (WINDOW, WINDOW, 1), mode="nearest").clip(0))
            for key, field in (("within", within), ("between", between),
                               ("combined", np.concatenate((within, between), axis=2)))}


def occupancy(mask):
    h, w = (dim // CELL for dim in mask.shape)
    return mask[:h * CELL, :w * CELL].reshape(h, CELL, w, CELL).mean(axis=(1, 3))


def samples(groups):
    by_quality = defaultdict(lambda: defaultdict(list))
    positions = {}
    for row, body, mask in views(groups, "fit"):
        source, q, role = (row[k] for k in ("source", "quality", "role"))
        key = source, q
        pixels = descriptor(body)
        if role == "ordinary":
            by_quality[q][source].append((role, pixels, mask))
            continue
        if role == "sharp":
            covered = occupancy(mask)
            ys, xs = np.where(covered >= .75)
            keep = sorted(zip(ys.tolist(), xs.tolist(), strict=True),
                          key=lambda cell: digest(f"{source}/{cell}".encode()))[:MAX_CELLS]
            if len(keep) >= 3:
                positions[key] = keep
        by_quality[q][source].append((role, pixels, mask))
    for q, photos in by_quality.items():
        for source, entries in photos.items():
            selected = positions.get((source, q))
            if not selected:
                continue
            y, x = zip(*selected, strict=True)
            ordinary = entries[0][1]
            sharp = next(fields for role, fields, _ in entries if role == "sharp")
            feather = next(fields for role, fields, _ in entries if role == "feather")
            blur = next(fields for role, fields, _ in entries if role == "local_blur")
            tone = next(fields for role, fields, _ in entries if role == "local_tone")
            yield q, {arm: (np.concatenate([sharp[arm][y, x], feather[arm][y, x]]),
                            np.concatenate([ordinary[arm][y, x], blur[arm][y, x], tone[arm][y, x]]))
                      for arm in ARMS}


def train(groups):
    stats = defaultdict(list)
    for q, data in samples(groups):
        for arm, (edit, control) in data.items():
            # Matched local controls intentionally include changed pixels.
            delta = edit.mean(axis=0) - control.mean(axis=0)
            center = (edit.mean(axis=0) + control.mean(axis=0)) / 2
            covariance = ((edit - center).T @ (edit - center) / len(edit)
                          + (control - center).T @ (control - center) / len(control)) / 2
            stats[q, arm].append((delta, covariance))
    weights = {q: {} for q in QUALITIES}
    for (q, arm), group in stats.items():
        require(len(group) >= 12, "Insufficient independent exact training footprints")
        delta = np.mean([row[0] for row in group], axis=0)
        covariance = np.mean([row[1] for row in group], axis=0)
        weights[q][arm] = np.linalg.solve(covariance + RIDGE * np.eye(len(delta)), delta).tolist()
    return {str(q): arms for q, arms in weights.items()}


def proposal(values):
    h, w = values.shape
    values = (values - np.median(values)) / max(.012, 1.4826 * np.median(np.abs(values - np.median(values))))
    table = geometry.integral(values)
    score, best = None, None
    for height in BOX_SIDES:
        for width in BOX_SIDES:
            if max(height, width) > 2 * min(height, width) or height + 5 > h or width + 5 > w:
                continue
            t, l = np.mgrid[2:h - height - 1, 2:w - width - 1]
            b, r = t + height, l + width
            inside = geometry.rectangle(table, t, l, b, r)
            outside = geometry.rectangle(table, t - 2, l - 2, b + 2, r + 2) - inside
            area = height * width
            response = (inside / area - outside / (4 * (height + width) + 16)) * (area / 36) ** .12
            ij = np.unravel_index(response.argmax(), response.shape)
            value = float(response[ij])
            if score is None or value > score:
                score, best = value, [int(part[ij]) * CELL for part in (l, t, r, b)]
    return {"score": score, "box": best}


def measure(groups, part, frozen, weights, points=None):
    return measure_views(views(groups, part), part, frozen, weights, points)


def measure_views(deliveries, part, frozen, weights, points=None):
    rows = []
    for (row, body, mask), bound in zip(deliveries, frozen, strict=True):
        require(row == bound, "Frozen encoded input or native mask differs")
        truth = mask if row["role"] in ("sharp", "feather") else np.zeros_like(mask)
        fields = descriptor(body)
        check = {}
        for arm, field in fields.items():
            result = proposal(field @ np.asarray(weights[str(row["quality"])][arm]))
            result["overlap"] = geometry.pixels(result["box"], truth)
            result["useful"] = geometry.useful(result["overlap"])
            if points is not None:
                result["flagged"] = result["score"] is not None and result["score"] > points[str(row["quality"])][arm]
            check[arm] = result
        if part == "held":
            baseline = jpeg_dq.analyse(body)
            region = baseline.localised if baseline is not None else None
            if region is not None:
                x, y, w, h = (getattr(region, key) * CELL for key in ("x", "y", "w", "h"))
                box = [x, y, min(x + w, mask.shape[1]), min(y + h, mask.shape[0])]
            else:
                box = None
            pixel = geometry.pixels(box, truth)
            check["jpeg_compression"] = {"flagged": box is not None, "useful": geometry.useful(pixel),
                                          "box": box, "overlap": pixel}
        rows.append({**row, "checks": check})
        if len(rows) % 48 == 0:
            print(part, len(rows), flush=True)
    return rows


def calibrate(rows):
    points = {str(q): {} for q in QUALITIES}
    for q in QUALITIES:
        for arm in ARMS:
            groups = defaultdict(float)
            for row in rows:
                if row["quality"] == q:
                    groups[row["source"]] = max(groups[row["source"]], row["checks"][arm]["score"] or 0.)
            require(len(groups) == 24, "Calibration source role changed")
            points[str(q)][arm] = sorted(groups.values())[-2]
    return points


def summarize(rows):
    strata, extra, got, false = defaultdict(Counter), defaultdict(set), defaultdict(set), defaultdict(Counter)
    for row in rows:
        family, source, role, quality = (row[k] for k in ("family", "source", "role", "quality"))
        base = row["checks"]["jpeg_compression"]
        for arm, check in row["checks"].items():
            strata[f"{quality}/{family}/{role}/{arm}"].update(total=1, flagged=check["flagged"],
                useful=check["flagged"] and check["useful"], useful_before_point=check["useful"])
            if arm == "jpeg_compression" or not check["flagged"]:
                continue
            key = f"{quality}/{arm}"
            if role not in ("sharp", "feather"):
                if not base["flagged"]:
                    extra[key].add(source)
            elif not check["useful"]:
                false[key]["wrong_edit_views"] += 1
            elif role == "feather" and not base["useful"]:
                got[key].add((family, source))
    results = {}
    for q in QUALITIES:
        for arm in ARMS:
            key = f"{q}/{arm}"
            actual = {"added_feathered_sources": len(got[key]), "added_families": len({fam for fam, _ in got[key]}),
                      "extra_control_groups": len(extra[key]), "wrong_edit_views": false[key]["wrong_edit_views"]}
            bounds = DESIGN["advance"]
            results[key] = {"actual": actual, "passed": all(
                actual[k] >= bound if k.startswith("added") else actual[k] <= bound for k, bound in bounds.items())}
    return {"strata": strata, "outcomes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path, default=Path("backend/data/localization-eval"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    groups = sources(args)
    paths = tuple(Path(module.__file__) for module in (jpeg_dq, publisher, geometry)) + (Path(__file__),)
    frozen = {"schema": SCHEMA, "design": DESIGN, "code": {p.name: digest(p.read_bytes()) for p in paths},
              "manifests": {name: digest((args.sources / name / "manifest.json").read_bytes())
                            for plans in PARTS.values() for name, _ in plans},
              "inputs": {part: [row for row, *_ in views(groups, part)] for part in PARTS}}
    frozen = json.loads(json.dumps(frozen))
    plan = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(plan, frozen)
        print("Frozen", digest(plan.read_bytes()), {part: len(rows) for part, rows in frozen["inputs"].items()}, flush=True)
        return
    require(read(plan) == frozen, "Method or sources differ from the freeze")
    calibration = args.run / "calibration.json"
    if args.stage in ("calibrate", "review"):
        weights = train(groups)
        rows = measure(groups, "calibration", frozen["inputs"]["calibration"], weights)
        calibrated = {"freeze_sha256": digest(plan.read_bytes()), "weights": weights, "points": calibrate(rows), "rows": rows}
        if args.stage == "calibrate":
            write(calibration, calibrated)
            print("Calibration", digest(calibration.read_bytes()), flush=True)
            return
        require(read(calibration) == calibrated, "Calibration replay differs")
    fitted = read(calibration)
    require(fitted["freeze_sha256"] == digest(plan.read_bytes()), "Calibration freeze differs")
    rows = measure(groups, "held", frozen["inputs"]["held"], fitted["weights"], fitted["points"])
    result = {"freeze_sha256": fitted["freeze_sha256"], "calibration_sha256": digest(calibration.read_bytes()),
              "rows": rows, "summary": summarize(rows)}
    output = args.run / "held.json"
    if args.stage == "review":
        require(read(output) == result, "Exact held replay differs")
    else:
        write(output, result)
    print(json.dumps(result["summary"]["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
