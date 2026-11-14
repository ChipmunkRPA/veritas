# SPDX-License-Identifier: Apache-2.0
"""Compare local third-order color and luma relationships on single JPEG uploads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-residual-third-order-transfer/1"
BG_PLAN = "915b08fbdb26a4a49fe9c585eddbb9d495459af7bfb63b8e5d460daa43ea70a0"
FORMATS = ("jpeg75", "jpeg90", "png")
ARMS = ("second_order", "third_order", "joint")
POSITIVE = ("sharp", "feather")
CELL = 8
BOX_SIDES = (4, 6, 8, 10, 12, 16, 20, 24, 30, 36)
FLOORS = (1.5, 1.5, 1.5)
CHANNELS = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 1, 2))
LAGS = ((1, 0, 0, 1), (1, 0, -1, 0), (0, 1, 0, -1),
        (1, 1, 1, -1), (2, 0, 0, 2), (2, 0, -2, 0), (0, 2, 0, -2))
DESIGN = {
    "measurement": "Clipped, locally normalized signed luma and two chroma residuals. On each native eight-pixel cell, estimate pair products and three-point products with fixed 2D offsets, pooled over three by three cells.",
    "comparison": "Second and third order separately and jointly. Fit group-balanced paired covariance for matching native locations, including blur and tone controls. Select the maximum rectangle relative to its ring. Compare a fixed sharp boundary or aligned JPEG luma check.",
    "geometry": "Native 8 px cells, two-cell reference ring, rectangle aspect ratio at most 2 and coverage at most 12% of image area.",
    "threshold": "Strict exceedance of second-highest ordinary calibration-group maximum per delivery and arm.",
    "useful": "At least half the exact edited pixels and a native intersection-over-union of at least 0.30.",
    "advance": {"TGIF_feathered_groups": 6, "BGREAL_feathered_groups": 2, "max_extra_ordinary_groups_each": 3,
                "max_misplaced_each": 4},
    "affects_overall_confidence": False,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def write(path, document):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def useful(counts):
    overlap, target, reported = (counts[key] for key in ("overlap", "target", "reported"))
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (reported + target - overlap))


def inputs(pairs, part):
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    from scripts import image_jpeg_signed_transitions as previous

    for row, body, mask in previous.views_from_pairs(pairs, part):
        yield {**row, "format": f"jpeg{row['quality']}"}, body, mask
    # Native roles use exactly the corresponding pre-JPEG arrays and masks.
    for key, family, original, edited, exact in pairs:
        photo = Image.fromarray(original)
        blur = np.asarray(photo.filter(ImageFilter.GaussianBlur(1.4)))
        tone = np.clip(np.asarray(ImageEnhance.Color(photo).enhance(.72)).astype(np.int16) + [12, -6, 7], 0, 255)
        dist = cv2.distanceTransform(exact.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = ((1 - np.cos(np.clip(dist / 8, 0, 1) * np.pi)) / 2)[..., None]
        variants = {role: np.rint(original * (1 - alpha) + rgb * alpha).astype(np.uint8)
                    for role, rgb in (("local_blur", blur), ("local_tone", tone), ("feather", edited))}
        delivered = {"ordinary": original, "local_blur": variants["local_blur"], "local_tone": variants["local_tone"]}
        if part == "calibration":
            delivered["global_contrast"] = np.asarray(ImageEnhance.Contrast(photo).enhance(1.12))
        else:
            delivered.update(sharp=edited, feather=variants["feather"])
        for role, rgb in delivered.items():
            mask = np.any(original != rgb, axis=2)
            with io.BytesIO() as stream:
                Image.fromarray(rgb).save(stream, "PNG")
                body = stream.getvalue()
            yield {"source": key, "family": family, "role": role, "format": "png",
                   "input_sha256": digest(body), "native_mask_sha256": digest(mask.tobytes()),
                   "changed_pixels": int(mask.sum()), "shape": list(mask.shape)}, body, mask


def features(rgb):
    import cv2
    import numpy as np

    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    colors = (.299 * r + .587 * g + .114 * b, r - g, b - g)
    signals = []
    for plane, floor in zip(colors, FLOORS, strict=True):
        residual = plane - cv2.GaussianBlur(plane, (0, 0), 1.1)
        power = cv2.boxFilter(residual ** 2, cv2.CV_32F, (15, 15), normalize=True)
        signals.append(np.clip(residual / np.sqrt(np.maximum(power, floor ** 2)), -2.5, 2.5))
    height, width = rgb.shape[:2]
    rows, columns = height // CELL, width // CELL

    def tile(plane):
        mean = plane[:rows * CELL, :columns * CELL].reshape(rows, CELL, columns, CELL).mean(axis=(1, 3))
        return cv2.blur(mean, (3, 3))

    # Two reflected pixels at the source edge. No opposite-edge wrap.
    coords = {coords for lag in LAGS for coords in (lag[:2], lag[2:])}
    padded = [np.pad(plane, 2, mode="reflect") for plane in signals]
    shifted = [{(dy, dx): plane[2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
                for dy, dx in coords} for plane in padded]

    pair, triple = [], []
    for a, b, c in CHANNELS:
        for dy, dx, ey, ex in LAGS:
            product = shifted[b][dy, dx] * shifted[c][ey, ex]
            pair.append(tile(product))
            triple.append(tile(signals[a] * product))
    second, third = np.stack(pair, axis=-1), np.stack(triple, axis=-1)
    return (second, third, np.concatenate((second, third), axis=-1))


def decoded(body):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(body), formats=["PNG", "JPEG"]) as image:
        return np.asarray(image.convert("RGB"))


def fit(pairs):
    import numpy as np

    from scripts import image_jpeg_signed_transitions as previous

    statistics = defaultdict(list)
    for pair in pairs:
        locations = np.argwhere(previous.occupancy(pair[-1]) >= .75)
        if len(locations) < 3:
            continue
        yy, xx = locations[np.linspace(0, len(locations) - 1, min(48, len(locations))).astype(int)].T
        groups = defaultdict(dict)
        for row, body, _ in inputs([pair], "fit"):
            groups[row["format"]][row["role"]] = tuple(field[yy, xx] for field in features(decoded(body)))
        for fmt, group in groups.items():
            for index, arm in enumerate(ARMS):
                positive = np.concatenate([group[role][index] for role in POSITIVE])
                ordinary = np.concatenate([group[role][index] for role in ("ordinary", "local_blur", "local_tone")])
                difference = positive.mean(axis=0) - ordinary.mean(axis=0)
                center = (positive.mean(axis=0) + ordinary.mean(axis=0)) * .5
                a, b = positive - center, ordinary - center
                covariance = (a.T @ a / len(a) + b.T @ b / len(b)) * .5
                statistics[fmt, arm].append((difference, covariance))
    weights = defaultdict(dict)
    for fmt in FORMATS:
        for arm in ARMS:
            differences, covariances = zip(*statistics[fmt, arm], strict=True)
            require(len(differences) >= 12, "Insufficient covered training groups")
            weights[fmt][arm] = np.linalg.solve(np.mean(covariances, axis=0) + .035 * np.eye(len(differences[0])),
                                                 np.mean(differences, axis=0)).tolist()
    return weights


def proposal(values):
    import numpy as np

    from scripts import image_matched_detail_transfer as geometry

    table = geometry.integral(values / max(.01, 1.4826 * np.median(np.abs(values - np.median(values)))))
    rows, cols = values.shape
    best = None
    rect = geometry.rectangle
    for h in BOX_SIDES:
        for w in BOX_SIDES:
            area = h * w
            if max(h, w) > 2 * min(h, w) or area > rows * cols * .12 or h + 5 > rows or w + 5 > cols:
                continue
            y, x = np.mgrid[2:rows - h - 1, 2:cols - w - 1]
            core = rect(table, y, x, y + h, x + w)
            ring = rect(table, y - 2, x - 2, y + h + 2, x + w + 2) - core
            score = (core / area - ring / (4 * (h + w) + 16)) * (area / 36) ** .12
            at = np.unravel_index(score.argmax(), score.shape)
            if best is None or float(score[at]) > best["score"]:
                best = {"score": float(score[at]), "box": [int(value[at]) * CELL for value in (x, y, x + w, y + h)]}
    return best


def measure(pairs, part, weights):
    import numpy as np

    from scripts import image_matched_detail_transfer as geometry
    from scripts.image_residual_color_transfer import format_reference

    for count, (key, body, changed) in enumerate(inputs(pairs, part), start=1):
        image = decoded(body)
        truth = changed if key["role"] in POSITIVE else np.zeros_like(changed)
        checks = {}
        for arm, field in zip(ARMS, features(image), strict=True):
            chosen = proposal(field @ np.asarray(weights[key["format"]][arm]))
            checks[arm] = chosen | {"overlap": geometry.pixels(chosen["box"], truth)} if chosen else None
        if part != "calibration":
            base = format_reference(key["format"], body, image)
            bounds = base["box"] if base and base["selected"] else None
            if bounds:
                height, width = truth.shape
                bounds = [bounds[0], bounds[1], min(bounds[2], width), min(bounds[3], height)]
            checks["incumbent"] = {"box": bounds, "overlap": geometry.pixels(bounds, truth)} if base else None
        yield key | {"checks": checks}
        if count % 72 == 0:
            print(f"{part}: measured {count} deliveries", flush=True)


def operating_points(rows):
    groups = {fmt: defaultdict(lambda: defaultdict(float)) for fmt in FORMATS}
    for row in rows:
        for arm, result in row["checks"].items():
            source = groups[row["format"]][arm]
            source[row["source"]] = max(source[row["source"]], result["score"] if result else 0.)
    require(all(set(group) == set(ARMS) and all(len(arm) == 24 for arm in group.values())
                for group in groups.values()), "Calibration roster changed")
    return {fmt: {arm: sorted(by_source.values())[-2] for arm, by_source in arms.items()} for fmt, arms in groups.items()}


def summarize(rows, points):
    measures = defaultdict(Counter)
    gains, flags, gain_families = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        fmt, role, checks = (row[k] for k in ("format", "role", "checks"))
        positive = role in POSITIVE
        base = checks["incumbent"]
        base_flag = bool(base and base["box"])
        base_useful = bool(base and useful(base["overlap"]))
        for arm, selected in checks.items():
            group = (fmt, arm)
            count = measures["/".join(group)]
            expected = bool(selected and useful(selected["overlap"]))
            flagged = bool(selected and selected["box"] and (arm == "incumbent" or selected["score"] > points[fmt][arm]))
            count.update(deliveries=1, assessed=selected is not None, positive=int(positive), ordinary=int(not positive))
            if positive:
                count.update(useful=int(flagged and expected), useful_before_point=int(expected),
                             misplaced=int(flagged and not expected), missed=int(not flagged), incumbent_plus=int(base_useful or flagged and expected))
                if arm != "incumbent" and role == "feather" and flagged and expected and not base_useful:
                    gains[group].add(row["source"])
                    gain_families[group].add(row["family"])
            elif flagged:
                count.update(ordinary_flags=1)
                if arm != "incumbent" and not base_flag:
                    count.update(extra_ordinary_files=1)
                    flags[group].add(row["source"])
    for fmt in FORMATS:
        for arm in ARMS:
            group = (fmt, arm)
            measures["/".join(group)].update(new_feathered_groups=len(gains[group]), new_families=len(gain_families[group]),
                                               new_ordinary_groups=len(flags[group]))
    return dict(measures)


def audit(receipt):
    for part in ("calibration", "tgif", "bgreal"):
        rows = receipt[part]
        planned = receipt["freeze"]["inputs"][part]
        require(all({key: row[key] for key in sealed} == sealed for row, sealed in zip(rows, planned, strict=True)),
                "Frozen deliveries differ")
        for row in rows:
            h, w = row["shape"]
            arms = set(ARMS) if part == "calibration" else {*ARMS, "incumbent"}
            require(set(row["checks"]) == arms, "Missing comparison arm")
            for arm, record in row["checks"].items():
                if record is None:
                    continue
                area = 0
                if record["box"]:
                    left, top, right, bottom = record["box"]
                    require(0 <= left < right <= w and 0 <= top < bottom <= h, "Region outside source")
                    area = (right - left) * (bottom - top)
                counts = record["overlap"]
                target = row["changed_pixels"] if row["role"] in POSITIVE else 0
                require(counts["target"] == target and counts["reported"] == area
                        and max(0, area + target - h * w) <= counts["overlap"] <= min(area, target),
                        "Impossible overlap counts")
                require(arm == "incumbent" or math.isfinite(record["score"]), "Nonfinite evidence")
    points = operating_points(receipt["calibration"])
    require(receipt["points"] == points, "Calibration arithmetic differs")
    measured = {part: summarize(receipt[part], points) for part in ("tgif", "bgreal")}
    require(measured == receipt["summary"], "Held outcomes differ")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "calibrate", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path, default=Path("backend/data/localization-eval"))
    parser.add_argument("--bg", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(audit(read(args.run)), sort_keys=True))
        return

    import cv2
    import numpy as np

    from scripts import image_jpeg_signed_transitions as previous
    from scripts import image_jpeg_small_bg_real_holdout as external

    cv2.setNumThreads(1)
    groups = previous.sources(args)
    pairs = {name: list(previous.archive_pairs(groups, name)) for name in groups}
    pairs["tgif"] = pairs.pop("held")
    bg = read(args.bg / "plan.json")
    require(digest((args.bg / "plan.json").read_bytes()) == BG_PLAN and len(bg["rows"]) == 12, "Unexpected BG-REAL roster")
    pairs["bgreal"] = list(external.pixel_pairs(args.bg, bg))
    all_ids = [row[0] for group in pairs.values() for row in group]
    require(len(set(all_ids)) == len(all_ids) == 108, "Cross-role input identity collision")
    freeze = {"schema": SCHEMA, "design": DESIGN, "runner_sha256": digest(Path(__file__).read_bytes()),
              "inputs": {part: [row for row, *_ in inputs(group, part)] for part, group in pairs.items()}}
    locked = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(locked, freeze)
        for role in ("fit", "calibration", "tgif", "bgreal"):
            print(f"Locked {role}: {len(freeze['inputs'][role])} deliveries", flush=True)
        return
    require(freeze == read(locked), "Score inputs differ from frozen plan")
    calibration = args.run / "calibration.json"
    weights = fit(pairs["fit"])
    require(all(np.isfinite(w).all() for arm in weights.values() for w in arm.values()), "Nonfinite fit")
    if args.phase in {"calibrate", "replay"}:
        rows = list(measure(pairs["calibration"], "calibration", weights))
        record = {"weights": weights, "points": operating_points(rows), "calibration": rows}
        if args.phase == "calibrate":
            write(calibration, record)
            print(json.dumps(record["points"], sort_keys=True))
            return
        require(read(calibration) == record, "Calibration replay differs")
    matched = read(calibration)
    require(weights == matched["weights"], "Fitted weights differ")
    results = {part: list(measure(pairs[part], part, weights)) for part in ("tgif", "bgreal")}
    summary = {part: summarize(rows, matched["points"]) for part, rows in results.items()}
    receipt = {"freeze": freeze, **matched, **results, "summary": summary}
    audit(receipt)
    result_path = args.run / "receipt.json"
    if args.phase == "replay":
        require(read(result_path) == receipt, "Native measures differ from prior execution")
    else:
        write(result_path, receipt)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
