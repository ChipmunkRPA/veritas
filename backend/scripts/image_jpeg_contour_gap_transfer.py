# SPDX-License-Identifier: Apache-2.0
"""Test whether two compatible line ends locate small JPEG edits without a reference."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts import image_bg_real_acquisition_v221 as source
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_jpeg_small_bg_real_holdout as f5
from scripts import image_jpeg_texture_rank_transfer as recent
from scripts import image_residual_encoder_transfer as common
from scripts.review_image_residual_encoder_transfer import overlap

SCHEMA = "chipmunk.image-jpeg-contour-gap-transfer/1"
OLD_FREEZE = "7ecd5f065260606e193a837513022ae79f9337740cfe0b8a9a7e968b56f86417"
RECENT_FREEZE = "35e3b219ba6908d3ae4ad213049800db205541c4b560fc42e48527754842de35"
STEP = 8
GAPS = (40, 64, 88, 112, 152)
FLANK = np.arange(6, 31, 4)
ANGLES = np.arange(8) * np.pi / 8
PROBE = np.arange(-4, 5, 2)
MIN_EDGE, MAX_MIDDLE = 3.0, .45
POSITIVE, QUALITIES = common.POSITIVE, common.QUALITIES
DESIGN = {
    "quality": list(QUALITIES), "subsampling": "4:2:0", "native_step_px": STEP, "gap_lengths_px": list(GAPS),
    "external_flank_sample_distances_px": FLANK.tolist(), "orientations": len(ANGLES), "normal_offsets_px": PROBE.tolist(),
    "preprocessing": "luminance Gaussian sigma 1.6 then signed directional Sobel with fixed polarity",
    "supports": "seven directional samples per flank, at least six strong at one normal position, 50% minimum cross-flank strength ratio",
    "min_flank_strength": MIN_EDGE, "maximum_middle_flank_ratio": MAX_MIDDLE,
    "score": "geometric mean of the two signed directional supports minus signed interior support and fixed scale penalty",
    "geometry": "native square centered on the broken segment, side length gap plus 16 source pixels",
    "calibration": "12 distinct originals, second-largest source maximum per JPEG quality, all nuisance renders considered",
    "reference": "current aligned JPEG luma compression localization on identical input bytes",
    "useful": {"iou": .30, "recall": .50}, "confidence_weight": 0,
    "advance": {"useful_feathered_by_both_families": 3, "additional_ordinary_groups": 2, "wrong_edit_locations": 4},
}
require, read, write, sha = common.require, common.read, common.write, common.sha


def block_sample(field, x, y):
    return cv2.remap(field, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def traces(field, centers, tangent, normal, offsets):
    xx, yy = centers
    (tx, ty), (nx, ny) = tangent, normal
    length = np.asarray(offsets)[None, :, None, None]
    width = PROBE[:, None, None, None]
    x = xx + tx * length + nx * width
    y = yy + ty * length + ny * width
    return block_sample(field, x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])).reshape(x.shape)


def best_gap(rgb):
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.6)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    best = None
    for gap in GAPS:
        half = gap // 2
        extent = half + int(FLANK[-1]) + int(PROBE[-1]) + 2
        x, y = np.meshgrid(np.arange(extent, width - extent, STEP), np.arange(extent, height - extent, STEP))
        if not x.size:
            continue
        middle_offsets = np.arange(-half + 5, half - 4, STEP)
        for angle in ANGLES:
            tx, ty = np.cos(angle), np.sin(angle)
            nx, ny = -ty, tx
            field = gx * nx + gy * ny
            left, right, middle = (traces(field, (x, y), (tx, ty), (nx, ny), values) for values in
                                   (-(half + FLANK), half + FLANK, middle_offsets))
            index = int(np.ceil(.75 * middle.shape[1])) - 1
            inner = np.partition(np.abs(middle), index, axis=1)[:, index]
            # Allow one weak sample at each end but keep a single offset and polarity across both flanks.
            for polarity in (1., -1.):
                a = np.partition(left * polarity, 1, axis=1)[:, 1]
                b = np.partition(right * polarity, 1, axis=1)[:, 1]
                lower, upper = np.minimum(a, b), np.maximum(a, b)
                valid = (lower >= MIN_EDGE) & (lower >= .5 * upper) & (inner <= MAX_MIDDLE * lower)
                response = np.sqrt(np.maximum(0, a) * np.maximum(0, b)) - inner - .006 * gap
                ranked = np.where(valid, response, -np.inf)
                p, j, i = np.unravel_index(ranked.argmax(), ranked.shape)
                score = float(ranked[p, j, i])
                if np.isfinite(score) and (best is None or score > best["score"]):
                    mid_x, mid_y = int(x[j, i]), int(y[j, i])
                    radius = half + 8
                    best = {"box": [mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius], "score": score,
                            "gap_pixels": gap, "angle": float(angle), "normal_offset": int(PROBE[p]), "polarity": int(polarity),
                            "flank_minima": [float(a[p, j, i]), float(b[p, j, i])], "middle_p75": float(inner[p, j, i])}
    return best


def overlay(rgb, shape, identity):
    """Represent conventional annotation that can genuinely interrupt a scene line."""
    height, width = rgb.shape[:2]
    if shape is None:
        rng = np.random.default_rng(int(sha(identity.encode())[:12], 16))
        x, y = rng.uniform(.23, .76, 2)
        w, h = rng.uniform(.08, .15, 2)
        left, right = round((x - w) * width), round((x + w) * width)
        top, bottom = round((y - h) * height), round((y + h) * height)
    else:
        y, x = np.where(shape)
        left, right, top, bottom = x.min(), x.max(), y.min(), y.max()
    picture = Image.fromarray(rgb)
    pen = ImageDraw.Draw(picture)
    pen.rounded_rectangle((left, top, right, bottom), radius=min(9, (bottom - top) // 3), fill=(241, 239, 228))
    if right - left >= 45 and bottom - top >= 30:
        pen.text((left + 8, top + 8), "NOTE", fill=(30, 35, 43))
    return np.asarray(picture)


def prior_groups(args):
    folder = args.earlier
    plan = read(folder / "plan.json")
    for case, (_, family, original, edited, mask) in zip(plan["rows"], f5.pixel_pairs(folder, plan), strict=True):
        identity = {"recipient": case["source_id"], "generator": "lama" if family == "lama" else "sd15",
                    "credit": case["attribution"], "original_sha256": case["source"]["sha256"], "size": case["shape"]}
        yield identity, {"original": original, "edited": edited}, mask


def later_groups(args):
    plan = read(args.later / "plan.json")
    require(roster.sha(args.later / "plan.json") == RECENT_FREEZE, "Previously qualified later cohort changed")
    for item, native, truth in recent.held_inputs(args.later, plan):
        identity = {"recipient": item["recipient"], "generator": item["generator"], "credit": item["photo_attribution"],
                    "original_sha256": item["original"]["sha256"], "size": item["size"]}
        yield identity, native, truth


def groups(args, phase):
    if phase == "calibration":
        for item, native, truth in common.frozen_inputs(args, phase, read(args.encoder / "freeze.json")):
            yield {**item, "credit": item["attribution"]}, native, truth
    elif phase == "retained_a":
        yield from prior_groups(args)
    else:
        yield from later_groups(args)


def identity(item, native, truth):
    profile = item["credit"]
    owner, photo = roster.landing_identity(profile["OriginalLandingURL"])
    author = roster.canonical_owner(profile["AuthorProfileURL"].rstrip("/").rsplit("/", 1)[1])
    require(profile["License"] == roster.PHOTO_LICENCE and profile["Author"], "Missing source photo credit")
    picture = source.fingerprint(b"", native["original"])
    return {"recipient": item["recipient"], "generator": item["generator"], "credit": profile, "owner": owner,
            "author_profile": author, "photo": photo, "original_sha256": item["original_sha256"],
            "rgb_sha256": picture["rgb_sha256"], "size": picture["size"], "sketch": picture["sketch"],
            "truth_sha256": sha(truth.tobytes()) if truth is not None else None}


def plan(args):
    require(roster.sha(args.encoder / "freeze.json") == OLD_FREEZE, "Retained calibration release changed")
    phases = ("calibration", "retained_a", "retained_b")
    parts = {phase: [identity(item, native, truth) for item, native, truth in groups(args, phase)] for phase in phases}
    require([len(parts[phase]) for phase in phases] == [12, 12, 16], "Source denominators changed")
    all_rows = [r for part in parts.values() for r in part]
    for key in ("recipient", "photo", "original_sha256", "rgb_sha256"):
        require(len({item[key] for item in all_rows}) == 40, f"Source duplicate: {key}")
    creators = [{author for item in parts[phase] for author in (item["owner"], item["author_profile"], item["credit"]["Author"].casefold())}
                for phase in phases]
    require(sum(map(len, creators)) == len(set.union(*creators)), "A source photographer appears in multiple roles")
    require(not any(source.near(a, b) for n, a in enumerate(all_rows) for b in all_rows[:n]), "Retained source thumbnail overlap")
    paths = (args.encoder / "freeze.json", args.earlier / "plan.json", args.earlier / "acquired.json",
             args.later / "plan.json", args.later / "intake.json")
    return {"schema": SCHEMA, "design": DESIGN, "sources": parts, "code_sha256": roster.sha(Path(__file__)),
            "source_commitments": [roster.sha(path) for path in paths]}


def assess(args, locked, phase, points=None):
    rows = []
    for (item, native, exact), check in zip(groups(args, phase), locked["sources"][phase], strict=True):
        require(identity(item, native, exact) == check, "Native source does not match freeze")
        ordinary, generated = common.authored(native, exact, item["recipient"])
        ordinary["annotation_overlay"] = overlay(native["original"], exact, item["recipient"])
        for role, rgb in (ordinary | generated).items():
            truth = np.any(native["original"] != rgb, axis=2) if role in POSITIVE else np.zeros(rgb.shape[:2], dtype=bool)
            for quality in QUALITIES:
                body = common.encode(rgb, quality)
                delivered = common.decode(body)
                candidate = best_gap(delivered)
                result = {"contour": candidate or {"box": None, "score": None}}
                if points is not None:
                    base = common.comparison.format_reference(f"jpeg{quality}", body, delivered)
                    box = base["box"] if base and base["selected"] else None
                    result["current_luma"] = {"box": box}
                for arm, hit in result.items():
                    hit["overlap"] = overlap(truth, hit["box"])
                    hit["useful"] = common.geometry.useful(hit["overlap"])
                    if points is not None:
                        point = points[str(quality)]
                        hit["flagged"] = bool(hit["box"]) and (arm == "current_luma" or (point is not None and hit["score"] > point))
                rows.append({"cohort": phase, "generator": item["generator"], "source": item["recipient"], "role": role, "quality": quality,
                             "input_sha256": sha(body), "truth_sha256": sha(truth.tobytes()), "result": result})
        print(phase, len(rows), flush=True)
    return rows


def thresholds(rows):
    maxima = defaultdict(dict)
    for row in rows:
        group, source = maxima[str(row["quality"])], row["source"]
        value = row["result"]["contour"]["score"]
        group[source] = max(group.get(source, -np.inf), value if value is not None else -np.inf)
    require(all(len(group) == 12 for group in maxima.values()), "Expected 12 calibration groups per quality")
    return {quality: (float(value) if np.isfinite(value) else None) for quality, group in maxima.items()
            for value in [sorted(group.values())[-2]]}


def summarize(rows):
    strata, overall = defaultdict(Counter), {}
    for quality in QUALITIES:
        ordinary, additional = defaultdict(set), set()
        wrong = 0
        for row in (x for x in rows if x["quality"] == quality):
            role, family, identity = row["role"], row["generator"], row["source"]
            for arm, hit in row["result"].items():
                strata[f"{quality}/{row['cohort']}/{family}/{role}/{arm}"].update(
                    views=1, ranked=bool(hit["box"]), useful_before_point=hit["useful"], flagged=hit["flagged"],
                    useful=hit["flagged"] and hit["useful"])
                if role not in POSITIVE and hit["flagged"]:
                    ordinary[arm].add(identity)
                if arm == "contour" and role in POSITIVE and hit["flagged"]:
                    wrong += not hit["useful"]
                    if role == "feathered" and hit["useful"] and not row["result"]["current_luma"]["useful"]:
                        additional.add((family, identity))
        actual = {"added_feathered": len(additional), "families": len({family for family, _ in additional}),
                  "extra_ordinary_groups": len(ordinary["contour"] - ordinary["current_luma"]), "wrong_edit_locations": wrong}
        overall[str(quality)] = {"actual": actual, "passes": len(additional) >= 3 and actual["families"] == 2 and
                                 actual["extra_ordinary_groups"] <= 2 and wrong <= 4}
    return {"outcomes": overall, "strata": strata}


def audit(args, locked, rows, phase, points=None):
    records = {(row["source"], row["role"], row["quality"]): row for row in rows}
    require(len(records) == len(rows), "Repeated JPEG record")
    checks = 0
    for (item, native, exact), bound in zip(groups(args, phase), locked["sources"][phase], strict=True):
        require(identity(item, native, exact) == bound, "Native source does not match freeze")
        negatives, positives = common.authored(native, exact, item["recipient"])
        negatives["annotation_overlay"] = overlay(native["original"], exact, item["recipient"])
        for role, rgb in (negatives | positives).items():
            truth = np.any(native["original"] != rgb, axis=2) if role in POSITIVE else np.zeros(rgb.shape[:2], dtype=bool)
            for quality in QUALITIES:
                row = records.pop((item["recipient"], role, quality))
                require(sha(common.encode(rgb, quality)) == row["input_sha256"] and sha(truth.tobytes()) == row["truth_sha256"],
                        "File bytes or exact truth changed")
                for arm, hit in row["result"].items():
                    metric = overlap(truth, hit["box"])
                    useful = metric["iou"] is not None and metric["iou"] >= .3 and metric["recall"] >= .5
                    require(hit["overlap"] == metric and hit["useful"] == useful, "Native overlap differs")
                    if points is not None:
                        point = points[str(quality)]
                        flagged = bool(hit["box"]) and (arm == "current_luma" or (point is not None and hit["score"] > point))
                        require(hit["flagged"] == flagged, "Flag differs from frozen point")
                    checks += 1
    require(not records, "Unexpected JPEG records")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("encoder", "earlier", "later", "archive", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.rosters = roster.DATA
    cv2.setNumThreads(1)
    profile = plan(args)
    folder = args.output
    if args.stage == "freeze":
        folder.mkdir(mode=0o700)
        write(folder / "freeze.json", profile)
        print(roster.sha(folder / "freeze.json"), {key: len(value) for key, value in profile["sources"].items()}, flush=True)
        return
    require(profile == read(folder / "freeze.json"), "Frozen design or sources changed")
    if args.stage == "run":
        start = time.monotonic()
        calibration = assess(args, profile, "calibration")
        write(folder / "calibration.json", calibration)
        points = thresholds(calibration)
        write(folder / "points.json", points)
        print("points", points, flush=True)
        held = [row for phase in ("retained_a", "retained_b") for row in assess(args, profile, phase, points)]
        write(folder / "held.json", held)
        result = {"schema": SCHEMA, "design": DESIGN, "points": points, "held_rows": len(held), "calibration_rows": len(calibration),
                  "groups": {key: Counter(r["generator"] for r in group) for key, group in profile["sources"].items()},
                  "commitments": {name: roster.sha(folder / name) for name in ("freeze.json", "calibration.json", "points.json", "held.json")},
                  "code_sha256": profile["code_sha256"], "elapsed_s": round(time.monotonic() - start, 2), **summarize(held)}
        write(folder / "result.json", result)
        print(result["outcomes"], flush=True)
        return
    result = read(folder / "result.json")
    require(all(roster.sha(folder / name) == digest for name, digest in result["commitments"].items()), "Stored result commitment changed")
    calibration, held = (read(folder / name) for name in ("calibration.json", "held.json"))
    points = read(folder / "points.json")
    require(points == thresholds(calibration) == result["points"], "Operating points do not reproduce")
    count = audit(args, profile, calibration, "calibration")
    count += sum(audit(args, profile, [r for r in held if r["cohort"] == phase], phase, points)
                 for phase in ("retained_a", "retained_b"))
    require(summarize(held) == {key: result[key] for key in ("outcomes", "strata")}, "Aggregate result does not reproduce")
    require(args.publication is not None, "A publication destination is required for the verified receipt")
    result["readback"] = {"file_mask_bindings": len(calibration) + len(held), "native_box_overlaps": count}
    write(args.publication, result)
    print(result["readback"], flush=True)


if __name__ == "__main__":
    main()
