# SPDX-License-Identifier: Apache-2.0
"""Compare same-framing JPEGs with spatially cross-checked global tone curves."""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import jpeg_bitstream, reference_alignment, reference_compare
from scripts import image_regional_tile_rescue_transfer as panel
from scripts import resume_image_regional_tile_rescue as prior
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-reference-monotone-jpeg-transfer/1"
SOURCE_QUALITIES = (75, 90, 95)
TARGET_QUALITIES = (75, 90)
ROLES = ("sharp", "feathered", "local_blur", "local_tone", "local_copy", "resize_roundtrip",
         "no_tone", "ordinary_roundtrip", "ordinary_graded", "fractional", "translated", "unrelated")
LOCAL = {"sharp", "feathered", "local_blur", "local_tone", "local_copy"}
METHODS = ("current", "cross_checked")
BLOCK = 32
SIGMA = 1.6
DESIGN = {
    "scope": "Same dimensions, standard baseline upright RGB JPEGs from supported quality tables, 4:2:0.",
    "geometry": "Retain zero alignment only with globally corroborated identity and local luma phase/gradient matches distributed across image.",
    "curve": "Two disjoint checkerboard block folds. Robust monotone per-channel brightness functions fitted to Gaussian-smoothed RGB.",
    "background": "Hold each fitted curve to the opposite fold; at least 80% of held background pixels within 6 RGB levels.",
    "difference": "Absolute RGB residual beyond the current codec and modeled global tones after smoothing. Use lower residual of both fold curves.",
    "strong_threshold": 8., "component_threshold": 4.,
    "coordinate_space": "Unmoved submitted and earlier pixel grids",
    "useful": {"iou": .30, "recall": .50},
    "success": "Held gains >=3 useful feathered LaMa sources versus current, >=4/6 total; at most one extra ordinary delivery group, no unsupported pairs accepted.",
    "confidence_weight": 0,
}
read, write, require, sha, file_sha = panel.read, panel.write, panel.require, panel.sha, panel.file_sha


def jpeg_profile(image, payload, qualities):
    spec = jpeg_bitstream.read(payload)
    return bool(image.mode == "RGB" and image.format == "JPEG" and image.getexif().get(274, 1) == 1
                and not {"icc_profile", "progressive", "progression"} & image.info.keys()
                and spec and spec.frame == "baseline" and spec.precision == 8 and spec.ijg_quality in qualities
                and [(c.id, c.h, c.v) for c in spec.components] == [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
                and all(spec.table_bits.get(c.table_id) == 8 for c in spec.components)
                and spec.table(1) == spec.table(2))


def tile_blocks(shape):
    y, x = np.indices(shape)
    return (x // BLOCK + y // BLOCK) % 2 == 0


def geometry(a, b):
    width, height = a.shape[1::-1]
    gray = [cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.)
            for rgb in (a, b)]
    shift, response = reference_compare._alignment(a, b)
    if response < .24 or max(map(abs, shift)) > .65:
        return None
    size = 64
    hann = cv2.createHanningWindow((size, size), cv2.CV_32F)
    zones = Counter()
    eligible, matches = 0, []
    for y in range(0, height - size + 1, size):
        for x in range(0, width - size + 1, size):
            patches = [g[y:y+size, x:x+size].copy() for g in gray]
            if min(float(p.std()) for p in patches) < 6:
                continue
            eligible += 1
            gradients = [np.concatenate([np.diff(p, axis=k).ravel() for k in (0, 1)]) for p in patches]
            one, two = gradients
            similarity = float((one @ two) / max(np.linalg.norm(one) * np.linalg.norm(two), 1.))
            local, peak = cv2.phaseCorrelate(*patches, hann)
            if similarity >= .80 and peak >= .28 and max(map(abs, local)) <= .85:
                zones[((y + size//2) * 3 // height, (x + size//2) * 3 // width)] += 1
                matches.append([x, y])
    supported = [list(z) for z, count in sorted(zones.items()) if count >= 2]
    if (len(matches) < max(24, .65 * eligible) or len(supported) < 6
            or any(len({z[k] for z in supported}) != 3 for k in (0, 1))):
        return None
    return {"global_shift": shift, "global_response": response, "detailed_tiles": eligible,
            "aligned_tiles": matches, "supported_zones": supported}


def isotonic(values, weights):
    """Weighted pool-adjacent-violators reduction of measured knot medians."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([i, i + 1, value * weight, weight])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    out = np.empty(len(values))
    for left, right, total, weight in blocks:
        out[left:right] = total / weight
    return out


def fit_curve(x, y):
    """Brightness-conditioned median response with bounded endpoint continuation."""
    bins = np.minimum(x.astype(np.int16) // 8, 31)
    samples = [(float(np.median(x[p])), float(np.median(y[p])), int(p.sum()))
               for i in np.unique(bins) if (p := bins == i).sum() >= 32]
    if len(samples) < 9:
        return None
    source, target, weights = map(np.asarray, zip(*samples, strict=True))
    target = isotonic(target, weights)
    if source[-1] - source[0] < 75 or np.any(np.diff(target) > 2.4 * np.diff(source) + 3.):
        return None
    left_slope = np.clip((target[2] - target[0]) / (source[2] - source[0]), 0, 2.25)
    right_slope = np.clip((target[-1] - target[-3]) / (source[-1] - source[-3]), 0, 2.25)
    xnodes = np.concatenate(([0.], source, [255.]))
    ynodes = np.clip(np.concatenate(([target[0] - left_slope * source[0]], target,
                                    [target[-1] + right_slope * (255. - source[-1])])), 0, 255)
    keep = np.r_[np.diff(xnodes) > 0, True]
    return xnodes[keep], ynodes[keep]


def compare(before, after, earlier, submitted, rgb_a, rgb_b):
    if (rgb_a.shape != rgb_b.shape or not 512 <= min(rgb_a.shape[:2]) <= max(rgb_a.shape[:2]) <= 768
            or not jpeg_profile(earlier, before, SOURCE_QUALITIES) or not jpeg_profile(submitted, after, TARGET_QUALITIES)):
        return {"status": "outside_profile", "regions": []}, None
    proof = geometry(rgb_a, rgb_b)
    if proof is None:
        return {"status": "same_framing_unresolved", "regions": []}, None
    expected, _ = reference_compare._codec(rgb_a, submitted)
    a, b = [cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), SIGMA) for rgb in (expected, rgb_b)]
    halves = tile_blocks(a.shape[:2])
    curves, supports, residuals = [], [], []
    for fold in (halves, ~halves):
        measured = []
        for i in range(3):
            train_x = a[::3, ::3, i][fold[::3, ::3]]
            train_y = b[::3, ::3, i][fold[::3, ::3]]
            knots = fit_curve(train_x, train_y)
            if knots is None:
                return {"status": "insufficient_tone_support", "regions": []}, None
            measured.append(knots)
        # Reapply each measured curve to the earlier source before matching the submitted codec.
        corrected = np.stack([np.rint(np.interp(rgb_a[:, :, c], *knots)).astype(np.uint8)
                              for c, knots in enumerate(measured)], axis=-1)
        delivered, _ = reference_compare._codec(corrected, submitted)
        predicted = cv2.GaussianBlur(delivered.astype(np.float32), (0, 0), SIGMA)
        residual = np.abs(predicted - b).max(axis=2)
        support = float((residual[~fold] <= 6.).mean())
        if support < .80:
            return {"status": "unexplained_background", "regions": []}, None
        curves.append([[[float(x), float(y)] for x, y in zip(*knots, strict=True)] for knots in measured])
        supports.append(support)
        residuals.append(residual)
    residual = np.minimum(*residuals)
    low, high = DESIGN["component_threshold"], DESIGN["strong_threshold"]
    seeds = (residual > high).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    candidates = (residual > low).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    core = np.bincount(labels[seeds > 0], minlength=count)
    valid = (core >= 24) & (core >= .07 * stats[:, cv2.CC_STAT_AREA])
    valid[0] = False
    regions, mask, omitted = reference_compare._regions(valid[labels], max(48, round(residual.size * .0008)))
    for region in regions:
        region["reference_region"] = {k: region[k] for k in ("x", "y", "w", "h")}
    return {"status": "comparable", "regions": regions, "additional_regions": omitted, "alignment": proof,
            "background_support": supports, "tone_knots": curves,
            "source_residual_sha256": sha(residual.tobytes())}, mask


def profile(args):
    path = args.tile_run / "freeze.json"
    require(file_sha(path) == prior.FROZEN_SHA256, "Prior source freeze changed")
    roster = read(path)
    require(panel.profile(args) == roster, "Prior JPEG identities or source code differs")
    views = list(cases(args, roster))
    groups = {key: {row["group"] for row, *_ in views if row["bank"] == key} for key in ("retained_b", "external")}
    require([len(groups[key]) for key in ("retained_b", "external")] == [7, 6]
            and not groups["retained_b"] & groups["external"], "Both source-disjoint LaMa banks required")
    return {"schema": SCHEMA, "method": DESIGN, "prior_freeze": file_sha(path),
            "runner_sha256": file_sha(Path(__file__)),
            "support": {f.__name__: file_sha(Path(f.__file__)) for f in (reference_compare, reference_alignment, jpeg_bitstream)},
            "views": [r for r, *_ in views]}, roster


def grade(rgb, mode):
    x = rgb.astype(np.float32) / 255
    if mode == 0:
        y = (.03 + .94 * x**.73)
    elif mode == 1:
        y = .07 + .85 * (x + .12 * np.sin((2*x - 1) * np.pi))
    else:
        y = .035 + .90 * x**np.array([1.34, .80, 1.10], dtype=np.float32)
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)


def motion(rgb, amount):
    return cv2.warpAffine(rgb, np.float32([[1, 0, amount], [0, 1, -amount]]), rgb.shape[1::-1],
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def cases(args, roster):
    native = {}
    frozen_roles = {(r["panel"], r["group"], r["role"]): r for r in roster["rows"] if r["family"] == "lama"}

    def bind(bank, group, authoring, exact, recipient):
        controls, edits = panel.source_panel.common.authored(authoring, exact, recipient)
        uses = {role: (controls | edits)[role] for role in ("original", *sorted(LOCAL))}
        for role, rgb in uses.items():
            record = frozen_roles[(bank, group, role)]
            truth = (np.any(rgb != authoring["original"], axis=2) if role in panel.previous.POSITIVE
                     else np.zeros(exact.shape, bool))
            require(record["input_sha256"] == sha(panel.source_panel.common.encode(rgb, 75))
                    and record["truth_sha256"] == sha(truth.tobytes()), "Prior exact JPEG or truth differs")
        native[(bank, group)] = uses

    for item, pixels, exact in panel.source_panel.later_groups(SimpleNamespace(later=args.later_f5)):
        if item["generator"] == "lama" and min(pixels["original"].shape[:2]) >= 512:
            credit = panel.source_panel.identity(item, pixels, exact)
            bind("retained_b", credit["photo"], pixels, exact, item["recipient"])

    external = panel.previous.panel.external
    plan, intake = (read(args.external / name) for name in ("plan.json", "intake.json"))
    require(file_sha(args.external / "plan.json") == intake["plan_sha256"], "Held intake plan differs")
    for item, receipt in zip(plan["cases"], intake["rows"], strict=True):
        if item["generator"] != "lama" or receipt["status"] != "qualified":
            continue
        assets = {role: (args.external / "assets" / f"{item['slot']:02d}-{role}.png").read_bytes()
                  for role in ("original", "edited", "semantic_mask")}
        for role, body in assets.items():
            proof = receipt["proofs"][role]
            require(sha(body) == proof["sha256"] == (item[role]["sha256"] or proof["publisher_asset_commitment"])
                    and len(body) == proof["bytes"] and proof["resolver_revision"] == external.roster.REVISION,
                    "Held source carrier differs")
        pixels, exact, _, _ = external.bound(item, assets)
        require(sha(exact.tobytes()) == receipt["mask_sha256"], "Held exact LaMa footprint differs")
        bind("external", str(item["flickr_photo_id"]), pixels, exact, item["recipient"])
    for index, ((bank, group), sources) in enumerate(native.items()):
        require(set(sources) == {"original", *LOCAL}, "Missing LaMa edit/control counterpart")
        original = sources["original"]
        before_quality, after_quality = SOURCE_QUALITIES[index % 3], TARGET_QUALITIES[index % 2]
        before = panel.source_panel.common.encode(original, before_quality)
        transformed = grade(original, index % 3)
        other_source = next(v["original"] for key, v in native.items() if key[0] == bank and key[1] != group)
        other_source = cv2.resize(other_source, original.shape[1::-1], interpolation=cv2.INTER_AREA)
        roundtrip = cv2.resize(cv2.resize(original, None, fx=.78, fy=.78, interpolation=cv2.INTER_AREA),
                              original.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        variants = {role: grade(sources[role], index % 3) for role in ROLES if role in LOCAL}
        variants.update(no_tone=original, ordinary_roundtrip=panel.source_panel.common.decode(
            panel.source_panel.common.encode(original, 82)), ordinary_graded=transformed, resize_roundtrip=grade(roundtrip, index % 3),
            fractional=motion(transformed, 1.45), translated=motion(transformed, 9.), unrelated=grade(other_source, index % 3))
        for role in ROLES:
            variant = variants[role]
            truth = np.any(variant != transformed, axis=2) if role in LOCAL else np.zeros(original.shape[:2], bool)
            require(bool(truth.any()) == (role in LOCAL), "Local edit truth differs")
            after = panel.source_panel.common.encode(variant, after_quality)
            yield {"bank": bank, "group": group, "role": role, "before_sha256": sha(before), "after_sha256": sha(after),
                   "truth_sha256": sha(truth.tobytes()), "shape": list(truth.shape),
                   "quality": [before_quality, after_quality]}, before, after, truth


def assess(args, roster, frozen):
    rows = []
    for (row, before, after, truth), locked in zip(cases(args, roster), frozen["views"], strict=True):
        require(row == locked, "Authored JPEG or exact changed-pixel support differs")
        with Image.open(io.BytesIO(before)) as reference, Image.open(io.BytesIO(after)) as submitted:
            rgb_a, rgb_b = np.asarray(reference), np.asarray(submitted)
            current = reference_alignment.compare(rgb_a, rgb_b, submitted)
            updated = compare(before, after, reference, submitted, rgb_a, rgb_b)
        output = {}
        for method, (result, mask) in zip(METHODS, (current, updated), strict=True):
            if mask is None:
                mask = np.zeros_like(truth)
            output[method] = {"status": result["status"], "mask_sha256": sha(mask.tobytes()),
                              "result": result, "metrics": mask_metrics(mask, truth)}
        rows.append(row | {"measurements": output})
        if len(rows) % len(ROLES) == 0:
            print(len(rows), "/", len(frozen["views"]), flush=True)
    return rows


def summary(rows):
    out = {}
    for bank in ("retained_b", "external"):
        groups = {method: set() for method in METHODS}
        wins = set()
        totals = Counter()
        for r in rows:
            if r["bank"] != bank:
                continue
            role = r["role"]
            wins_this_file = {}
            for name, result in r["measurements"].items():
                m = result["metrics"]
                found = role in LOCAL and panel.previous.is_useful(m)
                flag = m["predicted_positive_pixels"] > 0
                totals.update({f"{name}/{role}/files": 1, f"{name}/{role}/useful": found,
                               f"{name}/{role}/flagged": flag, f"{name}/{role}/compared": result["status"] == "comparable"})
                if role in {"no_tone", "ordinary_roundtrip", "ordinary_graded", "resize_roundtrip"} and flag:
                    groups[name].add(r["group"])
                wins_this_file[name] = found
            if role == "feathered" and wins_this_file["cross_checked"] and not wins_this_file["current"]:
                wins.add(r["group"])
        extra = len(groups["cross_checked"] - groups["current"])
        rejected_types = ("fractional", "translated", "unrelated")
        wrong_geometry = sum(totals[f"cross_checked/{r}/compared"] for r in rejected_types)
        out[bank] = {"counts": dict(totals), "ordinary_groups_flagged": {k: len(v) for k, v in groups.items()},
                     "additional_ordinary_groups": extra, "new_feathered_groups": len(wins),
                     "wrong_geometry": wrong_geometry,
                     "passes": len(wins) >= 3 and totals["cross_checked/feathered/useful"] >= 4
                     and extra <= 1 and wrong_geometry == 0}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    for name in ("tile-run", "archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    freeze, roster = profile(args)
    path = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(path, freeze)
        return
    require(freeze == read(path), "Frozen image pairs or method differ")
    rows = assess(args, roster, freeze)
    result = {k: v for k, v in freeze.items() if k != "views"} | {"results": summary(rows),
             "views": len(rows), "freeze_sha256": file_sha(path)}
    if args.phase == "run":
        write(args.output / "rows.json", rows)
        write(args.output / "result.json", result)
    else:
        require(read(args.output / "rows.json") == rows and read(args.output / "result.json") == result,
                "Exact paired source/mask replay differs")
        write(args.publication, result | {"exact_replays": len(rows), "rows_sha256": file_sha(args.output / "rows.json")})
    print({k: {field: r[field] for field in ("new_feathered_groups", "additional_ordinary_groups", "wrong_geometry", "passes")}
           for k, r in result["results"].items()}, flush=True)


if __name__ == "__main__":
    main()
