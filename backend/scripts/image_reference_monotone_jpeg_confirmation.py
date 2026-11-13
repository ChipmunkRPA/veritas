# SPDX-License-Identifier: Apache-2.0
"""Freeze and evaluate a sharpness scope check on distinct paired JPEGs."""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts import image_reference_delivery_envelope_lama as intake
from scripts import image_reference_monotone_jpeg_transfer as earlier

SCHEMA = "chipmunk.image-reference-monotone-jpeg-confirmation/1"
ARMS = ("current", "cross_checked", "fallback", "sharpness_fallback")
CONTROLS = ("no_tone", "ordinary_roundtrip", "ordinary_graded", "resize_roundtrip",
            "resize_lanczos", "global_blur")
WRONG = ("fractional", "translated", "unrelated")
COUNT = 8
DESIGN = {
    "scope": earlier.DESIGN["scope"],
    "geometry": "Same earlier crop and pixel coordinates as the frozen monotone pair method.",
    "ordinary_scale": "Compare local fine-to-coarse energy in corresponding 64px luma tiles.",
    "retention": "Require at least twelve comparable detailed tiles. Withhold the fallback when the upper quartile of log relative fine-to-coarse energy is below -0.22 for both fitted tone deliveries.",
    "registration_scope": "An accepted difference is a source comparison and carries zero AI confidence weight.",
    "admission": "Gain at least two sharp and two feathered known locations; add at most two nuisance files, at most one nuisance source group and zero incompatible-framing outcomes.",
}
read, write, require, sha, file_sha = earlier.read, earlier.write, earlier.require, earlier.sha, earlier.file_sha


def sharpness(rgb):
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(luma, (0, 0), 1.25)
    horizontal = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=.125)
    vertical = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=.125)
    fine = luma - cv2.GaussianBlur(luma, (0, 0), .85)
    return horizontal**2 + vertical**2, fine**2


def scope_check(rgb_a, rgb_b, submitted, curves):
    coarse, fine = sharpness(rgb_b)
    readings = []
    for measured in curves:
        corrected = np.stack([np.rint(np.interp(rgb_a[:, :, channel], *zip(*knots, strict=True))).astype(np.uint8)
                              for channel, knots in enumerate(measured)], axis=-1)
        expected, _ = earlier.reference_compare._codec(corrected, submitted)
        base_coarse, base_fine = sharpness(expected)
        ratios = []
        for y in range(0, coarse.shape[0] - 63, 64):
            for x in range(0, coarse.shape[1] - 63, 64):
                tile = np.s_[y:y+64, x:x+64]
                a, b = float(coarse[tile].mean()), float(fine[tile].mean())
                c, d = float(base_coarse[tile].mean()), float(base_fine[tile].mean())
                if a > 5 and c > 5 and d > 1:
                    ratios.append(float(np.log(max(1e-6, (b / a) / (d / c)))))
        upper = float(np.quantile(ratios, .75)) if len(ratios) >= 12 else None
        readings.append({"tiles": len(ratios), "upper_log_energy_ratio": upper})
    if any(r["upper_log_energy_ratio"] is None for r in readings):
        return "unassessed_sharpness_support", readings
    if all(r["upper_log_energy_ratio"] < -.22 for r in readings):
        return "unassessed_global_detail_loss", readings
    return "comparable", readings


def sources(args):
    primary = args.source_roster / "qualified-private.json"
    require(file_sha(primary) == read(args.source_roster / "summary-v225.json")["qualified_private_sha256"],
            "Earlier qualified source roster differs")
    roster = read(primary)
    known = set()
    for group, assets in ((args.later_f5, "plan.json"), (args.external, "plan.json")):
        for row in read(group / assets)["cases"]:
            known.update(identities(row))
    results = []
    cases = sorted(roster["qualified"], key=lambda r: sha(f"{SCHEMA}/{r['source']['flickr_photo_id']}".encode()))
    for row in cases:
        item = row["source"]
        keys = identities(item)
        if keys & known or not 512 <= min(item["size"]) <= max(item["size"]) <= 768:
            continue
        require(item["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                "Unexpected photo licence")
        a, b, _ = intake.source_pixels(Path(roster["cache"]), row)
        exact = np.any(a != b, axis=2)
        require(.004 <= float(exact.mean()) <= .08, "Small known removal required")
        known.update(keys)
        ordinary, edited = earlier.panel.source_panel.common.authored({"original": a, "edited": b}, exact, item["recipient"])
        results.append((sha(str(item["flickr_photo_id"]).encode()), ordinary, edited))
        if len(results) == COUNT:
            break
    require(len(results) == COUNT, "Insufficient separate eligible source photographs")
    return results


def identities(item):
    credit = item["photo_attribution"]
    return {str(item["flickr_photo_id"]), item["flickr_owner"].strip().casefold(),
            credit["Author"].strip().casefold(), item["original"]["sha256"]}


def cases(args):
    bank = sources(args)
    encode, decode = earlier.panel.source_panel.common.encode, earlier.panel.source_panel.common.decode
    for index, (source, ordinary, edited) in enumerate(bank):
        original = ordinary["original"]
        before_quality = earlier.SOURCE_QUALITIES[index % 3]
        after_quality = earlier.TARGET_QUALITIES[index % 2]
        grade_mode = (index + 1) % 3
        graded = earlier.grade(original, grade_mode)
        before = encode(original, before_quality)
        unrelated = cv2.resize(bank[(index + 1) % len(bank)][1]["original"], original.shape[1::-1])
        reduced = cv2.resize(original, None, fx=.78, fy=.78, interpolation=cv2.INTER_AREA)
        resized = cv2.resize(reduced, original.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        picture = Image.fromarray(original)
        lz = picture.resize(tuple(round(x*.76) for x in picture.size), Image.Resampling.LANCZOS)
        lz = np.asarray(lz.resize(picture.size, Image.Resampling.LANCZOS))
        variants = {"no_tone": original, "ordinary_roundtrip": decode(encode(original, 82)), "ordinary_graded": graded,
                    "resize_roundtrip": earlier.grade(resized, grade_mode), "resize_lanczos": earlier.grade(lz, grade_mode),
                    "global_blur": cv2.GaussianBlur(graded, (0, 0), .85), "fractional": earlier.motion(graded, 1.45),
                    "translated": earlier.motion(graded, 9.), "unrelated": earlier.grade(unrelated, grade_mode)}
        for role, pixels in (ordinary | edited).items():
            if role in earlier.LOCAL:
                variants[role] = earlier.grade(pixels, grade_mode)
        for role, pixels in variants.items():
            truth = np.any(pixels != graded, axis=2) if role in earlier.LOCAL else np.zeros(original.shape[:2], bool)
            after = encode(pixels, after_quality)
            row = {"source": source, "role": role, "dimensions": list(truth.shape), "quality": [before_quality, after_quality],
                   "before_sha256": sha(before), "after_sha256": sha(after), "truth_sha256": sha(truth.tobytes())}
            yield row, before, after, truth


def score(args, rows):
    report = []
    for (row, before, after, truth), witness in zip(cases(args), rows, strict=True):
        require(row == witness, "Held delivery or exact mask differs")
        with Image.open(io.BytesIO(before)) as reference, Image.open(io.BytesIO(after)) as submitted:
            rgb_a, rgb_b = np.asarray(reference), np.asarray(submitted)
            current = earlier.reference_alignment.compare(rgb_a, rgb_b, submitted)
            cross = earlier.compare(before, after, reference, submitted, rgb_a, rgb_b)
            current_ok = current[0]["status"] == "comparable"
            fallback = current if current_ok else cross
            guarded, support = fallback, None
            if not current_ok and cross[0]["status"] == "comparable":
                status, support = scope_check(rgb_a, rgb_b, submitted, cross[0]["tone_knots"])
                if status != "comparable":
                    guarded = ({"status": status}, None)
        metrics = {}
        for name, (result, mask) in zip(ARMS, (current, cross, fallback, guarded), strict=True):
            selected = mask if mask is not None else np.zeros_like(truth)
            metrics[name] = {"status": result["status"], "mask_sha256": sha(selected.tobytes()),
                             "pixels": earlier.mask_metrics(selected, truth)}
        report.append(row | {"arms": metrics, "sharpness": support})
    return report


def summary(rows):
    result = {}
    flagged_groups = {arm: set() for arm in ARMS}
    gains = {arm: set() for arm in ARMS}
    for arm in ARMS:
        totals = Counter()
        for row in rows:
            role, group = row["role"], row["source"]
            hit = row["arms"][arm]
            flags = bool(hit["pixels"]["predicted_positive_pixels"])
            useful = earlier.panel.previous.is_useful(hit["pixels"])
            totals.update({f"{role}/compared": hit["status"] == "comparable", f"{role}/flagged": flags,
                           f"{role}/useful": useful, f"{role}/views": 1})
            if role in CONTROLS and flags:
                flagged_groups[arm].add(group)
            if role in ("sharp", "feathered") and useful and not earlier.panel.previous.is_useful(row["arms"]["current"]["pixels"]):
                gains[arm].add((group, role))
        result[arm] = dict(totals)
    comparisons = {}
    for arm in ARMS:
        newly_flagged = sum(row["role"] in CONTROLS and row["arms"][arm]["pixels"]["predicted_positive_pixels"] > 0
                            and not row["arms"]["current"]["pixels"]["predicted_positive_pixels"] for row in rows)
        new_groups = len(flagged_groups[arm] - flagged_groups["current"])
        extra = Counter(role for _, role in gains[arm])
        wrong = sum(result[arm][f"{role}/compared"] for role in WRONG)
        comparisons[arm] = {"new_nuisance_files": newly_flagged, "new_nuisance_groups": new_groups,
                            "new_sharp": extra["sharp"], "new_feathered": extra["feathered"],
                            "incompatible_compared": wrong, "ordinary_groups_flagged": len(flagged_groups[arm]),
                            "passes": extra["sharp"] >= 2 and extra["feathered"] >= 2
                            and newly_flagged <= 2 and new_groups <= 1 and wrong == 0}
    return {"strata": result, "paired": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    for name in ("source-roster", "later-f5", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    profile = {"schema": SCHEMA, "design": DESIGN, "method_sha256": file_sha(Path(earlier.__file__)),
               "runner_sha256": file_sha(Path(__file__)),
               "source_plan": file_sha(args.source_roster / "qualified-private.json"),
               "excluded_plans": [file_sha(path / "plan.json") for path in (args.later_f5, args.external)],
               "rows": [row for row, *_ in cases(args)]}
    path = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(path, profile)
        return
    require(profile == read(path), "Confirmed inputs or code differ")
    rows = score(args, profile["rows"])
    result = {k: v for k, v in profile.items() if k != "rows"} | summary(rows) | {
        "pairs": len(rows), "sources": COUNT, "freeze_sha256": file_sha(path)}
    if args.phase == "run":
        write(args.output / "rows.json", rows)
        write(args.output / "result.json", result)
    else:
        require(rows == read(args.output / "rows.json") and result == read(args.output / "result.json"),
                "Separate exact source and mask replay differs")
        write(args.publication, result | {"exact_replays": len(rows), "rows_sha256": file_sha(args.output / "rows.json")})
    print(result["paired"], flush=True)


if __name__ == "__main__":
    main()
