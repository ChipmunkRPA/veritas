# SPDX-License-Identifier: Apache-2.0
"""Test local crop response dependence and printed layout on known native image edits."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import commfor, sharp_boundary
from scripts import image_reference_orientation_transfer as sources
from scripts import image_reference_oriented_crop_transfer as native
from scripts import image_remote_crop_editor_transfer as baseline

SCHEMA = "chipmunk.image-crop-layout-transfer/1"
REGIONAL = baseline.panel
SHA = sources.sha
GEOMETRY = (736, 608)
FIT_ROLES = ("original", "contrast", "blur")
LAMA_ROLES = ("original", "edited", "feathered_edit", "local_contrast", "blur", "contrast")
SCREEN_ROLES = ("template", "photo", "edit")
POSITIVE = {"edited", "feathered_edit", "edit"}
ARMS = ("remote", "layout", "intervention", "combined")
POLICY = {"minimum_useful_feathered": 4, "minimum_feathered_source_families": 2, "maximum_extra_ordinary_photo_views": 3,
          "maximum_control_screenshot_groups": 1, "minimum_ordinary_screenshot_challenge_groups": 3,
          "minimum_useful_screen_edits": 2, "maximum_lost_useful_screen_edits": 1}
RUNTIME = {"blur_sigma": 1.5, "tile": 96, "fade": 16, "inner": [96, 192], "outer": [0, 288],
           "ocr_psm": 11, "minimum_ocr_word_confidence": 65, "minimum_word_height": 6, "maximum_word_height": 32,
           "minimum_aligned_rows": 3, "minimum_words_per_row": 2, "line_start_tolerance": 6,
           "minimum_layout_coverage": .08, "minimum_baseline_spread": .24}


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    return SHA(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def png(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def qualified_lama(folder):
    summary, roster = read(folder / "summary.json"), read(folder / "qualified-private.json")
    require(file_sha(folder / "qualified-private.json") == summary["qualified_private_sha256"]
            and len(roster["qualified"]) == summary["qualified"] == 12, "Twelve sealed LaMa sources required")
    return Path(roster["cache"]), roster["qualified"]


def lama_sources(folder):
    cache, records = qualified_lama(folder)
    for slot, row in enumerate(records):
        images = {}
        for role in ("original", "edited"):
            asset = row["measurement"]["files"][role]
            path = cache / asset["file"]
            require(file_sha(path) == asset["sha256"], "Publisher file bytes changed")
            images[role] = pixels(path)
        path = cache / row["truth_file"]
        require(file_sha(path) == row["truth_png_sha256"], "Qualified truth file changed")
        with Image.open(path) as image:
            truth = np.asarray(image.convert("L")) > 0
        require(truth.any() and np.array_equal((images["original"] != images["edited"]).any(axis=2), truth),
                "Publisher edits differ from qualified native mask")
        yield slot, images["original"], images["edited"], truth


def feather(before, after, mask):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = (1-np.cos(np.clip(dist / 12, 0., 1.)*np.pi))[:, :, None] / 2
    return np.rint(before * (1-alpha) + after * alpha).astype(np.uint8)


def source_box(mask):
    y, x = np.nonzero(mask)
    return int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)


def layout_source(original, edited, truth, slot):
    height, width = truth.shape
    left, top, right, bottom = source_box(truth)
    w, h = min(460, width), min(430, height)
    x0 = max(0, min(width-w, (left+right-w)//2))
    y0 = max(0, min(height-h, (top+bottom-h)//2))
    y, x = 79 + slot*4, 156 + slot*5
    region = np.s_[y0:y0+h, x0:x0+w]
    local_truth = truth[region]
    require(x+w <= GEOMETRY[0] and y+h <= GEOMETRY[1] and local_truth.sum() >= .8*truth.sum(),
            "Prepared screenshot crop lost too much of the exact generated edit")
    return {"slot": slot, "width": GEOMETRY[0], "height": GEOMETRY[1], "x": x, "y": y, "w": w, "h": h}, original[region], edited[region]


def prepare(args):
    args.run.mkdir(mode=0o700)
    photo_dir = args.run / "photo-fixtures"
    photo_dir.mkdir()
    plan = []
    for slot, original, edited, mask in lama_sources(args.lama):
        blended = feather(original, edited, mask)
        truth = (original != blended).any(axis=2)
        group, clear, changed = layout_source(original, blended, truth, slot)
        for role, rgb in (("photo", clear), ("edit", changed)):
            (photo_dir / f"{slot}-{role}.png").write_bytes(png(rgb))
        plan.append(group)
    sources.write(args.run / "layout-plan.json", plan)


def screenshot_views(args):
    plan, capture = (read(args.run / name) for name in ("layout-plan.json", "captures.json"))
    captures = {(row["slot"], row["role"]): row for row in capture["rows"]}
    require(len(captures) == len(capture["rows"]) == 36 and len(plan) == 12, "Expected 12 native screenshot groups")
    for group in plan:
        slot, x, y, w, h = (group[key] for key in ("slot", "x", "y", "w", "h"))
        pairs = {}
        for role in SCREEN_ROLES:
            row = captures[slot, role]
            path = args.run / "captures" / row["name"]
            require(row["name"] == f"{slot}-{role}.png" and file_sha(path) == row["png_sha256"], "Screenshot bytes changed")
            rgb = pixels(path)
            require(rgb.shape[:2] == (GEOMETRY[1], GEOMETRY[0]), "Chromium native viewport changed")
            if role != "template":
                fixture = args.run / "photo-fixtures" / f"{slot}-{role}.png"
                require(file_sha(fixture) == row["photo_sha256"] and np.array_equal(rgb[y:y+h, x:x+w], pixels(fixture)),
                        "Displayed photo differs from native RGB fixture")
            pairs[role] = (path.read_bytes(), rgb)
        truth = (pairs["photo"][1] != pairs["edit"][1]).any(axis=2)
        require(0 < truth.sum() == truth[y:y+h, x:x+w].sum(), "Screenshot changes escaped exact photo area")
        for role, (data, rgb) in pairs.items():
            yield "screen", slot, role, data, rgb, truth if role == "edit" else np.zeros_like(truth), None


def freeze(args):
    prior = read(args.baseline / "result.json")
    fit = sorted(sources.binding(args.commons)["sources"], key=lambda item: SHA(f"crop-layout/{item['sha256']}".encode()))[:8]
    snapshots = list(screenshot_views(args))
    old = read(args.baseline / "freeze.json")
    journal = args.baseline / "fresh-rows.json"
    require(file_sha(journal) == prior["journals"]["fresh"] and file_sha(args.weight) == old["weight_sha256"] == baseline.source.WEIGHT_SHA,
            "The frozen response ledger or MIT weights changed")
    cache, records = qualified_lama(args.lama)
    old_plan = read(args.generated / "plan.json")["cases"]
    require(not ({row["flickr_owner"] for row in old_plan} & {row["source"]["flickr_owner"] for row in records}),
            "Qualified photo contributor groups overlap")
    paths = (Path(__file__), Path(__file__).with_name("capture_image_crop_layout_transfer.cjs"),
             Path(baseline.__file__), Path(baseline.crops.__file__), Path(commfor.__file__), Path(sharp_boundary.__file__))
    return {"schema": SCHEMA, "code": {path.name: file_sha(path) for path in paths}, "fit_sources": fit,
            "capture": file_sha(args.run / "captures.json"), "capture_plan": file_sha(args.run / "layout-plan.json"),
            "screens": [{"slot": slot, "role": role, "input": SHA(data), "truth": SHA(truth.tobytes())}
                        for _, slot, role, data, _, truth, _ in snapshots],
            "lama_roster": file_sha(args.lama / "qualified-private.json"),
            "lama_cache_truth": [file_sha(cache / row["truth_file"]) for row in records],
            "generated_intake": {key: file_sha(args.generated / key) for key in ("result.json", "plan.json", "attempts.jsonl")},
            "weight_sha256": file_sha(args.weight), "cached_journal": file_sha(journal), "baseline_points": old["points"],
            "ocr_binary_sha256": file_sha(args.tesseract), "ocr_data_sha256": file_sha(args.tessdata / "eng.traineddata"),
            "runtime": RUNTIME, "criteria": POLICY}


def fit_views(args, lock):
    for slot, record in enumerate(lock["fit_sources"]):
        image = native.source(args.commons, record, GEOMETRY)
        variants = image, ImageEnhance.Contrast(image).enhance(.88), image.filter(ImageFilter.GaussianBlur(1.25))
        for role, item in zip(FIT_ROLES, variants, strict=True):
            rgb = np.asarray(item)
            yield "fit", slot, role, png(rgb), rgb, np.zeros(rgb.shape[:2], dtype=bool), None


def held_views(args):
    cache = {(row["key"][1], row["key"][2]): row for row in read(args.baseline / "fresh-rows.json") if row["key"][3] == "png"}
    for case, native_rgb, known in REGIONAL.paired.inputs(args.generated):
        for role, rgb in REGIONAL.native_variants(native_rgb, case["box"]).items():
            truth = (rgb != native_rgb["original"]).any(axis=2) if role in POSITIVE else np.zeros_like(known)
            data, decoded = REGIONAL.paired.encoded(rgb, "png")
            saved = cache.pop((case["slot"], role))
            require(saved["input_sha256"] == SHA(data) and saved["truth_sha256"] == SHA(truth.tobytes()), "Saved crop response input changed")
            yield "patch", case["slot"], role, data, np.asarray(decoded), truth, saved
    require(not cache, "The saved PNG response panel is incomplete")
    for slot, original, edited, known in lama_sources(args.lama):
        image = Image.fromarray(original)
        contrast = np.asarray(ImageEnhance.Contrast(image).enhance(.87))
        blur = np.asarray(image.filter(ImageFilter.GaussianBlur(1.4)))
        variants = (original, edited, feather(original, edited, known), feather(original, contrast, known),
                    feather(original, blur, known), contrast)
        for role, rgb in zip(LAMA_ROLES, variants, strict=True):
            truth = (original != rgb).any(axis=2) if role in POSITIVE else np.zeros_like(known)
            yield "lama", slot, role, png(rgb), rgb, truth, None
    yield from screenshot_views(args)


def model_loader(weight):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = commfor.load_model(weight, "cpu")
    mean = torch.tensor(commfor.MEAN)[None, :, None, None]
    std = torch.tensor(commfor.STD)[None, :, None, None]
    memo, calls = {}, Counter()

    def score(images):
        key = tuple(SHA(img.tobytes()) for img in images)
        if key not in memo:
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            memo[key] = commfor.logits(model, (tensor-mean)/std, "cpu")
            calls["model_batches"] += 1
            calls["model_inputs"] += len(images)
        else:
            calls["reused_batches"] += 1
        return memo[key]

    return score, calls


def stencils(crop):
    tile, fade = RUNTIME["tile"], RUNTIME["fade"]
    indexes = np.arange(tile, dtype=np.float64)
    border = np.minimum(indexes+1, tile-indexes)
    line = (1-np.cos(np.minimum(border/fade, 1.)*np.pi))/2
    window = np.outer(line, line)[:, :, None]
    smooth = cv2.GaussianBlur(crop, (0, 0), RUNTIME["blur_sigma"])
    for label in ("inner", "outer"):
        measured = crop.copy()
        for x in RUNTIME[label]:
            for y in RUNTIME[label]:
                area = np.s_[y:y+tile, x:x+tile]
                measured[area] = np.rint(crop[area]*(1-window) + smooth[area]*window).astype(np.uint8)
        yield measured


def ocr_boxes(args, data):
    task = subprocess.run([str(args.tesseract), "stdin", "stdout", "--tessdata-dir", str(args.tessdata), "-l", "eng",
                           "--psm", str(RUNTIME["ocr_psm"]), "tsv"], input=data, capture_output=True, check=True, timeout=8)
    rows = csv.DictReader(io.StringIO(task.stdout.decode("utf-8")), delimiter="\t")
    boxes = []
    for row in rows:
        if (row["level"] == "5" and float(row["conf"]) >= RUNTIME["minimum_ocr_word_confidence"]
                and sum(char.isalnum() for char in row["text"] or "") >= 2):
            x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
            if w > 0 and RUNTIME["minimum_word_height"] <= h <= RUNTIME["maximum_word_height"]:
                boxes.append([x, y, x+w, y+h])
    return sorted(boxes, key=lambda box: (box[3], box[0]))


def layout(boxes, crop):
    x, y, right, bottom = crop
    inside = [box for box in boxes if x <= box[0] < box[2] <= right and y <= box[1] < box[3] <= bottom]
    groups = []
    for box in inside:
        if not groups or abs(box[3]-groups[-1][0][3]) > max(3, (box[3]-box[1])//3):
            groups.append([])
        groups[-1].append(box)
    lines = [(min(box[0] for box in row), min(box[3] for box in row)) for row in groups if len(row) >= RUNTIME["minimum_words_per_row"]]
    aligned = max(([point for point in lines if abs(point[0]-start) <= RUNTIME["line_start_tolerance"]] for start, _ in lines),
                  key=len, default=[])
    spread = (max(base for _, base in aligned)-min(base for _, base in aligned))/(bottom-y) if aligned else 0.
    coverage = sum((r-a)*(b-t) for a, t, r, b in inside) / ((right-x)*(bottom-y))
    return {"words": len(inside), "aligned_rows": len(aligned), "baseline_spread": spread, "word_coverage": coverage}


def layout_confounded(measurement):
    return (measurement["aligned_rows"] >= RUNTIME["minimum_aligned_rows"]
            and measurement["baseline_spread"] >= RUNTIME["minimum_baseline_spread"]
            and measurement["word_coverage"] >= RUNTIME["minimum_layout_coverage"])


def overlap(box, truth):
    x, y, r, b = box
    return [int(truth.sum()), (r-x)*(b-y), int(truth[y:b, x:r].sum())]


def observe(args, lock, view, score, *, force=False):
    domain, slot, role, data, rgb, truth, saved = view
    boxes, prepared = zip(*baseline.crops.crops(Image.fromarray(rgb)), strict=True)
    bindings = [SHA(item.tobytes()) for item in prepared]
    if saved is None:
        logits = [value for offset in range(0, 25, 5) for value in score(prepared[offset:offset+5])]
    else:
        require(bindings == saved["crop_sha256"], "Saved crop tensor input changed")
        logits = saved["logits"]
    response = baseline.inspect(logits, boxes, lock["baseline_points"])
    index = boxes.index(tuple(response["box"]))
    intervention = layout_values = None
    if force or response["remote_control"]:
        changed = tuple(stencils(prepared[index]))
        inner, outer = score(changed)
        intervention = {"inner": inner, "outer": outer, "effect": outer-inner, "inputs": [SHA(item.tobytes()) for item in changed]}
        layout_values = layout(ocr_boxes(args, data), response["box"])
    h, w = truth.shape
    fixed = sharp_boundary.measure(rgb) if sharp_boundary.supported_size(w, h) else None
    return {"domain": domain, "slot": slot, "role": role, "size": [w, h], "input_sha256": SHA(data),
            "truth_sha256": SHA(truth.tobytes()), "crop_sha256": bindings[index], "logits": logits,
            "response": response, "counts": overlap(response["box"], truth), "layout": layout_values, "intervention": intervention,
            "sharp_boundary": {"score": fixed.score, "box": list(fixed.box), "counts": overlap(fixed.box, truth)} if fixed else None}


def point(rows):
    maxima = defaultdict(lambda: -math.inf)
    for row in rows:
        maxima[row["slot"]] = max(maxima[row["slot"]], row["intervention"]["effect"])
    require(len(maxima) == 8, "Incomplete separate fit panel")
    return {"strict_point": max(0., sorted(maxima.values())[-2]), "maxima": dict(sorted(maxima.items()))}


def useful(counts):
    target, reported, shared = counts
    return target > 0 and 10*shared >= 3*(target+reported-shared) and 2*shared >= target


def summary(rows, threshold):
    counts, groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        domain, role, slot = row["domain"], row["role"], row["slot"]
        remote = row["response"]["remote_control"]
        confound = bool(remote and layout_confounded(row["layout"]))
        support = bool(remote and row["intervention"]["effect"] > threshold)
        for arm, flag in zip(ARMS, (remote, remote and not confound, support, support and not confound), strict=True):
            key = f"{domain}/{role}/{arm}"
            hit = flag and useful(row["counts"])
            counts[key].update({"views": 1, "flags": flag, "useful": hit, "layout_confounds": confound})
            if domain == "screen" and flag and role != "edit":
                groups[f"screen_controls/{arm}"].add(slot)
            if domain != "screen" and flag and role not in POSITIVE:
                counts["photo_control_flags"][arm] += 1
            if role == "feathered_edit" and hit:
                groups[f"feathered_families/{arm}"].add(domain)
                counts["feathered_useful"][arm] += 1
        fixed = row["sharp_boundary"]
        flagged = bool(fixed and fixed["score"] > sharp_boundary.REVIEW_POINT)
        counts[f"{domain}/{role}/sharp_boundary"].update({"views": 1, "assessed": bool(fixed), "flags": flagged,
                                                          "useful": flagged and useful(fixed["counts"])})
        if domain != "screen" and flagged and role not in POSITIVE:
            counts["photo_control_flags"]["sharp_boundary"] += 1
    policy = POLICY
    advanced = (counts["feathered_useful"]["combined"] >= policy["minimum_useful_feathered"]
                and len(groups["feathered_families/combined"]) >= policy["minimum_feathered_source_families"]
                and counts["photo_control_flags"]["combined"] <= counts["photo_control_flags"]["sharp_boundary"] + policy["maximum_extra_ordinary_photo_views"]
                and len(groups["screen_controls/combined"]) <= policy["maximum_control_screenshot_groups"]
                and len(groups["screen_controls/remote"]) >= policy["minimum_ordinary_screenshot_challenge_groups"]
                and counts["screen/edit/combined"]["useful"] >= policy["minimum_useful_screen_edits"]
                and counts["screen/edit/remote"]["useful"] - counts["screen/edit/combined"]["useful"] <= policy["maximum_lost_useful_screen_edits"])
    return {"counts": {key: dict(value) for key, value in counts.items()}, "groups": {key: sorted(value) for key, value in groups.items()},
            "advances": bool(advanced)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "fit", "held"))
    for key in ("commons", "lama", "generated", "baseline", "weight", "tesseract", "tessdata", "run"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "prepare":
        prepare(args)
        return
    lock = freeze(args)
    freeze_path = args.run / "freeze.json"
    if args.stage == "freeze":
        sources.write(freeze_path, lock)
        print(file_sha(freeze_path), flush=True)
        return
    require(read(freeze_path) == lock, "Frozen source commitments or method changed")
    score, calls = model_loader(args.weight)
    if args.stage == "fit":
        rows = [observe(args, lock, row, score, force=True) for row in fit_views(args, lock)]
        sources.write(args.run / "fit.json", {"rows": rows, "points": point(rows), "calls": dict(calls)})
        print(f"Completed {len(rows)} fitting views, {calls['model_inputs']} actual inputs", flush=True)
        return
    fit = read(args.run / "fit.json")
    calculated = point(fit["rows"])
    require(calculated["strict_point"] == fit["points"]["strict_point"], "Frozen fit point changed")
    rows = []
    for view in held_views(args):
        rows.append(observe(args, lock, view, score))
        if len(rows) % 24 == 0:
            print(f"Scored {len(rows)} held views", flush=True)
    result = {"schema": SCHEMA, "freeze": lock, "freeze_sha256": file_sha(freeze_path), "fit": fit,
              "held": rows, "counts": summary(rows, calculated["strict_point"]), "held_model_calls": dict(calls)}
    sources.write(args.run / "result.json", result)
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
