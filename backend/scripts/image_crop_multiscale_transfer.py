# SPDX-License-Identifier: Apache-2.0
"""Freeze and evaluate dimension-only crop scales with qualified native image pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_crop_layout_transfer as prior

SCHEMA = "chipmunk.image-crop-multiscale-transfer/1"
GRID = 24
SCALES = (6, 4, 3)
BATCH = 10
SHAPE = (768, 640)
FIT_ROLES = ("original", "tone", "blur")
PRIMARY_ROLES = ("original", "edited", "feathered_edit", "tone", "local_blur", "resample")
LEGACY_ROLES = ("original", "edited", "feathered_edit", "global_processing")
SCREEN_ROLES = ("template", "photo", "edit")
POSITIVE = {"edited", "feathered_edit", "edit"}
GATE = {"new_primary_feathered_groups": 3, "new_screen_groups": 1, "lost_useful_views": 0,
        "extra_ordinary_photo_groups": 2, "extra_ordinary_ui_groups": 1}
DESIGN = {"grid": GRID, "divisors": list(SCALES), "stride": "half a crop", "scale_order_for_ties": list(SCALES),
          "resize": "Pillow RGB bilinear to 384 square", "response": "maximum logit less median at each scale",
          "combined": "maximum scale response; report the winning source rectangle only",
          "point": "strictly greater than the second highest of eight fit group maxima",
          "baseline": "frozen one-third relative crop with disjoint crop control",
          "useful": {"minimum_iou": .3, "minimum_recall": .5}, "confidence_weight": 0, "gate": GATE}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positions(width, height, divisor):
    side = GRID // divisor
    ticks = range(0, GRID - side + 1, side // 2)
    def bound(length, tick):
        return (length * tick + GRID - 1) // GRID
    return [(bound(width, x), bound(height, y), bound(width, x + side), bound(height, y + side))
            for y, x in product(ticks, repeat=2)]


def primary_roster(args):
    name = "qualified-private.json"
    roster = read(args.primary / name)
    previous = read(args.backup / name)
    require(file_sha(args.primary / name) == read(args.primary / "summary-v225.json")["qualified_private_sha256"]
            and file_sha(args.backup / name) == read(args.backup / "summary.json")["qualified_private_sha256"],
            "The qualified source rosters changed")
    fields = ("flickr_owner", "flickr_photo_id")
    for field in fields:
        groups = [{row["source"][field] for row in panel["qualified"]} for panel in (roster, previous)]
        require([len(group) for group in groups] == [24, 12] and not groups[0] & groups[1], "Source roles overlap")
        legacy = {row[field] for row in read(args.legacy / "plan.json")["cases"]}
        require(len(legacy) == 7 and not groups[0] & legacy, "Legacy and primary source roles overlap")
    strata = sorted({row["source"]["publisher_split"] for row in roster["qualified"]})
    chosen = []
    for split in strata:
        rows = [row for row in roster["qualified"] if row["source"]["publisher_split"] == split]
        chosen.extend(sorted(rows, key=lambda row: digest(f"crop-multiscale/1/{row['measurement']['files']['original']['sha256']}".encode()))[:3])
    require(len(strata) == 4 and len(chosen) == 12, "Expected three groups in each publisher split")
    return Path(roster["cache"]), chosen


def fit_roster(args):
    former = {row["slot"] for row in read(args.previous / "freeze.json")["fit_sources"]}
    current = prior.sources.binding(args.commons)["sources"]
    names = read(args.commons / "plan.json")["sources"]
    selected = sorted((item for item in current if item["slot"] not in former),
                      key=lambda row: digest(f"crop-multiscale/1/{row['sha256']}".encode()))[:8]
    require(len({names[row["slot"]]["author"] for row in selected}) == 8, "Fitting photo authors overlap")
    primary = primary_roster(args)[1]
    legacy = read(args.legacy / "plan.json")["cases"]
    held = {row["measurement"]["files"]["original"]["sha256"] for row in primary}
    held |= {json.loads(line)["assets"]["original"]["sha256"] for line in
             (args.legacy / "attempts.jsonl").read_text().splitlines()}
    require(not {row["sha256"] for row in selected} & held, "Exact fitting and held photos overlap")
    owners = {str(names[row["slot"]]["author"]).strip().casefold() for row in selected}
    held_owners = {str(row["flickr_owner"]).strip().casefold() for row in legacy}
    held_owners |= {str(row["source"]["flickr_owner"]).strip().casefold() for row in primary}
    require(not owners & held_owners, "Recorded fitting and held contributors overlap")
    return selected


def primary_sources(args):
    cache, records = primary_roster(args)
    for slot, row in enumerate(records):
        images = {}
        for role in ("original", "edited"):
            asset = row["measurement"]["files"][role]
            path = cache / asset["file"]
            require(file_sha(path) == asset["sha256"], "Qualified photo bytes changed")
            images[role] = prior.pixels(path)
        path = cache / row["truth_file"]
        require(file_sha(path) == row["truth_png_sha256"], "Qualified truth file changed")
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        original, edited = (images[key] for key in ("original", "edited"))
        require(mask.any() and np.array_equal((original != edited).any(axis=2), mask), "Qualified native edit changed")
        yield slot, row["source"]["publisher_split"], original, edited, mask


def prepare(args):
    args.run.mkdir(mode=0o700)
    fixtures = args.run / "photo-fixtures"
    fixtures.mkdir()
    plan = []
    for slot, _, original, edited, known in primary_sources(args):
        blended = prior.feather(original, edited, known)
        truth = (original != blended).any(axis=2)
        left, top, right, bottom = prior.source_box(truth)
        h, w = truth.shape
        cw, ch = min(w, 468), min(h, 438)
        a = max(0, min(w-cw, (left+right-cw)//2))
        b = max(0, min(h-ch, (top+bottom-ch)//2))
        x, y = ((152, 95), (241, 78), (58, 148), (187, 132))[slot % 4]
        roi = np.s_[b:b+ch, a:a+cw]
        require(x+cw <= SHAPE[0] and y+ch <= SHAPE[1] and truth[roi].sum() >= .8*truth.sum(),
                "Screenshot could not preserve the known region at native scale")
        plan.append(dict(slot=slot, width=SHAPE[0], height=SHAPE[1], x=x, y=y, w=cw, h=ch,
                         layout=slot % 4, source_box=[a, b, a+cw, b+ch]))
        for role, rgb in (("photo", original), ("edit", blended)):
            (fixtures / f"{slot}-{role}.png").write_bytes(prior.png(rgb[roi]))
    write(args.run / "layout-plan.json", plan)


def screen_views(args):
    plan = read(args.run / "layout-plan.json")
    capture = read(args.run / "captures.json")
    captures = {(row["slot"], row["role"]): row for row in capture["rows"]}
    require(len(captures) == len(capture["rows"]) == 36 and len(plan) == 12, "Incomplete screenshot capture")
    for group, source in zip(plan, primary_sources(args), strict=True):
        slot, _, original, edited, known = source
        a, b, right, bottom = group["source_box"]
        x, y, w, h = (group[key] for key in ("x", "y", "w", "h"))
        require(group["slot"] == slot and (right-a, bottom-b) == (w, h), "Screenshot position differs")
        expected = {"photo": original[b:bottom, a:right], "edit": prior.feather(original, edited, known)[b:bottom, a:right]}
        views = {}
        for role in SCREEN_ROLES:
            row = captures[slot, role]
            path = args.run / "captures" / f"{slot}-{role}.png"
            body, rgb = path.read_bytes(), prior.pixels(path)
            require(digest(body) == row["png_sha256"] and rgb.shape[:2] == SHAPE[::-1], "Screenshot bytes changed")
            if role in expected:
                fixture = args.run / "photo-fixtures" / f"{slot}-{role}.png"
                require(file_sha(fixture) == row["photo_sha256"] and np.array_equal(rgb[y:y+h, x:x+w], expected[role]),
                        "Displayed photo changed its native pixels")
            views[role] = body, rgb
        truth = (views["photo"][1] != views["edit"][1]).any(axis=2)
        require(0 < truth.sum() == truth[y:y+h, x:x+w].sum(), "Screenshot change escaped the photo")
        for role, (body, rgb) in views.items():
            yield ("screen", slot, role), f"layout-{group['layout']}", body, rgb, truth if role == "edit" else np.zeros_like(truth)


def case_views(args, fitting=False):
    if fitting:
        for slot, source in enumerate(fit_roster(args)):
            image = prior.native.source(args.commons, source, SHAPE)
            views = (image, ImageEnhance.Contrast(image).enhance(.89), image.filter(ImageFilter.GaussianBlur(1.3)))
            for role, view in zip(FIT_ROLES, views, strict=True):
                rgb = np.asarray(view)
                yield ("fit", slot, role), "credited-photo", prior.png(rgb), rgb, np.zeros(rgb.shape[:2], dtype=bool)
        return
    for slot, split, original, edited, known in primary_sources(args):
        image = Image.fromarray(original)
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(1.5)))
        w, h = image.size
        resample = image.resize((round(w*.77), round(h*.77)), Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)
        views = (original, edited, prior.feather(original, edited, known), np.asarray(ImageEnhance.Contrast(image).enhance(.9)),
                 prior.feather(original, blurred, known), np.asarray(resample))
        for role, rgb in zip(PRIMARY_ROLES, views, strict=True):
            truth = (original != rgb).any(axis=2) if role in POSITIVE else np.zeros_like(known)
            yield ("primary", slot, role), split, prior.png(rgb), rgb, truth
    for case, images, known in prior.REGIONAL.paired.inputs(args.legacy):
        variants = prior.REGIONAL.native_variants(images, case["box"])
        for role in LEGACY_ROLES:
            rgb = variants[role]
            data, decoded = prior.REGIONAL.paired.encoded(rgb, "png")
            truth = (rgb != images["original"]).any(axis=2) if role in POSITIVE else np.zeros_like(known)
            yield ("legacy", case["slot"], role), case["generator"], data, np.asarray(decoded), truth
    yield from screen_views(args)


def identity(case):
    key, family, body, rgb, truth = case
    require(bool(truth.any()) == (key[-1] in POSITIVE), "A case lost its known target")
    return {"key": list(key), "family": family, "input": digest(body), "rgb": digest(rgb.tobytes()),
            "truth": digest(truth.tobytes()), "shape": list(truth.shape), "target": int(truth.sum())}


def freeze(args):
    baseline = read(args.baseline / "freeze.json")
    require(file_sha(args.weight) == prior.baseline.source.WEIGHT_SHA == baseline["weight_sha256"], "MIT checkpoint differs")
    paths = (Path(__file__), Path(__file__).with_name("capture_image_crop_multiscale_transfer.cjs"),
             Path(prior.__file__), Path(prior.baseline.__file__), Path(prior.baseline.crops.__file__), Path(prior.commfor.__file__))
    cases = {role: [identity(case) for case in case_views(args, fitting)] for role, fitting in (("fit", True), ("held", False))}
    require(len(cases["fit"]) == 24 and len(cases["held"]) == 136
            and len({tuple(row["key"]) for rows in cases.values() for row in rows}) == 160,
            "Frozen panel denominator differs")
    return {"schema": SCHEMA, "design": DESIGN, "code": {path.name: file_sha(path) for path in paths},
            "baseline_points": baseline["points"], "checkpoint": file_sha(args.weight),
            "source_commitments": {"commons": file_sha(args.commons / "acquisition.json"),
                "primary": file_sha(args.primary / "qualified-private.json"), "backup": file_sha(args.backup / "qualified-private.json"),
                "legacy": file_sha(args.legacy / "attempts.jsonl"), "layouts": file_sha(args.run / "layout-plan.json"),
                "captures": file_sha(args.run / "captures.json")},
            "versions": {name: version(name) for name in ("pillow", "numpy", "torch", "timm", "safetensors")}
            | {"opencv": prior.cv2.__version__},
            "batch": BATCH, "cases": cases}


def measure(box, integral, target):
    x, y, right, bottom = box
    overlap = int(integral[bottom, right]-integral[y, right]-integral[bottom, x]+integral[y, x])
    area = (right-x)*(bottom-y)
    return {"reported": area, "overlap": overlap, "target": target,
            "useful": bool(target and 2*overlap >= target and 10*overlap >= 3*(area+target-overlap))}


def geometry(truth, boxes):
    target = int(truth.sum())
    integral = np.pad(truth.cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))
    output = {}
    for scale, regions in boxes.items():
        counts = [measure(box, integral, target) for box in regions] if target else []
        output[str(scale)] = {"useful_proposals": sum(value["useful"] for value in counts),
                              "best_iou": max((v["overlap"]/(v["reported"]+target-v["overlap"]) for v in counts), default=0.)}
    return output, integral


def geometry_only(args, lock):
    rows = []
    for case, expected in zip(case_views(args), lock["cases"]["held"], strict=True):
        require(identity(case) == expected, "Held input differs from the freeze")
        key, _, _, rgb, truth = case
        if key[-1] in POSITIVE:
            h, w = truth.shape
            result, _ = geometry(truth, {scale: positions(w, h, scale) for scale in SCALES})
            rows.append({"key": list(key), "geometry": result})
    write(args.run / "geometry.json", {"rows": rows, "summary": geometry_summary(rows)})


def geometry_summary(rows):
    counts = {}
    for row in rows:
        key = "/".join(row["key"][::2])
        value = counts.setdefault(key, Counter())
        value["views"] += 1
        for scale, summary in row["geometry"].items():
            value[scale] += summary["useful_proposals"] > 0
    return {key: dict(count) for key, count in counts.items()}


def loader(weight):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = prior.commfor.load_model(weight, "cpu")
    mean = torch.tensor(prior.commfor.MEAN)[None, :, None, None]
    std = torch.tensor(prior.commfor.STD)[None, :, None, None]
    cache, counter = {}, Counter()

    def score(image, boxes):
        data, inputs = {}, []
        for box in boxes:
            with image.crop(box) as source, source.resize((384, 384), Image.Resampling.BILINEAR) as crop:
                rgb = np.asarray(crop)
            key = digest(rgb.tobytes())
            inputs.append(key)
            if key not in cache:
                data.setdefault(key, rgb)
        missing = list(data)
        for start in range(0, len(missing), BATCH):
            keys = missing[start:start+BATCH]
            batch = torch.from_numpy(np.stack([data[k] for k in keys])).permute(0, 3, 1, 2).float() / 255
            values = prior.commfor.logits(model, (batch-mean)/std, "cpu")
            cache.update(zip(keys, values, strict=True))
            counter["batches"] += 1
        counter["new_model_inputs"] += len(missing)
        counter["exact_input_reuses"] += len(inputs)-len(missing)
        return inputs, [cache[key] for key in inputs]

    return score, counter


def responses(regions, scores, baseline_points):
    peaks = {}
    for scale in SCALES:
        values = scores[scale]
        require(all(math.isfinite(value) for value in values), "A crop response is nonfinite")
        index = max(range(len(values)), key=values.__getitem__)
        peaks[str(scale)] = {"box": list(regions[scale][index]), "response": values[index]-float(np.median(values))}
    scale = max(peaks, key=lambda item: peaks[item]["response"])
    box, relative, remote = prior.baseline.method.readout(regions[3], scores[3])
    return peaks, scale, {"box": list(box), "relative": relative, "remote": remote,
                           "flag": relative >= baseline_points["relative"] and remote < baseline_points["remote_upper_strict"]}


def collect(args, lock, fitting, point=None):
    role = "fit" if fitting else "held"
    score, counters = loader(args.weight)
    start = time.monotonic()
    path = args.run / f"{role}-rows.jsonl"
    with path.open("x") as journal:
        for i, (case, expected) in enumerate(zip(case_views(args, fitting), lock["cases"][role], strict=True), 1):
            require(identity(case) == expected, "Scored input differs from the freeze")
            _, _, _, rgb, truth = case
            h, w = truth.shape
            boxes = {scale: positions(w, h, scale) for scale in SCALES}
            with Image.fromarray(rgb) as image:
                hashes, values = {}, {}
                for scale, rects in boxes.items():
                    hashes[scale], values[scale] = score(image, rects)
            peaks, winner, baseline = responses(boxes, values, lock["baseline_points"])
            ceiling, integral = geometry(truth, boxes)
            target = expected["target"]
            for row in [*peaks.values(), baseline]:
                row["overlap"] = measure(row["box"], integral, target)
            row = {**expected, "hashes": hashes, "logits": values, "peaks": peaks, "winner": winner,
                   "baseline": baseline, "geometry": ceiling,
                   "candidate_flag": None if fitting else peaks[winner]["response"] > point}
            journal.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            journal.flush()
            if i % 6 == 0 or i == len(lock["cases"][role]):
                print(role, i, len(lock["cases"][role]), dict(counters), round(time.monotonic()-start, 1), flush=True)
    return dict(counters) | {"elapsed_s": round(time.monotonic()-start, 3)}


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def fitted(rows):
    maxima = {}
    for row in rows:
        slot = str(row["key"][1])
        score = row["peaks"][row["winner"]]["response"]
        maxima[slot] = max(maxima.get(slot, -math.inf), score)
    require(len(maxima) == 8, "Expected eight fitting maxima")
    return {"point": sorted(maxima.values())[-2], "maxima": maxima}


def summary(rows):
    breakdown, groups = {}, {"baseline": set(), "candidate": set()}
    changes = {"added_primary_feathered": [], "added_screen": [], "lost_useful": []}
    for row in rows:
        panel, slot, role = row["key"]
        cell = breakdown.setdefault(panel + "/" + role, Counter())
        cell["views"] += 1
        old, new = row["baseline"], row["peaks"][row["winner"]]
        for arm, flag, item in (("baseline", old["flag"], old), ("candidate", row["candidate_flag"], new)):
            cell[arm+"_flags"] += flag
            cell[arm+"_useful"] += flag and item["overlap"]["useful"]
            if role not in POSITIVE and flag:
                groups[arm].add((panel, slot))
        for scale, item in row["peaks"].items():
            cell["top_"+scale+"_useful"] += item["overlap"]["useful"]
        cell["top_winner_useful"] += new["overlap"]["useful"]
        old_useful = old["flag"] and old["overlap"]["useful"]
        new_useful = row["candidate_flag"] and new["overlap"]["useful"]
        if old_useful and not new_useful:
            changes["lost_useful"].append(row["key"])
        if new_useful and not old_useful and panel == "primary" and role == "feathered_edit":
            changes["added_primary_feathered"].append(slot)
        if new_useful and not old_useful and panel == "screen" and role == "edit":
            changes["added_screen"].append(slot)
    group_counts = {arm: {key: sum(p == key for p, _ in items) for key in ("primary", "legacy", "screen")}
                    for arm, items in groups.items()}
    extra = groups["candidate"] - groups["baseline"]
    extra_photos = sum(p != "screen" for p, _ in extra)
    extra_screens = sum(p == "screen" for p, _ in extra)
    passed = (len(changes["added_primary_feathered"]) >= GATE["new_primary_feathered_groups"]
              and len(changes["added_screen"]) >= GATE["new_screen_groups"]
              and len(changes["lost_useful"]) <= GATE["lost_useful_views"]
              and extra_photos <= GATE["extra_ordinary_photo_groups"] and extra_screens <= GATE["extra_ordinary_ui_groups"])
    return {"breakdown": {key: dict(value) for key, value in breakdown.items()}, "control_groups": group_counts,
            "changes": changes, "extra_control_photo_groups": extra_photos, "extra_control_ui_groups": extra_screens,
            "geometry": geometry_summary(row for row in rows if row["key"][-1] in POSITIVE), "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "geometry", "fit", "held"))
    for field in ("commons", "primary", "backup", "legacy", "previous", "baseline", "weight", "run"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return
    freeze_path = args.run / "freeze.json"
    lock = freeze(args)
    if args.stage == "freeze":
        write(freeze_path, lock)
        print(file_sha(freeze_path), flush=True)
        return
    require(read(freeze_path) == lock, "The frozen design, sources or code changed")
    if args.stage == "geometry":
        geometry_only(args, lock)
        print(read(args.run / "geometry.json")["summary"], flush=True)
        return
    fit_path, held_path = (args.run / f"{role}-rows.jsonl" for role in ("fit", "held"))
    if args.stage == "fit":
        counters = collect(args, lock, True)
        write(args.run / "fit.json", {**fitted(journal(fit_path)), "journal_sha256": file_sha(fit_path), "counters": counters})
        return
    fit = read(args.run / "fit.json")
    require(fit["journal_sha256"] == file_sha(fit_path) and all(fit[k] == v for k, v in fitted(journal(fit_path)).items()),
            "Fit point differs from the ordinary-group journal")
    counters = collect(args, lock, False, fit["point"])
    rows = journal(held_path)
    require(read(args.run / "geometry.json")["summary"] == geometry_summary(row for row in rows if row["key"][-1] in POSITIVE),
            "The measured geometry differs from the pre-inference readout")
    result = {"schema": SCHEMA, "freeze_sha256": file_sha(freeze_path), "geometry_sha256": file_sha(args.run / "geometry.json"),
              "fit": fit, "held_sha256": file_sha(held_path), "held_counters": counters, "result": summary(rows)}
    write(args.run / "result.json", result)
    print(result["result"], flush=True)


if __name__ == "__main__":
    main()
