# SPDX-License-Identifier: Apache-2.0
"""Test exact source-patch residual agreement across two native tilings."""
from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import binary_dilation, uniform_filter

from scripts import image_crop_multiscale_transfer as old
from scripts import image_source_groups_v22 as source_groups

SCHEMA = "chipmunk.image-native-tile-agreement-transfer/1"
BASE = old.prior
TILE, PATCH = 384, 16
LAYERS = (1, 5)
OFFSETS = (0, 192)
DIMS = (3, 4, 6, 8, 10, 12, 16)
BATCH = 5
RIDGE = .02
MAX_TRAIN = 24
PRIMARY = old.PRIMARY_ROLES
GENERATED = ("original", "edited", "feathered_edit", "global_processing", "feathered_processing")
CAL = ("original", "tone", "blur", "resample")
ARMS = ("single_context", "agreement")
ROLES = {"fit_lama": (12, PRIMARY), "fit_generated": (7, GENERATED), "cal": (8, CAL),
         "held_lama": (12, PRIMARY), "held_generated": (6, GENERATED), "screen": (12, old.SCREEN_ROLES)}
POS = old.POSITIVE
GATE = {"added_primary_feathered": 2, "added_any_other_feathered": 1,
        "maximum_lost_useful": 1, "maximum_extra_photo_groups": 2, "maximum_extra_screen_groups": 1}
DESIGN = {"tile": TILE, "patch": PATCH, "layers_zero_based": list(LAYERS), "offsets": list(OFFSETS),
          "head": "ridge on native token ring residuals and matched exact-mask controls",
          "fit": {"maximum_cells_each_per_group": MAX_TRAIN, "ridge": RIDGE,
                  "positive": "high-occupancy changed cells on sharp and feathered pairs",
                  "negative": "same cells in ordinary views plus remote cells of edited views",
                  "equal_group_class_weights": True},
          "score": "within-view standardized patch response; elementwise minimum in agreement arm",
          "proposals": {"sides_in_source_cells": list(DIMS), "maximum_aspect_ratio": 2, "power": .15},
          "calibration": "strictly exceed second-highest independent ordinary-group maximum per arm",
          "eligibility": "upright RGB PNG, short side at least 384, long side at most 768",
          "baseline": "earlier frozen one-third relative crop with disjoint context control",
          "useful": {"iou": .30, "recall": .50}, "confidence_weight": 0, "gate": GATE}
require, read, write, sha, file_sha = old.require, old.read, old.write, old.digest, old.file_sha


def sources(args):
    original = read(args.primary / "qualified-private.json")
    backup = read(args.backup / "qualified-private.json")
    former = old.primary_roster(args)[1]
    require(file_sha(args.primary / "qualified-private.json") == read(args.primary / "summary-v225.json")["qualified_private_sha256"]
            and file_sha(args.backup / "qualified-private.json") == read(args.backup / "summary.json")["qualified_private_sha256"],
            "Qualified photo roster changed")
    already = {row["source"]["flickr_photo_id"] for row in former}
    remainder = [row for row in original["qualified"] if row["source"]["flickr_photo_id"] not in already]
    fit = read(args.legacy / "plan.json")["cases"]
    alternate = read(args.alternate / "plan.json")["cases"]
    chosen = []
    for editor in ("kandinsky", "sdxl"):
        available = [row for row in alternate if row["generator"] == editor]
        chosen.extend(sorted(available, key=lambda row: sha(f"native-tile/1/{row['flickr_photo_id']}".encode()))[:3])
    require(len(remainder) == 12 and len(chosen) == 6 and len(fit) == 7 and len(backup["qualified"]) == 12,
            "Expected sealed primary, backup and generator roles")
    identities = [[row["source"] for row in archive] for archive in (backup["qualified"], remainder, former)] + [fit, chosen]
    for key in ("flickr_photo_id", "flickr_owner"):
        names = [[str(item[key]).casefold() for item in panel] for panel in identities]
        flat = [item for group in names for item in group]
        require(len(flat) == len(set(flat)), "Recorded photo roles or contributors overlap")
    return {"fit_lama": (Path(backup["cache"]), backup["qualified"]),
            "held_lama": (Path(original["cache"]), remainder),
            "fit_generated": [row["slot"] for row in fit], "held_generated": [row["slot"] for row in chosen]}


def calibration_sources(args):
    recent = {row["slot"] for row in old.fit_roster(args)}
    older = {row["slot"] for row in read(args.previous / "freeze.json")["fit_sources"]}
    photos = BASE.sources.binding(args.commons)["sources"]
    return sorted((item for item in photos if item["slot"] not in recent|older),
                  key=lambda row: sha(f"native-tile/1/cal/{row['sha256']}".encode()))[:8]


def lama_pairs(args, panel):
    cache, rows = sources(args)[panel]
    for slot, row in enumerate(rows):
        images = []
        for role in ("original", "edited"):
            asset = row["measurement"]["files"][role]
            path = cache / asset["file"]
            require(file_sha(path) == asset["sha256"], "Native publisher image changed")
            images.append(BASE.pixels(path))
        mask_path = cache / row["truth_file"]
        require(file_sha(mask_path) == row["truth_png_sha256"], "Native publisher truth changed")
        with Image.open(mask_path) as image:
            known = np.asarray(image.convert("L")) > 0
        original, edited = images
        require(np.array_equal((original != edited).any(axis=2), known), "Native truth differs from exact changed pixels")
        yield slot, row["source"]["publisher_split"], original, edited, known


def generated_pairs(args, panel):
    folder = args.legacy if panel == "fit_generated" else args.alternate
    slots = sources(args)[panel]
    selected = {case["slot"]: (case, images, known) for case, images, known in BASE.REGIONAL.paired.inputs(folder)
                if case["slot"] in slots}
    require(len(selected) == len(slots), "A selected generated pair is absent")
    for slot, source_slot in enumerate(slots):
        case, images, known = selected[source_slot]
        yield slot, case["generator"], BASE.REGIONAL.native_variants(images, case["box"]), known


def tone(rgb):
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(rgb)).enhance(.9))


def resample(rgb):
    image = Image.fromarray(rgb)
    w, h = image.size
    return np.asarray(image.resize((round(w*.78), round(h*.78)), Image.Resampling.LANCZOS)
                      .resize((w, h), Image.Resampling.BICUBIC))


def png_case(panel, slot, role, family, rgb, truth):
    h, w = truth.shape
    require(min(w, h) >= TILE and max(w, h) <= 768 and rgb.shape == (h, w, 3), "Input outside native tile geometry")
    return (panel, slot, role), family, BASE.png(rgb), rgb, truth


def make_screens(args):
    args.run.mkdir(mode=0o700)
    assets = args.run / "photo-fixtures"
    assets.mkdir()
    plan = []
    for slot, _, original, edited, known in lama_pairs(args, "held_lama"):
        blend = BASE.feather(original, edited, known)
        truth = (original != blend).any(axis=2)
        left, top, right, bottom = BASE.source_box(truth)
        h, w = truth.shape
        cw, ch = min(w, 475), min(h, 456)
        a = max(0, min(w-cw, (left+right-cw)//2))
        b = max(0, min(h-ch, (top+bottom-ch)//2))
        x = (118, 219, 74, 167)[slot % 4] + 6*(slot//4)
        y = (93, 72, 143, 118)[slot % 4] + 5*(slot//4)
        crop = np.s_[b:b+ch, a:a+cw]
        require(truth[crop].sum() >= .8*truth.sum(), "Rendered preview omitted too much known truth")
        plan.append(dict(slot=slot, width=768, height=640, x=x, y=y, w=cw, h=ch,
                         layout=slot % 4, source_box=[a, b, a+cw, b+ch]))
        for role, rgb in (("photo", original), ("edit", blend)):
            (assets / f"{slot}-{role}.png").write_bytes(BASE.png(rgb[crop]))
    write(args.run / "layout-plan.json", plan)


def screens(args):
    plan, captures = (read(args.run / name) for name in ("layout-plan.json", "captures.json"))
    by_key = {(row["slot"], row["role"]): row for row in captures["rows"]}
    require(len(by_key) == len(captures["rows"]) == 36, "Expected complete native screenshot roster")
    for item, pair in zip(plan, lama_pairs(args, "held_lama"), strict=True):
        slot, _, original, edited, known = pair
        a, b, right, bottom = item["source_box"]
        x, y, w, h = (item[name] for name in ("x", "y", "w", "h"))
        require(slot == item["slot"] and (right-a, bottom-b) == (w, h), "Screenshot plan differs")
        preview = {"photo": original[b:bottom, a:right], "edit": BASE.feather(original, edited, known)[b:bottom, a:right]}
        grouped = {}
        for role in old.SCREEN_ROLES:
            path = args.run / "captures" / f"{slot}-{role}.png"
            data, rgb = path.read_bytes(), BASE.pixels(path)
            receipt = by_key[slot, role]
            require(sha(data) == receipt["png_sha256"] and rgb.shape == (640, 768, 3), "Captured image changed")
            if role in preview:
                fixture = args.run / "photo-fixtures" / f"{slot}-{role}.png"
                require(file_sha(fixture) == receipt["photo_sha256"] and np.array_equal(rgb[y:y+h, x:x+w], preview[role]),
                        "Native screenshot preview differs from the photo")
            grouped[role] = data, rgb
        truth = (grouped["photo"][1] != grouped["edit"][1]).any(axis=2)
        require(0 < truth.sum() == truth[y:y+h, x:x+w].sum(), "Screenshot differences leave the photo bounds")
        for role, (data, rgb) in grouped.items():
            yield ("screen", slot, role), f"layout-{item['layout']}", data, rgb, truth if role == "edit" else np.zeros_like(truth)


def views(args, panel):
    if panel.endswith("_lama"):
        for slot, family, original, edited, known in lama_pairs(args, panel):
            blur = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(1.5)))
            variants = original, edited, BASE.feather(original, edited, known), tone(original), BASE.feather(original, blur, known), resample(original)
            for role, rgb in zip(PRIMARY, variants, strict=True):
                truth = (rgb != original).any(axis=2) if role in POS else np.zeros_like(known)
                yield png_case(panel, slot, role, family, rgb, truth)
    elif panel.endswith("_generated"):
        for slot, family, variants, known in generated_pairs(args, panel):
            for role in GENERATED:
                rgb = variants[role]
                truth = (rgb != variants["original"]).any(axis=2) if role in POS else np.zeros_like(known)
                require(not (truth & ~known).any(), "Paired source edit exceeds the qualified footprint")
                yield png_case(panel, slot, role, family, rgb, truth)
    elif panel == "cal":
        for slot, photo in enumerate(calibration_sources(args)):
            image = BASE.native.source(args.commons, photo, (768, 640))
            rgb = np.asarray(image)
            variants = rgb, tone(rgb), np.asarray(image.filter(ImageFilter.GaussianBlur(1.25))), resample(rgb)
            for role, source in zip(CAL, variants, strict=True):
                yield png_case(panel, slot, role, "credited-photo", source, np.zeros(source.shape[:2], dtype=bool))
    else:
        yield from screens(args)


def freeze(args):
    root = args.run
    prior = read(args.baseline / "freeze.json")
    require(file_sha(args.weight) == prior["weight_sha256"] == BASE.baseline.source.WEIGHT_SHA, "Pinned MIT weights changed")
    cases, fingerprints = {}, {}
    for panel in ROLES:
        cases[panel], fingerprints[panel] = [], []
        for case in views(args, panel):
            cases[panel].append(old.identity(case))
            if case[0][-1] == "original":
                with Image.fromarray(case[3]) as image:
                    fingerprints[panel].append({"size": list(image.size), "sketch": f"{source_groups.sketch(image):064x}"})
    for panel, (groups, roles) in ROLES.items():
        require({tuple(row["key"]) for row in cases[panel]} == {(panel, i, role) for i in range(groups) for role in roles},
                "Frozen source roles differ")
    keys = {panel: {row["rgb"] for row in cases[panel] if row["key"][-1] == "original"} for panel in ROLES if panel != "screen"}
    require(sum(map(len, keys.values())) == len(set.union(*keys.values())), "Fitting and held source pixels overlap")
    panels = list(fingerprints)
    compared = sum(len(fingerprints[first])*len(fingerprints[second])
                   for index, first in enumerate(panels) for second in panels[index+1:])
    require(not any(source_groups.near(a, b) for index, first in enumerate(panels) for second in panels[index+1:]
                    for a in fingerprints[first] for b in fingerprints[second]), "Cross-role near-thumbnail overlap")
    paths = (Path(__file__), Path(old.__file__), Path(BASE.__file__), Path(BASE.commfor.__file__),
             Path(old.__file__).with_name("capture_image_crop_multiscale_transfer.cjs"), Path(BASE.baseline.crops.__file__),
             Path(source_groups.__file__), Path(BASE.REGIONAL.__file__), Path(BASE.REGIONAL.paired.__file__))
    return {"schema": SCHEMA, "design": DESIGN, "cases": cases, "weight_sha256": file_sha(args.weight),
            "cross_role_thumbnail_pairs": compared,
            "baseline_points": prior["points"], "code": {path.name: file_sha(path) for path in paths},
            "source_roles_sha256": {name: file_sha(path) for name, path in {
                "commons": args.commons/"acquisition.json", "primary": args.primary/"qualified-private.json",
                "backup": args.backup/"qualified-private.json", "earlier": args.legacy/"result.json",
                "alternate": args.alternate/"result.json", "layout": root/"layout-plan.json", "captures": root/"captures.json"}.items()},
            "versions": {name: version(name) for name in ("pillow", "numpy", "scipy", "torch", "timm", "safetensors")}}


def tiling(rgb, offset):
    h, w = rgb.shape[:2]
    starts = [(y, x) for y in range(-offset, h, TILE) for x in range(-offset, w, TILE)]
    padded = np.pad(rgb, ((offset, starts[-1][0]+TILE-h), (offset, starts[-1][1]+TILE-w), (0, 0)), mode="reflect")
    return starts, [padded[y+offset:y+offset+TILE, x+offset:x+offset+TILE] for y, x in starts]


def reader(weight):
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = BASE.commfor.load_model(weight, "cpu")
    mean = torch.tensor(BASE.commfor.MEAN)[None, :, None, None]
    std = torch.tensor(BASE.commfor.STD)[None, :, None, None]
    caches = {"feature": {}, "baseline": {}}
    calls = Counter()

    def infer(images, kind):
        cache = caches[kind]
        hashes = [sha(image.tobytes()) for image in images]
        pending = {key: image for key, image in zip(hashes, images, strict=True) if key not in cache}
        keys = list(pending)
        for start in range(0, len(keys), BATCH):
            group = keys[start:start+BATCH]
            batch = torch.from_numpy(np.stack([pending[key] for key in group])).permute(0, 3, 1, 2).float()/255
            batch = (batch-mean)/std
            with torch.inference_mode():
                if kind == "feature":
                    layers = model.forward_intermediates(batch, indices=list(LAYERS), stop_early=True,
                                                         intermediates_only=True, output_fmt="NLC")
                    value = torch.cat(layers, dim=-1).reshape(len(group), 24, 24, 768).numpy()
                else:
                    value = BASE.commfor.logits(model, batch, "cpu")
            cache.update(zip(group, value, strict=True))
        calls[kind+"_inputs"] += len(keys)
        calls[kind+"_reuses"] += len(hashes)-len(keys)
        return hashes, [cache[key] for key in hashes]

    def read_pixels(rgb):
        h, w = (math.ceil(n/PATCH) for n in rgb.shape[:2])
        maps, commitments = [], []
        for offset in OFFSETS:
            starts, tiles = tiling(rgb, offset)
            hashes, features = infer(tiles, "feature")
            dense = np.empty((h, w, 768), dtype=np.float32)
            for (top, left), feature in zip(starts, features, strict=True):
                y, x = top//PATCH, left//PATCH
                a, b, c, d = max(0, y), max(0, x), min(h, y+24), min(w, x+24)
                dense[a:c, b:d] = feature[a-y:c-y, b-x:d-x]
            blocks = dense.reshape(h, w, 2, 384)
            unit = blocks / np.maximum(np.linalg.norm(blocks, axis=3, keepdims=True), 1e-5)
            ring = (49*uniform_filter(unit, (7, 7, 1, 1), mode="reflect")
                    -9*uniform_filter(unit, (3, 3, 1, 1), mode="reflect"))/40
            residual = (unit-ring).reshape(h, w, 768)
            require(np.isfinite(residual).all(), "Native token residual has invalid values")
            maps.append(residual)
            commitments.append({"origins_yx": starts, "rgb_sha256": hashes, "residual_sha256": sha(residual.tobytes())})
        return maps, commitments

    def baseline(rgb, points):
        with Image.fromarray(rgb) as image:
            boxes, prepared = zip(*BASE.baseline.crops.crops(image), strict=True)
        hashes, scores = infer(prepared, "baseline")
        result = BASE.baseline.inspect(scores, boxes, points)
        return {"box": result["box"], "flag": result["remote_control"], "scores": scores, "rgb_sha256": hashes}

    def clear():
        for cache in caches.values():
            cache.clear()

    return read_pixels, baseline, calls, clear


def cells(mask):
    h, w = mask.shape
    y = np.arange(0, h, PATCH)
    x = np.arange(0, w, PATCH)
    count = np.add.reduceat(np.add.reduceat(mask, y, axis=0, dtype=np.int64), x, axis=1)
    size = np.diff(np.r_[y, h])[:, None]*np.diff(np.r_[x, w])[None, :]
    return count/size


def grouped(args, panel, lock):
    current = None
    items = {}
    for case, expected in zip(views(args, panel), lock["cases"][panel], strict=True):
        require(old.identity(case) == expected, "Input differs from its role freeze")
        slot, role = case[0][1:]
        if current is not None and slot != current:
            yield current, items
            items = {}
        current = slot
        items[role] = case
    yield current, items


def train(args, lock, extract, clear):
    gram = np.eye(768, dtype=np.float64)*RIDGE
    rhs = np.zeros(768, dtype=np.float64)
    receipts = []
    for panel in ("fit_lama", "fit_generated"):
        for slot, group in grouped(args, panel, lock):
            sharp, feathered = (group[role][4] for role in ("edited", "feathered_edit"))
            occupancy = cells(feathered)
            local = sorted(np.flatnonzero(occupancy.ravel() > 0), key=lambda i: (-occupancy.ravel()[i], int(i)))[:MAX_TRAIN]
            remote = np.flatnonzero(~binary_dilation(cells(sharp) > 0, iterations=2).ravel())
            far = sorted(remote, key=lambda i: sha(f"{panel}/{slot}/{int(i)}".encode()))[:MAX_TRAIN]
            require(bool(local and far), "A fitting group lacks measured interior or remote cells")
            pos, neg, commitments = [], [], []
            for role, case in group.items():
                features, tile = extract(case[3])
                vectors = np.concatenate([item.reshape(-1, 768) for item in features])
                length = features[0].shape[0]*features[0].shape[1]
                same = [*local, *(i+length for i in local)]
                away = [*far, *(i+length for i in far)]
                if role in POS:
                    pos.append(vectors[same])
                    neg.append(vectors[away])
                else:
                    neg.append(vectors[same])
                commitments.append({"role": role, "tiles": tile})
            for label, arrays in ((1., pos), (-1., neg)):
                x = np.concatenate(arrays).astype(np.float64)
                weight = .5 / (19*len(x))
                gram += (x.T@x)*weight
                rhs += label*x.sum(axis=0)*weight
            receipts.append({"panel": panel, "slot": slot, "local_cells": list(map(int, local)), "remote_cells": list(map(int, far)),
                             "positive_samples": sum(map(len, pos)), "negative_samples": sum(map(len, neg)), "tiles": commitments})
            clear()
            print(panel, slot+1, ROLES[panel][0], flush=True)
    direction = np.linalg.solve(gram, rhs)
    require(np.isfinite(direction).all(), "Ridge direction has nonfinite values")
    return direction, receipts


def normalized(a):
    center = np.median(a)
    return (a-center)/max(float(1.4826*np.median(np.abs(a-center))), .02)


def propose(response, shape):
    h, w = response.shape
    integral = np.pad(response.cumsum(0, dtype=np.float64).cumsum(1), ((1, 0), (1, 0)))
    best = -math.inf, None
    for dh, dw in product(DIMS, repeat=2):
        if max(dh, dw) > 2*min(dh, dw) or dh > h or dw > w:
            continue
        means = (integral[dh:, dw:]-integral[:-dh, dw:]-integral[dh:, :-dw]+integral[:-dh, :-dw])/(dh*dw)
        values = means*((dh*dw/36)**.15)
        y, x = np.unravel_index(np.argmax(values), values.shape)
        if float(values[y, x]) > best[0]:
            best = float(values[y, x]), [int(x)*PATCH, int(y)*PATCH,
                                        min(shape[1], (int(x)+dw)*PATCH), min(shape[0], (int(y)+dh)*PATCH)]
    return {"score": best[0], "box": best[1]}


def overlap(box, mask):
    x, y, right, bottom = box
    changed = int(mask[y:bottom, x:right].sum())
    area, target = (right-x)*(bottom-y), int(mask.sum())
    return {"area": area, "target": target, "overlap": changed,
            "useful": bool(target and 2*changed >= target and 10*changed >= 3*(area+target-changed))}


def score(args, panels, lock, direction, extract, baseline, clear, points=None):
    for panel in panels:
        for slot, group in grouped(args, panel, lock):
            for case in group.values():
                key, family, _, rgb, truth = case
                dense, tiles = extract(rgb)
                a, b = [normalized(values.astype(np.float64)@direction) for values in dense]
                maps = a, np.minimum(a, b)
                result = {}
                for arm, values in zip(ARMS, maps, strict=True):
                    chosen = propose(values, truth.shape)
                    result[arm] = chosen | {"pixels": overlap(chosen["box"], truth),
                                            "flag": None if points is None else chosen["score"] > points[arm],
                                            "score_map": values.tolist()}
                if points is not None:
                    old_result = baseline(rgb, lock["baseline_points"])
                    result["research_baseline"] = old_result | {"pixels": overlap(old_result["box"], truth)}
                yield old.identity(case) | {"tiles": tiles, "responses": result, "secondary_score_map": b.tolist()}
            clear()
            print(panel, slot+1, ROLES[panel][0], flush=True)


def threshold(rows):
    maxima = {arm: {} for arm in ARMS}
    for row in rows:
        slot = str(row["key"][1])
        for arm in ARMS:
            values = maxima[arm]
            values[slot] = max(values.get(slot, -math.inf), row["responses"][arm]["score"])
    require(all(len(values) == 8 for values in maxima.values()), "Incomplete calibration role")
    return {arm: sorted(values.values())[-2] for arm, values in maxima.items()}, maxima


def aggregate(rows):
    arms = ("research_baseline", *ARMS)
    controls, by_role = {arm: set() for arm in arms}, {}
    changes = {"added_primary_feathered": [], "added_other_feathered": [], "lost_useful": []}
    for row in rows:
        panel, slot, role = row["key"]
        cell = by_role.setdefault(f"{panel}/{role}", Counter(views=0))
        cell["views"] += 1
        for arm in arms:
            item = row["responses"][arm]
            cell[arm+"_flags"] += item["flag"]
            cell[arm+"_useful"] += item["flag"] and item["pixels"]["useful"]
            cell[arm+"_raw_useful"] += item["pixels"]["useful"]
            if item["flag"] and role not in POS:
                controls[arm].add((panel, slot))
        new, old_item = (row["responses"][arm] for arm in ("agreement", "research_baseline"))
        old_good, new_good = (item["flag"] and item["pixels"]["useful"] for item in (old_item, new))
        if old_good and not new_good:
            changes["lost_useful"].append(row["key"])
        if new_good and not old_good:
            if panel == "held_lama" and role == "feathered_edit":
                changes["added_primary_feathered"].append(row["key"])
            if (panel, role) in (("held_generated", "feathered_edit"), ("screen", "edit")):
                changes["added_other_feathered"].append(row["key"])
    extra = controls["agreement"]-controls["research_baseline"]
    extra_screen = sum(panel == "screen" for panel, _ in extra)
    extra_photo = len(extra)-extra_screen
    passed = (len(changes["added_primary_feathered"]) >= GATE["added_primary_feathered"]
              and len(changes["added_other_feathered"]) >= GATE["added_any_other_feathered"]
              and len(changes["lost_useful"]) <= GATE["maximum_lost_useful"]
              and extra_photo <= GATE["maximum_extra_photo_groups"] and extra_screen <= GATE["maximum_extra_screen_groups"])
    return {"by_role": {key: dict(value) for key, value in by_role.items()}, "changes": changes,
            "control_groups": {arm: {panel: sum(key[0] == panel for key in items)
                                     for panel in ("held_lama", "held_generated", "screen")} for arm, items in controls.items()},
            "extra_photo_groups": extra_photo, "extra_screen_groups": extra_screen, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "fit", "held"))
    for field in ("commons", "primary", "backup", "legacy", "alternate", "previous", "baseline", "weight", "run"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        make_screens(args)
        return
    lock = freeze(args)
    root = args.run
    frozen_path = root / "freeze.json"
    if args.stage == "freeze":
        write(frozen_path, lock)
        print(file_sha(frozen_path), {panel: len(rows) for panel, rows in lock["cases"].items()}, flush=True)
        return
    require(lock == read(frozen_path), "Frozen protocol or source binding changed")
    start = time.monotonic()
    extract, baseline, counters, clear = reader(args.weight)
    head_path, fit_path = root / "direction.npy", root / "fit.json"
    if args.stage == "fit":
        direction, training = train(args, lock, extract, clear)
        with head_path.open("xb") as stream:
            np.save(stream, direction, allow_pickle=False)
        cal = list(score(args, ("cal",), lock, direction, extract, baseline, clear))
        path = root / "cal.json"
        write(path, cal)
        points, maxima = threshold(cal)
        write(fit_path, {"freeze_sha256": file_sha(frozen_path), "direction_sha256": file_sha(head_path),
              "cal_sha256": file_sha(path), "points": points, "maxima": maxima, "train": training,
              "counters": dict(counters), "elapsed_s": time.monotonic()-start})
        return
    fit = read(fit_path)
    require(fit["freeze_sha256"] == file_sha(frozen_path) and fit["direction_sha256"] == file_sha(head_path)
            and fit["cal_sha256"] == file_sha(root / "cal.json") and threshold(read(root / "cal.json"))[0] == fit["points"],
            "Sealed fitting or calibration result differs")
    direction = np.load(head_path, allow_pickle=False)
    rows = list(score(args, ("held_lama", "held_generated", "screen"), lock, direction, extract, baseline, clear, fit["points"]))
    path = root / "held.json"
    write(path, rows)
    result = {"schema": SCHEMA, "freeze_sha256": file_sha(frozen_path), "fit_sha256": file_sha(fit_path),
              "held_sha256": file_sha(path), "counters": dict(counters), "elapsed_s": time.monotonic()-start,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "result": aggregate(rows)}
    write(root / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
