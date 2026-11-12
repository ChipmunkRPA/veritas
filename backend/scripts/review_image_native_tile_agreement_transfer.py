# SPDX-License-Identifier: Apache-2.0
"""Independently verify the native tile and exact-overlap study receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from types import SimpleNamespace

SCHEMA = "chipmunk.image-native-tile-agreement-transfer/1"
ARMS = ("research_baseline", "single_context", "agreement")
POS = {"edited", "feathered_edit", "edit"}
SIZES = (3, 4, 6, 8, 10, 12, 16)
PRIMARY = ("original", "edited", "feathered_edit", "tone", "local_blur", "resample")
GENERATED = ("original", "edited", "feathered_edit", "global_processing", "feathered_processing")
CAL = ("original", "tone", "blur", "resample")
ROLES = {"fit_lama": (12, PRIMARY), "fit_generated": (7, GENERATED), "cal": (8, CAL),
         "held_lama": (12, PRIMARY), "held_generated": (6, GENERATED), "screen": (12, ("template", "photo", "edit"))}


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def check_roster(rows, roles):
    expected = {(panel, slot, role) for panel in roles for slot in range(ROLES[panel][0]) for role in ROLES[panel][1]}
    actual = [tuple(row["key"]) for row in rows]
    need(len(actual) == len(expected) and set(actual) == expected, "Source roles or denominators differ")


def record_counts(box, changed, target):
    x, y, right, bottom = box
    pixels = (right-x)*(bottom-y)
    need(type(changed) is int and type(target) is int and pixels > 0 and 0 <= changed <= min(pixels, target),
         "Pixel overlap is invalid")
    useful = target > 0 and changed/target >= .50 and changed/(pixels+target-changed) >= .30
    return {"area": pixels, "target": target, "overlap": changed, "useful": useful}


def one_third(shape):
    h, w = shape
    return [((w*x+5)//6, (h*y+5)//6, (w*(x+2)+5)//6, (h*(y+2)+5)//6)
            for y in range(5) for x in range(5)]


def verify_box(box, shape, baseline):
    h, w = shape
    x, y, r, b = box
    need(all(type(item) is int for item in box) and 0 <= x < r <= w and 0 <= y < b <= h, "Crop exceeds native source")
    if baseline:
        need(tuple(box) in one_third(shape), "One-third crop geometry differs")
    else:
        dx, dy = math.ceil((r-x)/16), math.ceil((b-y)/16)
        need(x%16 == y%16 == 0 and (r%16 == 0 or r == w) and (b%16 == 0 or b == h)
             and dx in SIZES and dy in SIZES and max(dx, dy) <= 2*min(dx, dy), "Native token crop geometry differs")


def baseline_pick(scores, shape, points):
    boxes = one_third(shape)
    need(len(scores) == 25 and all(math.isfinite(v) for v in scores), "Baseline values differ")
    i = scores.index(max(scores))
    x0, y0, x1, y1 = boxes[i]
    others = [score for box, score in zip(boxes, scores, strict=True) if
              box[2] <= x0 or x1 <= box[0] or box[3] <= y0 or y1 <= box[1]]
    flag = scores[i]-median(scores) >= points["relative"] and median(others) < points["remote_upper_strict"]
    return list(boxes[i]), flag


def points_from_cal(rows):
    check_roster(rows, ("cal",))
    maxima = {arm: {str(slot): max(row["responses"][arm]["score"] for row in rows if row["key"][1] == slot)
                    for slot in range(8)} for arm in ARMS[1:]}
    return {arm: sorted(values.values())[-2] for arm, values in maxima.items()}, maxima


def readout(rows, point, historical):
    check_roster(rows, ("held_lama", "held_generated", "screen"))
    by_role, controls = {}, {arm: set() for arm in ARMS}
    changes = {"added_primary_feathered": [], "added_other_feathered": [], "lost_useful": []}
    for row in rows:
        panel, slot, role = row["key"]
        entries = row["responses"]
        need(set(entries) == set(ARMS), "A study response is missing")
        group = by_role.setdefault(f"{panel}/{role}", Counter(views=0))
        group["views"] += 1
        need((row["target"] > 0) == (role in POS) and row["target"] <= math.prod(row["shape"]), "Ground-truth size differs")
        for arm, item in entries.items():
            verify_box(item["box"], row["shape"], arm == ARMS[0])
            truth = record_counts(item["box"], item["pixels"]["overlap"], row["target"])
            need(truth == item["pixels"], "Measured region and count differ")
            if arm == ARMS[0]:
                choice, flag = baseline_pick(item["scores"], row["shape"], historical)
                need(choice == item["box"], "Frozen one-third candidate differs")
            else:
                need(math.isfinite(item["score"]), "Nonfinite region score")
                flag = item["score"] > point[arm]
            need(type(item["flag"]) is bool and item["flag"] == flag, "Review selection differs")
            group[arm+"_flags"] += flag
            group[arm+"_useful"] += flag and truth["useful"]
            group[arm+"_raw_useful"] += truth["useful"]
            if flag and role not in POS:
                controls[arm].add((panel, slot))
        previous = entries[ARMS[0]]["flag"] and entries[ARMS[0]]["pixels"]["useful"]
        both = entries["agreement"]["flag"] and entries["agreement"]["pixels"]["useful"]
        if previous and not both:
            changes["lost_useful"].append(row["key"])
        if both and not previous:
            if (panel, role) == ("held_lama", "feathered_edit"):
                changes["added_primary_feathered"].append(row["key"])
            if (panel, role) in (("held_generated", "feathered_edit"), ("screen", "edit")):
                changes["added_other_feathered"].append(row["key"])
    group_counts = {arm: {p: sum(panel == p for panel, _ in items) for p in ("held_lama", "held_generated", "screen")}
                    for arm, items in controls.items()}
    extra = controls["agreement"] - controls[ARMS[0]]
    photos, screens = (sum((panel == "screen") == is_screen for panel, _ in extra) for is_screen in (False, True))
    accepted = (len(changes["added_primary_feathered"]) >= 2 and len(changes["added_other_feathered"]) >= 1
                and len(changes["lost_useful"]) <= 1 and photos <= 2 and screens <= 1)
    return {"by_role": {key: dict(value) for key, value in by_role.items()}, "changes": changes,
            "control_groups": group_counts, "extra_photo_groups": photos, "extra_screen_groups": screens, "passed": accepted}


def public(record):
    need(record["schema"] == SCHEMA, "Unsupported tile agreement study")
    recipe = record["design"]
    need(recipe["tile"] == 384 and recipe["patch"] == 16 and recipe["offsets"] == [0, 192]
         and recipe["layers_zero_based"] == [1, 5] and recipe["confidence_weight"] == 0,
         "Frozen tile design differs")
    historical = record["baseline_points"]
    need(historical["relative"] == 8.313690900802614 and historical["remote_upper_strict"] == -4.78482723236084,
         "Historical point differs")
    fit, maxima = points_from_cal(record["cal"])
    need(fit == record["points"] and maxima == record["maxima"], "Calibration differs from its source maxima")
    result = readout(record["held"], fit, historical)
    need(result == record["result"], "Aggregate differs from the held receipts")
    audit = record["native_audit"]
    need(audit["training_groups"] == 19 and audit["views"] == 277 and audit["native_boxes"] == 414
         and audit["independent_score_maps"] == 340 and audit["baseline_crops"] == 3450
         and audit["cross_role_thumbnail_pairs"] == 794, "Independent audit coverage differs")
    return result


def native(args, output):
    import numpy as np
    from PIL import Image
    from scipy.signal import fftconvolve

    from scripts import image_native_tile_agreement_transfer as trial

    lock_path = args.run / "freeze.json"
    lock = read(lock_path)
    need(lock == trial.freeze(args), "Frozen input bytes differ")
    paths = {name: args.run / f"{name}.json" for name in ("fit", "cal", "held", "result")}
    fit, cal, held, result = (read(paths[key]) for key in ("fit", "cal", "held", "result"))
    need(result["freeze_sha256"] == fit["freeze_sha256"] == file_sha(lock_path)
         and result["fit_sha256"] == file_sha(paths["fit"]) and result["held_sha256"] == file_sha(paths["held"])
         and fit["cal_sha256"] == file_sha(paths["cal"]) and fit["direction_sha256"] == file_sha(args.run/"direction.npy"),
         "Saved training or inference receipt changed")
    fitted = np.load(args.run/"direction.npy", allow_pickle=False)
    need(fitted.shape == (768,) and np.isfinite(fitted).all(), "Fitted head has invalid geometry")
    metrics = Counter(cross_role_thumbnail_pairs=lock["cross_role_thumbnail_pairs"])

    def reflected(start, length):
        index = np.arange(start, start+384) % (2*length-2)
        return np.where(index < length, index, 2*length-2-index)

    def verify_tiles(rgb, records):
        h, w = rgb.shape[:2]
        need(len(records) == 2, "Two native tiled contexts are required")
        for offset, saved in zip((0, 192), records, strict=True):
            origins = [[top, left] for top in range(-offset, h, 384) for left in range(-offset, w, 384)]
            need(origins == saved["origins_yx"], "Tile origins differ from the native grid")
            need(len(origins) == len(saved["rgb_sha256"]) and len(saved["residual_sha256"]) == 64,
                 "Missing prepared tile or residual binding")
            for (top, left), expected in zip(origins, saved["rgb_sha256"], strict=True):
                rgb_tile = rgb[np.ix_(reflected(top, h), reflected(left, w))]
                need(sha(rgb_tile.tobytes()) == expected, "Independently reflected source tile changed")
                metrics["tile_inputs"] += 1

    def identity(case):
        key, family, body, rgb, mask = case
        return {"key": list(key), "family": family, "input": sha(body), "rgb": sha(rgb.tobytes()),
                "truth": sha(mask.tobytes()), "shape": list(mask.shape), "target": int(mask.sum())}

    def verify_maximum(values, shape, saved):
        a = np.asarray(values, dtype=np.float64)
        need(a.shape == tuple(math.ceil(n/16) for n in shape) and np.isfinite(a).all(), "Patch score grid invalid")
        verify_box(saved["box"], shape, False)
        x, y, r, b = saved["box"]
        dx, dy = math.ceil((r-x)/16), math.ceil((b-y)/16)
        local = a[y//16:y//16+dy, x//16:x//16+dx].mean()*((dx*dy/36)**.15)
        maximum = max(float(fftconvolve(a, np.ones((dy, dx)), mode="valid").max())/(dx*dy)*((dx*dy/36)**.15)
                      for dy in SIZES for dx in SIZES if max(dx, dy) <= 2*min(dx, dy))
        need(abs(saved["score"]-float(local)) <= 1e-10 and abs(saved["score"]-maximum) <= 1e-10,
             "Independent spatial maximum or coordinate differs")
        metrics["independent_score_maps"] += 1

    training = {(row["panel"], row["slot"]): row for row in fit["train"]}
    need(len(training) == len(fit["train"]) == 19, "Training roster differs")
    for panel in ("fit_lama", "fit_generated"):
        rows = {(row["key"][1], row["key"][-1]): row for row in lock["cases"][panel]}
        masks = {}
        for case in trial.views(args, panel):
            key, _, _, rgb, mask = case
            need(identity(case) == rows[key[1], key[-1]], "Training case pixels differ")
            train = training[panel, key[1]]
            record = next(record for record in train["tiles"] if record["role"] == key[-1])
            verify_tiles(rgb, record["tiles"])
            metrics["views"] += 1
            if key[-1] in POS:
                masks.setdefault(key[1], {})[key[-1]] = mask
        for slot, variants in masks.items():
            sharp, feather = (variants[name] for name in ("edited", "feathered_edit"))
            occupancy, occupied = [], set()
            w = math.ceil(sharp.shape[1]/16)
            h = math.ceil(sharp.shape[0]/16)
            for i in range(h*w):
                y, x = divmod(i, w)
                box = np.s_[16*y:16*(y+1), 16*x:16*(x+1)]
                fraction = float(feather[box].mean())
                if fraction > 0:
                    occupancy.append((-fraction, i))
                if sharp[box].any():
                    occupied.add((y, x))
            local = [i for _, i in sorted(occupancy)[:24]]
            remote = [i for i in range(h*w) if all((i//w+dy, i%w+dx) not in occupied
                      for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx)+abs(dy) <= 2)]
            remote = sorted(remote, key=lambda i: sha(f"{panel}/{slot}/{i}".encode()))[:24]
            train = training[panel, slot]
            negatives = len(local)*2*(len(ROLES[panel][1])-2)+len(remote)*4
            need((local, remote, len(local)*4, negatives) == (train["local_cells"], train["remote_cells"],
                 train["positive_samples"], train["negative_samples"]), "Training source-cell selection differs")
            metrics["training_groups"] += 1

    cal_out, held_out = [], []
    for panel in ("cal", "held_lama", "held_generated", "screen"):
        fitting = panel == "cal"
        journals = [row for row in (cal if fitting else held) if row["key"][0] == panel]
        for case, row, expected in zip(trial.views(args, panel), journals, lock["cases"][panel], strict=True):
            need(identity(case) == expected and all(row[key] == value for key, value in expected.items()), "Scoring input differs")
            _, _, _, rgb, mask = case
            verify_tiles(rgb, row["tiles"])
            metrics["views"] += 1
            responses = row["responses"]
            single, second = np.asarray(responses["single_context"]["score_map"]), np.asarray(row["secondary_score_map"])
            need(second.shape == single.shape and np.isfinite(second).all()
                 and np.array_equal(np.minimum(single, second), responses["agreement"]["score_map"]), "Two-context score fusion differs")
            prepared = {}
            for arm, item in responses.items():
                x, y, r, b = item["box"]
                overlap = int(np.count_nonzero(mask[y:b, x:r]))
                need(record_counts(item["box"], overlap, expected["target"]) == item["pixels"], "Native selected overlap differs")
                if arm == ARMS[0]:
                    boxes = one_third(row["shape"])
                    for (left, top, right, bottom), commitment in zip(boxes, item["rgb_sha256"], strict=True):
                        patch = Image.fromarray(rgb[top:bottom, left:right]).resize((384, 384), Image.Resampling.BILINEAR)
                        need(sha(patch.tobytes()) == commitment, "One-third reference crop pixels differ")
                        metrics["baseline_crops"] += 1
                else:
                    verify_maximum(item["score_map"], mask.shape, item)
                metrics["native_boxes"] += not fitting
                prepared[arm] = {k: v for k, v in item.items() if k not in {"score_map", "rgb_sha256"}}
            public_row = {k: row[k] for k in ("key", "family", "shape", "target")} | {"responses": prepared}
            (cal_out if fitting else held_out).append(public_row)
    record = {"schema": SCHEMA, "design": lock["design"], "weight_sha256": lock["weight_sha256"],
              "baseline_points": lock["baseline_points"], "code": lock["code"],
              "freeze_sha256": file_sha(lock_path), "result_sha256": file_sha(paths["result"]),
              "source_roles_sha256": lock["source_roles_sha256"], "fit_sha256": file_sha(paths["fit"]),
              "points": fit["points"], "maxima": fit["maxima"], "cal": cal_out, "held": held_out, "result": result["result"],
              "native_audit": dict(metrics), "training_inference": fit["counters"], "held_inference": result["counters"],
              "fit_seconds": fit["elapsed_s"], "held_seconds": result["elapsed_s"], "held_peak_rss_bytes": result["peak_rss_bytes"]}
    public(record)
    trial.write(output, record)
    print(dict(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--native", nargs=9, type=Path)
    args = parser.parse_args()
    if args.native:
        names = ["commons", "primary", "backup", "legacy", "alternate", "previous", "baseline", "weight", "run"]
        native(SimpleNamespace(**dict(zip(names, args.native, strict=True))), args.record)
    else:
        print(json.dumps(public(read(args.record)), sort_keys=True))


if __name__ == "__main__":
    main()
