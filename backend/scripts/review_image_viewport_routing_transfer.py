# SPDX-License-Identifier: Apache-2.0
"""Independent exact-crop and portable scalar review for viewport routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "chipmunk.image-viewport-routing-transfer/1"
POINT = .8292734622955322
ROLES = {"screen": (12, ("template", "photo", "sharp", "feather", "ambiguous")),
         "direct": (12, ("photo", "sharp", "feather", "tone")), "standalone": (8, ("photo", "tone"))}
POS = {"sharp", "feather"}
ARMS = ("whole", "routed", "exact_viewport")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def area(box):
    x, y, right, bottom = box
    return (right-x)*(bottom-y)


def box_overlap(first, second):
    if first is None or second is None:
        return 0.
    cross = math.prod(max(0, min(first[i+2], second[i+2])-max(first[i], second[i])) for i in (0, 1))
    return cross/(area(first)+area(second)-cross)


def boxed(box, bounds):
    return (len(box) == 4 and all(type(value) is int for value in box)
            and bounds[0] <= box[0] < box[2] <= bounds[2] and bounds[1] <= box[1] < box[3] <= bounds[3])


def measure(item, crop, target):
    sides = item["sides"]
    need(set(sides) == {"left", "right", "top", "bottom"} and all(-2e-6 <= value <= 3.000002 for value in sides.values())
         and math.isfinite(item["score"]) and abs(item["score"]-max(0., math.prod(sides.values()))**.25) <= 2e-5,
         "Four-sided boundary arithmetic differs")
    box = item["box"]
    need(boxed(box, crop) and area(box) == item["area"] and 0 <= item["overlap"] <= min(item["area"], target),
         "Selected native region differs from its crop")
    need(type(item["flag"]) is bool and item["flag"] == (item["score"] > POINT), "Frozen boundary flag changed")
    overlap = item["overlap"]
    useful = bool(target and overlap/target >= .50 and overlap/(item["area"]+target-overlap) >= .30)
    need(useful == item["useful"], "Changed-pixel localization rule differs")
    return item["flag"], useful


def summary(rows):
    expected = {(panel, slot, role) for panel,(count, roles) in ROLES.items() for slot in range(count) for role in roles}
    keys = [tuple(row["key"]) for row in rows]
    need(len(keys) == len(expected) and set(keys) == expected, "Frozen held coverage differs")
    totals, deltas, groups = {}, Counter(), set()
    for row in rows:
        panel, slot, role = row["key"]
        target, (h, w) = row["target"], row["shape"]
        bounds = [0, 0, w, h]
        route, viewport = row["route"], row["viewport"]
        need((target > 0) == (role in POS) and 0 <= target <= w*h, "Native changed-pixel denominator differs")
        need((viewport is not None) == (panel == "screen" and role in {"photo", *POS}), "Exact photo viewport assigned incorrectly")
        need(row["route_iou"] == box_overlap(route, viewport), "Photo viewport overlap differs")
        if route is not None:
            need(boxed(route, bounds) and min(route[2]-route[0], route[3]-route[1]) >= 512
                 and max(route[2]-route[0], route[3]-route[1]) <= 768 and len(row["aligned_exterior"]) >= 3,
                 "Photo viewport lacks external layout evidence or scope")
        else:
            need(row["whole"] == row["routed"], "Uncertain viewport changed whole-image behavior")
        results = {"whole": measure(row["whole"], bounds, target), "routed": measure(row["routed"], route or bounds, target),
                   "exact_viewport": measure(row["exact_viewport"], viewport, target) if viewport else (False, False)}
        slot_counts = totals.setdefault(f"{panel}/{role}", Counter(views=0))
        slot_counts.update(views=1, routed=route is not None, correct_viewports=row["route_iou"] >= .95)
        for arm in ARMS:
            flag, good = results[arm]
            slot_counts.update({arm+"_flags": flag, arm+"_useful": flag and good})
        base, routed = results["whole"], results["routed"]
        deltas.update(lost_useful=all(base) and not all(routed),
                      added_sharp_screen=panel == "screen" and role == "sharp" and all(routed) and not all(base),
                      added_ordinary_screen_flags=panel == "screen" and role not in POS and routed[0] and not base[0],
                      removed_ordinary_screen_flags=panel == "screen" and role not in POS and base[0] and not routed[0])
        if route and panel != "screen" and role not in POS:
            groups.add((panel, slot))
    deltas["misrouted_standalone_groups"] = len(groups)
    passed = (totals["screen/photo"]["correct_viewports"] >= 8 and totals["screen/ambiguous"]["routed"] <= 2
              and deltas["lost_useful"] <= 1 and deltas["added_sharp_screen"] >= 2 and deltas["removed_ordinary_screen_flags"] >= 6
              and deltas["added_ordinary_screen_flags"] == 0 and deltas["misrouted_standalone_groups"] <= 1)
    return {"by_role": {key: dict(value) for key, value in totals.items()}, "changes": dict(deltas), "passed": passed}


def public(record):
    need(record["schema"] == SCHEMA and record["point"] == POINT, "Study identity differs")
    result = summary(record["rows"])
    need(result == record["result"], "Held aggregate differs")
    audit = record["native_audit"]
    need(audit["views"] == 124 and audit["oracle_pairs"] == 36 and audit["cross_role_photo_pairs"] == 96
         and audit["crop_replays"] == 160+sum(row["route"] is not None for row in record["rows"]),
         "Native audit coverage differs")
    geometry = record["scale_diagnostic"]
    useful = result["by_role"]["screen/sharp"]["whole_useful"]
    need(geometry["full_sharp_useful"] == useful and geometry["full_sharp_exact_crop_same_edges"] <= useful
         and geometry["full_sharp_exact_crop_same_edges_but_size_excluded"] <= geometry["full_sharp_exact_crop_same_edges"]
         and geometry["full_sharp_fixed_box_clean_flags"] <= useful and geometry["full_sharp_box_has_viewport_side"] <= useful,
         "Scale diagnostic coverage differs")
    return result


def native(args, output):
    import numpy as np

    from scripts import image_viewport_routing_transfer as trial

    freeze, journal, result = (args.run/f"{name}.json" for name in ("freeze", "rows", "result"))
    lock, rows, observed = read(freeze), read(journal), read(result)
    need(lock == trial.freeze(args) and file_sha(freeze) == observed["freeze_sha256"] and file_sha(journal) == observed["rows_sha256"],
         "Original files or source seal changed")
    counts = Counter(cross_role_photo_pairs=lock["cross_role_photo_pairs"])
    indexed = {tuple(row["key"]): row for row in rows}

    def side(horizontal_crop, line, start, stop):
        rgb = horizontal_crop[start:stop].astype(np.float64)
        def differences(x):
            return np.abs(rgb[:, x]-rgb[:, x-1]).mean(axis=1) if x else np.zeros(len(rgb))
        center = differences(line)
        around = (differences(line-3)+differences(line+3))/2
        return float(np.clip(np.log((center+3)/(around+3)), 0, 3).mean())

    def sides(pixels, box):
        a, b, c, d = box
        transposed = pixels.transpose(1, 0, 2)
        return dict(left=side(pixels, a, b, d), right=side(pixels, c, b, d),
                    top=side(transposed, b, a, c), bottom=side(transposed, d, a, c))

    def dimension_choices(shape):
        return {(round(shape[1]*x), round(shape[0]*y)) for x in (1/6, 1/4, 1/3) for y in (1/6, 1/4, 1/3)}

    def possible(mask, sizes):
        h, w = mask.shape
        target = int(mask.sum())
        integral = np.pad(mask.cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))
        for dw, dh in sizes:
            if dw+6 >= w or dh+6 >= h:
                continue
            counts = integral[3+dh:h-3, 3+dw:w-3]-integral[3:h-3-dh, 3+dw:w-3]
            counts -= integral[3+dh:h-3, 3:w-3-dw]-integral[3:h-3-dh, 3:w-3-dw]
            if ((counts*2 >= target) & (counts*10 >= 3*(dw*dh+target-counts))).any():
                return True
        return False

    geometry = Counter(full_sharp_useful=0, full_sharp_exact_crop_same_edges=0, full_sharp_exact_crop_same_edges_but_size_excluded=0,
                       full_sharp_fixed_box_clean_flags=0, full_sharp_box_has_viewport_side=0, full_sharp_box_leaves_viewport=0,
                       full_sharp_fixed_box_is_clean_winner=0)
    clean_screens = {}
    for case, row, expected in zip(trial.views(args), rows, lock["cases"], strict=True):
        key, family, body, rgb, mask, viewport = case
        identity = {"key": list(key), "family": family, "input_sha256": sha(body), "rgb_sha256": sha(rgb.tobytes()),
                    "mask_sha256": sha(mask.tobytes()), "shape": list(mask.shape), "target": int(np.count_nonzero(mask)), "viewport": viewport}
        need(identity == expected and all(row[name] == value for name, value in identity.items()), "Exact source pixels changed")
        if key[0] == "screen" and key[-1] == "photo":
            clean_screens[key[1]] = rgb
        locator = trial.viewport_pixels(rgb)
        need(locator == row["locator"], "Native texture viewport replay differs")
        if locator is not None:
            words = trial.prior.ocr_boxes(args, body)
            counts["ocr_replays"] += 1
            need(words == row["ocr_boxes"] and trial.aligned_exterior(words, locator["box"]) == row["aligned_exterior"],
                 "Exterior word geometry differs")
        need(row["route"] == (locator["box"] if row["aligned_exterior"] else None), "Exterior gating differs")
        h, w = mask.shape
        for arm, crop in (("whole", [0, 0, w, h]), ("routed", row["route"]), ("exact_viewport", viewport)):
            if crop is None:
                continue
            x, y, right, bottom = crop
            pixels = rgb[y:bottom, x:right]
            item = row[arm]
            need(sha(pixels.tobytes()) == item["input_crop_sha256"], "Rebuilt inspection crop differs")
            a, b, c, d = item["box"]
            need(int(np.count_nonzero(mask[b:d,a:c])) == item["overlap"], "Native changed-pixel overlap differs")
            expected_sides = sides(pixels, (a-x, b-y, c-x, d-y))
            need(all(abs(value-item["sides"][name]) <= 2e-5 for name, value in expected_sides.items()),
                 "Independent native edge evidence differs")
            counts["crop_replays"] += 1
        if viewport:
            direct = indexed[("direct", key[1], key[-1])]["whole"]
            exact = row["exact_viewport"]
            expected_box = [value+viewport[i%2] for i,value in enumerate(direct["box"])]
            need(exact["box"] == expected_box and all(exact[name] == direct[name] for name in
                 ("input_crop_sha256", "score", "sides", "flag", "overlap", "useful")),
                 "The exact screenshot crop differs from the standalone photo check")
            counts["oracle_pairs"] += 1
            if key[-1] in POS:
                x, y, right, bottom = viewport
                clipped = mask[y:bottom,x:right]
                size_set, crop_sizes = dimension_choices(mask.shape), dimension_choices(clipped.shape)
                role = key[-1]
                geometry[role+"_full_dimension_ceiling"] += possible(mask, size_set)
                geometry[role+"_crop_dimension_ceiling"] += possible(clipped, crop_sizes)
                geometry[role+"_full_sizes_within_crop_ceiling"] += possible(clipped, size_set)
                whole = row["whole"]
                if role == "sharp" and whole["flag"] and whole["useful"]:
                    geometry["full_sharp_useful"] += 1
                    a, b, c, d = whole["box"]
                    clean = sides(clean_screens[key[1]], whole["box"])
                    geometry["full_sharp_fixed_box_clean_flags"] += math.prod(clean.values())**.25 > POINT
                    geometry["full_sharp_box_has_viewport_side"] += any(abs(p-q) <= 3 for p,q in zip(whole["box"], viewport, strict=True))
                    geometry["full_sharp_box_leaves_viewport"] += not boxed(whole["box"], viewport)
                    geometry["full_sharp_fixed_box_is_clean_winner"] += whole["box"] == indexed[("screen",key[1],"photo")]["whole"]["box"]
                    if a-x >= 4 and b-y >= 4 and right-c >= 4 and bottom-d >= 4:
                        replay = sides(rgb[y:bottom,x:right], (a-x,b-y,c-x,d-y))
                        need(all(abs(value-whole["sides"][name]) <= 2e-5 for name, value in replay.items()),
                             "Whole-image region evidence changed after exact cropping")
                        geometry["full_sharp_exact_crop_same_edges"] += 1
                        geometry["full_sharp_exact_crop_same_edges_but_size_excluded"] += (c-a,d-b) not in crop_sizes
        counts["views"] += 1
    safe = [{k: v for k,v in row.items() if k not in {"input_sha256", "rgb_sha256", "mask_sha256", "ocr_boxes"}}
            for row in rows]
    for row in safe:
        for arm in ARMS:
            if row[arm] is not None:
                row[arm] = {k: v for k,v in row[arm].items() if k != "input_crop_sha256"}
    report = {"schema": SCHEMA, "point": lock["point"], "parameters": lock["parameters"], "code": lock["code"],
              "source_sha256": lock["source_sha256"], "freeze_sha256": file_sha(freeze), "result_sha256": file_sha(result),
              "native_audit": dict(counts), "scale_diagnostic": dict(geometry),
              "elapsed_s": observed["elapsed_s"], "rows": safe, "result": observed["result"]}
    public(report)
    trial.write(output, report)
    print(dict(counts), dict(geometry), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--native", nargs=6, type=Path)
    args = parser.parse_args()
    if args.native:
        keys = ("commons", "earlier", "later", "tesseract", "tessdata", "run")
        native(SimpleNamespace(**dict(zip(keys, args.native, strict=True))), args.record)
    else:
        print(json.dumps(public(read(args.record)), sort_keys=True))


if __name__ == "__main__":
    main()
