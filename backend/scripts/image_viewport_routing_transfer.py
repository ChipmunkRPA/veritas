# SPDX-License-Identifier: Apache-2.0
"""Prospective test of single-file photo viewport routing before boundary scoring."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.detectors.image import sharp_boundary
from scripts import image_crop_layout_transfer as prior
from scripts import image_crop_multiscale_transfer as multi
from scripts import image_source_groups_v22 as source_groups

SCHEMA = "chipmunk.image-viewport-routing-transfer/1"
SCREEN = ("template", "photo", "sharp", "feather", "ambiguous")
DIRECT = ("photo", "sharp", "feather", "tone")
ORDINARY = ("photo", "tone")
PANELS = {"screen": (12, SCREEN), "direct": (12, DIRECT), "standalone": (8, ORDINARY)}
POSITIVE = {"sharp", "feather"}
ORIGIN = ((158, 98), (25, 108), (124, 65), (101, 135))
PARAMETERS = {"block": 8, "active_neighbor_change": .50, "component_close": 5, "component_open": 3,
              "min_component_side": 360, "min_component_fill": .40, "second_component_maximum": .40,
              "edge_search": 52, "edge_strip": 12, "edge_outer_limit": .25, "edge_minimum_step": .35,
              "perpendicular_trim": 28, "ocr_words_per_line": 2, "aligned_ocr_rows": 3,
              "ocr_baseline_span": 28, "ocr_word_confidence": 65, "minimum_viewport_iou": .95,
              "maximum_misrouted_standalone_groups": 1, "minimum_correct_photo_routes": 8,
              "maximum_routed_ambiguous": 2, "minimum_removed_ordinary_screenshot_flags": 6,
              "minimum_added_sharp_ui_locations": 2, "maximum_lost_useful": 1,
              "no_new_ordinary_screenshot_flags": True, "confidence_weight": 0}
ARMS = ("whole", "routed", "exact_viewport")
sha, read, file_sha, write, require = multi.digest, multi.read, multi.file_sha, multi.write, multi.require


def photo_groups(args):
    manifests = [read(folder / "plan.json")["cases"] for folder in (args.earlier, args.later)]
    selected = []
    for index, records in enumerate(manifests):
        slots = []
        for editor in ("sdxl", "kandinsky"):
            family = [row for row in records if row["generator"] == editor]
            require(len(family) == 6, "Expected the complete qualified source family")
            if index == 0:
                chosen = sorted(family, key=lambda row: sha(f"viewport/1/{row['flickr_photo_id']}".encode()))[:3]
            else:
                earlier_use = {row["slot"] for row in sorted(family, key=lambda row:
                               sha(f"native-tile/1/{row['flickr_photo_id']}".encode()))[:3]}
                chosen = [row for row in family if row["slot"] not in earlier_use]
            slots.extend(chosen)
        selected.append(slots)
    for field in ("flickr_photo_id", "flickr_owner"):
        keys = [str(row[field]).casefold() for rows in selected for row in rows]
        require(len(keys) == len(set(keys)) == 12, "The two source roles overlap")
    return selected


def pairs(args):
    slot = 0
    for folder, selected in zip((args.earlier, args.later), photo_groups(args), strict=True):
        inputs = {case["slot"]: (case, views, known) for case, views, known in prior.REGIONAL.paired.inputs(folder)}
        for row in selected:
            case, native, mask = inputs[row["slot"]]
            variants = prior.REGIONAL.native_variants(native, case["box"])
            selected_views = {"photo": variants["original"], "sharp": variants["edited"], "feather": variants["feathered_edit"]}
            edited = (selected_views["photo"] != selected_views["sharp"]).any(axis=2)
            softened = (selected_views["photo"] != selected_views["feather"]).any(axis=2)
            require(edited.any() and softened.any() and not ((edited | softened) & ~mask).any(),
                    "Qualified exact changes differ")
            yield slot, row["generator"], selected_views, edited | softened
            slot += 1


def crop_plan(slot, mask):
    height, width = mask.shape
    a, b, c, d = prior.source_box(mask)
    w, h = min(width, 536+8*(slot%3)), min(height, 528+8*(slot%3))
    x = max(0, min(width-w, (a+c-w)//2))
    y = max(0, min(height-h, (b+d-h)//2))
    require(sharp_boundary.supported_size(w, h) and x <= a <= c <= x+w and y <= b <= d <= y+h,
            "The authored viewport loses pixels in the known edit")
    px, py = ORIGIN[slot%4]
    px, py = px+5*(slot//4), py+3*(slot//4)
    require(px+w <= 768 and py+h <= 768, "The authored viewport leaves the capture")
    return {"slot": slot, "layout": slot%4, "x": px, "y": py, "w": w, "h": h, "crop": [x, y, x+w, y+h]}


def prepare(args):
    args.run.mkdir(mode=0o700)
    fixture = args.run / "photo-fixtures"
    fixture.mkdir()
    plan = []
    for slot, _, views, mask in pairs(args):
        row = crop_plan(slot, mask)
        plan.append(row)
        a, b, c, d = row["crop"]
        for role, rgb in views.items():
            (fixture/f"{slot}-{role}.png").write_bytes(prior.png(rgb[b:d, a:c]))
    write(args.run/"layout-plan.json", plan)


def documentary_controls(args):
    archive = prior.sources.binding(args.commons)["sources"]
    catalogue = read(args.commons/"plan.json")["sources"]
    ranked = sorted(archive, key=lambda row: (str(catalogue[row["slot"]]["category"]).casefold(),
                                              sha(f"viewport/1/photo/{row['sha256']}".encode())))
    by_kind = {}
    for row in ranked:
        by_kind.setdefault(catalogue[row["slot"]]["category"], []).append(row)
    round_robin = [row for group in zip(*(rows[:4] for rows in by_kind.values()), strict=False) for row in group]
    roster = round_robin[:8]
    require(len(roster) == len({catalogue[row["slot"]]["author"] for row in roster}) == 8,
            "Expected eight ordinary controls from distinct credited sources")
    for slot, row in enumerate(roster):
        image = prior.native.source(args.commons, row, (768, 680))
        for role, item in (("photo", image), ("tone", ImageEnhance.Contrast(image).enhance(.88))):
            rgb = np.asarray(item)
            yield ("standalone", slot, role), "credited-photo", prior.png(rgb), rgb, np.zeros(rgb.shape[:2], dtype=bool), None


def views(args):
    plan = read(args.run/"layout-plan.json")
    screenshots = read(args.run/"captures.json")["rows"]
    by_key = {(row["slot"], row["role"]): row for row in screenshots}
    require(len(by_key) == len(screenshots) == 60 and len(plan) == 12, "Screenshot roster is incomplete")
    for row, case in zip(plan, pairs(args), strict=True):
        slot, family, source, mask = case
        require(row == crop_plan(slot, mask), "Photo crop and screenshot plan differ")
        a, b, c, d = row["crop"]
        x, y, w, h = (row[key] for key in ("x", "y", "w", "h"))
        cropped = {role: rgb[b:d, a:c] for role, rgb in source.items()}
        viewport = [x, y, x+w, y+h]
        rgb_screen = {}
        for role in SCREEN:
            data = (args.run/"captures"/f"{slot}-{role}.png").read_bytes()
            saved = by_key[slot, role]
            require(sha(data) == saved["png_sha256"], "Screenshot bytes differ")
            rgb = prior.pixels(args.run/"captures"/f"{slot}-{role}.png")
            require(rgb.shape == (768, 768, 3), "Unexpected capture geometry")
            if role in cropped:
                fixture = args.run/"photo-fixtures"/f"{slot}-{role}.png"
                require(file_sha(fixture) == saved["photo_sha256"] and np.array_equal(rgb[y:y+h,x:x+w], cropped[role]),
                        "Displayed native photo differs from the sealed preview")
            rgb_screen[role] = (data, rgb)
        for role, (data, rgb) in rgb_screen.items():
            truth = (rgb != rgb_screen["photo"][1]).any(axis=2) if role in POSITIVE else np.zeros(rgb.shape[:2], dtype=bool)
            require(truth.sum() == truth[y:y+h,x:x+w].sum(), "Native screenshot change outside the displayed photo")
            yield ("screen", slot, role), f"layout-{row['layout']}", data, rgb, truth, viewport if role in cropped else None
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(cropped["photo"])).enhance(.88))
        for role, rgb in (cropped | {"tone": tone}).items():
            truth = (rgb != cropped["photo"]).any(axis=2) if role in POSITIVE else np.zeros(rgb.shape[:2], dtype=bool)
            yield ("direct", slot, role), family, prior.png(rgb), rgb, truth, None
    yield from documentary_controls(args)


def identity(case):
    key, family, data, rgb, truth, viewport = case
    require(bool(truth.any()) == (key[-1] in POSITIVE), "Known native target missing")
    return {"key": list(key), "family": family, "input_sha256": sha(data), "rgb_sha256": sha(rgb.tobytes()),
            "mask_sha256": sha(truth.tobytes()), "shape": list(truth.shape), "target": int(truth.sum()), "viewport": viewport}


def freeze(args):
    cases = list(views(args))
    receipts = [identity(case) for case in cases]
    expected = {(panel, i, role) for panel, (count, roles) in PANELS.items() for i in range(count) for role in roles}
    require(len(receipts) == len(expected) and {tuple(row["key"]) for row in receipts} == expected, "Incomplete study roster")
    photo_fingerprints, other_fingerprints = [], []
    for key, _, _, rgb, _, _ in cases:
        if key[0] == "screen" or key[-1] != "photo":
            continue
        with Image.fromarray(rgb) as image:
            item = {"size": list(image.size), "sketch": f"{source_groups.sketch(image):064x}", "rgb": sha(rgb.tobytes())}
        (photo_fingerprints if key[0] == "direct" else other_fingerprints).append(item)
    require(not any(a["rgb"] == b["rgb"] or source_groups.near(a, b)
                    for a in photo_fingerprints for b in other_fingerprints), "Standalone control image overlaps a paired source")
    code = (Path(__file__), Path(__file__).with_name("capture_image_viewport_routing.cjs"), Path(prior.__file__),
            Path(sharp_boundary.__file__), Path(source_groups.__file__))
    return {"schema": SCHEMA, "parameters": PARAMETERS, "cases": receipts, "cross_role_photo_pairs": 96,
            "code": {path.name: file_sha(path) for path in code}, "point": sharp_boundary.REVIEW_POINT,
            "source_sha256": {name: file_sha(path) for name, path in {
                "earlier": args.earlier/"result.json", "later": args.later/"result.json", "commons": args.commons/"acquisition.json",
                "layout": args.run/"layout-plan.json", "captures": args.run/"captures.json"}.items()},
            "opencv": cv2.__version__, "ocr": {"binary": file_sha(args.tesseract), "eng": file_sha(args.tessdata/"eng.traineddata")}}


def rough_component(rgb):
    h, w = rgb.shape[:2]
    rows, cols = h//8, w//8
    tiles = rgb[:rows*8,:cols*8].reshape(rows, 8, cols, 8, 3).transpose(0, 2, 1, 3, 4)
    along_x = (tiles[:,:, :,1:] != tiles[:,:, :,:-1]).any(axis=4).mean(axis=(2, 3))
    along_y = (tiles[:,:,1:] != tiles[:,:,:-1]).any(axis=4).mean(axis=(2, 3))
    active = ((along_x >= .5) & (along_y >= .5)).astype(np.uint8)
    closed = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    dense = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    if n < 2:
        return None
    ordered = sorted(stats[1:].tolist(), key=lambda rect: -rect[4])
    x, y, width, height, count = ordered[0]
    if (min(width, height)*8 < 360 or count < .4*width*height
            or (len(ordered) > 1 and ordered[1][4] > .4*count)):
        return None
    return [8*x, 8*y, 8*(x+width), 8*(y+height)]


def edge(profile, around, inward):
    step = 12
    stop = len(profile)-step+1
    options = []
    for i in range(max(step, around-52), min(stop, around+53)):
        before, after = float(profile[i-step:i].mean()), float(profile[i:i+step].mean())
        inside, outside = (after, before) if inward > 0 else (before, after)
        if outside <= .25 and inside-outside >= .35:
            options.append((inside-outside, -abs(i-around), -i))
    if not options:
        return None
    support, _, location = max(options)
    return -location, support


def viewport_pixels(rgb):
    rough = rough_component(rgb)
    if rough is None:
        return None
    x, y, right, bottom = rough
    trim = 28
    height_strip = rgb[y+trim:bottom-trim]
    width_strip = rgb[:,x+trim:right-trim]
    along_x = (height_strip[1:] != height_strip[:-1]).any(axis=2).mean(axis=0)
    along_y = (width_strip[:,1:] != width_strip[:,:-1]).any(axis=2).mean(axis=1)
    measured = [edge(profile, start, side) for profile, start, side in
                ((along_x, x, 1), (along_y, y, 1), (along_x, right, -1), (along_y, bottom, -1))]
    if any(value is None for value in measured):
        return None
    box = [row[0] for row in measured]
    if not sharp_boundary.supported_size(box[2]-box[0], box[3]-box[1]):
        return None
    return {"rough": rough, "box": box, "edges": [row[1] for row in measured]}


def aligned_exterior(boxes, viewport):
    x, y, right, bottom = viewport
    gutters = ([word for word in boxes if word[2] <= x-4], [word for word in boxes if word[0] >= right+4],
               [word for word in boxes if word[3] <= y-4], [word for word in boxes if word[1] >= bottom+4])
    best = []
    for gutter in gutters:
        rows = []
        for word in sorted(gutter, key=lambda item: (item[3], item[0])):
            if not rows or abs(word[3]-rows[-1][0][3]) > 4:
                rows.append([])
            rows[-1].append(word)
        starts = [(min(word[0] for word in row), min(word[3] for word in row)) for row in rows if len(row) >= 2]
        for left, _ in starts:
            cluster = [item for item in starts if abs(item[0]-left) <= 8]
            if len(cluster) >= 3 and cluster[-1][1]-cluster[0][1] >= 28 and len(cluster) > len(best):
                best = cluster
    return [list(item) for item in best]


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    x, y, right, bottom = a
    sx, sy, sr, sb = b
    cross = max(0, min(right, sr)-max(x, sx))*max(0, min(bottom, sb)-max(y, sy))
    return cross/((right-x)*(bottom-y)+(sr-sx)*(sb-sy)-cross)


def measure(rgb, truth, crop=None):
    h, w = truth.shape
    x, y, right, bottom = crop or [0, 0, w, h]
    source = np.ascontiguousarray(rgb[y:bottom, x:right])
    boundary = sharp_boundary.measure(source)
    a, b, c, d = boundary.box
    box = [a+x, b+y, c+x, d+y]
    changed = int(truth[b+y:d+y,a+x:c+x].sum())
    target = int(truth.sum())
    area = (c-a)*(d-b)
    return {"box": box, "input_crop_sha256": sha(source.tobytes()), "score": boundary.score,
            "sides": boundary.sides, "flag": boundary.score > sharp_boundary.REVIEW_POINT,
            "overlap": changed, "area": area,
            "useful": bool(target and 2*changed >= target and 10*changed >= 3*(target+area-changed))}


def inspect(args, case):
    key, _, data, rgb, truth, viewport = case
    locator = viewport_pixels(rgb)
    text = prior.ocr_boxes(args, data) if locator is not None else []
    aligned = aligned_exterior(text, locator["box"]) if locator else []
    route = locator["box"] if aligned else None
    whole = measure(rgb, truth)
    routed = measure(rgb, truth, route) if route else whole
    exact = measure(rgb, truth, viewport) if viewport is not None else None
    return identity(case) | {"locator": locator, "ocr_boxes": text, "aligned_exterior": aligned, "route": route,
                             "route_iou": box_iou(route, viewport), "whole": whole, "routed": routed, "exact_viewport": exact}


def aggregate(rows):
    counts, changes = {}, Counter()
    control_groups = set()
    for row in rows:
        panel, slot, role = row["key"]
        cell = counts.setdefault(f"{panel}/{role}", Counter(views=0))
        cell["views"] += 1
        cell["routed"] += row["route"] is not None
        cell["correct_viewports"] += row["route_iou"] >= .95
        for arm in ARMS:
            item = row[arm]
            cell[arm+"_flags"] += bool(item and item["flag"])
            cell[arm+"_useful"] += bool(item and item["flag"] and item["useful"])
        a, b = (row[arm]["flag"] for arm in ("whole", "routed"))
        useful_a, useful_b = (row[arm]["useful"] and row[arm]["flag"] for arm in ("whole", "routed"))
        changes["lost_useful"] += useful_a and not useful_b
        changes["added_sharp_screen"] += panel == "screen" and role == "sharp" and useful_b and not useful_a
        changes["added_ordinary_screen_flags"] += panel == "screen" and role not in POSITIVE and b and not a
        changes["removed_ordinary_screen_flags"] += panel == "screen" and role not in POSITIVE and a and not b
        if row["route"] and panel != "screen" and role not in POSITIVE:
            control_groups.add((panel, slot))
    changes["misrouted_standalone_groups"] = len(control_groups)
    correct = counts["screen/photo"]["correct_viewports"]
    ambiguous = counts["screen/ambiguous"]["routed"]
    accepted = (correct >= 8 and ambiguous <= 2 and changes["lost_useful"] <= 1
                and changes["added_sharp_screen"] >= 2 and changes["removed_ordinary_screen_flags"] >= 6
                and changes["added_ordinary_screen_flags"] == 0 and changes["misrouted_standalone_groups"] <= 1)
    return {"by_role": {key: dict(value) for key, value in counts.items()}, "changes": dict(changes), "passed": accepted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "run"))
    for key in ("commons", "earlier", "later", "tesseract", "tessdata", "run"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return
    frozen = freeze(args)
    path = args.run/"freeze.json"
    if args.stage == "freeze":
        write(path, frozen)
        print(file_sha(path), len(frozen["cases"]), flush=True)
        return
    require(frozen == read(path), "Frozen sources or measurement settings changed")
    start = time.monotonic()
    rows = []
    for case, expected in zip(views(args), frozen["cases"], strict=True):
        require(identity(case) == expected, "A native source changed after freeze")
        rows.append(inspect(args, case))
        if len(rows)%16 == 0:
            print(len(rows), len(frozen["cases"]), flush=True)
    journal = args.run/"rows.json"
    write(journal, rows)
    result = {"schema": SCHEMA, "freeze_sha256": file_sha(path), "rows_sha256": file_sha(journal),
              "elapsed_s": time.monotonic()-start, "result": aggregate(rows)}
    write(args.run/"result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
