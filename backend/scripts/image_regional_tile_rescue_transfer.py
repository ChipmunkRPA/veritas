# SPDX-License-Identifier: Apache-2.0
"""Check two native crop positions for a small editing region missed by the JPEG pass."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import localiser as loc
from app.detectors.image import regional, regional_support
from scripts import image_jpeg_contour_gap_transfer as source_panel
from scripts import image_regional_jpeg_transfer as previous
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-regional-tile-rescue-transfer/1"
TILE = 512
SHIFT = 32
MARGIN = 24
MIN_OVERLAP = .50
PHASES = ("retained_b", "external")
DESIGN = {
    "tile": TILE, "shift": SHIFT, "artificial_edge_margin": MARGIN, "mutual_region_overlap": MIN_OVERLAP,
    "first_view": "At most four source-corner 512-pixel squares without resampling.",
    "repeat_view": "Shift each crop origin up to 32 pixels toward the source center along each available axis.",
    "quality": "Apply the full-upload Q75 reliability mask and the original model response points to every crop.",
    "ranking": "Choose the strongest mutually supported candidate by original component mean, then by mutual overlap and exact area.",
    "geometry": "Keep the original crop component on its native source coordinates.",
    "additive": "Inspect crops only if the existing whole-image pass supplied no region.",
    "delivery": "Standard JPEG75 4:2:0 upright RGB, long side 768 and short side at least 512.",
    "useful": {"iou": .30, "recall": .50},
    "advance": {"feathered_lama_groups": 2, "additional_ordinary_groups": 2, "additional_wrong_edit_files": 3},
    "confidence_weight": 0,
}
read, write, require, sha, file_sha = previous.read, previous.write, previous.require, previous.sha, previous.file_sha


def crop_origins(width, height):
    return sorted(set(product((0, width - TILE), (0, height - TILE))), key=lambda xy: xy[::-1])


def shifted_origin(origin, shape):
    return tuple(min(SHIFT, side - TILE) if start == 0 else max(0, start - SHIFT)
                 for start, side in zip(origin, shape, strict=True))


def supported_component(first, second, first_xy, second_xy):
    x0, y0 = np.maximum(first_xy, second_xy)
    x1, y1 = np.minimum(np.asarray(first_xy) + TILE, np.asarray(second_xy) + TILE)
    a, b = ((mask[y0-y:y1-y, x0-x:x1-x]) for mask, (x, y) in
            ((first, first_xy), (second, second_xy)))
    shared = int(np.count_nonzero(a & b))
    return shared / max(int(first.sum()), int(second.sum()))


def tile_result(model, rgb, reliable, origin):
    x, y = origin
    pixels = rgb[y:y+TILE, x:x+TILE]
    support = reliable[y:y+TILE, x:x+TILE].copy()
    if x:
        support[:, :MARGIN] = False
    if y:
        support[:MARGIN] = False
    if x+TILE < rgb.shape[1]:
        support[:, -MARGIN:] = False
    if y+TILE < rgb.shape[0]:
        support[-MARGIN:] = False
    predicted = model.predict(Image.fromarray(pixels))
    active = predicted.candidate_eligibility["eligible"]
    values = np.rint(predicted.raw_mask.astype(np.float64) * 255) / 255.
    selected = ((values >= regional.PIXEL_THRESHOLD) & support).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    threshold = np.ceil(loc.MIN_REGION_FRACTION * TILE**2)
    components = []
    if active:
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area >= threshold:
                mask = labels == index
                components.append((float(values[mask].mean()), area, mask))
    components.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return components[:2], {
        "origin": list(origin), "input_sha256": predicted.geometry["input_float32_sha256"],
        "response_sha256": sha(predicted.raw_mask.tobytes()), "active": active,
    }


def rescue(model, rgb):
    width, height = rgb.shape[1], rgb.shape[0]
    excluded, _ = loc.conditions_mask(rgb, 75)
    reliable = ~excluded
    best, measurements = None, {}

    def measure(origin):
        if origin not in measurements:
            measurements[origin] = tile_result(model, rgb, reliable, origin)
        return measurements[origin][0]

    for origin in crop_origins(width, height):
        first = measure(origin)
        if not first:
            continue
        shifted = shifted_origin(origin, (width, height))
        require(shifted != origin, "Repeat crop needs new source context")
        second = measure(shifted)
        for score, area, a in first:
            for _, _, b in second:
                match = supported_component(a, b, origin, shifted)
                rank = (score, match, area)
                if match >= MIN_OVERLAP and (best is None or rank > best[0]):
                    best = (rank, origin, shifted, a)
    selected = np.zeros((height, width), bool)
    proof = None
    if best:
        (score, match, area), (x, y), shifted, component = best
        selected[y:y+TILE, x:x+TILE] = component
        proof = {"origin": [x, y], "repeat_origin": list(shifted), "mean": score, "mutual_overlap": match,
                 "selected_cells": area}
    return selected, [trace for _, trace in measurements.values()], proof


def smoke():
    one, two = np.zeros((TILE, TILE), bool), np.zeros((TILE, TILE), bool)
    one[70:90, 80:110] = two[70:90, 48:78] = True
    require(supported_component(one, two, (0, 0), (SHIFT, 0)) == 1.,
            "Exact common native pixels must survive a shifted origin")
    require(supported_component(one, two, (0, 0), (0, 0)) == 0., "An uncorrected origin should not agree")


def material(args):
    for item, pixels, exact in source_panel.later_groups(SimpleNamespace(later=args.later_f5)):
        credit = source_panel.identity(item, pixels, exact)
        if min(pixels["original"].shape[:2]) < 512:
            continue
        controls, edits = source_panel.common.authored(pixels, exact, item["recipient"])
        for role, rgb in (controls | edits).items():
            truth = np.any(rgb != pixels["original"], axis=2) if role in previous.POSITIVE else np.zeros(exact.shape, bool)
            key = {"panel": "retained_b", "source": item["recipient"], "group": credit["photo"],
                   "family": item["generator"], "role": role}
            yield key, source_panel.common.encode(rgb, 75), truth, credit

    args.exclude = []
    lock, _, _ = previous.panel.previous.fresh.previous(args)
    ext = previous.panel.external
    plan, known = ext.source_plan(args, lock)
    saved = read(args.external / "plan.json")
    require(all(saved[key] == value for key, value in plan.items()), "Fresh-editor source plan differs")
    items = {case["slot"]: case for case in plan["cases"]}
    external = SimpleNamespace(**(vars(args) | {"output": args.external}))
    for row, body, truth in ext.views(external, plan, known):
        item = items[row["slot"]]
        credit = {"photo": str(item["flickr_photo_id"]), "owner": item["flickr_owner"],
                  "author_profile": item["flickr_owner"].casefold(), "credit": item["photo_attribution"],
                  "original_sha256": item["original"]["sha256"]}
        key = {"panel": "external", "source": row["source"], "group": credit["photo"],
               "family": row["family"], "role": row["role"]}
        require(row["input_sha256"] == sha(body) and row["truth_sha256"] == sha(truth.tobytes()),
                "Earlier exact JPEG evidence differs")
        yield key, body, truth, credit


def witness(key, body, truth):
    require(bool(truth.any()) == (key["role"] in previous.POSITIVE), "Only known edit views may contain edit truth")
    return key | {"input_sha256": sha(body), "truth_sha256": sha(truth.tobytes()), "shape": list(truth.shape)}


def profile(args):
    parts = {name: {} for name in PHASES}
    rows = []
    for key, body, truth, credit in material(args):
        bank = parts[key["panel"]]
        require(key["group"] not in bank or bank[key["group"]] == credit, "Shared source provenance changed")
        bank[key["group"]] = credit
        rows.append(witness(key, body, truth))
    counts = Counter(r["panel"] for r in rows)
    require(counts == {"retained_b": 136, "external": 64}, "Unexpected frozen held delivery count")
    a, b = (list(parts[k].values()) for k in PHASES)
    require([len(a), len(b)] == [15, 7], "Need 22 source groups across the two banks")
    for name in ("photo", "owner", "author_profile", "original_sha256"):
        require(not {str(r[name]).casefold() for r in a} & {str(r[name]).casefold() for r in b},
                f"Cross-bank source overlap: {name}")
    require(not {r["credit"]["Author"].casefold() for r in a} & {r["credit"]["Author"].casefold() for r in b},
            "Cross-bank photographer overlap")
    checkpoints = (Path(__file__), Path(regional.__file__), Path(regional_support.__file__),
                   Path(source_panel.__file__), Path(previous.panel.external.__file__),
                   args.later_f5 / "plan.json", args.later_f5 / "intake.json", args.external / "plan.json",
                   args.external / "intake.json")
    require(file_sha(previous.CHECKPOINT) == regional.CHECKPOINT_SHA256, "Expected signed regional checkpoint")
    return {"schema": SCHEMA, "design": DESIGN, "checkpoint_sha256": regional.CHECKPOINT_SHA256,
            "bindings": {f"{path.parent.name}/{path.name}": file_sha(path) for path in checkpoints},
            "source_counts": {name: len(parts[name]) for name in PHASES}, "rows": rows}


def readout(model, body):
    rgb = source_panel.common.decode(body)
    require(max(rgb.shape[:2]) == 768 and min(rgb.shape[:2]) >= TILE, "JPEG lies outside the evaluated dimensions")
    image = Image.fromarray(rgb)
    original = model.predict(image)
    native = previous.native_readout(original, body, rgb).source_mask(*image.size)
    measured = source_panel.common.comparison.format_reference("jpeg75", body, rgb)
    luma = np.zeros_like(native)
    if measured and measured["selected"]:
        x0, y0, x1, y1 = measured["box"]
        luma[y0:y1, x0:x1] = True
    added, calls, proof = rescue(model, rgb) if not native.any() else (np.zeros_like(native), [], None)
    return (luma, native, native | added), {"tiles": calls, "added": proof,
                                           "native_raw_sha256": sha(original.raw_mask.tobytes())}


ARMS = ("current_luma", "native", "native_plus_tiles")


def summary(rows):
    counts, groups = {}, {}
    for key in ("retained_b", "external", "all"):
        subset = [row for row in rows if key == "all" or row["panel"] == key]
        counts[key], groups[key] = {}, {}
        for arm in ARMS:
            control_flags = set()
            table = Counter()
            for row in subset:
                metric = row["measures"][arm]
                flagged = metric["predicted_positive_pixels"] > 0
                role, family = row["role"], row["family"]
                if role in previous.POSITIVE:
                    table.update({f"{family}/{role}/files": 1, f"{family}/{role}/useful": previous.is_useful(metric),
                                  f"{family}/{role}/wrong": flagged and not previous.is_useful(metric)})
                else:
                    table.update({f"{family}/{role}/files": 1, f"{family}/{role}/flags": flagged})
                    if flagged:
                        control_flags.add(row["group"])
            counts[key][arm] = dict(table)
            groups[key][arm] = control_flags
        counts[key]["ordinary_groups"] = {arm: len(values) for arm, values in groups[key].items()}
        counts[key]["ordinary_groups_added_by_tiles"] = len(groups[key]["native_plus_tiles"] - groups[key]["native"])
        counts[key]["rescue_forwards"] = sum(len(row["tiles"]) for row in subset)
    gained = {row["group"] for row in rows if row["family"] == "lama" and row["role"] == "feathered"
              and previous.is_useful(row["measures"]["native_plus_tiles"]) and not previous.is_useful(row["measures"]["native"])}
    wrong = sum(row["role"] in previous.POSITIVE and row["added"] is not None
                and not previous.is_useful(row["measures"]["native_plus_tiles"]) for row in rows)
    gate = DESIGN["advance"]
    costs = counts["all"]["ordinary_groups_added_by_tiles"]
    return {"results": counts, "paired": {"added_feathered_lama_groups": len(gained), "additional_wrong_edit_files": wrong,
                                          "additional_ordinary_groups": costs},
            "passes": len(gained) >= gate["feathered_lama_groups"] and wrong <= gate["additional_wrong_edit_files"]
            and costs <= gate["additional_ordinary_groups"]}


def run(args, frozen):
    model = regional.load_model(previous.CHECKPOINT)
    rows = []
    with (args.output / "rows.jsonl").open("x") as journal:
        for (key, body, truth, _), expected in zip(material(args), frozen["rows"], strict=True):
            require(witness(key, body, truth) == expected, "Delivered JPEG, exact truth or source role differs")
            masks, trace = readout(model, body)
            packed = b"".join(np.packbits(mask, axis=None).tobytes() for mask in masks)
            mask_path = args.output / "masks" / f"{len(rows):03}.bits"
            mask_path.write_bytes(packed)
            row = expected | trace | {"mask_sha256": sha(packed), "measures": {
                arm: mask_metrics(mask, truth) for arm, mask in zip(ARMS, masks, strict=True)}}
            rows.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            if len(rows) % 8 == 0:
                print(len(rows), "/", len(frozen["rows"]), flush=True)
    return rows


def audit(args, frozen):
    rows = [json.loads(line) for line in (args.output / "rows.jsonl").read_text().splitlines()]
    for index, ((key, body, truth, _), expected, row) in enumerate(zip(material(args), frozen["rows"], rows, strict=True)):
        require(witness(key, body, truth) == expected and all(row[k] == v for k, v in expected.items()),
                "Delivered JPEG, truth or evidence role differs")
        payload = (args.output / "masks" / f"{index:03}.bits").read_bytes()
        stride = (truth.size + 7) // 8
        require(len(payload) == stride * len(ARMS) and sha(payload) == row["mask_sha256"], "Native masks differ")
        masks = [np.unpackbits(np.frombuffer(payload[i*stride:(i+1)*stride], np.uint8), count=truth.size)
                 .astype(bool).reshape(truth.shape) for i in range(len(ARMS))]
        require(all(mask_metrics(mask, truth) == row["measures"][arm] for arm, mask in zip(ARMS, masks, strict=True))
                and not (masks[1] & ~masks[2]).any(), "Overlap or additive masks differ on replay")
        if row["added"]:
            x, y = row["added"]["origin"]
            require(not masks[1].any() and int(masks[2].sum()) == row["added"]["selected_cells"]
                    == int(masks[2][y:y+TILE, x:x+TILE].sum()), "Rescue escapes the recorded native tile")
        else:
            require(np.array_equal(masks[1], masks[2]), "An unrecorded rescue changed the native mask")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    smoke()
    frozen = profile(args)
    lock = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        (args.output / "masks").mkdir()
        write(lock, frozen)
        print(file_sha(lock), flush=True)
        return
    require(frozen == read(lock), "Frozen study inputs differ")
    start = time.monotonic()
    rows = run(args, frozen) if args.stage == "run" else audit(args, frozen)
    result = {k: v for k, v in frozen.items() if k != "rows"} | summary(rows) | {
        "view_count": len(rows), "freeze_sha256": file_sha(lock), "rows_sha256": file_sha(args.output / "rows.jsonl")}
    if args.stage == "run":
        write(args.output / "result.json", result)
    else:
        require(result == read(args.output / "result.json") and args.publication is not None, "Study replay differs")
        write(args.publication, result | {"exact_source_mask_replays": len(rows)})
    print(result["paired"], "pass", result["passes"], round(time.monotonic() - start, 2), flush=True)


if __name__ == "__main__":
    main()
