"""Confirm the existing optional PNG repeat pass on additional matched banks."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-regional-flip-confirmation/1"
TGIF_FAMILIES = ("firefly", "sd2", "flux-fill")
FAMILIES = (*TGIF_FAMILIES, "sd15")
FORMAT_PROFILES = {"tgif": ("ordinary", "local_blur", "local_tone", "sharp", "feather"),
                   "bg": ("ordinary", "roundtrip", "sharp")}
METHOD = {
    "sources": "Use the next four geometry-eligible held TGIF/TGIF2 photos per family after the six used in the "
               "progressive JPEG comparison. Use all eight fixed slots in each of the two retained BG-REAL insertion "
               "banks. These photos were exposed in earlier work.",
    "delivery": "Upright 768 px long side and at least 512 px short side, lossless PNG. TGIF/TGIF2 uses sharp, "
                "feathered, untouched, local blur and local tone. Each BG-REAL source has the publisher edit, "
                "untouched file and matched whole-image autoencoder round trip.",
    "condition": "Compare the original native app mask with the deployed exact horizontal flip and "
                 "same original component retained at half aligned-grid overlap. Follow the runtime: repeat only "
                 "if the original pass supplied at least one component. Keep the original source coordinates.",
    "useful": "At least .30 IoU and .50 edited-pixel recall. False highlights include ordinary processing controls.",
    "advance": "Across at least two editor families with useful original masks, retain at least 75% of the useful "
               "locations, lose at most one useful source per family, and reduce control flags by at least 20% per "
               "source bank with control flags. Require at least four native useful locations and at least four "
               "native ordinary flags overall for a positive transfer. Scope is editing evidence only.",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sets(sources, bg):
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    from scripts import image_bg_real_insertion_score as insertion
    from scripts import image_jpeg_signed_transitions as archive

    held = archive.sources(argparse.Namespace(sources=sources))["held"]
    for family in TGIF_FAMILIES:
        entries = []
        for folder, original, edit in held:
            if edit["family"] == family:
                with Image.open(folder / original["image"]) as image:
                    if min(image.size) * 3 >= max(image.size) * 2:
                        entries.append((folder, original, edit))
        entries.sort(key=lambda entry: entry[1]["image_sha256"])
        require(len(entries) >= 10, "Too few TGIF/TGIF2 sources at the app dimensions")
        for key, _, original, edit, truth in archive.archive_pairs({"held": entries[6:10]}, "held"):
            h, w = truth.shape
            size = (w * 768 // max(h, w), h * 768 // max(h, w))
            original, edit = (cv2.resize(array, size, interpolation=cv2.INTER_AREA) for array in (original, edit))
            image = Image.fromarray(original)
            blur = np.asarray(image.filter(ImageFilter.GaussianBlur(1.4)))
            tone = np.asarray(ImageEnhance.Color(image).enhance(.72)).astype(np.int16)
            tone = np.clip(tone + [12, -6, 7], 0, 255).astype(np.uint8)
            truth = np.any(original != edit, axis=2)
            distance = cv2.distanceTransform(truth.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            alpha = ((1 - np.cos(np.clip(distance / 8, 0, 1) * np.pi)) / 2)[..., None]
            local_blur, local_tone, feather = (
                np.rint(original * (1 - alpha) + other * alpha).astype(np.uint8) for other in (blur, tone, edit)
            )
            variants = (("ordinary", original), ("local_blur", local_blur), ("local_tone", local_tone),
                        ("sharp", edit), ("feather", feather))
            for role, rgb in variants:
                yield ("tgif", family, key, role, rgb, original)
    creators = set()
    for trial in bg:
        _, plan, attempts = insertion.checked_inputs(trial)
        require(len(plan["cases"]) == 8, "Unexpected BG-REAL plan")
        ids = {str(record["flickr_photo_id"]) for record in plan["cases"]}
        require(len(ids) == 8 and not (ids & creators), "BG-REAL photo reused")
        creators.update(ids)
        for _, raw, pixels, _ in insertion.panels(trial, plan, attempts):
            for role, src in (("ordinary", "original"), ("roundtrip", "processed"), ("sharp", "edited")):
                yield ("bg", "sd15", sha(raw["original"]), role, pixels[src], pixels["original"])


def materialize(args):
    import numpy as np
    from PIL import Image

    for bank, family, group, role, rgb, original in sets(args.sources, args.bg):
        h, w = rgb.shape[:2]
        require(max(w, h) == 768 and min(w, h) >= 512, "Input violates the runtime scope")
        truth = np.any(original != rgb, axis=2) if role in {"sharp", "feather"} else np.zeros((h, w), bool)
        with io.BytesIO() as stream:
            Image.fromarray(rgb).save(stream, "PNG")
            png = stream.getvalue()
        yield ({"bank": bank, "family": family, "group": group, "role": role, "shape": [h, w],
                "png_sha256": sha(png), "truth_sha256": sha(truth.tobytes()), "edited_pixels": int(truth.sum())},
               png, rgb, truth)


def useful(values, target):
    selected, overlap = values
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (target + selected - overlap))


def audit(freeze, rows):
    require(freeze["schema"] == SCHEMA and freeze["method"] == METHOD
            and freeze["runner_sha256"] == file_sha(Path(__file__)), "Unexpected frozen method")
    require(len(rows) == len(freeze["roster"]) == 108, "Wrong number of planned deliveries")
    groups = defaultdict(list)
    outcome = defaultdict(Counter)
    lost_sources = defaultdict(set)
    useful_families = set()
    for row, plan in zip(rows, freeze["roster"], strict=True):
        require(row["input"] == plan, "Observed input differs from the frozen roster")
        bank, role, family, group = (plan[k] for k in ("bank", "role", "family", "group"))
        groups[group].append((bank, family, role))
        native, repeated = (row[key] for key in ("native", "repeat"))
        require((row["comparison"] is not None) == bool(native[0]), "Repeat schedule differs from serving")
        h, w = plan["shape"]
        target = plan["edited_pixels"]
        require(max(h, w) == 768 and min(h, w) >= 512 and bool(target) == (role in {"sharp", "feather"}),
                "Invalid frozen geometry")
        for selected, common in (native, repeated):
            require(0 <= selected <= h * w and max(0, selected + target - h * w) <= common <= min(target, selected),
                    "Invalid area or overlap")
        require(repeated[0] <= native[0] and repeated[1] <= native[1], "Repeat mask expanded beyond its source")
        if cmp := row["comparison"]:
            require(1 <= len(cmp) <= 2 and all(0 <= match <= pixels for pixels, match in cmp)
                    and bool(repeated[0]) == any(2 * match >= pixels for pixels, match in cmp), "Invalid component support")
        for name in (bank, family, f"{bank}/{role}", "total"):
            count = outcome[name]
            count.update(views=1, repeats=int(native[0] > 0))
            if target:
                count.update(edited=1, useful_native=useful(native, target), useful_repeat=useful(repeated, target),
                             lost=useful(native, target) and not useful(repeated, target),
                             gained=useful(repeated, target) and not useful(native, target))
            else:
                count.update(controls=1, ordinary_native=bool(native[0]), ordinary_repeat=bool(repeated[0]))
        if useful(native, target):
            useful_families.add(family)
            if not useful(repeated, target):
                lost_sources[family].add(group)
    rosters = Counter()
    for items in groups.values():
        bank, family, _ = items[0]
        require(len(items) == len(FORMAT_PROFILES[bank])
                and set(items) == {(bank, family, role) for role in FORMAT_PROFILES[bank]}, "Source variants differ")
        rosters[family] += 1
    require(dict(rosters) == {**dict.fromkeys(TGIF_FAMILIES, 4), "sd15": 16}, "Source roster scope differs")
    total = outcome["total"]
    reduced = all(5 * outcome[b]["ordinary_repeat"] <= 4 * outcome[b]["ordinary_native"] for b in FORMAT_PROFILES)
    passes = (total["useful_native"] >= 4 and total["ordinary_native"] >= 4 and len(useful_families) >= 2
              and 4 * total["useful_repeat"] >= 3 * total["useful_native"] and reduced
              and all(len(ids) <= 1 for ids in lost_sources.values()))
    return {"counts": {key: dict(value) for key, value in outcome.items()}, "qualifies": passes,
            "useful_families": sorted(useful_families), "lost_groups": {k: len(v) for k, v in lost_sources.items()}}


def measure(args, freeze):
    import time

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import regional, regional_support
    from app.detectors.image.adapters.regional import flip_components

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    predictor = regional.load_model(args.checkpoint)
    rows = []
    with args.output.joinpath("observations.jsonl").open("x") as journal:
        for (plan, png, rgb, truth), expected in zip(materialize(args), freeze["roster"], strict=True):
            require(plan == expected, "Source or exact truth changed since the freeze")
            started = time.monotonic()
            prediction = predictor.predict(Image.fromarray(rgb))
            native = regional_support.readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                              regional.PIXEL_THRESHOLD, png, rgb)
            h, w = truth.shape
            mask = native.source_mask(w, h)
            combined, proof = mask, None
            if native.regions:
                mirrored = np.ascontiguousarray(rgb[:, ::-1])
                repeat = predictor.predict(Image.fromarray(mirrored))
                support = regional_support.readout(repeat.raw_mask, repeat.candidate_eligibility["eligible"],
                                                   regional.PIXEL_THRESHOLD, png, mirrored)
                aligned = np.ascontiguousarray(support.selected_grid_mask[:, ::-1])
                _, _, found = flip_components(native, aligned)
                proof = [[r["source_pixels"], r["shared_pixels"]] for r in found["candidates"]]
                labels = np.zeros_like(native.selected_grid_mask)
                for component, (area, matched) in zip(native.component_masks, proof, strict=True):
                    if 2 * matched >= area:
                        labels |= component
                combined = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) > 0
            row = {"input": plan, "native": [int(mask.sum()), int((mask & truth).sum())],
                   "repeat": [int(combined.sum()), int((combined & truth).sum())], "comparison": proof,
                   "native_mask_sha256": sha(mask.tobytes()), "repeat_mask_sha256": sha(combined.tobytes())}
            rows.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            if len(rows) % 6 == 0:
                print(f"{len(rows)}/108, last delivery {time.monotonic() - started:.2f} s", flush=True)
    report = {"freeze": freeze, "rows": rows, "result": audit(freeze, rows)}
    write(args.output / "receipt.json", report)
    return report["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "measure", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--sources", type=Path, default=Path("backend/data/localization-eval"))
    parser.add_argument("--bg", type=Path, nargs=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/weights/native-regional-v1/regional-model.safetensors"))
    args = parser.parse_args()
    if args.stage == "review":
        packet = json.loads(args.output.read_bytes())
        result = audit(packet["freeze"], packet["rows"])
        require(result == packet["result"], "Published result differs from recount")
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    freeze = args.output / "freeze.json"
    require(args.bg is not None, "Provide both BG-REAL trial directories")
    identity = {"schema": SCHEMA, "method": METHOD, "runner_sha256": file_sha(Path(__file__)),
                "bg_intake_sha256": [file_sha(path / "result.json") for path in args.bg],
                "checkpoint_sha256": file_sha(args.checkpoint)}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        roster = [plan for plan, _, _, _ in materialize(args)]
        require(len(roster) == 108, "Source intake incomplete")
        write(freeze, identity | {"roster": roster})
    else:
        saved = json.loads(freeze.read_bytes())
        require(all(saved[k] == v for k, v in identity.items()), "Frozen runtime or inputs changed")
        print(json.dumps(measure(args, saved), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
