# SPDX-License-Identifier: Apache-2.0
"""Compare color-only and brightness evidence on qualified lossy WebP pairs."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts import image_jpeg_native_rotation_transfer as cameras
from scripts import image_reference_resized_webp_transfer as webp

photos = webp.parent
SOURCE = photos.development.SOURCE
CAMERAS = SOURCE.with_name("image-jpeg-native-rotation-held-2026-11-15.json")
SCHEMA = "chipmunk.image-reference-webp-color-transfer/2"
SOURCE_ORDER = "chipmunk.image-reference-webp-color-transfer/1"
PROFILES = {
    "same": (None, None, "bicubic", 73),
    "crop": (.9, None, "bicubic", 82),
    "resize": (None, .83, "lanczos", 86),
    "crop_resize": (.89, .87, "area", 89),
}
ARMS = ("brightness", "color_offset", "color_affine", "color_gamut")
EDITS = {"small_recolor", "feathered_recolor", "separate_luma_and_color", "editor_color", "editor_original"}
CONTROLS = {"unchanged", "prior_jpeg", "prior_webp", "whole_tint", "whole_hue", "whole_saturation", "local_luma"}
SCOPE = {"unrelated", "translated"}
METHOD = {
    "sources": "Reserve eight credited BG-REAL photos for development and 18 for held evaluation. Partition each editor "
               "by fixed source hash before scoring. Transfer the same rules to 10 separate CC0 camera photos. No models are fitted.",
    "carrier": "Each original is tested with same-framing WebP and one of unscaled crop, proportional resize, or "
               "crop and resize. First require existing brightness-based registration and framing. Reuse its exact projected crop.",
    "conditions": "Measure an unchanged re-encode, two intermediate codecs, three whole-image color changes, separate local "
                  "brightness, an unrelated image and a translated image. Add sharp and feathered recolors, recolor with "
                  "a separate brightness edit, and actual BG-REAL edits with and without their luma change.",
    "color": "Compare smoothed blue and red color differences with a fixed minimum of 8 levels. Compare a color offset "
             "with a robust affine fit for hue and saturation and with two clipping-aware adjustments. "
             "Select clipping-aware alternatives using a separate sampling grid with capped per-pixel errors. "
             "Only measure color with enough shared framing and keep earlier brightness findings.",
    "readout": "A useful outline covers at least 50% of the exact changed pixels and reaches 0.30 intersection over union. "
               "Measure scope, missed edits, misplaced and extra ordinary locations. Count color separately from the total mask.",
}


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write(path, packet):
    values = json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(values, mtime=0))


def inventory():
    from scripts import image_edge_context_new_sources as bank

    source = photos.read(SOURCE)
    photos.require(bank.summary(source) == source["summary"], "The attributed earlier bank does not replay")
    for family, count in (("lama", 3), ("sd15", 3), ("kandinsky", 2)):
        items = sorted((item for item in source["sources"] if item["generator"] == family),
                       key=lambda item: photos.sha(f"{SOURCE_ORDER}/{item['original']['sha256']}".encode()))
        photos.require(len(items) > count, f"Incomplete credited family {family}")
        for index, item in enumerate(items):
            yield "development" if index < count else "held", item


def delivery_profiles(group):
    return "same", tuple(PROFILES)[1 + bytes.fromhex(group)[0] % 3]


def source_specs():
    for split, item in inventory():
        group = item["original"]["sha256"]
        yield {"split": split, "family": item["generator"], "group": group, "source": item["slot"],
               "source_sha256": item["proof"]["assets"]["original"]}
    old = cameras.read(CAMERAS)
    unchanged = [row for row in old["freeze"]["rows"] if row["role"] == "ordinary" and row["geometry"] == "medium"]
    photos.require(len(unchanged) == 10, "The published CC0 camera files differ")
    for row in unchanged:
        pin = row["sha256"]
        yield {"split": "cameras", "family": "cc0_camera", "group": pin, "source": row["file"], "source_sha256": pin}


def source_pixels(sources, camera_dir, spec):
    from scripts import image_edge_context_new_sources as bank

    if spec["split"] == "cameras":
        data = (camera_dir / spec["source"]).read_bytes()
        photos.require(photos.sha(data) == spec["source_sha256"], "The original CC0 delivery changed")
        rgb = photos.image_array(data, "JPEG")
        width, height = rgb.shape[1::-1]
        adjusted = round(width * 768 / max(width, height)), round(height * 768 / max(width, height))
        return webp.resize(rgb, adjusted, "area"), None
    source = next(item for _, item in inventory() if item["slot"] == spec["source"])
    expected = source["proof"]["assets"]
    inputs = {role: (sources / str(source["slot"]) / f"{role}.png").read_bytes() for role in expected}
    photos.require({role: photos.sha(value) for role, value in inputs.items()} == expected, "A credited source image changed")
    views, truth, box, _ = bank.read_source(source, inputs)
    photos.require(box == source["proof"]["box"] and photos.sha(truth.tobytes()) == source["proof"]["truth_sha256"],
                   "The earlier publisher edit mask changed")
    return views["original"], views["edited"]


def shift_color(rgb, change):
    import numpy as np

    # Use one per-pixel fraction to stay within RGB range and maintain the
    # authored relative channel changes even in saturated source pixels.
    change = np.broadcast_to(change, rgb.shape)
    upper = np.where(change > 0, 255. - rgb, rgb)
    fraction = np.minimum(1., np.min(upper / np.maximum(abs(change), 1e-8), axis=2, keepdims=True))
    return np.rint(np.clip(rgb + fraction * change, 0, 255)).astype(np.uint8)


def pure_color(change):
    import numpy as np

    return change - (change @ np.array([.299, .587, .114]))[..., None]


def tint(rgb, vector):
    import numpy as np

    y, channels = rgb.astype(np.float32), np.asarray(vector, dtype=np.float32)
    return shift_color(y, pure_color(channels))


def alterations(rgb, authored, foreign, group):
    import numpy as np

    h, w = rgb.shape[:2]
    seed = bytes.fromhex(group)
    x, y = w // 4 + seed[1] % (w // 5), h // 4 + seed[2] % (h // 5)
    tint_vector = np.array(((44., -8., -64.), (-50., 12., 63.))[seed[3] % 2])
    color = pure_color(tint_vector)

    def patch(source, left, top, size, vector, feather):
        pixels = source.copy()
        crop = source[top:top+size, left:left+size].astype(np.float32)
        if feather:
            axis = np.minimum.reduce((np.arange(size) + 1, np.arange(size, 0, -1), np.full(size, feather))) / feather
            vector = np.minimum(axis[:, None], axis[None, :])[:, :, None] * vector
        pixels[top:top+size, left:left+size] = shift_color(crop, vector)
        return pixels

    small = patch(rgb, x, y, 40, color, 0)
    soft = patch(rgb, x, y, 64, color * .72, 10)
    luma = patch(rgb, w * 2 // 3, h // 2, 72, np.array([37., 37., 37.]), 8)
    dual = luma.copy()
    dual[y:y+64, x:x+64] = soft[y:y+64, x:x+64]
    chroma = pure_color(rgb.astype(np.float32))
    angle = .27
    cb = chroma[:, :, 2] * .564
    cr = chroma[:, :, 0] * .713
    rotated_cb = np.cos(angle) * cb - np.sin(angle) * cr
    rotated_cr = np.sin(angle) * cb + np.cos(angle) * cr
    delta_b, delta_r = (rotated_cb - cb) / .564, (rotated_cr - cr) / .713
    delta_g = -(.299 * delta_r + .114 * delta_b) / .587
    variants = {"unchanged": rgb, "small_recolor": small, "feathered_recolor": soft,
                "separate_luma_and_color": dual, "local_luma": luma, "whole_tint": tint(rgb, [18, -5, -22]),
                "whole_hue": shift_color(rgb, np.stack((delta_r, delta_g, delta_b), axis=-1)),
                "whole_saturation": shift_color(rgb, .28 * chroma), "translated": np.roll(rgb, 4, axis=1),
                "unrelated": webp.resize(foreign, (w, h), "bilinear"),
                "prior_webp": photos.image_array(photos.encode(rgb, "WEBP", quality=61, method=4), "WEBP"),
                "prior_jpeg": photos.image_array(photos.encode(rgb, "JPEG", quality=78, subsampling=2), "JPEG")}
    if authored is not None:
        variants["editor_original"] = authored
        variants["editor_color"] = shift_color(rgb, pure_color(authored.astype(np.float32) - rgb))
    return variants


def project(pixels, crop, size, method):
    x, y, width, height = crop
    patch = pixels[y:y+height, x:x+width]
    return webp.resize(patch, size, method) if size != (width, height) else patch


def deliveries(spec, sources, camera_dir, foreign, *, profile=None, role=None):
    import numpy as np

    rgb, generated = source_pixels(sources, camera_dir, spec)
    variants = alterations(rgb, generated, foreign, spec["group"])
    reference = photos.encode(rgb, "PNG")
    for name in delivery_profiles(spec["group"]):
        if profile is not None and name != profile:
            continue
        crop, scale, filter_name, quality = PROFILES[name]
        height, width = rgb.shape[:2]
        crop_x = crop_y = 0
        if crop:
            dw, dh = round(width * (1. - crop)), round(height * (1. - crop))
            crop_x, crop_y = dw * 2 // 5, dh * 3 // 4
            width, height = width - dw, height - dh
        target_size = (round(width * scale), round(height * scale)) if scale else (width, height)
        bounds = crop_x, crop_y, width, height
        base = project(rgb, bounds, target_size, filter_name)
        for key, variant in sorted(variants.items()):
            if role is not None and key != role:
                continue
            authored = project(variant, bounds, target_size, filter_name)
            truth_source = (project(variants["feathered_recolor"], bounds, target_size, filter_name)
                            if key == "separate_luma_and_color" else authored)
            changed = np.any(base != truth_source, axis=2) if key in EDITS else np.zeros(base.shape[:2], bool)
            submission = photos.encode(authored, "WEBP", quality=quality, method=2)
            yield {**spec, "profile": name, "role": key, "expected_crop": list(bounds),
                   "submitted_size": list(target_size), "reference_sha256": photos.sha(reference),
                   "submitted_sha256": photos.sha(submission), "truth_sha256": photos.sha(changed.tobytes()),
                   "changed": int(changed.sum())}, reference, submission, changed


def baseline(earlier, submitted, before, after):
    from app.detectors.image import (
        reference_lossy_webp,
        reference_lossy_webp_crop,
        reference_lossy_webp_crop_resize,
        reference_lossy_webp_resize,
    )

    if earlier.size == submitted.size:
        return reference_lossy_webp.compare(earlier, submitted, before, after)
    for detector in (reference_lossy_webp_resize, reference_lossy_webp_crop, reference_lossy_webp_crop_resize):
        result, mask = detector.compare(earlier, submitted, before, after)
        if result["status"] == "comparable":
            return result, mask
    return result, mask


def measure(case, earlier, submitted, truth):
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_webp_color

    def mask_record(mask, regions):
        return {"selected": int(mask.sum()), "overlap": int((mask & truth).sum()),
                "mask_sha256": photos.sha(mask.tobytes()), "regions": regions}

    with Image.open(io.BytesIO(earlier)) as first, Image.open(io.BytesIO(submitted)) as second:
        initial, before = baseline(first, second, earlier, submitted)
        if before is None:
            return {"case": case, "status": initial["status"], "measurements": None}
        first_rgb, second_rgb = np.asarray(first), np.asarray(second)
        states = {"brightness": mask_record(before, initial["regions"])}
        for name in ARMS[1:]:
            method = name.removeprefix("color_")
            proof, regions, found, omitted = reference_webp_color.compare(first_rgb, second_rgb, initial, method=method)
            photos.require(not (found & before).any(), "Color evidence overlaps a retained brightness mask")
            states[name] = mask_record(before | found, initial["regions"] + regions) | {
                "color": mask_record(found, regions), "proof": proof, "omitted": omitted,
            }
        crop = initial.get("lossy_webp_crop_resize") or initial.get("lossy_webp_crop")
        observed = (list(first.size) if crop is None else crop["reference_crop"])
        bounds = [0, 0, *observed] if crop is None else [observed[key] for key in ("x", "y", "w", "h")]
    return {"case": case, "status": initial["status"], "measured_crop": bounds, "measurements": states}


def source_run(args):
    spec, pair, sources, camera_dir = args
    import cv2

    cv2.setNumThreads(1)
    foreign, _ = source_pixels(sources, camera_dir, pair)
    return [measure(*delivery) for delivery in deliveries(spec, sources, camera_dir, foreign)]


def parent_pins():
    return {path.name: photos.sha(path.read_bytes()) for path in (SOURCE, CAMERAS)}


def review(packet):
    expected = {spec["group"]: spec for spec in source_specs() if spec["split"] == packet["split"]}
    photos.require(packet["schema"] == SCHEMA and packet["method"] == METHOD and packet["parents"] == parent_pins(),
                   "The fixed evaluation recipe differs")
    totals, inventory = defaultdict(Counter), defaultdict(set)
    for row in packet["rows"]:
        case = row["case"]
        group, profile, role = (case[key] for key in ("group", "profile", "role"))
        photos.require(all(case[key] == value for key, value in expected[group].items())
                       and (profile, role) not in inventory[group], "A split member changed or appears twice")
        inventory[group].add((profile, role))
        width, height = case["submitted_size"]
        states = row["measurements"]
        for arm in ARMS:
            entry = states[arm] if states else None
            selected, overlap = (entry["selected"], entry["overlap"]) if entry else (0, 0)
            colored = entry.get("color") if entry else None
            changed = case["changed"]
            photos.require(0 <= overlap <= min(selected, changed) and selected <= width * height, "Mask counts exceed native picture bounds")
            if colored is not None:
                base = states["brightness"]
                photos.require(entry["regions"] == base["regions"] + colored["regions"]
                               and selected == base["selected"] + colored["selected"]
                               and overlap == base["overlap"] + colored["overlap"]
                               and colored["selected"] == sum(item["selected_pixels"] for item in colored["regions"])
                               and all(item["difference_basis"] == "color" for item in colored["regions"]),
                               "A new color region must preserve the earlier brightness finding")
            right_crop = states is not None and row["measured_crop"] == case["expected_crop"]
            useful = bool(right_crop and changed and photos.useful((selected, overlap), changed))
            without = bool(right_crop and changed and photos.useful((states["brightness"]["selected"],
                                                                     states["brightness"]["overlap"]), changed))
            color_useful = bool(right_crop and changed and colored and photos.useful((colored["selected"], colored["overlap"]), changed))
            for band in (profile, "all"):
                totals[f"{band}/{arm}/{role}"].update(files=1, comparable=bool(states), exact_crop=right_crop,
                    any_flag=selected > 0, useful=useful, gained=useful and not without, lost=without and not useful,
                    added_color=bool(colored and colored["selected"]), color_useful=color_useful,
                    wrong_location=bool(changed and selected and not useful))
    photos.require(set(inventory) == set(expected), "A source group is missing from the receipt")
    for group, pairs in inventory.items():
        roles = EDITS | CONTROLS | SCOPE
        if packet["split"] == "cameras":
            roles = roles - {"editor_color", "editor_original"}
        photos.require(pairs == {(profile, role) for profile in delivery_profiles(group) for role in roles}, "A delivery is missing")
    return {key: dict(counts) for key, counts in totals.items()}


def capture(args):
    cases = [spec for spec in source_specs() if spec["split"] == args.phase]
    work = [(spec, cases[(i + len(cases) // 2) % len(cases)], args.sources, args.cameras) for i, spec in enumerate(cases)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for partial in pool.map(source_run, work):
            rows.extend(partial)
            print(f"{args.phase}: {len(rows)} files scored", flush=True)
    packet = {"schema": SCHEMA, "split": args.phase, "method": METHOD, "parents": parent_pins(), "rows": rows}
    packet["results"] = review(packet)
    write(args.receipt, packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "cameras", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.phase == "review":
        packet = read(args.receipt)
        result = review(packet)
        photos.require(packet["results"] == result, "The recorded results differ from review")
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        capture(args)


if __name__ == "__main__":
    main()
