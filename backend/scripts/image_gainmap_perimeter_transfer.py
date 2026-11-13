# SPDX-License-Identifier: Apache-2.0
"""Compare a coherent SDR-only dark perimeter against retained HDR gain maps."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_gainmap_cross_camera as cross

SCHEMA = "chipmunk.image-gainmap-perimeter-transfer/1"
PUBLIC = cross.PRIOR.parent
CROSS = PUBLIC / "image-gainmap-cross-camera-2026-11-12.json"
METHOD = {"working_long_edge": 384, "sample_step": [4, 8], "side_samples": 12, "side_support_order": 2,
          "maximum_map_step_codes": 12, "combine": "Mean of the lowest three of four signed side supports, clipped at zero.",
          "point": "Strictly above the largest ordinary response in the first four published Pixel 6 Pro groups.",
          "controls": {"local_lift": 1.18, "local_retouch": .64},
          "prior_role": "Previously scored by other cues. Use for transfer diagnostics only.", "confidence_weight": 0}


def freeze(fresh):
    cross_roster = cross.read(CROSS)["freeze"]["source_roster"]
    cross_hashes = {row["local_sha256"] for row in cross_roster}
    prior_hashes = {row["sha256"] for row in cross.read(cross.PRIOR)["intake"]["files"]}
    cross.require(len({row["sha256"] for row in fresh}) == len(fresh) and all(
        row["source"] == "Own work" and row["license"] == "CC BY-SA 4.0"
        and row["sha256"] not in cross_hashes | prior_hashes for row in fresh), "Unexpected fresh sources")
    return {"schema": SCHEMA, "method": METHOD, "sources": fresh, "prior_sha256": cross.sha(cross.PRIOR.read_bytes()),
            "cross_sha256": cross.sha(CROSS.read_bytes()), "scripts": cross.script_hashes() | {
                Path(__file__).name: cross.sha(Path(__file__).read_bytes())}}


def measure(image, gain, mask):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts import image_gainmap_localization_transfer as prior

    work = prior.working(image, METHOD["working_long_edge"]).convert("L")
    width, height = work.size
    luminance = np.log2((np.asarray(work, dtype=np.float32) + 12) / 267)
    luminance = cv2.GaussianBlur(luminance, (0, 0), 1)
    map_codes = np.asarray(gain.resize(work.size, Image.Resampling.BILINEAR), dtype=np.float32)
    map_codes = cv2.GaussianBlur(map_codes, (0, 0), 1)
    samples = (np.arange(METHOD["side_samples"]) + .5) / METHOD["side_samples"]
    offsets = np.array(METHOD["sample_step"])

    def side(outer_axis, along, direction, horizontal):
        centered = outer_axis[..., None, None] + direction * offsets
        reflected = outer_axis[..., None, None] - direction * offsets
        along = along[..., :, None]
        inner, outer = ((centered, along), (reflected, along)) if horizontal else ((along, centered), (along, reflected))
        step = (luminance[outer] - luminance[inner]).mean(axis=-1)
        map_step = np.abs(map_codes[outer] - map_codes[inner]).mean(axis=-1)
        supported = np.where(map_step <= METHOD["maximum_map_step_codes"], step, 0)
        support_index = METHOD["side_support_order"]
        return np.partition(supported, support_index, axis=-1)[..., support_index]

    highest, box = 0., None
    for left, top, right, bottom in prior._windows(height // prior.CELL, width // prior.CELL, prior.CELL):
        if not left.size:
            continue
        across = np.rint(left[..., None] + (.15 + .7 * samples) * (right - left)[..., None]).astype(np.intp)
        down = np.rint(top[..., None] + (.15 + .7 * samples) * (bottom - top)[..., None]).astype(np.intp)
        sides = np.stack((side(top, across, 1, True), side(bottom, across, -1, True),
                          side(left, down, 1, False), side(right, down, -1, False)), axis=-1)
        scores = np.sort(sides, axis=-1)[..., :3].mean(axis=-1)
        index = np.unravel_index(scores.argmax(), scores.shape)
        if scores[index] > highest:
            highest = float(scores[index])
            box = [round(coordinate[index] * native / rendered) for coordinate, native, rendered in
                   zip((left, top, right, bottom), image.size * 2, work.size * 2, strict=True)]
    hit = int(mask[box[1]:box[3], box[0]:box[2]].sum()) if box is not None and mask is not None else 0
    return {"score": highest.hex(), "box": box, "hit": hit}


def retouches(image, gain, source_hash):
    import numpy as np
    from PIL import Image

    from scripts import image_gainmap_localization_transfer as prior

    base = prior.working(image, prior.SIDE * 2)
    width, height = base.size
    left, top, right, bottom = (round(value * length) for value, length in zip(
        prior._box_from_gain(gain, source_hash), (width, height, width, height), strict=True))
    for name, factor in METHOD["controls"].items():
        pixels = np.asarray(base).copy()
        tile = pixels[top:bottom, left:right]
        y, x = np.ogrid[:tile.shape[0], :tile.shape[1]]
        inset = np.minimum(np.minimum(x, tile.shape[1]-1-x), np.minimum(y, tile.shape[0]-1-y))
        blend = np.clip(inset / max(1, min(tile.shape[:2]) * .14), 0, 1)[..., None]
        tile[:] = np.rint(np.clip(tile * (1 + (factor - 1) * blend), 0, 255)).astype(np.uint8)
        out = io.BytesIO()
        Image.fromarray(pixels).save(out, "JPEG", quality=90, subsampling=2)
        with Image.open(out) as delivered:
            yield name, delivered.convert("RGB")


def row(key, author, role, image, gain, mask, prior_scores=None):
    return {"key": key, "author": author, "role": role, "base": cross.geometry(image), "gain": cross.geometry(gain),
            "truth_pixels": int(mask.sum()) if mask is not None else 0,
            "truth_sha256": cross.sha(mask.tobytes()) if mask is not None else None,
            "perimeter": measure(image, gain, mask) if gain is not None else None,
            "baseline": cross.measure(image, gain, mask) if prior_scores is None and gain is not None else prior_scores}


def selection(archive):
    from scripts import image_gainmap_localization_transfer as prior

    for slot, role, body, _ in prior.cases(archive):
        if slot < 4 and role in prior.ORDINARY:
            image, gain, _ = prior.split(body)
            yield row(str(slot), "Mishaal Rahman", role, image, gain, None)


def held(archive):
    prior_receipt = cross.read(CROSS)
    sources = {item["key"]: item for item in prior_receipt["freeze"]["source_roster"]}
    for (key, role, image, gain, mask, _), old in zip(cross.cases(archive, list(sources.values())), prior_receipt["measured"], strict=True):
        cross.require((key, role) == (old["key"], old["role"]), "Cross-camera input differs")
        source = sources[key]
        yield row(key, source["author"], role, image, gain, mask, old["scores"])
        if role == "original" and gain is not None:
            for name, delivered in retouches(image, gain, source["local_sha256"]):
                yield row(key, source["author"], name, delivered, gain, None)


def carrier_profile(data, admitted):
    from scripts import image_gainmap_iso as reader

    base = reader._scan(data, 0, len(data))
    gain = reader._scan(data, base.end, len(data))
    (index,) = reader._markers(data, base, 0xE2, reader.MPF)
    start, end = index
    tiff = data[start:end]
    byte_orders = {b"II": "<", b"MM": ">"}
    cross.require(len(tiff) >= 82 and tiff[:2] in byte_orders, "Invalid intake MPF header")
    endian = byte_orders[tiff[:2]]
    magic, ifd = struct.unpack_from(endian + "HI", tiff, 2)
    cross.require(magic == 42 and 8 <= ifd <= len(tiff) - 6, "Invalid intake MPF offset")
    count, = struct.unpack_from(endian + "H", tiff, ifd)
    cross.require(count == 3 and ifd + 6 + count * 12 <= len(tiff), "Invalid intake MPF tag count")
    tags = {tag: (kind, length, value) for tag, kind, length, value in struct.iter_unpack(
        endian + "HHII", tiff[ifd + 2:ifd + 2 + count * 12])}
    cross.require(len(tags) == count and tags.get(0xB001) == (4, 1, 2) and tags.get(0xB002, ())[:2] == (7, 32),
                  "Invalid intake MPF entries")
    table = tags[0xB002][2]
    cross.require(ifd + 6 + 12 * count <= table <= len(tiff) - 32, "Intake MPF table outside TIFF")
    _, primary_size, primary_offset, _, _ = struct.unpack_from(endian + "IIIHH", tiff, table)
    aux_type, aux_size, offset, _, _ = struct.unpack_from(endian + "IIIHH", tiff, table + 16)
    declared_start = start + offset
    aux_metadata = reader._iso_marker(data, gain)
    cross.require(primary_offset == 0 and declared_start == base.end and declared_start + aux_size == gain.end == len(data),
                  "Published input has different auxiliary boundaries")
    return {"admitted_grayscale": admitted, "primary_end": base.end, "mpf_primary_size": primary_size,
            "gain_start": declared_start, "gain_end": gain.end, "gain_components": gain.components,
            "base_dimensions": list(base.size), "gain_dimensions": list(gain.size), "mpf_aux_type": aux_type,
            "iso_primary_marker": reader._iso_marker(data, base) == b"\0" * 4,
            "iso_gain_metadata_bytes": len(aux_metadata) if aux_metadata is not None else 0,
            "iso_gain_flags": aux_metadata[4] if aux_metadata is not None and len(aux_metadata) >= 5 else None,
            "gain_sha256": cross.sha(data[declared_start:gain.end])}


def fresh_views(archive, sources):
    from scripts import image_gainmap_iso as reader

    for source in sources:
        key = source["key"]
        data = (archive / (key + ".jpg")).read_bytes()
        cross.require(len(data) == source["bytes"] and cross.sha(data) == source["sha256"]
                      and hashlib.sha1(data).hexdigest() == source["publisher_sha1"], "Fresh source changed")
        parsed = reader.split(data)
        image, gain = parsed[:2] if parsed else (None, None)
        original = row(key, source["author"], "original", image, gain, None)
        original["carrier"] = carrier_profile(data, parsed is not None)
        yield original
        if parsed:
            for role, delivered in retouches(image, gain, source["sha256"]):
                yield row(key, source["author"], role, delivered, gain, None)


def point(rows):
    cross.require(len(rows) == 24 and len({(item["key"], item["role"]) for item in rows}) == 24, "Wrong calibration controls")
    return max(float.fromhex(item["perimeter"]["score"]) for item in rows).hex()


def counts(rows, threshold):
    prior_points = cross.read(CROSS)["freeze"]["points"]
    metrics = defaultdict(Counter)
    flagged_authors = defaultdict(set)
    for item in rows:
        role, truth = item["role"], item["truth_pixels"]
        arms = {"perimeter": {"perimeter": item["perimeter"]}, "existing_union": item["baseline"]}
        points = prior_points | {"perimeter": threshold}
        cross.require(bool(truth) == (role in cross.EDITED), "Wrong mask role")
        for method, reads in arms.items():
            cell = metrics[method]
            cell["files"] += 1
            cell["unassessed"] += item["gain"] is None
            if item["gain"] is None:
                cross.require(reads is None or not any(reads.values()), "Unassessed image contains a finding")
                continue
            decisions = []
            for name, candidate in reads.items():
                score = float.fromhex(candidate["score"])
                box, hit = candidate["box"], candidate["hit"]
                cross.require(math.isfinite(score) and score >= 0 and type(hit) is int and 0 <= hit <= truth, "Invalid score or mask overlap")
                if box is None:
                    cross.require(score == hit == 0, "Missing source region")
                    area = 0
                else:
                    w, h = item["base"]["dimensions"]
                    a, b, c, d = box
                    area = (c - a) * (d - b)
                    cross.require(0 <= a < c <= w and 0 <= b < d <= h and hit <= area, "Source region outside the image")
                useful = bool(truth and hit * 2 >= truth and hit * 10 >= 3 * (truth + area - hit))
                decisions.append((score > float.fromhex(points[name]), useful))
            if truth:
                cell[f"useful_{role}"] += any(flag and useful for flag, useful in decisions)
                cell["misplaced"] += any(flag and not useful for flag, useful in decisions)
                cell[f"possible_{role}"] += any(useful for _, useful in decisions)
            else:
                flagged = any(flag for flag, _ in decisions)
                cell[f"ordinary_{role}"] += flagged
                cell["ordinary_flags"] += flagged
                if flagged:
                    flagged_authors[method].add(item["author"])
    return {name: dict(metric, ordinary_authors=len(flagged_authors[name])) for name, metric in metrics.items()}


def summary(held_rows, fresh_rows, threshold):
    profiles = [item["carrier"] for item in fresh_rows if item["role"] == "original"]
    cross.require(len(profiles) == len({item["key"] for item in fresh_rows if item["role"] == "original"}), "Duplicate carrier")
    for profile in profiles:
        cross.require(profile["primary_end"] == profile["gain_start"] < profile["gain_end"]
                      and profile["gain_components"] in {1, 3}, "Unexpected carrier geometry")
    rgb = [profile for profile in profiles if profile["gain_components"] == 3 and profile["iso_primary_marker"]]
    return {"cross_camera_prior_views": counts([item for item in held_rows if item["role"] not in METHOD["controls"]], threshold),
            "new_local_retouches": counts([item for item in held_rows if item["role"] in METHOD["controls"]], threshold),
            "fresh_acquisition": counts(fresh_rows, threshold),
            "fresh_carriers": {"files": len(profiles), "admitted_grayscale": sum(p["admitted_grayscale"] for p in profiles),
                               "rgb_iso": len(rgb), "rgb_iso_primary_size_mismatch": sum(
                                   p["mpf_primary_size"] != p["primary_end"] for p in rgb)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--pixel", type=Path)
    parser.add_argument("--cross", type=Path)
    parser.add_argument("--fresh", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        if not (args.pixel and args.cross and args.fresh):
            parser.error("run needs --pixel, --cross and --fresh")
        cross.configure()
        lock = freeze(cross.read(args.fresh / "roster.json"))
        cross.write(args.receipt.with_name("freeze.json"), lock)
        controls = list(selection(args.pixel))
        threshold = point(controls)
        cross.write(args.receipt.with_name("point.json"), {"selection": controls, "point": threshold})
        outcomes, new = list(held(args.cross)), list(fresh_views(args.fresh, lock["sources"]))
        result = summary(outcomes, new, threshold)
        cross.write(args.receipt, {"freeze": lock, "selection": controls, "point": threshold, "held": outcomes,
                                   "fresh": new, "summary": result})
    else:
        receipt = cross.read(args.receipt)
        cross.require(receipt["freeze"] == freeze(receipt["freeze"]["sources"]), "Frozen definition changed")
        threshold = point(receipt["selection"])
        cross.require(threshold == receipt["point"], "Ordinary operating point changed")
        result = summary(receipt["held"], receipt["fresh"], threshold)
        cross.require(result == receipt["summary"], "Held results changed")
        if args.pixel and args.cross and args.fresh:
            cross.configure()
            cross.require(receipt["selection"] == list(selection(args.pixel)) and receipt["held"] == list(held(args.cross))
                          and receipt["fresh"] == list(fresh_views(args.fresh, receipt["freeze"]["sources"])), "Native replay differs")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
