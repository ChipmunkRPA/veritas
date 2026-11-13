# SPDX-License-Identifier: Apache-2.0
"""Evaluate coordinate-only hidden RGB observations on known PNG samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import struct
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from scripts import image_png_hidden_rgb_candidate as candidate
from scripts import image_three_class_patch_fresh as prior

SCHEMA = "chipmunk.png-hidden-rgb-transfer/1"
PUBLIC = {
    "flowers": ("DuckPlant67", "Plastic_Flowers_Transparent_Background,_Blue.png", "a/a9", "51756091d3c186e487ec6fc8317f5f32266ac137"),
    "cactus": ("Andrikkos", "Cactus_flower_extracted.png", "f/fd", "74d3b590a6d730674423819ae0e73be82d3db927"),
    "strawberry": ("Paolo Neo", "Strawberry_(transparent_background).png", "8/86", "0162b0a28502c1f1c7b959570314c1024dea50d3"),
    "blank": ("David Louis Hughes", "Blank_Canvas_on_Transparent_Background.png", "5/54", "a05960ad19b5f03da367e96aa5c70c78de2dc786"),
    "blue": ("The People's Internet", "Blue_Flower_Transparent_Background.png", "f/fc", "336f729c608ffa86791723306cae59f6b48b2d98"),
}
DESIGN = {
    "claim": "Fully transparent native PNG tiles contain spatially varying RGB samples; intent and AI origin remain unassessed.",
    "private_outputs": "Coordinates, counts, and mask hash only. No hidden RGB and no image artifact.",
    "groups": {"development": 8, "confirmation": 12, "public_controls": 5},
    "useful": {"minimum_iou": .30, "minimum_hidden_support_recall": .50},
    "advance": {"minimum_large": 10, "minimum_small": 6, "maximum_negative_highlights": 0, "maximum_invalid_admissions": 0},
    "grid": {"tile_px": candidate.TILE, "stride_px": candidate.STEP, "visible_margin_px": candidate.CLEAR_MARGIN,
             "minimum_step": candidate.DETAIL_STEP, "minimum_pairs_per_axis": candidate.MIN_DETAIL_PAIRS,
             "minimum_component_tiles": candidate.MIN_TILES},
}
read, write, require, sha = prior.read, prior.write, prior.require, prior.sha
HIDDEN = ("small_retained", "large_retained", "cutout_retained")
NEGATIVE = ("opaque", "scrubbed", "constant_matte", "partial_alpha", "alpha_fringe", "isolated_rgb")
BAD = ("bad_idat_crc", "bad_iend_crc", "tail", "extra_raster", "invalid_filter", "bad_stream_footer", "interlaced", "rotated_metadata", "apng")


def encode(array, *, fmt="PNG", **kw):
    with io.BytesIO() as stream:
        Image.fromarray(array).save(stream, fmt, **kw)
        return stream.getvalue()


def public_sources(folder):
    from urllib.parse import quote

    records = {}
    for key, (author, title, hash_path, expected) in PUBLIC.items():
        body = (folder / f"{key}.png").read_bytes()
        require(hashlib.sha1(body, usedforsecurity=False).hexdigest() == expected, "Public source file SHA-1 changed")
        records[key] = {
            "author": author, "file": title, "license": "CC0-1.0",
            "source": f"https://commons.wikimedia.org/wiki/File:{quote(title, safe='')}",
            "original": f"https://upload.wikimedia.org/wikipedia/commons/{hash_path}/{quote(title, safe='')}",
            "sha1": expected, "sha256": sha(body),
        }
    return records


def sources(args, stage):
    lock, _, _ = prior.previous(args)
    args.later = args.later_f5
    if stage == "development":
        for item, native, _, _ in prior.probe.bound_cases(args, lock["profile"], "calibration"):
            yield item["original_sha256"], native["original"]
    else:
        plan, known = prior.planned(args)
        require(plan == read(args.output_fresh / "plan.json"), "Independent source plan changed")
        for item, native, _ in prior.groups(args.output_fresh, plan, known):
            yield item["original"]["sha256"], native["original"]


def authored(rgb, source):
    image = np.array(ImageOps.fit(Image.fromarray(rgb), (384, 384), method=Image.Resampling.LANCZOS))
    height, width = image.shape[:2]
    y, x = np.indices((height, width))
    key = int(sha(source.encode())[:8], 16)
    cx, cy = 151 + key % 80, 149 + (key >> 8) % 80
    small = (abs(x - cx) < 30) & (abs(y - cy) < 30)
    large = ((x - cx) / 86)**2 + ((y - cy) / 68)**2 <= 1
    cutout = ((x - cx) / 104)**2 + ((y - cy) / 129)**2 > 1
    all_visible = np.full((height, width), 255, dtype=np.uint8)
    empty = np.zeros((height, width), dtype=bool)

    def rgba(color, alpha):
        return np.dstack((color, alpha))

    def clear(mask):
        return np.where(mask, 0, 255).astype(np.uint8)

    yield "small_retained", rgba(image, clear(small)), small
    yield "large_retained", rgba(image, clear(large)), large
    yield "cutout_retained", rgba(image, clear(cutout)), cutout
    yield "opaque", rgba(image, all_visible), empty
    yield "scrubbed", rgba(np.where(large[..., None], 0, image), clear(large)), empty
    yield "constant_matte", rgba(np.where(large[..., None], [154, 65, 218], image).astype(np.uint8), clear(large)), empty
    yield "partial_alpha", rgba(image, np.where(large, 1, 255).astype(np.uint8)), empty
    distance = cv2.distanceTransform(large.astype(np.uint8), cv2.DIST_C, 3)
    yield "alpha_fringe", rgba(np.where((distance > 4)[..., None], 0, image), clear(large)), empty
    isolated = (x % 31 == 0) & (y % 29 == 0)
    yield "isolated_rgb", rgba(np.where((large & ~isolated)[..., None], 0, image), clear(large)), empty


def chunk(name, payload):
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))


def malformed():
    rgba = np.zeros((128, 128, 4), dtype=np.uint8)
    body = encode(rgba)
    first_idat_end = 33 + 12 + int.from_bytes(body[33:37], "big")
    require(body[37:41] == b"IDAT" and body[first_idat_end + 4:first_idat_end + 8] == b"IEND", "Unexpected synthetic PNG layout")

    def flip(data, offset):
        return data[:offset] + bytes([data[offset] ^ 1]) + data[offset + 1:]

    scan = zlib.decompress(body[41:first_idat_end - 4])
    header, iend = body[:33], body[first_idat_end:]
    stream = body[41:first_idat_end - 4]
    rotated = Image.Exif()
    rotated[274] = 6
    return {
        "bad_idat_crc": flip(body, first_idat_end - 1),
        "bad_iend_crc": flip(body, len(body) - 1),
        "tail": body + b"trailing",
        "extra_raster": header + chunk(b"IDAT", zlib.compress(scan + b"\0")) + iend,
        "invalid_filter": header + chunk(b"IDAT", zlib.compress(b"\x05" + scan[1:])) + iend,
        "bad_stream_footer": header + chunk(b"IDAT", flip(stream, len(stream) - 1)) + iend,
        "interlaced": body[:8] + chunk(b"IHDR", body[16:28] + b"\x01") + body[33:],
        "rotated_metadata": encode(rgba, exif=rotated),
        "apng": body[:33] + chunk(b"acTL", struct.pack(">II", 1, 0)) + body[33:],
    }


def deliveries(args, stage):
    for group, rgb in sources(args, stage):
        for role, pixels, truth in authored(rgb, group):
            yield group, role, encode(pixels), truth
    if stage == "confirmation":
        for key, record in public_sources(args.public).items():
            yield record["sha256"], f"public_{key}", (args.public / f"{key}.png").read_bytes(), None
        for key, body in malformed().items():
            yield "synthetic_structural", key, body, None


def identity(group, role, body, truth):
    return {"source": group, "role": role, "input_sha256": sha(body),
            "truth_sha256": sha(truth.tobytes()) if truth is not None else None}


def evaluate(group, role, body, truth):
    result, mask = candidate.inspect(body)
    if mask is not None:
        with Image.open(io.BytesIO(body)) as image:
            expected = np.asarray(image.getchannel("A"))
        require(np.all(expected[mask] == 0), "Selected pixels must all be fully transparent")
        require(mask.shape == expected.shape, "Native dimensions changed")
        require(int(mask.sum()) == result.get("selected_pixels", 0), "Native mask count differs")
    match = int((truth & mask).sum()) if truth is not None and mask is not None else 0
    changed = int(truth.sum()) if truth is not None else 0
    selected = result.get("selected_pixels", 0)
    return identity(group, role, body, truth) | {"result": result, "intersection": match, "target": changed,
            "useful": changed > 0 and 2*match >= changed and 10*match >= 3*(changed + selected - match)}


def summarize(rows):
    strata = defaultdict(Counter)
    for row in rows:
        role, measured = row["role"], row["result"]
        strata[role].update(views=1, assessed=measured["status"] == "measured", highlights=bool(measured["regions"]),
                            useful=row["useful"], selected_pixels=measured.get("selected_pixels", 0))
    counts = {"large": strata["large_retained"]["useful"], "small": strata["small_retained"]["useful"],
              "negative_highlights": sum(strata[role]["highlights"] for role in NEGATIVE),
              "invalid_admissions": sum(strata[role]["assessed"] for role in BAD)}
    limits = DESIGN["advance"]
    passed = (counts["large"] >= limits["minimum_large"] and counts["small"] >= limits["minimum_small"]
              and counts["negative_highlights"] <= limits["maximum_negative_highlights"]
              and counts["invalid_admissions"] <= limits["maximum_invalid_admissions"])
    return {"strata": strata, "confirmation_rule": counts, "passes": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for key in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "output", "public"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    freeze_path = args.output / "freeze.json"
    stages = ("development", "confirmation")
    frozen = {
        "schema": SCHEMA, "design": DESIGN, "public": public_sources(args.public),
        "candidate_sha256": prior.roster.sha(Path(candidate.__file__)), "runner_sha256": prior.roster.sha(Path(__file__)),
        "view_bindings": {phase: [identity(*item) for item in deliveries(args, phase)] for phase in stages},
    }
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, frozen)
        return
    require(read(freeze_path) == frozen, "Source or method freeze differs")
    result = {}
    for phase in stages:
        start = time.monotonic()
        rows = [evaluate(*item) for item in deliveries(args, phase)]
        path = args.output / f"{phase}.json"
        metrics = summarize(rows)
        if args.stage == "run":
            write(path, rows)
        else:
            require(rows == read(path), "Native measurement replay differs")
        result[phase] = metrics | {"delivery_count": len(rows), "private_rows_sha256": prior.roster.sha(path),
                                   "elapsed_seconds": round(time.monotonic() - start, 2)}
        print(phase, len(rows), metrics["confirmation_rule"], flush=True)
    if args.stage == "review":
        publication = {key: value for key, value in frozen.items() if key != "view_bindings"}
        write(args.publication, publication | {"results": result, "freeze_sha256": prior.roster.sha(freeze_path),
                                               "source_or_revealed_pixels_published": False})


if __name__ == "__main__":
    main()
