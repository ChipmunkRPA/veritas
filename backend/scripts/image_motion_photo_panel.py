# SPDX-License-Identifier: Apache-2.0
"""Replay the Motion Photo comparison on checksum-pinned camera originals."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import motion_photo
from app.evidence.image_motion_photo import inspect

VARIANTS = ("native", "reencoded", "tone", "staged_center", "staged_offcenter")


def variant(data, carrier, kind):
    if kind == "native":
        return data, None
    with Image.open(BytesIO(data[:carrier.image_end]), formats=["JPEG"]) as source:
        with ImageOps.exif_transpose(source) as oriented:
            pixels = np.array(oriented.convert("RGB"))
    height, width = pixels.shape[:2]
    truth = None
    if kind == "tone":
        pixels = np.clip((pixels.astype(np.float32) - 128) * 1.05 + 136, 0, 255).astype(np.uint8)
    elif kind.startswith("staged"):
        rx, ry = (.50, .50) if kind == "staged_center" else (.33, .65)
        w, h = round(width * .14), round(height * .14)
        x, y = round(width * rx) - w // 2, round(height * ry) - h // 2
        other_x = round(width * (rx + .18))
        xs, ys = np.arange(w), np.arange(h)
        horizontal = np.minimum(xs, xs[::-1])[None] / max(1, w * .08)
        vertical = np.minimum(ys, ys[::-1])[:, None] / max(1, h * .08)
        mix = np.minimum(horizontal, vertical).clip(0, 1)[..., None]
        target = pixels[y:y + h, x:x + w]
        replacement = pixels[y:y + h, other_x:other_x + w]
        target[:] = np.rint(target * (1 - mix) + replacement * mix)
        truth = (x / width, y / height, w / width, h / height)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=91)
    jpeg = buffer.getvalue()
    packet = motion_photo.XMP + carrier.xmp
    app1 = b"\xff\xe1" + (2 + len(packet)).to_bytes(2, "big") + packet
    # Leave the gain map, video and timing declaration byte-for-byte intact.
    return jpeg[:2] + app1 + jpeg[2:] + data[carrier.image_end:], truth


def overlap(reading, mask, truth):
    width, height = reading["comparison"]["still_grid"]
    x, y, w, h = (round(unit * edge) for unit, edge in zip(truth, (width, height, width, height), strict=True))
    mx, my, _, _ = reading["region"]["mask"]
    ys, xs = np.nonzero(np.asarray(mask)[:, :, 0])
    hit = int(((xs + mx >= x) & (xs + mx < x + w) & (ys + my >= y) & (ys + my < y + h)).sum())
    total = w * h + reading["region"]["marked_pixels"] - hit
    return round(hit / max(1, total), 6)


def study(body, folder=None, name=None):
    try:
        carrier = motion_photo.carrier(body)
    except ValueError as error:
        return {"container": str(error), "cases": []}
    if carrier is None:
        return {"container": "No Motion Photo declaration", "cases": []}
    rows = []
    for kind in VARIANTS:
        data, truth = variant(body, carrier, kind)
        digest = sha256(data).hexdigest()
        reading, pictures = inspect(data, digest)
        if reading is None:
            raise ValueError("A case lost its Motion Photo declaration")
        iou = overlap(reading, pictures["mask"], truth) if truth and reading["region"] is not None else None
        rows.append({"variant": kind, "sha256": digest, "truth_rectangle": truth, "iou": iou, "reading": reading})
        if folder is not None:
            (folder / f"{name}-{kind}.jpg").write_bytes(data)
    return {"container": "JPEG and MP4", "cases": rows}


def recount(sources):
    cases = [case for source in sources for case in source["cases"]]
    staged = [case for case in cases if case["truth_rectangle"] is not None]
    ordinary = [case for case in cases if case["truth_rectangle"] is None]
    return {"source_files": len(sources), "supported_attachments": sum(bool(source["cases"]) for source in sources),
            "ordinary_deliveries": len(ordinary), "ordinary_assessed": sum(c["reading"]["status"] != "unassessed" for c in ordinary),
            "ordinary_flags": sum(c["reading"]["status"] == "different" for c in ordinary),
            "staged_deliveries": len(staged), "staged_assessed": sum(c["reading"]["status"] != "unassessed" for c in staged),
            "staged_with_region": sum(c["reading"]["status"] == "different" for c in staged),
            "correct_regions": sum(c["iou"] is not None and c["iou"] >= .10 for c in staged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="pinned source list")
    parser.add_argument("sources", type=Path, help="folder containing the downloaded originals")
    parser.add_argument("output", type=Path, help="JSON receipt")
    parser.add_argument("--example", help="source filename for an optional set of local visual test cases")
    parser.add_argument("--example-folder", type=Path)
    args = parser.parse_args()
    if bool(args.example) != bool(args.example_folder):
        parser.error("--example and --example-folder must be used together")
    if args.example_folder:
        args.example_folder.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    receipts = []
    for entry in json.loads(args.manifest.read_bytes()):
        filename = entry["file"]
        if filename != Path(filename).name:
            raise ValueError("Expected a source filename")
        path = args.sources / filename
        data = path.read_bytes()
        if len(data) != entry["size"] or sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source bytes differ from the manifest for {filename}")
        example = args.example_folder if filename == args.example else None
        receipts.append({"file": filename, "sha256": entry["sha256"], **study(data, example, path.stem)})
        print(filename, receipts[-1]["container"], flush=True)
    result = {"schema": "motion-photo-held-camera-panel/1", "sources": receipts, "summary": recount(receipts)}
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
