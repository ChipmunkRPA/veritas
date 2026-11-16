"""Transfer the small geometric copy check to web-compressed image deliveries."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
from collections import Counter
from pathlib import Path

from scripts import image_small_affine_jpeg as parent

SCHEMA = "chipmunk.image-small-affine-webp/1"
QUALITIES = (75, 90)
METHOD = {
    "jpeg_input_quality": 90,
    "webp_qualities": list(QUALITIES),
    "webp_method": 4,
    "minimum_iou": .30,
    "minimum_recall": .50,
    "minimum_detail_correlation": parent.MIN_DETAIL,
    "maximum_color_error": parent.MAX_COLOR_ERROR,
}
SCENE_METHOD = METHOD | {"jpeg_input_quality": None, "edit_stage": "after prior JPEG decode, before WebP"}


def encode(image, row, stem, output):
    from app.detectors.image.reference_lossy_webp import webp_size

    for quality in QUALITIES:
        encoded = io.BytesIO()
        image.save(encoded, "WEBP", quality=quality, method=METHOD["webp_method"], lossless=False)
        delivery = encoded.getvalue()
        parent.require(webp_size(delivery, minimum_side=384) == image.size, "Unexpected lossy WebP profile")
        file = f"{stem}-{quality}.webp"
        output.joinpath(file).write_bytes(delivery)
        yield row | {"file": file, "quality": quality, "sha256": parent.sha(delivery)}


def save_freeze(output, prior, rows, **extra):
    import PIL
    from PIL import features

    parent.write(output / "freeze.json", {
        "schema": SCHEMA, "method": SCENE_METHOD if "staged_after_jpeg" in extra else METHOD,
        "prior_freeze_sha256": parent.sha(prior.read_bytes()),
        "encoder": {"pillow": PIL.__version__, "libwebp": features.version("webp")}, "rows": rows,
        **extra,
    })


def prepare(jpeg_inputs, output):
    from PIL import Image

    freeze = jpeg_inputs / "freeze.json"
    plan = parent.read(freeze)
    parent.require(plan["schema"] == parent.SCHEMA, "Expected the reviewed camera study roster")
    output.mkdir(mode=0o700)
    rows = []
    for row in plan["rows"]:
        if row["quality"] != METHOD["jpeg_input_quality"]:
            continue
        data = (jpeg_inputs / row["file"]).read_bytes()
        parent.require(parent.sha(data) == row["sha256"], "JPEG input differs from the prior freeze")
        stem = Path(row["file"]).stem.rsplit("-", 1)[0]
        if row["truth"] is not None:
            output.joinpath(stem + ".npz").write_bytes(jpeg_inputs.joinpath(stem + ".npz").read_bytes())
        with Image.open(io.BytesIO(data), formats=("JPEG",)) as image:
            parent.require(image.mode == "RGB" and list(image.size) == row["size"], "Unexpected source raster")
            rows.extend(encode(image, row | {"jpeg_sha256": row["sha256"]}, stem, output))
    save_freeze(output, freeze, rows)


def prepare_scenes(camera_inputs, earlier_receipt, output):
    import cv2
    import numpy as np
    from PIL import Image

    cv2.setNumThreads(1)
    freeze = camera_inputs / "freeze.json"
    plan = parent.read(freeze)
    previous = parent.read(earlier_receipt)["freeze"]
    used = {row["group"] for row in previous["rows"]}
    sources = sorted((row for row in plan["rows"] if row["role"] == "ordinary" and row["geometry"] == "medium"),
                     key=lambda row: row["group"])
    groups = {row["group"] for row in sources}
    parent.require(plan["phase"] == "held" and len(sources) == len(groups) == 10 and not groups & used,
                   "Expected ten separate camera groups from the right-angle JPEG study")
    output.mkdir(mode=0o700)
    rows = []
    for index, row in enumerate(sources):
        data = camera_inputs.joinpath(row["file"]).read_bytes()
        parent.require(parent.sha(data) == row["sha256"], "Input differs from the prior freeze")
        with Image.open(io.BytesIO(data), formats=("JPEG",)) as image:
            parent.require(image.mode == "RGB" and list(image.size) == row["size"], "Unexpected camera raster")
            oriented = image.transpose(Image.Transpose.ROTATE_90) if index % 3 == 1 else image
            edge = (768, 1024)[index % 2]
            size = tuple(round(edge * value / max(oriented.size)) for value in oriented.size)
            rgb = np.asarray(oriented.resize(size, Image.Resampling.LANCZOS))
        for role, pixels, truth in parent.views(rgb, index):
            stem = f"{row['group']}-{role}"
            labels = None
            if truth is not None:
                np.savez_compressed(output / (stem + ".npz"), donor=truth[0], receiver=truth[1])
                labels = [parent.sha(mask.tobytes()) for mask in truth]
            variant = {"group": row["group"], "role": role, "truth": labels, "size": list(size),
                       "jpeg_sha256": row["sha256"]}
            rows.extend(encode(Image.fromarray(pixels), variant, stem, output))
    save_freeze(output, freeze, rows, staged_after_jpeg=sources, comparison_receipt_sha256=parent.sha(earlier_receipt.read_bytes()))


def observe(directory, row):
    import numpy as np
    from PIL import Image

    data = directory.joinpath(row["file"]).read_bytes()
    parent.require(parent.sha(data) == row["sha256"], "WebP input differs from the freeze")
    with Image.open(io.BytesIO(data), formats=("WEBP",)) as image:
        parent.require(image.mode == "RGB" and list(image.size) == row["size"], "Unexpected decoded raster")
        rgb = np.asarray(image)
    truth = None
    if row["truth"] is not None:
        stem = Path(row["file"]).stem.rsplit("-", 1)[0]
        with np.load(directory / (stem + ".npz")) as labels:
            truth = tuple(labels[key] for key in ("donor", "receiver"))
        parent.require([parent.sha(mask.tobytes()) for mask in truth] == row["truth"], "Changed-pixel truth differs")
    return {"sha256": row["sha256"], "decoded_rgb_sha256": parent.sha(rgb.tobytes()), **parent.measure(rgb, truth)}


def review(receipt):
    freeze = receipt["freeze"]
    rows = freeze["rows"]
    scenes = "staged_after_jpeg" in freeze
    parent.require(freeze["schema"] == SCHEMA and freeze["method"] == (SCENE_METHOD if scenes else METHOD),
                   "Wrong study protocol")
    counts = Counter((row["quality"], row["role"]) for row in rows)
    roles = (*parent.EDIT_SPECS, *parent.CONTROLS)
    if scenes:
        expected = dict.fromkeys(roles, 10)
    else:
        expected = dict.fromkeys(roles, 18) | {"repetition": 9}
    parent.require(counts == {(quality, role): count for quality in QUALITIES for role, count in expected.items()},
                   "Incomplete edit and ordinary control roster")
    return parent.tally(rows, receipt["observed"])


def capture(directory, receipt, workers):
    from functools import partial

    from app.detectors.image import copy_affine_small

    parent.require(copy_affine_small.MAX_COLOR_ERROR == parent.MAX_COLOR_ERROR
                   and copy_affine_small.MIN_DETAIL_CORRELATION == parent.MIN_DETAIL, "The live confirmation rules differ")
    freeze = parent.read(directory / "freeze.json")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(observe, directory), freeze["rows"]))
    result = {"freeze": freeze, "observed": rows}
    result["summary"] = review(result)
    parent.write(receipt, result)
    return result["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase, fields in (("prepare", ("jpeg_inputs", "output")), ("capture", ("directory", "receipt")),
                          ("prepare-scenes", ("camera_inputs", "earlier_receipt", "output")),
                          ("review", ("receipt",))):
        command = commands.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
        if phase == "capture":
            command.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.jpeg_inputs, args.output)
    elif args.phase == "prepare-scenes":
        prepare_scenes(args.camera_inputs, args.earlier_receipt, args.output)
    elif args.phase == "capture":
        capture(args.directory, args.receipt, args.workers)
    else:
        result = parent.read(args.receipt)
        parent.require(review(result) == result["summary"], "Saved summary differs from the row-wise recount")
        print(f"Verified {len(result['observed'])} WebP measurements")


if __name__ == "__main__":
    main()
