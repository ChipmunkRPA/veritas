# SPDX-License-Identifier: Apache-2.0
"""Measure localization from a linked HEIC thumbnail against exact changed-pixel masks."""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import libheif_info

from app.evidence.image_editor_preview_binding import MAX_REGIONS
from app.provenance.heif_thumbnail import read_pair
from app.provenance.image_thumbnail_localization import CONFIRMATION_COMPONENT_FRACTION, measure_heif

RECIPES = ((320, 82, 74), (192, 66, 58))
CAMERA_REVISION = "5332a9cf0220e9c6c93f88a187daa90808051e10"
CAMERA_SHA256 = {
    "IMG_5195.HEIC": "d9b9ebe21c4127080414f2b86f6ecb900dc2ca902be82b39be93b61ccb622680",
    "iphone_13_pro_max.HEIC": "e760c80eed310e4f27c092d5487693ca8e104e7cc01d25ba4828deb28f679676",
    "HMD_Nokia_8.3_5G.heif": "c2820b3e4e0368c6f031b28b62945658a97446895cb446dfc37acdcf7109e0f5",
    "HMD_Nokia_8.3_5G_hdr.heif": "17fccf795b0e2b749b9f5667ffcea31e665cb6964fb104b001684c60a14acb4d",
}
CONFIGS = (
    ("sharp", "sharp", "ordinary", True),
    ("feather", "feather", "ordinary", True),
    ("ordinary", "ordinary", "ordinary", False),
    ("refreshed_edit", "sharp", "sharp", False),
    ("refreshed_tone", "local_tone", "local_tone", False),
    ("refreshed_blur", "local_blur", "local_blur", False),
    ("refreshed_copy", "local_copy", "local_copy", False),
)


class HeifError(c.Structure):
    _fields_ = [("code", c.c_int), ("subcode", c.c_int), ("message", c.c_char_p)]


class Encoder:
    """Use libheif's public encoder to build real linked thumbnails from known pixels."""

    def __init__(self, library):
        self.lib = c.CDLL(str(library))
        ptr, integer, string = c.c_void_p, c.c_int, c.c_char_p
        out, stride = c.POINTER(ptr), c.POINTER(integer)
        self.bind("context_alloc", ptr)
        for name in ("context_free", "encoder_release", "image_release", "image_handle_release"):
            self.bind(name, None, ptr)
        self.bind("context_get_encoder_for_format", HeifError, ptr, integer, out)
        self.bind("encoder_set_lossy_quality", HeifError, ptr, integer)
        self.bind("encoder_set_parameter_string", HeifError, ptr, string, string)
        self.bind("image_create", HeifError, integer, integer, integer, integer, out)
        self.bind("image_add_plane", HeifError, ptr, integer, integer, integer, integer)
        self.bind("image_get_plane", ptr, ptr, integer, stride)
        self.bind("context_encode_image", HeifError, ptr, ptr, ptr, ptr, out)
        self.bind("context_encode_thumbnail", HeifError, ptr, ptr, ptr, ptr, ptr, integer, out)
        self.bind("context_write_to_file", HeifError, ptr, string)

    def bind(self, name, result, *params):
        function = getattr(self.lib, "heif_" + name)
        function.argtypes, function.restype = params, result
        setattr(self, name, function)

    def pixels(self, data, cleanup):
        height, width, _ = data.shape
        image = c.c_void_p()
        check(self.image_create(width, height, 1, 10, c.byref(image)))  # RGB, interleaved RGB
        cleanup.callback(self.image_release, image)
        check(self.image_add_plane(image, 10, width, height, 8))  # Interleaved channel
        stride = c.c_int()
        plane = self.image_get_plane(image, 10, c.byref(stride))
        if not plane or stride.value < 3 * width:
            raise ValueError("Invalid fixture encoder plane")
        pixels = np.ascontiguousarray(data)
        for row in range(height):
            c.memmove(plane + row * stride.value, pixels.ctypes.data + row * 3 * width, width * 3)
        return image

    def write(self, output, current, preserved, recipe):
        edge, full_quality, thumb_quality = recipe
        with ExitStack() as cleanup:
            context = self.context_alloc()
            if not context:
                raise ValueError("Fixture encoder context unavailable")
            cleanup.callback(self.context_free, context)
            encoder = c.c_void_p()
            check(self.context_get_encoder_for_format(context, 1, c.byref(encoder)))  # HEVC
            cleanup.callback(self.encoder_release, encoder)
            check(self.encoder_set_parameter_string(encoder, b"preset", b"ultrafast"))
            check(self.encoder_set_lossy_quality(encoder, full_quality))
            main = c.c_void_p()
            check(self.context_encode_image(context, self.pixels(current, cleanup), encoder, None, c.byref(main)))
            cleanup.callback(self.image_handle_release, main)
            if preserved is not None:
                check(self.encoder_set_lossy_quality(encoder, thumb_quality))
                thumbnail = c.c_void_p()
                check(self.context_encode_thumbnail(context, self.pixels(preserved, cleanup), main,
                                                    encoder, None, edge, c.byref(thumbnail)))
                if not thumbnail.value:
                    raise ValueError("Fixture encoder omitted the linked thumbnail")
                cleanup.callback(self.image_handle_release, thumbnail)
            check(self.context_write_to_file(context, os.fsencode(output)))


def check(result):
    if result.code:
        raise ValueError(result.message.decode("utf-8", errors="replace"))


def digest(body):
    return hashlib.sha256(body).hexdigest()


def pixels(path, *, mask=False):
    with Image.open(path) as image:
        return np.asarray(image.convert("L" if mask else "RGB"))


def measure(body, truth=None):
    try:
        with read_pair(body) as (status, pair):
            if pair is None:
                return {"status": status, "regions": []}
            reading, _ = measure_heif(*pair, minimum_component_fraction=CONFIRMATION_COMPONENT_FRACTION)
    except (ValueError, EOFError, OSError, RuntimeError, SyntaxError):
        return {"status": "unreadable_carrier", "regions": []}
    boxes = [region["submitted"] for region in reading.get("regions", [])[:MAX_REGIONS]]
    if truth is None:
        return {"status": reading["status"], "regions": boxes}
    selected = np.zeros_like(truth)
    for box in boxes:
        x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
        selected[y:y+h, x:x+w] = True
    return {"status": reading["status"], "regions": boxes, "truth": int(truth.sum()),
            "selected": int(selected.sum()), "intersection": int((selected & truth).sum())}


def counts(rows):
    summary = {}
    for row in rows:
        reading = row["reading"]
        tally = summary.setdefault((row["family"], row["edge"]), Counter())
        if reading["status"] != "comparable":
            tally["unassessed"] += 1
        if row["mismatch"]:
            tally["mismatched"] += 1
            shared, truth, selected = (reading.get(key, 0) for key in ("intersection", "truth", "selected"))
            if truth and shared * 2 >= truth and shared * 10 >= (truth + selected - shared) * 3:
                tally["located"] += 1
            elif reading["regions"]:
                tally["misplaced"] += 1
        else:
            tally["controls"] += 1
            tally["control_flags"] += bool(reading["regions"])
    return {f"{family}-{edge}": dict(tally) for (family, edge), tally in sorted(summary.items())}


def evaluate(library, roots, provider, output, *, cameras=None):
    cv2.setNumThreads(1)
    encoder = Encoder(library)
    output.mkdir(mode=0o700, exist_ok=True)
    media = output / "deliveries"
    media.mkdir(mode=0o700, exist_ok=True)
    rows, inputs = [], []
    for label, root in roots:
        manifest = (root / "manifest.json").read_bytes()
        inputs.append({"label": label, "sha256": digest(manifest)})
        plan = json.loads(manifest)
        deliveries = {(row["group"], row["role"]): row for row in plan["images"] if row["quality"] == 90}
        for group in plan["groups"]:
            tag = group["tag"]
            family = group["family"]
            variants = {name: deliveries[tag, name] for name in {item for _, main, thumb, _ in CONFIGS for item in (main, thumb)}}
            for row in variants.values():
                if (digest((root / row["file"]).read_bytes()) != row["sha256"]
                        or digest((root / row["mask"]).read_bytes()) != row["mask_sha256"]):
                    raise ValueError("A source delivery or ground-truth mask differs from the frozen manifest")
            views = {name: pixels(root / row["file"]) for name, row in variants.items()}
            for recipe in RECIPES:
                for condition, main, preview, mismatch in CONFIGS:
                    file = media / f"{label}-{tag}-{condition}-{recipe[0]}.heic"
                    encoder.write(file, views[main], views[preview], recipe)
                    body = file.read_bytes()
                    truth = pixels(root / variants[main]["mask"], mask=True) > 0 if mismatch else None
                    rows.append({"input": label, "source": group["source"], "family": family, "condition": condition,
                                 "edge": recipe[0], "file": file.name, "sha256": digest(body), "mismatch": mismatch, "shape": group["shape"],
                                 "truth_sha256": variants[main]["mask_sha256"] if mismatch else None, "reading": measure(body, truth)})
            print(f"{label} {tag} · {len(rows)} deliveries", flush=True)
    external = []
    for line in (provider / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if not name.endswith((".heic", ".heics", ".heif", ".hif", ".avif", ".jpg")):
            continue
        body = (provider / name).read_bytes()
        if digest(body) != expected:
            raise ValueError("Publisher test file differs from its commitment")
        external.append({"file": name, "sha256": expected, **measure(body)})
    controls = []
    if cameras:
        for name, expected in sorted(CAMERA_SHA256.items()):
            source = cameras / name
            body = source.read_bytes()
            if digest(body) != expected:
                raise ValueError("Camera control differs from its pinned download")
            with read_pair(body) as (_, pair):
                if pair:
                    with Image.open(source) as displayed:
                        if displayed.getexif().get(274, 1) != 1 or not np.array_equal(np.asarray(displayed), np.asarray(pair[0])):
                            raise ValueError("The browser evidence axes differ from the serving decoder")
            controls.append({"file": name, "sha256": expected, **measure(body)})
    receipt = {"schema": "chipmunk.heif-thumbnail-validation/1", "codec": libheif_info(),
               "recipes": RECIPES, "inputs": inputs, "rows": rows, "external": external, "camera_revision": CAMERA_REVISION,
               "cameras": controls, "counts": counts(rows)}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(receipt["counts"], indent=2), flush=True)


if __name__ == "__main__":
    from pillow_heif import register_heif_opener

    register_heif_opener()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("provider", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--input", nargs=2, action="append", required=True, metavar=("LABEL", "DELIVERIES"))
    args = parser.parse_args()
    evaluate(args.library, [(name, Path(path)) for name, path in args.input], args.provider, args.output, cameras=args.cameras)
