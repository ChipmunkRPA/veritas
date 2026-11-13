# SPDX-License-Identifier: Apache-2.0
"""Read back saved JPEG editor-thumbnail evidence against the original source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.provenance import image_thumbnail_localization as localization

BOX = ("x", "y", "w", "h")
MAX_BYTES = 60 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounds(box, size):
    x, y, width, height = (box[k] for k in BOX)
    require(all(type(n) is int for n in (x, y, width, height)) and width > 0 and height > 0
            and 0 <= x < x + width <= size[0] and 0 <= y < y + height <= size[1], "Evidence box is outside its source")
    return x, y, x + width, y + height


def _match_asset(assets, record, image):
    pin = record["sha256"]
    require(len(pin) == 64 and all(c in "0123456789abcdef" for c in pin), "Crop does not carry a SHA-256 commitment")
    raw = assets(pin)
    require(sha(raw) == pin, "Saved PNG differs from its commitment")
    with Image.open(io.BytesIO(raw), formats=["PNG"]) as observed:
        require(observed.mode == "RGB" and observed.size == (record["display_width"], record["display_height"]) == image.size
                and observed.tobytes() == image.tobytes(), "Saved PNG differs from the source pixels")


def verify(body, document, assets):
    require(sha(body) == document.get("source_sha256"), "Submitted source differs from the export")
    require(document.get("preview_carrier") == "jpeg_editor_resource"
            and document.get("main_exif_orientation") == 1 and document.get("preview_alignment_orientation") == 0
            and document.get("authorship_confidence_contribution") == 0, "Unsupported preview carrier, orientation or confidence")
    measurement, masks = localization.measure(body, minimum_component_fraction=localization.CONFIRMATION_COMPONENT_FRACTION)
    rows = document["regions"]
    require(masks is not None and measurement.get("preview_carrier") == "jpeg_editor_resource"
            and len(rows) == min(3, len(measurement["regions"])) and bool(rows), "Source comparison differs from the saved regions")
    checked = {"regions": len(rows), "source_crops": 0, "footprints": 0}
    with Image.open(io.BytesIO(body), formats=["JPEG"]) as source:
        _, blob = localization.retained_preview(source, source.getexif())
        with Image.open(io.BytesIO(blob), formats=["JPEG"]) as preview:
            for index, (saved, region) in enumerate(zip(rows, measurement["regions"]), 1):
                require(saved["region"] == index and saved["main_image"] == region["submitted"]
                        and saved["marked_preview_pixels"] == region["selected_thumbnail_pixels"],
                        "Saved main-image region differs from the measurement")
                x, y, width, height = (region["embedded_thumbnail"][k] for k in BOX)
                left, top, right, bottom = measurement["trim"]
                shifts = [support["shift_and_response"] for support in measurement["support"]]
                x0 = max(left, int(np.floor(x - max(dx for dx, _, _ in shifts))))
                y0 = max(top, int(np.floor(y - max(dy for _, dy, _ in shifts))))
                x1 = min(right, int(np.ceil(x + width - 1 - min(dx for dx, _, _ in shifts))) + 1)
                y1 = min(bottom, int(np.ceil(y + height - 1 - min(dy for _, dy, _ in shifts))) + 1)
                require(bounds(saved["embedded_preview"], preview.size) == (x0, y0, x1, y1),
                        "Saved thumbnail region differs from the measured input pixels")
                for key, original in (("main_image", source), ("embedded_preview", preview)):
                    with original.crop(bounds(saved[key], original.size)).convert("RGB") as crop:
                        reduced = max(crop.size) > 384
                        if reduced:
                            crop.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        require(saved["crops"][key]["display_reduced"] == reduced, "Display crop reduction differs")
                        _match_asset(assets, saved["crops"][key], crop)
                        checked["source_crops"] += 1
                if grid := saved.get("footprint_grid"):
                    require(grid == dict(zip(BOX, (x - left, y - top, width, height), strict=True))
                            | {"coordinate_space": "aligned-preview-grid"}, "Saved footprint position differs")
                    mask = masks[0].astype(np.uint8)
                    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    candidates = np.flatnonzero(np.all(stats[1:] == [x-left, y-top, width, height, saved["marked_preview_pixels"]], axis=1)) + 1
                    require(len(candidates) == 1, "Saved footprint does not identify one source component")
                    pixels = (labels[y-top:y-top+height, x-left:x-left+width] == candidates[0]).astype(np.uint8) * 255
                    with Image.fromarray(pixels).convert("RGB") as plate:
                        _match_asset(assets, saved["crops"]["measured_footprint"], plate)
                    checked["footprints"] += 1
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("assets", type=Path, help="Directory containing each crop as <sha256>.png")
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        body = stream.read(MAX_BYTES + 1)
    require(len(body) <= MAX_BYTES, "Source is larger than the verification limit")
    value = json.loads(args.export.read_bytes())
    print(json.dumps(verify(body, value.get("thumbnail_evidence", value),
                            lambda pin: (args.assets / f"{pin}.png").read_bytes()), sort_keys=True))
