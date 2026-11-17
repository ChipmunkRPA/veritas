# SPDX-License-Identifier: Apache-2.0
"""Measure and replay an earlier JPEG grid from the uploaded bytes."""

from __future__ import annotations

import io

import cv2
from PIL import Image, ImageDraw, ImageOps

from app.core.canonical import canonical_hash
from app.detectors.image import jpeg_bitstream, jpeg_previous_grid
from app.evidence import image_previous_grid_binding as binding

_REVIEW_POINTS = {85: 0.34823856868844094, 90: 0.4593661360933643}


def _geometry(body, schema):
    jpeg = jpeg_bitstream.read(body)
    qualities = _REVIEW_POINTS if schema == binding.SCHEMA else (90,)
    if (jpeg is None or jpeg.precision != 8 or jpeg.ijg_quality not in qualities
            or min(jpeg.width, jpeg.height) < 320 or jpeg.width * jpeg.height > 1_048_576
            or [(c.id, c.h, c.v) for c in jpeg.components] != [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
            or any(jpeg.table_bits.get(c.table_id) != 8 for c in jpeg.components)
            or jpeg.table(1) != jpeg.table(2)):
        return None
    if jpeg.frame != "baseline" and (schema == binding.LEGACY_SCHEMA or jpeg.frame != "progressive"
                                     or jpeg_bitstream.progressive_frame(body) is None):
        return None
    with Image.open(io.BytesIO(body)) as image:
        orientation = image.getexif().get(274, 1)
        orientations = (1,) if schema == binding.LEGACY_SCHEMA else range(1, 9)
        if image.mode != "RGB" or type(orientation) is not int or orientation not in orientations:
            return None
        return jpeg.width, jpeg.height, orientation, jpeg.ijg_quality


def record(source_sha256, body=None, *, stopped=False, prior=False, check=None, schema=binding.SCHEMA):
    status, measurement, bounds = "unavailable", None, None
    if stopped:
        status = "provenance"
    elif body is not None:
        try:
            if (geometry := _geometry(body, schema)) is None:
                status = "outside_profile"
            elif prior:
                status = "earlier_region"
            elif (measured := jpeg_previous_grid.measure(body, check=check)) is None:
                status = "no_prior_grid"
            else:
                status = "no_region"
                measurement = {"grid_offset": list(measured.phase), "estimated_earlier_quality": measured.previous_quality,
                               "background_support": [round(measured.background, 6), round(measured.separated_from_next, 6)]}
                width, height, orientation, quality = geometry
                if schema == binding.SCHEMA:
                    measurement["delivery_quality"] = quality
                region = measured.regions["surrounded_scan"]
                if region is not None and region.score > _REVIEW_POINTS[quality]:
                    status = "matched"
                    x, y, w, h = region.box
                    bounds = binding.orient_region([x, y, x + w, y + h], orientation, width, height)
                    measurement.update(inside_support=[round(n, 6) for n in region.inside],
                                       surrounding_support=[round(n, 6) for n in region.surrounding],
                                       supported_cells=list(region.cells))
        except (OSError, ValueError, cv2.error, MemoryError):
            status, measurement, bounds = "unavailable", None, None
    return {"schema": schema, "source_sha256": source_sha256, "affects_overall_confidence": False,
            "status": status, "observation": binding.STATUS_TEXT[status], "scope": binding.SCOPES[schema], "limitation": binding.LIMITATION,
            "basis": binding.BASIS if status == "matched" else None, "measurement": measurement, "region": bounds}


def plates(original, bounds):
    crop = original.crop(tuple(bounds))
    context = original.copy()
    x, y, right, bottom = bounds
    outline = (x, y, right - 1, bottom - 1)
    draw = ImageDraw.Draw(context)
    draw.rectangle(outline, outline="#243832", width=4)
    draw.rectangle((x + 1, y + 1, right - 2, bottom - 2), outline="#ffbf47", width=2)
    return {"context": context, "crop": crop}


def verify(body, review, read_asset):
    frozen = review["record"]
    status = frozen["status"]
    if status in {"provenance", "unavailable"}:
        return False
    measured = record(frozen["source_sha256"], body, prior=status == "earlier_region", schema=frozen["schema"])
    if canonical_hash(measured) != canonical_hash(frozen):
        raise ValueError("The earlier JPEG grid did not replay from the uploaded image")
    if region := frozen["region"]:
        with Image.open(io.BytesIO(body)) as source:
            expected = plates(ImageOps.exif_transpose(source).convert("RGB"), region)
        for name, picture in expected.items():
            with Image.open(io.BytesIO(read_asset(review["artifacts"][name]["sha256"]))) as actual:
                if actual.mode != picture.mode or actual.size != picture.size or actual.tobytes() != picture.tobytes():
                    raise ValueError("A JPEG evidence picture differs from the original pixels")
    return True
