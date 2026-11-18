# SPDX-License-Identifier: Apache-2.0
"""Validate an exact small copy against the submitted static RGB still."""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
from PIL import Image

from app.detectors.image import exact_small_copy
from app.detectors.image.lossless_webp import lossless_webp_size
from app.evidence.image_copy_similarity_binding import (
    PNG_COPY_TURNS,
    ROLES,
    SMALL_PNG_MAX_SIDE,
    SMALL_PNG_MIN_SIDE,
    exact_copy_size,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ENCODED_BYTES = 4 * 1024 * 1024
MAX_CHUNKS = 4096


def native_rgb(data):
    if len(data) > MAX_ENCODED_BYTES or not data.startswith(PNG_SIGNATURE):
        return None
    chunks = memoryview(data)
    offset = len(PNG_SIGNATURE)
    saw_idat = False
    for _ in range(MAX_CHUNKS):
        if offset + 12 > len(data):
            return None
        length, kind = struct.unpack_from(">I4s", data, offset)
        start = offset + 8
        end = start + length
        offset = end + 4
        if (offset > len(data) or kind in (b"acTL", b"fcTL", b"fdAT", b"tRNS")
                or zlib.crc32(chunks[start - 4:end]) != struct.unpack_from(">I", data, end)[0]):
            return None
        if start == 16:
            if kind != b"IHDR" or length != 13 or data[start + 8:start + 10] != b"\x08\x02":
                return None
            width, height = struct.unpack_from(">II", data, start)
            if not SMALL_PNG_MIN_SIDE <= min(width, height) <= max(width, height) <= SMALL_PNG_MAX_SIDE:
                return None
        elif kind == b"IHDR":
            return None
        elif kind == b"IDAT":
            saw_idat = True
        elif kind == b"IEND":
            if not saw_idat or length or offset != len(data):
                return None
            break
    else:
        return None
    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        image.load()
        if image.mode != "RGB" or image.getexif().get(274, 1) != 1:
            return None
        return np.array(image)


def webp_rgb(data):
    if len(data) > MAX_ENCODED_BYTES or (size := lossless_webp_size(data)) is None or not exact_copy_size(*size, webp=True):
        return None
    with Image.open(io.BytesIO(data), formats=["WEBP"]) as image:
        if image.size != size or image.mode != "RGB" or image.n_frames != 1 or image.getexif().get(274, 1) != 1:
            return None
        return np.array(image)


def review(data, *, include_rotation=False, webp=False):
    pixels = webp_rgb(data) if webp else native_rgb(data)
    if pixels is None:
        return {"status": "unsupported_profile", "measurement": None}, [], 0
    item = exact_small_copy.analyse(pixels)
    if include_rotation and item["status"] == "no_pair":
        from app.detectors.image import exact_turned_copy

        item = exact_turned_copy.analyse(pixels)
    if not item["pairs"]:
        return {"status": item["status"], "measurement": None}, [], 0
    match, = item["pairs"]
    return {"status": item["status"], "measurement": match["measurement"]}, [match["pair"]], match.get("turns", 0)


def review_reflection(data, *, webp=True):
    from app.detectors.image import exact_reflected_copy

    pixels = webp_rgb(data) if webp else native_rgb(data)
    if pixels is None:
        return {"status": "unsupported_profile", "measurement": None}, [], None
    item = exact_reflected_copy.analyse(pixels)
    if not item["pairs"]:
        return {"status": item["status"], "measurement": None}, [], None
    match, = item["pairs"]
    return {"status": item["status"], "measurement": match["measurement"]}, [match["pair"]], match["transform"]


def review_multiple(data, *, webp=False):
    from app.detectors.image import exact_multi_copy

    pixels = webp_rgb(data) if webp else native_rgb(data)
    if pixels is None:
        return {"status": "unsupported_profile", "measurements": []}, []
    result = exact_multi_copy.analyse(pixels)
    pairs = result["pairs"]
    return {"status": result["status"], "measurements": [pair["measurement"] for pair in pairs]}, pairs


def verify(data, pair, measurement, *, webp=False, reflected=False):
    _verify(_verified_pixels(data, webp=webp), pair, measurement, reflected=reflected)


def _verified_pixels(data, *, webp):
    pixels = webp_rgb(data) if webp else native_rgb(data)
    if pixels is None:
        carrier = "lossless WebP" if webp else "PNG"
        raise ValueError(f"Exact copy comparison requires an upright static RGB {carrier} within its tested size.")
    return pixels


def _verify(pixels, pair, measurement, *, reflected):
    boxes = [tuple(pair["regions"][role][key] for key in "xywh") for role in ROLES]
    transform = pair.get("transform")
    if reflected:
        from app.detectors.image.exact_reflected_copy import AXES

        axis = AXES.get(transform)
        recorded = exact_small_copy.witness(pixels, *boxes, reflection_axis=axis) if axis is not None else None
    else:
        turns = PNG_COPY_TURNS.get(transform)
        recorded = exact_small_copy.witness(pixels, *boxes, turns=turns) if turns is not None else None
    if recorded is None or recorded != measurement:
        raise ValueError("Exact copy measurement differs from the submitted RGB pixels or surroundings.")


def verify_multiple(data, translation, *, webp):
    from app.evidence.image_copy_similarity_binding import MULTI_EXACT_KEY, REFLECTION_TRANSFORMS

    pixels = _verified_pixels(data, webp=webp)
    for pair, measurement in zip(translation["pairs"], translation[MULTI_EXACT_KEY]["measurements"], strict=True):
        _verify(pixels, pair, measurement, reflected=pair["transform"] in REFLECTION_TRANSFORMS)
