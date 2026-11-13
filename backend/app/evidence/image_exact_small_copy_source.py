# SPDX-License-Identifier: Apache-2.0
"""Validate an exact small copy against the submitted static RGB PNG."""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
from PIL import Image

from app.detectors.image import exact_small_copy
from app.evidence.image_copy_similarity_binding import ROLES, SMALL_PNG_MAX_SIDE, SMALL_PNG_MIN_SIDE

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_CHUNKS = 4096


def native_rgb(data):
    if len(data) > MAX_PNG_BYTES or not data.startswith(PNG_SIGNATURE):
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


def review(data):
    pixels = native_rgb(data)
    if pixels is None:
        return {"status": "unsupported_profile", "measurement": None}, []
    item = exact_small_copy.analyse(pixels)
    if not item["pairs"]:
        return {"status": item["status"], "measurement": None}, []
    match, = item["pairs"]
    return {"status": item["status"], "measurement": match["measurement"]}, [match["pair"]]


def verify(data, pair, measurement):
    pixels = native_rgb(data)
    if pixels is None:
        raise ValueError("Exact copy comparison requires an upright static RGB PNG within its tested size.")
    boxes = [tuple(pair["regions"][role][key] for key in "xywh") for role in ROLES]
    if exact_small_copy.witness(pixels, *boxes) != measurement:
        raise ValueError("Exact copy measurement differs from the submitted RGB pixels or surroundings.")
