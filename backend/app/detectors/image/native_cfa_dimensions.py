# SPDX-License-Identifier: Apache-2.0
"""Native interpolation inspection for independently checked PNG dimensions."""
from __future__ import annotations

import hashlib
import struct
import zlib

import cv2
import numpy as np

from app.detectors.image import native_cfa

# Width, height as stored in the submitted PNG.
ADDITIONAL_SIZES = frozenset({(2304, 1728), (2368, 1344), (1792, 1792)})
LEGACY_SIZE = native_cfa.SCOPE[::-1]
SIZES = ADDITIONAL_SIZES | {LEGACY_SIZE}


def png_size(body: bytes | memoryview) -> tuple[int, int] | None:
    if len(body) < 33 or body[:16] != native_cfa.PNG + b"\x00\x00\x00\x0dIHDR":
        return None
    return struct.unpack_from(">II", body, 16)


def _additional_png(body: bytes | memoryview, size: tuple[int, int]) -> bool:
    if size not in ADDITIONAL_SIZES or len(body) > native_cfa.MAX_BODY_BYTES or body[24:29] != native_cfa.PROFILE:
        return False
    view = memoryview(body)
    offset = 8
    data = False
    for index in range(native_cfa.MAX_CHUNKS):
        if offset + 12 > len(body):
            return False
        length, kind = struct.unpack_from(">I4s", body, offset)
        end = offset + 8 + length
        if end + 4 > len(body) or zlib.crc32(view[offset+4:end]) != struct.unpack_from(">I", body, end)[0]:
            return False
        if kind == b"IDAT":
            data = True
        elif kind == b"IEND":
            return data and length == 0 and end + 4 == len(body)
        elif kind != b"IHDR" or index != 0:
            return False
        offset = end + 4
    return False


def decoded_rgb(body: bytes | memoryview) -> np.ndarray | None:
    size = png_size(body)
    if size == LEGACY_SIZE:
        return native_cfa.decoded_rgb(body)
    if size is None or not _additional_png(body, size):
        return None
    try:
        pixels = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None
    if pixels is None or pixels.dtype != np.uint16 or pixels.shape != (*size[::-1], 3):
        return None
    return np.ascontiguousarray(pixels[..., ::-1])


def measure(rgb: np.ndarray) -> native_cfa.InterpolationReview:
    if rgb.dtype != np.uint16 or rgb.shape[2:] != (3,) or rgb.shape[1::-1] not in SIZES:
        return native_cfa.InterpolationReview("unsupported_png")
    if rgb.shape[:2] == native_cfa.SCOPE:
        return native_cfa.measure(rgb)
    rates = native_cfa.stencil(rgb)
    reference = native_cfa.learn(rates)
    if reference is None:
        return native_cfa.InterpolationReview("no_repeatable_stencil")
    phase, closure, eligible = reference
    strongest_other = np.max(rates[..., [p for p in range(4) if p != phase]], axis=-1)
    proposal = native_cfa.best_region(strongest_other - rates[..., phase] - .20,
                                      eligible & (strongest_other >= .55))
    box = tuple(proposal["box"]) if proposal["box"] else None
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(rgb))).hexdigest() if box and proposal["score"] > native_cfa.POINT else None
    return native_cfa.InterpolationReview("assessed", phase, proposal["score"], box, int(eligible.sum()), closure, digest)


def inspect_png(body: bytes | memoryview) -> native_cfa.InterpolationReview:
    rgb = decoded_rgb(body)
    if rgb is None:
        return native_cfa.InterpolationReview("unsupported_png")
    return measure(rgb)
