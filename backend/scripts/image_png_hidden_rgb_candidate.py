# SPDX-License-Identifier: Apache-2.0
"""Measure fully transparent PNG areas with retained RGB variation, without revealing pixels."""

from __future__ import annotations

import hashlib
import io
import struct
import zlib

import cv2
import numpy as np
from PIL import Image

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_SIDE, MAX_SIDE = 64, 5000
MAX_BYTES, MAX_PIXELS = 12_000_000, 17_000_000
TILE, STEP, CLEAR_MARGIN = 16, 8, 5
MIN_DETAIL_PAIRS, MIN_TILES, MAX_REGIONS = 9, 6, 4
DETAIL_STEP = 7
FORBIDDEN = {b"acTL", b"fcTL", b"fdAT", b"tRNS", b"eXIf", b"iTXt", b"zTXt", b"cICP", b"mDCV", b"cLLI"}


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def valid_raster(parts, row_bytes, height):
    """Validate the decoded scan extent and row filters in bounded chunks."""
    decoder = zlib.decompressobj()
    received, expected = 0, row_bytes*height
    try:
        for part in parts:
            while part:
                data = decoder.decompress(part, min(262_144, expected - received + 1))
                part = decoder.unconsumed_tail
                if decoder.unused_data or any(value > 4 for value in data[(-received) % row_bytes::row_bytes]):
                    return False
                received += len(data)
                if received > expected:
                    return False
        return decoder.eof and received == expected
    except zlib.error:
        return False


def native_rgba(body):
    """Admit bounded, complete, unambiguous noninterlaced 8-bit truecolor-alpha PNGs."""
    if len(body) > MAX_BYTES or not body.startswith(SIGNATURE):
        return None
    view, offset, state = memoryview(body), len(SIGNATURE), "before_idat"
    parts, palette = [], False
    for index in range(4096):
        if offset + 12 > len(body):
            return None
        length, kind = struct.unpack_from(">I4s", body, offset)
        start, end = offset + 8, offset + 8 + length
        offset = end + 4
        if (offset > len(body) or kind in FORBIDDEN or not kind.isalpha() or not kind[2:3].isupper()
                or zlib.crc32(view[start - 4:end]) != struct.unpack_from(">I", body, end)[0]):
            return None
        if index == 0:
            if kind != b"IHDR" or length != 13 or body[start + 8:end] != b"\x08\x06\x00\x00\x00":
                return None
            width, height = struct.unpack_from(">II", body, start)
            if not MIN_SIDE <= min(width, height) <= max(width, height) <= MAX_SIDE or width*height > MAX_PIXELS:
                return None
        elif kind == b"IHDR":
            return None
        elif kind == b"IDAT":
            if state == "after_idat":
                return None
            parts.append(view[start:end])
            state = "inside_idat"
        elif kind == b"IEND":
            if state == "before_idat" or length or offset != len(body):
                return None
            break
        elif kind[0] & 32 == 0:
            if kind != b"PLTE" or state != "before_idat" or palette or not 0 < length <= 768 or length % 3:
                return None
            palette = True
        elif state == "inside_idat":
            state = "after_idat"
    else:
        return None
    if not valid_raster(parts, width*4 + 1, height):
        return None
    try:
        with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
            image.load()
            if image.mode != "RGBA" or image.size != (width, height) or image.n_frames != 1:
                return None
            return np.asarray(image).copy()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return None


def _counts(plane, y, x, height, width):
    accumulated = cv2.integral(plane, sdepth=cv2.CV_32S)
    return (accumulated[y+height, x+width] + accumulated[y, x]
            - accumulated[y+height, x] - accumulated[y, x+width])


def detailed_tiles(rgba):
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    # Avoid color values at a visible edge where ordinary interpolation commonly extends edge colors.
    zero = (alpha == 0).astype(np.uint8)
    protected = cv2.erode(zero, np.ones((2*CLEAR_MARGIN + 1, 2*CLEAR_MARGIN + 1), np.uint8), borderValue=1)
    height, width = zero.shape
    ys, xs = np.meshgrid(np.arange(0, height-TILE+1, STEP), np.arange(0, width-TILE+1, STEP), indexing="ij")
    fully_hidden = _counts(protected, ys, xs, TILE, TILE) == TILE*TILE
    horizontal, vertical = np.zeros_like(zero), np.zeros_like(zero)
    for channel in range(3):
        plane = rgb[..., channel].astype(np.int16)
        horizontal[:, :-1] |= (np.abs(np.diff(plane, axis=1)) >= DETAIL_STEP).astype(np.uint8)
        vertical[:-1, :] |= (np.abs(np.diff(plane, axis=0)) >= DETAIL_STEP).astype(np.uint8)
    detail_x = _counts(horizontal, ys, xs, TILE, TILE - 1)
    detail_y = _counts(vertical, ys, xs, TILE - 1, TILE)
    return fully_hidden & (detail_x >= MIN_DETAIL_PAIRS) & (detail_y >= MIN_DETAIL_PAIRS)


def inspect(body):
    """Return neutral coordinates, counts, and binary-mask commitment only."""
    rgba = native_rgba(body)
    if rgba is None:
        return {"status": "unsupported_profile", "regions": []}, None
    alpha = rgba[..., 3]
    fully_transparent = int(np.count_nonzero(alpha == 0))
    result = {"status": "measured", "dimensions": [alpha.shape[1], alpha.shape[0]],
              "fully_transparent_pixels": fully_transparent, "regions": []}
    if not fully_transparent:
        return result, np.zeros(alpha.shape, bool)
    allowed = detailed_tiles(rgba)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(allowed.astype(np.uint8), connectivity=8)
    candidates = sorted((index for index, entry in enumerate(stats) if index and entry[-1] >= MIN_TILES),
                        key=lambda index: (-stats[index, -1], index))
    native = np.zeros(alpha.shape, bool)
    for index in candidates[:MAX_REGIONS]:
        x, y, w, h, supported = (int(value) for value in stats[index])
        coords = np.argwhere(labels == index)
        left, top, width, height = x*STEP, y*STEP, (w - 1)*STEP + TILE, (h - 1)*STEP + TILE
        footprint = np.zeros((height, width), bool)
        for row, col in coords:
            dy, dx = (int(row) - y)*STEP, (int(col) - x)*STEP
            footprint[dy:dy+TILE, dx:dx+TILE] = True
        native[top:top+height, left:left+width] |= footprint
        result["regions"].append({"x": left, "y": top, "w": width, "h": height,
                                  "selected_pixels": int(footprint.sum()), "supporting_tiles": supported})
    if result["regions"]:
        if np.any(alpha[native] != 0):
            raise ValueError("Internal hidden RGB selection included visible pixels")
        result["selected_pixels"] = int(native.sum())
        result["mask_sha256"] = hash_bytes(native.tobytes())
        result["additional_components"] = len(candidates) - len(result["regions"])
    return result, native
