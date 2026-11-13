# SPDX-License-Identifier: Apache-2.0
"""Extract one unambiguous RGB thumbnail from a JPEG image-resource segment."""

from __future__ import annotations

import io
import struct

from PIL import Image

SIGNATURE = b"Photoshop 3.0\0"
RESOURCE_ID = 1036
PREVIEW_MIN_SIDE, PREVIEW_MAX_SIDE = 64, 512
MAX_HEADER_BYTES = 1 << 20


def _resources(data: bytes):
    pos = len(SIGNATURE)
    while pos < len(data):
        if pos + 7 > len(data) or data[pos:pos + 4] != b"8BIM":
            raise ValueError("Invalid image resource signature")
        resource_id = int.from_bytes(data[pos + 4:pos + 6], "big")
        name = 1 + data[pos + 6]
        offset = pos + 6 + name + name % 2
        if offset + 4 > len(data):
            raise ValueError("Truncated image resource name")
        length = int.from_bytes(data[offset:offset + 4], "big")
        start = offset + 4
        end = start + length
        pos = end + length % 2
        if pos > len(data):
            raise ValueError("Truncated image resource")
        yield resource_id, data[start:end]


def _preview(block: bytes) -> bytes:
    if len(block) < 28:
        raise ValueError("Truncated image resource thumbnail")
    mode, width, height, stride, raw_size, jpeg_size, depth, planes = struct.unpack_from(">6I2H", block)
    blob = block[28:]
    if (mode != 1 or depth != 24 or planes != 1 or not PREVIEW_MIN_SIDE <= min(width, height)
            <= max(width, height) <= PREVIEW_MAX_SIDE or stride != ((3 * width + 3) & ~3)
            or raw_size != stride * height or jpeg_size != len(blob) or not blob.endswith(b"\xff\xd9")):
        raise ValueError("Unsupported image resource thumbnail geometry")
    with Image.open(io.BytesIO(blob), formats=["JPEG"]) as image:
        if (image.mode != "RGB" or image.size != (width, height)
                or image.getexif().get(274, 1) != 1):
            raise ValueError("Image resource thumbnail header disagrees with the JPEG")
    return blob


def extract(source: Image.Image) -> bytes | None:
    """Return the preview bytes, declining duplicate or legacy competing thumbnails."""
    header_size = 0
    matches = []
    for kind, body in source.applist:
        header_size += 4 + len(body)
        if header_size > MAX_HEADER_BYTES:
            return None
        if kind == "APP13" and body.startswith(SIGNATURE):
            try:
                for resource_id, block in _resources(body):
                    if resource_id == 1033:
                        return None
                    if resource_id == RESOURCE_ID:
                        if matches:
                            return None
                        matches.append(_preview(block))
            except (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError):
                return None
    return matches[0] if matches else None
