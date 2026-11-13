# SPDX-License-Identifier: Apache-2.0
"""Read the retained Exif JPEG thumbnail of an upright static RGB PNG."""

from __future__ import annotations

import io
import struct
import zlib

from PIL import ExifTags, Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_EXIF_BYTES = 1 << 20
MAX_CHUNKS = 4096
_INCOMPATIBLE_COLOR = frozenset({b"iCCP", b"cICP", b"mDCV", b"cLLI", b"tRNS"})
_ANIMATION = frozenset({b"acTL", b"fcTL", b"fdAT"})
_SRGB_GAMMA = struct.pack(">I", 45455)
_SRGB_CHROMATICITIES = struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)


def extract(data: bytes) -> bytes | None:
    try:
        return _extract(data)
    except (OSError, SyntaxError, ValueError, TypeError, KeyError, IndexError, OverflowError, struct.error,
            Image.DecompressionBombError):
        return None


def _extract(data: bytes) -> bytes | None:
    if not data.startswith(PNG_SIGNATURE):
        return None
    chunks = memoryview(data)
    tiff = None
    started = False
    color = set()
    offset = len(PNG_SIGNATURE)
    # Read the full chunk roster to distinguish a single standard eXIf from
    # competing metadata, while skipping the compressed image bytes.
    for _ in range(MAX_CHUNKS):
        if offset + 12 > len(data):
            return None
        length, kind = struct.unpack_from(">I4s", data, offset)
        start = offset + 8
        end = start + length
        offset = end + 4
        if (offset > len(data) or kind in _ANIMATION or kind in _INCOMPATIBLE_COLOR
                or zlib.crc32(chunks[start - 4:end]) != struct.unpack_from(">I", data, end)[0]):
            return None
        if kind == b"IHDR":
            if start != 16 or length != 13 or data[start + 8:start + 10] != b"\x08\x02":
                return None
        elif kind in (b"gAMA", b"cHRM", b"sRGB"):
            if started or kind in color or length != {b"gAMA": 4, b"cHRM": 32, b"sRGB": 1}[kind]:
                return None
            value = data[start:end]
            if ((kind == b"gAMA" and value != _SRGB_GAMMA)
                    or (kind == b"cHRM" and value != _SRGB_CHROMATICITIES)
                    or (kind == b"sRGB" and value[0] > 3)):
                return None
            color.add(kind)
        elif kind == b"eXIf":
            if started or tiff is not None or not 8 <= length <= MAX_EXIF_BYTES:
                return None
            tiff = data[start:end]
            if tiff[:4] not in (b"II\x2a\0", b"MM\0\x2a"):
                return None
        elif kind == b"IDAT":
            started = True
        elif kind == b"IEND":
            if not started or tiff is None or length or offset != len(data):
                return None
            break
    else:
        return None

    exif = Image.Exif()
    exif.load(b"Exif\0\0" + tiff)
    if exif.get(274, 1) != 1:
        return None
    thumbnail = exif.get_ifd(ExifTags.IFD.IFD1)
    start, length = thumbnail.get(0x0201), thumbnail.get(0x0202)
    if (thumbnail.get(0x0103) != 6 or thumbnail.get(274, 1) != 1 or type(start) is not int or type(length) is not int
            or start < 8 or length < 4 or start + length > len(tiff)):
        return None
    blob = tiff[start:start + length]
    if not blob.startswith(b"\xff\xd8") or not blob.endswith(b"\xff\xd9"):
        return None
    with Image.open(io.BytesIO(blob), formats=["JPEG"]) as preview:
        if (preview.mode != "RGB" or preview.getexif().get(274, 1) != 1 or preview.info.get("icc_profile")
                or not 64 <= min(preview.size) <= max(preview.size) <= 512):
            return None
    return blob
