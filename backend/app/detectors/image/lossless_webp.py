"""Bounded recognition of a single opaque lossless image in a simple WebP container."""
from __future__ import annotations


def lossless_webp_size(data: bytes) -> tuple[int, int] | None:
    if (len(data) < 26 or data[:4] != b"RIFF" or data[8:16] != b"WEBPVP8L"
            or int.from_bytes(data[4:8], "little") != len(data) - 8 or data[20] != 0x2F):
        return None
    size = int.from_bytes(data[16:20], "little")
    padding = size % 2
    if size < 6 or 20 + size + padding != len(data) or (padding and data[-1] != 0):
        return None
    fields = int.from_bytes(data[21:25], "little")
    width, height = (fields & 0x3FFF) + 1, ((fields >> 14) & 0x3FFF) + 1
    if fields >> 28 or min(width, height) < 512 or max(width, height) > 1024:
        return None
    return width, height
