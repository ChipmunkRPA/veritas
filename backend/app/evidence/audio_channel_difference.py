"""Native PCM projection used to compare audio shared differently across channels."""
from __future__ import annotations

import sys
from array import array

PROJECTION = "floor((left_s16 - right_s16) / 2)"


def project(stereo_s16le: bytes) -> bytes:
    if len(stereo_s16le) % 4:
        raise ValueError("Expected complete stereo PCM sample frames")
    source = array("h")
    source.frombytes(stereo_s16le)
    if sys.byteorder == "big":
        source.byteswap()
    projected = array("h", ((left - right) // 2 for left, right in zip(source[::2], source[1::2], strict=True)))
    if sys.byteorder == "big":
        projected.byteswap()
    return projected.tobytes()
