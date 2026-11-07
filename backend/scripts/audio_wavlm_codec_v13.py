"""Bounded, lossless float32 transport; model parameters remain float64 JSON."""
from __future__ import annotations

import base64
import hashlib
import math
import struct

FORMAT = "chipmunk.wavlm-f32le/13"
WIDTH = 4608
BYTES = WIDTH * 4
ENCODED = (BYTES + 2) // 3 * 4


def pack(value, *, key=None):
    if key == "wavlm" and value is not None:
        if (not isinstance(value, list) or len(value) != WIDTH
                or any(type(v) is not float or not math.isfinite(v) for v in value)):
            raise ValueError("Complete finite original4608 float32 values required")
        raw = struct.pack(f"<{WIDTH}f", *value)
        if list(struct.unpack(f"<{WIDTH}f", raw)) != value:
            raise ValueError("Transport cannot round original features")
        return {"format": FORMAT, "shape": [WIDTH], "sha256": hashlib.sha256(raw).hexdigest(),
                "data": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, dict):
        return {k: pack(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, dict) and value.get("format") == FORMAT:
        if (set(value) != {"format", "shape", "sha256", "data"} or value["shape"] != [WIDTH]
                or type(value["data"]) is not str or len(value["data"]) != ENCODED):
            raise ValueError("Exact bounded feature payload required")
        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) != BYTES or hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("Packed feature bytes changed")
        vector = list(struct.unpack(f"<{WIDTH}f", raw))
        if any(not math.isfinite(v) for v in vector):
            raise ValueError("Nonfinite feature payload")
        return vector
    if isinstance(value, dict):
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value
