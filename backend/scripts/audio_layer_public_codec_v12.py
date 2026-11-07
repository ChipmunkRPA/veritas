"""Lossless float32 transport for public benchmark vectors; standard library only."""
from __future__ import annotations

import base64
import hashlib
import math
import struct

FORMAT = "chipmunk.audio-layer-f32le/12"
REFERENCE = "chipmunk.audio-original-vector-reference/12"


def binary(value, shape):
    if shape == [1024]:
        flat = value
    elif shape == [3, 1024] and isinstance(value, list) and len(value) == 3:
        if any(not isinstance(row, list) or len(row) != 1024 for row in value):
            raise ValueError("Three complete original layer vectors required")
        flat = [v for row in value for v in row]
    else:
        raise ValueError("Unexpected vector shape")
    size = math.prod(shape)
    if (not isinstance(flat, list) or len(flat) != size
            or any(type(v) is not float or not math.isfinite(v) for v in flat)):
        raise ValueError("Complete finite float32 values required")
    raw = struct.pack(f"<{size}f", *flat)
    if list(struct.unpack(f"<{size}f", raw)) != flat:
        raise ValueError("Float32 transport must not round original values")
    return raw


def pack(value, previous=None, *, key=None, source_id=None):
    if key in ("embedding", "layer_embeddings") and value is not None:
        shape = [1024] if key == "embedding" else [3, 1024]
        raw = binary(value, shape)
        digest = hashlib.sha256(raw).hexdigest()
        if key == "embedding" and previous is not None:
            if source_id not in previous or binary(previous[source_id]["embedding"], shape) != raw:
                raise ValueError("Original pooled vector differs from its public reference")
            return {"format": REFERENCE, "id": source_id, "sha256": digest}
        return {"format": FORMAT, "shape": shape, "sha256": digest,
                "data": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, dict):
        return {k: pack(v, previous, key=k, source_id=value.get("id", source_id)) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v, previous) for v in value]
    return value


def unpack(value, previous=None):
    if isinstance(value, dict) and value.get("format") == REFERENCE:
        if set(value) != {"format", "id", "sha256"} or previous is None or value["id"] not in previous:
            raise ValueError("Exact original vector reference required")
        vector = previous[value["id"]]["embedding"]
        if hashlib.sha256(binary(vector, [1024])).hexdigest() != value["sha256"]:
            raise ValueError("Original vector reference bytes changed")
        return list(vector)
    if isinstance(value, dict) and value.get("format") == FORMAT:
        if set(value) != {"format", "shape", "sha256", "data"} or value["shape"] not in ([1024], [3, 1024]):
            raise ValueError("Exact bounded float32 payload required")
        shape = value["shape"]
        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) != math.prod(shape) * 4 or hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("Packed vector payload changed")
        flat = list(struct.unpack(f"<{math.prod(shape)}f", raw))
        if not all(math.isfinite(v) for v in flat):
            raise ValueError("Packed vector contains non-finite values")
        return flat if shape == [1024] else [flat[i:i + 1024] for i in range(0, len(flat), 1024)]
    if isinstance(value, dict):
        return {k: unpack(v, previous) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v, previous) for v in value]
    return value
