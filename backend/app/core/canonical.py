"""Canonical JSON (ARCHITECTURE §4.8): the byte form behind `result_hash`, `snapshot_hash` and audit `event_hash`.

Rules:
- keys sorted, no whitespace, UTF-8 (not ASCII-escaped);
- floats rounded to 6 decimal places, then written with Python's shortest `repr`; `-0.0` becomes `0.0`; NaN and infinity
  are refused;
- timestamps as ISO-8601 UTC with microseconds (`2026-09-12T08:20:03.000000Z`); naive datetimes are refused because their
  zone is ambiguous;
- enums serialise as their values, pydantic models through `model_dump(mode="python")`, tuples as arrays;
- anything without one canonical form is refused: sets (no order), bytes, non-string keys, arbitrary objects.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

FLOAT_DP = 6


class CanonicalError(TypeError):
    """The value has no canonical JSON form."""


def _normalise(value: Any, path: str) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Enum):
        return _normalise(value.value, path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError(f"{path}: NaN and infinity have no canonical form")
        rounded = round(value, FLOAT_DP)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalError(f"{path}: naive datetime (the zone is ambiguous)")
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError(f"{path}: non-string key {key!r}")
            out[key] = _normalise(item, f"{path}.{key}")
        return out
    if isinstance(value, list | tuple):
        return [_normalise(item, f"{path}[{i}]") for i, item in enumerate(value)]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalise(model_dump(mode="python"), path)
    raise CanonicalError(f"{path}: {type(value).__name__} has no canonical JSON form")


def canonical_json(value: Any) -> bytes:
    return json.dumps(_normalise(value, "$"), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """sha256 hex of the canonical JSON bytes."""
    return hashlib.sha256(canonical_json(value)).hexdigest()
