"""Measure or replay the native PNG and its bounded presentation images."""
from __future__ import annotations

import cv2
import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.image import native_cfa
from app.evidence import image_interpolation_binding as binding


def record(source_sha256, *, body=None, skipped=False):
    try:
        measurement = native_cfa.inspect_png(body) if body is not None and not skipped else None
    except (cv2.error, MemoryError):
        measurement = None
    status = "provenance" if skipped else "unavailable"
    if measurement is not None:
        status = ("matched" if measurement.region else "no_match") if measurement.status == "assessed" else measurement.status
    return {"schema": binding.SCHEMA, "source_sha256": source_sha256, "status": status,
            "scope": binding.SCOPE, "limitation": binding.LIMITATION, "observation": binding.STATUS[status],
            "affects_overall_confidence": False,
            "region": list(measurement.region) if measurement and measurement.region else None,
            "phase": measurement.phase if measurement else None,
            "supported_cells": measurement.supported_cells if measurement else 0,
            "reference_closure": measurement.reference_closure if measurement else None,
            "decoded_rgb16_sha256": measurement.decoded_rgb_sha256 if measurement else None}


def plates(body, region):
    rgb = native_cfa.decoded_rgb(body)
    if rgb is None:
        raise ValueError("Could not reopen the checked 16-bit PNG")
    x0, y0, x1, y1 = region
    bgr = rgb[..., ::-1]
    crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
    context = cv2.resize((bgr >> 8).astype(np.uint8), (1024, 768), interpolation=cv2.INTER_AREA)
    cv2.rectangle(context, (x0 // 2, y0 // 2), (x1 // 2 - 1, y1 // 2 - 1), (35, 193, 255), 2)
    return {"context": context, "crop": crop}


def verify(body, review, read_asset):
    measured = review["record"]
    if measured["status"] in {"unavailable", "provenance"}:
        return False
    if canonical_hash(record(measured["source_sha256"], body=body)) != canonical_hash(measured):
        raise ValueError("Interpolation observation did not replay from the original pixels")
    if measured["region"] is not None:
        for role, expected in plates(body, measured["region"]).items():
            data = np.frombuffer(read_asset(review["artifacts"][role]["sha256"]), dtype=np.uint8)
            actual = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if actual is None or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError("Interpolation image differs from its original pixels")
    return True
