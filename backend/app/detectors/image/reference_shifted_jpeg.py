"""Exact JPEG comparison inside the common area of an integer translation."""

from __future__ import annotations

import numpy as np

from app.detectors.image import reference_alignment, reference_compare, reference_exact_jpeg

SCHEMA = "chipmunk.image-reference-comparison/12"
METHOD = "exact_decoded_jpeg_translated_overlap"
TILE = 16
MAX_OFFSET = 24


def _slice(bounds):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    return np.s_[y:y+h, x:x+w]


def _offset(reference, submitted, incumbent):
    measured = incumbent.get("alignment")
    if measured:
        offset = measured["reference_minus_submitted"]
        return (offset["x"], offset["y"]), "corroborated"
    shift, response = reference_compare._alignment(reference, submitted)
    estimate = np.asarray(shift)
    if not np.isfinite(estimate).all() or not np.isfinite(response) or response < .30:
        return None, None
    rounded = np.rint(estimate)
    if np.max(np.abs(estimate-rounded)) > .40:
        return None, None
    return tuple(int(value) for value in -rounded), "phase_nominated"


def _interior(bounds):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    x0, y0 = (((start + 2*TILE - 1) // TILE) * TILE for start in (x, y))
    x1, y1 = (((stop - TILE) // TILE) * TILE for stop in (x+w, y+h))
    return {"x": x0, "y": y0, "w": x1-x0, "h": y1-y0}


def _delivery(reference, offset, overlap, carrier, fill):
    dx, dy = offset
    canvas = np.full_like(reference, fill)
    before = overlap | {"x": overlap["x"]+dx, "y": overlap["y"]+dy}
    canvas[_slice(overlap)] = reference[_slice(before)]
    return reference_exact_jpeg.encoded_reference(canvas, carrier)


def compare(reference, carrier, earlier, submitted, incumbent):
    """Support a whole-pixel offset with exact JPEG blocks inside the shared image."""
    if incumbent["status"] == "comparable" or not reference_exact_jpeg.qualified_carriers(reference, carrier, earlier, submitted):
        return None, None
    native = np.array(reference)
    rgb = np.array(carrier)
    offset, method = _offset(native, rgb, incumbent)
    if offset is None or not 1 <= max(map(abs, offset)) <= MAX_OFFSET:
        return None, None
    dx, dy = offset
    overlap = reference_alignment._overlap(native.shape[:2], rgb.shape[:2], dx, dy)
    if overlap is None:
        return None, None
    bounds = _interior(overlap)
    if min(bounds["w"], bounds["h"]) < 160:
        return None, None
    inner = _slice(bounds)
    reconstructed = _delivery(native, offset, overlap, carrier, 0)[inner]
    if not np.array_equal(reconstructed, _delivery(native, offset, overlap, carrier, 255)[inner]):
        return None, None
    proof = reference_exact_jpeg.alignment_proof(reconstructed, rgb[inner])
    if proof is None:
        return None, None

    def clean_residual(candidate):
        report, mask = reference_compare.compare(candidate, rgb[inner])
        return mask if report["status"] == "comparable" else None

    primary = clean_residual(reconstructed)
    if primary is None:
        return None, None
    regions, supported, omitted, qualities = [], np.zeros(primary.shape, bool), 0, []
    if primary.any():
        support = np.zeros(primary.shape, np.uint8)
        for quality in reference_compare.PRIOR_JPEG_QUALITIES:
            earlier_jpeg = reference_compare._prior_jpeg(native, quality)
            alternate = clean_residual(_delivery(earlier_jpeg, offset, overlap, carrier, 0)[inner])
            if alternate is not None:
                support += alternate
                qualities.append(quality)
        if len(qualities) < 2:
            return None, None
        regions, supported, omitted = reference_compare._regions(
            primary & (support >= 2), max(48, round(primary.size*.0015)),
        )
    full_mask = np.zeros(rgb.shape[:2], bool)
    full_mask[inner] = supported
    for region in regions:
        region["x"] += bounds["x"]
        region["y"] += bounds["y"]
        region["reference_region"] = {"x": region["x"]+dx, "y": region["y"]+dy, "w": region["w"], "h": region["h"]}
    alignment = {**proof, "method": METHOD, "nomination": method, "reference_minus_submitted": {"x": dx, "y": dy},
                 "submitted_overlap": bounds,
                 "reference_overlap": bounds | {"x": bounds["x"]+dx, "y": bounds["y"]+dy}}
    return {"status": "comparable", "alignment": alignment, "regions": regions, "additional_regions": omitted,
            "prior_jpeg_qualities_compared": qualities, "affects_overall_confidence": False}, full_mask
