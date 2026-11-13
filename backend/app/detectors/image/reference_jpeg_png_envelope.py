"""Measure earlier JPEG to cropped PNG color differences beyond ordinary delivery."""

from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image import (
    reference_compare,
    reference_crop_resize,
    reference_png_exact_resize,
    reference_resize,
)

SCHEMA = "chipmunk.image-reference-comparison/10"
GAUSSIAN_SIGMA = 1.6
MINIMUM_EXCESS = 6.
ALTERNATE_JPEG_QUALITY = 83


def _projected_views(reference, crop, size):
    x, y, width, height = (crop[key] for key in ("x", "y", "w", "h"))
    rectangle = x, y, x + width, y + height
    reencoded = reference_compare._prior_jpeg(reference, ALTERNATE_JPEG_QUALITY)
    return [reference_png_exact_resize._project(variant, rectangle, size, kernel).astype(np.float32)
            for variant in (reference, reencoded) for kernel in reference_png_exact_resize.KERNELS.values()]


def _measure(reference, submitted, crop):
    actual = cv2.GaussianBlur(submitted.astype(np.float32), (0, 0), GAUSSIAN_SIGMA)
    lower = np.full_like(actual, np.inf)
    upper = np.full_like(actual, -np.inf)
    for view in _projected_views(reference, crop, submitted.shape[1::-1]):
        smoothed = cv2.GaussianBlur(view, (0, 0), GAUSSIAN_SIGMA)
        for channel in range(3):
            plane = smoothed[:, :, channel]
            gain, offset = reference_compare._tone(plane, actual[:, :, channel])
            corrected = np.clip(plane * gain + offset, 0, 255)
            np.minimum(lower[:, :, channel], corrected, out=lower[:, :, channel])
            np.maximum(upper[:, :, channel], corrected, out=upper[:, :, channel])
    margin = 1.25 * cv2.blur(upper - lower, (7, 7))
    residual = np.maximum((np.maximum(actual - upper, lower - actual) - margin).max(axis=2), 0)
    point = max(MINIMUM_EXCESS, 2.75 * float(np.percentile(residual, 70)))
    seeds = np.uint8(residual >= point)
    core = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contiguous = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return reference_compare._regions(contiguous, max(48, round(residual.size * .0015)))


def compare(reference, submitted, carrier, *, accept_crop):
    """Require corroborated framing before comparing the two delivery histories."""
    proposal, _ = reference_crop_resize.compare(reference, submitted, carrier)
    if not accept_crop(proposal, submitted.shape[1::-1]):
        return None, None
    crop = proposal["reference_crop_bounds"]
    regions, mask, omitted = _measure(reference, submitted, crop)
    for region in regions:
        native = reference_resize.native_crop(region, (crop["w"], crop["h"]), submitted.shape[1::-1])
        region["reference_region"] = {**native, "x": native["x"] + crop["x"], "y": native["y"] + crop["y"]}
    return {**proposal, "regions": regions, "additional_regions": omitted}, mask
