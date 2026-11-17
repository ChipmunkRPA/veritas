# SPDX-License-Identifier: Apache-2.0
"""Find and compare a proportionally reduced earlier photo inside a screenshot."""

from __future__ import annotations

from itertools import product

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

from app.detectors.image import reference_compare, reference_crop_resize, reference_heif, reference_resize

SCHEMA = "chipmunk.image-reference-comparison/27"
INTERPRETATION = ("The earlier photo was resized and compared inside the marked screenshot area. "
                  "Local edits or interface overlays can produce differences. This comparison carries no weight in the AI assessment.")
MIN_SIDE = 160
MAX_REFERENCE_SIDE = 1024
MAX_SCREEN_SIDE = 1280
MIN_SCALE, MAX_SCALE = .50, .94
MIN_MARGIN = 8
MIN_MATCHES = 20
GRID = 5
SEARCH_RADIUS = 2
MIN_CELLS = 15
MAX_DIFFERENCE_FRACTION = .24
RESIDUAL_RADIUS = 1.25


def _supported(reference, screen, carrier):
    if (min(*reference.shape[:2], *screen.shape[:2]) < MIN_SIDE
            or max(reference.shape[:2]) > MAX_REFERENCE_SIDE or max(screen.shape[:2]) > MAX_SCREEN_SIDE
            or carrier.mode != "RGB" or getattr(carrier, "n_frames", 1) != 1
            or carrier.getexif().get(274, 1) != 1):
        return False
    return (carrier.format == "PNG" or (carrier.format == "JPEG" and JpegImagePlugin.get_sampling(carrier) == 2
            and set(carrier.quantization) == {0, 1}))


def _rough_bounds(reference, screen):
    matches = reference_crop_resize.matching_points(reference, screen)
    if matches is None:
        return None
    later, earlier = matches
    affine, flags = cv2.estimateAffine2D(earlier, later, ransacReprojThreshold=1.35, maxIters=1200, confidence=.995)
    if affine is None or flags is None:
        return None
    kept = flags.ravel().astype(bool)
    height, width = reference.shape
    scales = np.diag(affine)
    if (kept.sum() < max(MIN_MATCHES, .65 * len(earlier))
            or np.any(np.ptp(earlier[kept], axis=0) < .52 * np.array((width, height)))
            or min(scales) < MIN_SCALE - .012 or max(scales) > MAX_SCALE + .012
            or abs(scales[0] - scales[1]) > .012 or max(abs(affine[0, 1]), abs(affine[1, 0])) > .006):
        return None
    origin = affine[:, 2] + .5 * (1 - scales)
    return np.rint(np.r_[origin, scales * (width, height)]).astype(int), int(kept.sum())


def _placement(reference, screen, guess):
    rh, rw = reference.shape
    sh, sw = screen.shape
    scale = np.mean(guess[2:] / (rw, rh))
    earlier = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), 1.2 / scale)
    later = cv2.GaussianBlur(screen.astype(np.float32), (0, 0), 1.2)
    ticks = (np.arange(50, dtype=np.float32) + .5) / 50
    py, px = np.meshgrid(ticks, ticks, indexing="ij")
    sample = cv2.remap(earlier, px * rw - .5, py * rh - .5, cv2.INTER_LINEAR)
    scores = []
    for change in product(range(-SEARCH_RADIUS, SEARCH_RADIUS + 1), repeat=4):
        x, y, width, height = (int(value) for value in guess + change)
        if (min(width, height) < MIN_SIDE or min(x, y, sw - x - width, sh - y - height) < MIN_MARGIN
                or not MIN_SCALE <= min(width / rw, height / rh) <= max(width / rw, height / rh) <= MAX_SCALE
                or abs(width - height * rw / rh) > max(1., rw / rh)
                or not .13 <= width * height / (sw * sh) <= .72):
            continue
        measured = cv2.remap(later, px * width + (x - .5), py * height + (y - .5), cv2.INTER_LINEAR)
        difference = np.abs(measured - sample)[2:-2, 2:-2]
        # Robust to one local edit, but distributed disagreement reduces the score.
        score = float(np.minimum(difference, 12.).mean())
        scores.append((score, (x, y, width, height)))
    return min(scores)[1] if scores else None


def _duplicate(reference, screen, bounds):
    x, y, width, height = bounds
    # A coarse full-screen search can find a second placement even when the
    # distinctive feature matches favored just one of the two.
    sw, sh = screen.shape[1] // 3, screen.shape[0] // 3
    small_screen = cv2.resize(screen, (sw, sh), interpolation=cv2.INTER_AREA)
    sx, sy = sw / screen.shape[1], sh / screen.shape[0]
    for factor in (.8, 1., 1.25):
        tw, th = round(width * sx * factor), round(height * sy * factor)
        if tw >= sw or th >= sh:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        scores = cv2.matchTemplate(small_screen, template, cv2.TM_CCOEFF_NORMED)
        # Ignore nearby replies around the already supported placement.
        left, right = round((x - width / 3) * sx), round((x + width / 3) * sx)
        top, bottom = round((y - height / 3) * sy), round((y + height / 3) * sy)
        scores[max(0, top):max(0, bottom), max(0, left):max(0, right)] = 0
        if scores.max() >= .975:
            return True
    return False


def _blur(pixels):
    return cv2.GaussianBlur(pixels.astype(np.float32), (0, 0), RESIDUAL_RADIUS)


def _residual(reference, screen, carrier, bounds):
    x, y, width, height = bounds
    photo = np.s_[y:y+height, x:x+width]
    measured = _blur(screen[photo])
    earlier = Image.fromarray(reference)
    size = (width, height)
    projected = (np.asarray(earlier.resize(size, Image.Resampling.BICUBIC)),
                 cv2.resize(reference, size, interpolation=cv2.INTER_LINEAR))
    residuals = []
    errors = []
    for candidate in projected:
        if carrier.format == "JPEG":
            canvas = screen.copy()
            canvas[photo] = candidate
            delivered, _ = reference_compare._codec(canvas, carrier)
            candidate = delivered[photo]
        predicted = _blur(candidate)
        error = np.abs(predicted - measured).max(axis=2)
        texture = _blur(np.abs(candidate.astype(np.float32) - predicted)).max(axis=2)
        residuals.append(error / (6. + .30 * texture))
        errors.append(error)
    return np.minimum.reduce(residuals), np.minimum.reduce(errors), projected[0]


def _witnesses(normalized, error, projected, screen, bounds):
    x, y, width, height = bounds
    ys, xs = (np.linspace(0, length, GRID + 1, dtype=int) for length in (height, width))
    good = []
    for row, column in product(range(GRID), repeat=2):
        tile = np.s_[ys[row]+4:ys[row+1]-4, xs[column]+4:xs[column+1]-4]
        if np.quantile(error[tile], .8) <= 5. and np.std(cv2.cvtColor(projected[tile], cv2.COLOR_RGB2GRAY)) >= 5.:
            good.append([row, column])
    if (len(good) < MIN_CELLS or any(sum(cell[axis] == edge for cell in good) < 2
                                    for axis, edge in product(range(2), (0, GRID - 1)))):
        return None
    interior = (normalized[3:7, 8:-8], normalized[-7:-3, 8:-8], normalized[8:-8, 3:7], normalized[8:-8, -7:-3])
    beyond = (screen[y-6:y-2, x+8:x+width-8], screen[y+height+2:y+height+6, x+8:x+width-8],
              screen[y+8:y+height-8, x-6:x-2], screen[y+8:y+height-8, x+width+2:x+width+6])
    edges = (projected[:1, 8:-8], projected[-1:, 8:-8], projected[8:-8, :1], projected[8:-8, -1:])
    for inside, outside, edge in zip(interior, beyond, edges, strict=True):
        exterior_error = np.max(np.abs(outside.astype(np.float32) - edge), axis=2)
        if np.mean(inside <= 1.) < .78 or np.mean(exterior_error >= 10.) < .65:
            return None
    return good


def compare(reference, screen, original, carrier):
    """Require corroborated placement, then retain supported local differences."""
    if (not _supported(reference, screen, carrier)
            or any("A" in image.getbands() or "transparency" in image.info for image in (original, carrier))):
        return None, None, reference
    profile_applied = bool(original.info.get("icc_profile"))
    try:
        # Replaying the submitted JPEG requires its decoded RGB values to be
        # in display space already. An earlier photo can be converted first.
        if carrier.info.get("icc_profile") and not np.array_equal(reference_heif.display_pixels(carrier), screen):
            return None, None, reference
        display_reference = reference_heif.display_pixels(original) if profile_applied else reference
    except ValueError:
        return None, None, reference
    gray_reference, gray_screen = (cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) for pixels in (display_reference, screen))
    rough = _rough_bounds(gray_reference, gray_screen)
    if rough is None:
        return None, None, reference
    guess, inliers = rough
    bounds = _placement(gray_reference, gray_screen, guess)
    if bounds is None or _duplicate(gray_reference, gray_screen, bounds):
        return None, None, reference
    normalized, error, projected = _residual(display_reference, screen, carrier, bounds)
    cells = _witnesses(normalized, error, projected, screen, bounds)
    if cells is None:
        return None, None, reference

    x, y, width, height = bounds
    candidates = normalized > 1.
    # Resampling and JPEG can mix interface pixels into the photo boundary.
    border = 8 if carrier.format == "JPEG" else 4
    candidates[:border] = candidates[-border:] = False
    candidates[:, :border] = candidates[:, -border:] = False
    candidates = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if candidates.mean() > MAX_DIFFERENCE_FRACTION:
        return None, None, reference
    regions, selected, omitted = reference_compare._regions(candidates, max(36, round(candidates.size * .00045)))
    mask = np.zeros(screen.shape[:2], dtype=bool)
    mask[y:y+height, x:x+width] = selected
    for region in regions:
        region["reference_region"] = reference_resize.native_crop(region, reference.shape[1::-1], (width, height))
        region["x"] += x
        region["y"] += y
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted,
        "affects_overall_confidence": False,
        "scaled_screenshot": {
            "reference_bounds": {"x": 0, "y": 0, "w": reference.shape[1], "h": reference.shape[0]},
            "submitted_bounds": {"x": x, "y": y, "w": width, "h": height},
            "compared_bounds": {"x": x + border, "y": y + border, "w": width - 2 * border, "h": height - 2 * border},
            "support_grid_size": GRID, "support_cells": cells, "aligned_features": inliers,
            "jpeg_delivery_modeled": carrier.format == "JPEG", "reference_profile_applied": profile_applied,
        },
    }, mask, display_reference
