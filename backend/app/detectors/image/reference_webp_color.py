"""Measure local color differences after the earlier WebP framing is established."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_compare, reference_lossy_webp, reference_resize

SCHEMA = "chipmunk.image-reference-comparison/23"
SIGMA = 3.0
POINT = 8.0
MAX_BRIGHTNESS_DIFFERENCE = 6.0
MIN_BACKGROUND = .85
LUMA = np.array((.299, .587, .114), dtype=np.float32)


def planes(rgb):
    smooth = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), SIGMA)
    return _components(smooth)


def _components(rgb):
    brightness = rgb @ LUMA
    color = np.stack((.564 * (rgb[:, :, 2] - brightness), .713 * (rgb[:, :, 0] - brightness)), axis=-1)
    return brightness, color


def _features(brightness, color):
    levels = np.empty((*brightness.shape, 4), dtype=np.float64)
    levels[:, :, 0] = 1.
    levels[:, :, 1] = (brightness - 128.) / 128.
    levels[:, :, 2:] = color / 128.
    return levels


def _global_color(brightness, color, target, *, eligible=None):
    """Fit one affine color transformation with a distinct check grid."""
    x = _features(brightness[4::8, 4::8], color[4::8, 4::8]).reshape(-1, 4)
    y = target[4::8, 4::8].reshape(-1, 2).astype(np.float64)
    if eligible is not None:
        x, y = x[eligible.ravel()], y[eligible.ravel()]
    identity = np.array(((0., 0.), (0., 0.), (128., 0.), (0., 128.)))
    regularizer = np.diag((.000001, .0001, .0001, .0001)) * len(x)
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        source, output = x[keep], y[keep]
        transform = np.linalg.solve(source.T @ source + regularizer, source.T @ output + regularizer @ identity)
        residual = np.linalg.norm(y - x @ transform, axis=1)
        keep = residual <= max(2., 3. * float(np.median(residual)))
    return transform


def _apply_affine(brightness, color, fit):
    return (((brightness - 128.) / 128.)[:, :, None] * fit[1] + (color / 128.) @ fit[2:] + fit[0])


def _fit_color_with_clipping(earlier, submitted):
    first, second = earlier.astype(np.float32), submitted[4::8, 4::8]
    sample = first[4::8, 4::8]
    # Fit on samples with room for a color change in every channel. Clipped
    # colors cannot directly witness the size of the whole-image adjustment.
    interior = np.all((sample >= 24) & (sample <= 231) & (second >= 16) & (second <= 239), axis=-1)
    if np.count_nonzero(interior) < 128:
        return
    brightness, color = _components(first)
    _, target = _components(submitted.astype(np.float32))
    fit = _global_color(brightness, color, target, eligible=interior)
    delta = _apply_affine(brightness, color, fit) - color
    blue, red = delta[:, :, 0] / .564, delta[:, :, 1] / .713
    rgb_delta = np.stack((red, -(.299 * red + .114 * blue) / .587, blue), axis=-1)
    # Allow both ordinary per-channel clipping and saturation that holds the
    # brightness constant by scaling the color change at the RGB boundary.
    yield planes(np.clip(first + rgb_delta, 0, 255))[1], "channel_clip"
    room = np.where(rgb_delta > 0, 255. - first, first)
    factor = np.minimum(1., np.min(room / np.maximum(abs(rgb_delta), 1e-8), axis=-1))
    yield planes(first + factor[..., None] * rgb_delta)[1], "brightness_preserving_clip"


def _color_residual(target, expected):
    delta = target - expected
    return np.hypot(delta[:, :, 0], delta[:, :, 1])


def _model_error(magnitude):
    grid = magnitude[::8, ::8]
    return float(np.mean(np.minimum(grid, POINT)))


def _mask(earlier, submitted, after, *, method):
    before = planes(earlier)
    brightness, color = before
    target_y, target_color = after
    y_diff = target_y - brightness
    y_diff -= np.median(y_diff[::8, ::8])
    if method != "offset":
        fit = _global_color(brightness, color, target_color)
        magnitude = _color_residual(target_color, _apply_affine(brightness, color, fit))
        adjustment = "affine"
    else:
        delta = target_color - color
        magnitude = _color_residual(target_color, color + np.median(delta[4::8, 4::8], axis=(0, 1)))
        adjustment = "offset"
    if method == "gamut":
        # Keep the original model unless the unused samples favor an
        # alternative across the picture. Large local errors are capped.
        best_error = _model_error(magnitude)
        for expected, label in _fit_color_with_clipping(earlier, submitted):
            alternative = _color_residual(target_color, expected)
            error = _model_error(alternative)
            if error + .1 < best_error:
                magnitude, best_error, adjustment = alternative, error, label
    calibration = magnitude[::8, ::8]
    med = float(np.median(calibration))
    background = float(np.mean(calibration <= POINT))
    proof = {"whole_image_adjustment": adjustment, "median_color_residual": round(med, 6),
             "matching_sample_fraction": round(background, 8)}
    if background < MIN_BACKGROUND:
        return None, proof
    point = max(POINT, 4. * med)
    candidates = (magnitude > point) & (np.abs(y_diff) <= MAX_BRIGHTNESS_DIFFERENCE)
    opened = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return opened.astype(bool), proof | {"color_residual_point": round(point, 6)}


def _geometry(earlier, result):
    crop = result.get("lossy_webp_crop") or result.get("lossy_webp_crop_resize")
    projection = result.get("lossy_webp_resize")
    x, y = 0, 0
    if crop is not None:
        x, y, w, h = (crop["reference_crop"][key] for key in ("x", "y", "w", "h"))
        earlier = earlier[y:y+h, x:x+w]
        projection = crop.get("projection")
    supported = {row["kernel"] for row in projection["supported_kernels"]} if projection else set()
    kernels = [kernel for kernel in reference_resize.KERNELS if kernel.name.lower() in supported]
    return earlier, (x, y), kernels, crop is not None


def _intersects(a, b):
    return all(a[start] < b[start] + b[length] and b[start] < a[start] + a[length]
               for start, length in (("x", "w"), ("y", "h")))


def _regions(candidates, previous, size, offset, *, mapped):
    height, width = candidates.shape
    margin = 4
    for row in previous:
        x, y, w, h = (row[key] for key in ("x", "y", "w", "h"))
        candidates[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin] = False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    area_floor = round(candidates.size * reference_lossy_webp.METHOD["minimum_region_area_fraction"])
    ranked = sorted((index for index, spec in enumerate(stats) if index and spec[4] >= area_floor),
                    key=lambda index: (-stats[index, 4], index))
    mask = np.zeros(candidates.shape, dtype=bool)
    rows, omitted = [], 0
    for index in ranked:
        x, y, w, h, area = (int(value) for value in stats[index])
        record = {"x": x, "y": y, "w": w, "h": h, "selected_pixels": area, "difference_basis": "color"}
        if any(_intersects(record, old) for old in (*previous, *rows)):
            continue
        if len(previous) + len(rows) >= reference_compare.MAX_REGIONS:
            omitted += 1
            continue
        if mapped:
            earlier = reference_resize.native_crop(record, size, (width, height))
            earlier["x"] += offset[0]
            earlier["y"] += offset[1]
            record["reference_region"] = earlier
        rows.append(record)
        mask |= labels == index
    return rows, mask, omitted


def compare(earlier, submitted, base, *, method="gamut"):
    """Read color on the validated native or projected crop and preserve earlier regions."""
    earlier, offset, kernels, cropped = _geometry(earlier, base)
    height, width = submitted.shape[:2]
    target = planes(submitted)
    votes = np.zeros((height, width), dtype=np.uint8)
    measurements = []
    for kernel in kernels or (None,):
        if kernel is None:
            projected = earlier
        else:
            with Image.fromarray(earlier).resize((width, height), kernel) as resized:
                projected = np.asarray(resized)
        mask, witness = _mask(projected, submitted, target, method=method)
        measurements.append({"kernel": kernel.name.lower() if kernel is not None else "native", **witness})
        if mask is not None:
            votes += mask
    required = reference_resize.MIN_VOTES if kernels else 1
    status = "measured" if sum("color_residual_point" in item for item in measurements) >= required else "unassessed"
    proof = {"status": status, "smoothing_sigma_px": SIGMA, "maximum_brightness_difference": MAX_BRIGHTNESS_DIFFERENCE,
             "minimum_agreeing_projections": required, "projections": measurements}
    if status == "unassessed":
        return proof, [], np.zeros((height, width), dtype=bool), 0
    regions, selection, omitted = _regions(votes >= required, base["regions"], earlier.shape[1::-1], offset,
                                         mapped=cropped or bool(kernels))
    return proof, regions, selection, omitted
