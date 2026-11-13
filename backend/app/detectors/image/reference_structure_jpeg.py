# SPDX-License-Identifier: Apache-2.0
"""Earlier-copy JPEG comparison with corroborated local structure changes."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from app.detectors.image import jpeg_bitstream, reference_compare

SCHEMA = "chipmunk.image-reference-comparison/13"
INTERPRETATION = (
    "The comparison accounts for image-wide tone and JPEG compression. Highlights show local changes in color and structure "
    "that both comparisons retained. Ordinary edits can produce these differences. The earlier copy's origin and the cause "
    "of the changes require separate evidence."
)
MIN_SIDE, MAX_SIDE = 512, 768
EARLIER_QUALITIES = (75, 90, 95)
SUBMITTED_QUALITIES = (75, 90)
TILE = 64
COLOR_SIGMA = 1.6
STRUCTURE_SIGMA = 2.5
STRUCTURE_WINDOW = 4.5
STRUCTURE_RADIUS = 9


def _jpeg_profile(image, payload, qualities):
    if (image.mode != "RGB" or image.format != "JPEG" or image.getexif().get(274, 1) != 1
            or {"icc_profile", "progressive", "progression"} & image.info.keys()):
        return False
    spec = jpeg_bitstream.read(payload)
    return bool(spec and spec.frame == "baseline" and spec.precision == 8 and spec.ijg_quality in qualities
                and [(c.id, c.h, c.v) for c in spec.components] == [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
                and all(spec.table_bits.get(c.table_id) == 8 for c in spec.components)
                and spec.table(1) == spec.table(2))


def _tiles(shape):
    height, width = shape
    for y in range(0, height - TILE + 1, TILE):
        for x in range(0, width - TILE + 1, TILE):
            yield x, y, np.s_[y:y + TILE, x:x + TILE]


def _geometry(reference, submitted):
    height, width = reference.shape[:2]
    shift, response = reference_compare._alignment(reference, submitted)
    if response < .24 or max(map(abs, shift)) > .65:
        return None
    gray = [cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.)
            for rgb in (reference, submitted)]
    hann = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)
    zones = Counter()
    eligible, matches = 0, []
    for x, y, tile in _tiles((height, width)):
        patches = [image[tile].copy() for image in gray]
        if min(float(patch.std()) for patch in patches) < 6:
            continue
        eligible += 1
        a, b = (np.concatenate([np.diff(patch, axis=axis).ravel() for axis in (0, 1)]) for patch in patches)
        similarity = float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.))
        local, peak = cv2.phaseCorrelate(*patches, hann)
        if similarity >= .80 and peak >= .28 and max(map(abs, local)) <= .85:
            zones[((y + TILE//2) * 3 // height, (x + TILE//2) * 3 // width)] += 1
            matches.append([x, y])
    supported = [list(zone) for zone, count in sorted(zones.items()) if count >= 2]
    if (len(matches) < max(24, .65 * eligible) or len(supported) < 6
            or any(len({zone[axis] for zone in supported}) != 3 for axis in (0, 1))):
        return None
    return {"global_shift": shift, "global_response": response, "detailed_tiles": eligible,
            "aligned_tiles": matches, "supported_zones": supported}


def _isotonic(values, weights):
    """Fit a nondecreasing brightness curve with weighted adjacent blocks."""
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([index, index + 1, value * weight, weight])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty(len(values))
    for left, right, total, weight in blocks:
        result[left:right] = total / weight
    return result


def _fit_curve(source, target):
    bins = np.minimum(source.astype(np.int16) // 8, 31)
    samples = [(float(np.median(source[group])), float(np.median(target[group])), int(group.sum()))
               for value in np.unique(bins) if (group := bins == value).sum() >= 32]
    if len(samples) < 9:
        return None
    x, y, weights = map(np.asarray, zip(*samples, strict=True))
    y = _isotonic(y, weights)
    if x[-1] - x[0] < 75 or np.any(np.diff(y) > 2.4 * np.diff(x) + 3.):
        return None
    left = np.clip((y[2] - y[0]) / (x[2] - x[0]), 0, 2.25)
    right = np.clip((y[-1] - y[-3]) / (x[-1] - x[-3]), 0, 2.25)
    xs = np.concatenate(([0.], x, [255.]))
    ys = np.clip(np.concatenate(([y[0] - left * x[0]], y, [y[-1] + right * (255. - x[-1])])), 0, 255)
    keep = np.r_[np.diff(xs) > 0, True]
    return xs[keep], ys[keep]


def _smooth_color(rgb):
    return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), COLOR_SIGMA)


def _tone_views(reference, submitted, carrier):
    expected, _ = reference_compare._codec(reference, carrier)
    before, after = _smooth_color(expected), _smooth_color(submitted)
    y, x = np.indices(before.shape[:2])
    fold = (x // 32 + y // 32) % 2 == 0
    predictions, residuals, supports = [], [], []
    for train in (fold, ~fold):
        curves = []
        training = train[::3, ::3]
        for channel in range(3):
            knots = _fit_curve(before[::3, ::3, channel][training], after[::3, ::3, channel][training])
            if knots is None:
                return None
            curves.append(knots)
        corrected = np.stack([np.rint(np.interp(reference[:, :, channel], *knots)).astype(np.uint8)
                              for channel, knots in enumerate(curves)], axis=-1)
        prediction, _ = reference_compare._codec(corrected, carrier)
        residual = np.abs(_smooth_color(prediction) - after).max(axis=2)
        support = float((residual[~train] <= 6.).mean())
        if support < .80:
            return None
        predictions.append(prediction)
        residuals.append(residual)
        supports.append(support)
    return predictions, np.minimum(*residuals), supports


def _sharpness(rgb):
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(luma, (0, 0), 1.25)
    horizontal = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=.125)
    vertical = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=.125)
    fine = luma - cv2.GaussianBlur(luma, (0, 0), .85)
    return horizontal**2 + vertical**2, fine**2


def _sharpness_support(predictions, submitted):
    coarse, fine = _sharpness(submitted)
    supports = []
    for prediction in predictions:
        before_coarse, before_fine = _sharpness(prediction)
        ratios = []
        for _, _, tile in _tiles(coarse.shape):
            a, b = float(coarse[tile].mean()), float(fine[tile].mean())
            c, d = float(before_coarse[tile].mean()), float(before_fine[tile].mean())
            if a > 5 and c > 5 and d > 1:
                ratios.append(float(np.log(max(1e-6, (b / a) / (d / c)))))
        if len(ratios) < 12:
            return None
        supports.append(float(np.quantile(ratios, .75)))
    return supports if any(value >= -.22 for value in supports) else None


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _color_mask(residual, area_floor):
    seeds = cv2.morphologyEx((residual > 8.).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3))
    components = cv2.morphologyEx((residual > 4.).astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(7))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(components, connectivity=8)
    core = np.bincount(labels[seeds > 0], minlength=count)
    valid = (core >= 24) & (core >= .07 * stats[:, cv2.CC_STAT_AREA])
    valid[0] = False
    _, mask, _ = reference_compare._regions(valid[labels], area_floor)
    return mask


def _gradients(rgb):
    smooth = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), STRUCTURE_SIGMA)
    return np.stack([cv2.Sobel(smooth, cv2.CV_32F, dx, dy, ksize=3, scale=.125)
                     for dx, dy in ((1, 0), (0, 1))], axis=3)


def _structure_candidates(predictions, submitted, color_mask):
    def smooth(matrix):
        return cv2.GaussianBlur(matrix, (0, 0), STRUCTURE_WINDOW)

    observed = _gradients(submitted)
    observed_energy = smooth(np.sum(observed**2, axis=(2, 3)))
    seeds = color_mask.copy()
    for prediction in predictions:
        gradient = _gradients(prediction)
        energy = smooth(np.sum(gradient**2, axis=(2, 3)))
        dot = smooth(np.sum(gradient * observed, axis=(2, 3)))
        strongest = np.maximum(energy, observed_energy)
        different_angle = dot < .78 * np.sqrt(energy * observed_energy)
        different_strength = np.minimum(energy, observed_energy) < .35**2 * strongest
        seeds &= (strongest >= 2.5**2) & (different_angle | different_strength)
    opened = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3))
    return (cv2.dilate(opened, _ellipse(2 * STRUCTURE_RADIUS + 1)) > 0) & color_mask


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    """Return the supported native-grid difference mask for a same-framing JPEG pair."""
    if (earlier.size != submitted.size or not MIN_SIDE <= min(earlier.size) <= max(earlier.size) <= MAX_SIDE
            or not _jpeg_profile(earlier, earlier_bytes, EARLIER_QUALITIES)
            or not _jpeg_profile(submitted, submitted_bytes, SUBMITTED_QUALITIES)):
        return None, None
    reference, questioned = np.asarray(earlier), np.asarray(submitted)
    framing = _geometry(reference, questioned)
    if framing is None or (views := _tone_views(reference, questioned, submitted)) is None:
        return None, None
    predictions, residual, tone_support = views
    if (sharpness_support := _sharpness_support(predictions, questioned)) is None:
        return None, None
    floor = max(48, round(residual.size * .0008))
    mask = _color_mask(residual, floor)
    regions, omitted = [], 0
    if mask.any():
        candidates = _structure_candidates(predictions, questioned, mask)
        regions, mask, omitted = reference_compare._regions(candidates, floor)
        if float(mask.mean()) > .10:
            return None, None
    for region in regions:
        region["reference_region"] = {key: region[key] for key in ("x", "y", "w", "h")}
    proof = {**framing, "background_support": tone_support, "sharpness_support": sharpness_support}
    return {"status": "comparable", "regions": regions, "additional_regions": omitted,
            "same_framing_structure": proof, "affects_overall_confidence": False}, mask
