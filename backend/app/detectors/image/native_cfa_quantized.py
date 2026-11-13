"""Measure remaining camera-grid structure in a lower-depth pixel region."""

from __future__ import annotations

import numpy as np

from app.detectors.image import native_cfa_rotation


def lattice(tile):
    """Identify a common stored-value step in every color channel."""
    size = tile.shape[0] * tile.shape[1]
    result = None
    for step in (256, 257):
        histogram = [np.bincount(channel.ravel() % step, minlength=step) for channel in tile.transpose(2, 0, 1)]
        fraction = min(float(channel.max() / size) for channel in histogram)
        if fraction >= .95 and (result is None or fraction > result[0]):
            result = fraction, step
    return result


def phase_strength(tile):
    """Require opposite red/blue curvature phase in both axes and all four quarters."""
    strengths = []
    for axis in (0, 1):
        along = np.moveaxis(tile, axis, 0)
        detail = np.clip(np.log1p(np.abs(along[2:, 1:-1] + along[:-2, 1:-1] - 2 * along[1:-1, 1:-1])), 0, 3)
        middle_row, middle_col = (n // 4 * 2 for n in detail.shape[:2])
        opposite = []
        for rows in (slice(0, middle_row), slice(middle_row, None)):
            for cols in (slice(0, middle_col), slice(middle_col, None)):
                block = detail[rows, cols]
                if min(block.shape[:2]) < 8:
                    return 0.
                difference = block[::2].mean(axis=(0, 1)) - block[1::2].mean(axis=(0, 1))
                opposite.extend((float(difference[0]), -float(difference[2])))
        strengths.append(max(0., min(opposite), -max(opposite)))
    return min(strengths)


def measure(rgb, box):
    x0, y0, x1, y1 = box
    tile = rgb[y0+10:y1-10, x0+10:x1-10]
    quantization = lattice(tile)
    if quantization is None:
        return None
    fraction, step = quantization
    return step, fraction, phase_strength(tile / step)


def inspect(rgb, *, check=None):
    candidate = native_cfa_rotation.inspect(rgb, check=check, include_candidate=True)
    if candidate["status"] == "local_rotation":
        return "rotation", candidate["box"], None
    if candidate["status"] == "no_rotation_support":
        if check is not None:
            check("native quantized interpolation")
        reading = measure(rgb, candidate["box"])
        if reading and reading[2] > 0.:
            return "quantized_resampling", candidate["box"], reading
    return "no_match", None, None
