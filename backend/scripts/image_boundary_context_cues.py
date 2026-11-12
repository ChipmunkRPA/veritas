# SPDX-License-Identifier: Apache-2.0
"""Model-free color continuity and texture periodicity near an exact candidate box."""

from __future__ import annotations

import cv2
import numpy as np

ALPHA = {"outer_std_min": 8., "beta_min": .14, "beta_max": .90, "max_beta_spread": .22,
         "held_gain_min": .50, "held_median_error_max": 8., "minimum_held_sides": 3, "matte_slack": 12.}
PERIOD = {"inset": 4, "search_radius": 10, "minimum_lag": 3, "min_interior_rms": 1.5,
          "min_correlation": .82, "max_outer_correlation": .50, "minimum_outer_pairs": 96}


def _sides(pixels, box):
    left, top, right, bottom = box
    if left < 3 or top < 3 or right + 2 >= pixels.shape[1] or bottom + 2 >= pixels.shape[0]:
        return []
    source = pixels.astype(np.float64)
    vertical = slice(top + 6, bottom - 6)
    horizontal = slice(left + 6, right - 6)
    return [
        (source[vertical, left - 2], source[vertical, left + 1]),
        (source[top - 2, horizontal], source[top + 1, horizontal]),
        (source[vertical, right + 1], source[vertical, right - 2]),
        (source[bottom + 1, horizontal], source[bottom - 2, horizontal]),
    ]


def alpha_measurement(pixels, box):
    """Fit an RGB source-over model to three sides and check the unused side."""
    pairs = _sides(pixels, box)
    results = []
    for held, (outer, inner) in enumerate(pairs):
        training = pairs[:held] + pairs[held + 1:]
        a = np.concatenate([side[0] for side in training])
        b = np.concatenate([side[1] for side in training])
        ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
        energy = float(np.square(ac).sum())
        beta = float((ac * bc).sum() / energy) if energy > 0 else 0.
        intercept = np.median(b - beta * a, axis=0)
        matte = intercept / (1 - beta) if beta < 1 else np.full(3, -1_000.)
        residual = inner - (beta * outer + intercept)
        held_energy = float(np.square(inner - inner.mean(axis=0)).sum())
        gain = 1 - float(np.square(residual).sum()) / held_energy if held_energy > 0 else -1.
        results.append({"side": held, "outer_std": float(outer.std(axis=0).mean()), "beta": beta,
                        "matte": [float(value) for value in matte], "gain": gain,
                        "median_error": float(np.median(np.abs(residual)))})
    return results


def _paired(image, dy, dx):
    height, width = image.shape
    a = image[max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)]
    b = image[max(0, dy):height - max(0, -dy), max(0, dx):width - max(0, -dx)]
    return a, b


def _correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    energy = float(np.square(a).sum() * np.square(b).sum()) ** .5
    return max(-1., min(1., float((a * b).sum() / energy))) if energy > 1.e-6 else 0.


def periodic_measurement(pixels, box):
    """Check repeated interior phase at one and two lags, and nearby exterior."""
    left, top, right, bottom = box
    extent = 24
    x, y = max(0, left - extent), max(0, top - extent)
    end_x, end_y = min(pixels.shape[1], right + extent), min(pixels.shape[0], bottom + extent)
    gray = cv2.cvtColor(pixels[y:end_y, x:end_x], cv2.COLOR_RGB2GRAY).astype(np.float64)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    inset = PERIOD["inset"]
    interior = residual[top - y + inset:bottom - y - inset, left - x + inset:right - x - inset]
    rms = float(np.sqrt(np.square(interior).mean()))
    if rms < PERIOD["min_interior_rms"]:
        return {"interior_rms": rms, "proposals": []}
    exterior = np.ones(residual.shape, dtype=bool)
    exterior[max(0, top - y - inset):bottom - y + inset, max(0, left - x - inset):right - x + inset] = False
    proposals = []
    radius = PERIOD["search_radius"]
    for dy in range(radius + 1):
        for dx in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) < PERIOD["minimum_lag"] or (dy == 0 and dx < 0):
                continue
            at_period = _correlation(*_paired(interior, dy, dx))
            if at_period < PERIOD["min_correlation"]:
                continue
            at_double = _correlation(*_paired(interior, 2 * dy, 2 * dx))
            if at_double < PERIOD["min_correlation"]:
                continue
            oa, ob = _paired(residual, dy, dx)
            ma, mb = _paired(exterior, dy, dx)
            valid = ma & mb
            if int(valid.sum()) >= PERIOD["minimum_outer_pairs"]:
                outer = _correlation(oa[valid], ob[valid])
                proposals.append({"lag": [dy, dx], "one": at_period, "two": at_double, "outer": outer,
                                  "outer_pairs": int(valid.sum())})
    return {"interior_rms": rms, "proposals": proposals}
