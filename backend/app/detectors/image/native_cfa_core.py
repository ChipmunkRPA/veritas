# SPDX-License-Identifier: Apache-2.0
"""Locate a small alternate interpolation core with source context around it."""

from __future__ import annotations

STEP = 4
BOX = 32
ALT_MIN, ALT_LEAD = .62, .25
REF_MIN, REF_LEAD = .72, .15
CORE = ((4, 4), (4, 6), (6, 4), (6, 6))
OUTER = ((4, 0), (6, 0), (0, 4), (0, 6), (4, 10), (6, 10), (10, 4), (10, 6))
ARMS = ("core", "surrounded")


def cell_support(core, outer):
    """The four core cells and eight exterior cells are fixed around the crop."""
    matching = [a >= ALT_MIN and a - b >= ALT_LEAD for b, a in core]
    strength = sum(a - b - ALT_LEAD for match, (b, a) in zip(matching, core, strict=True) if match) / 2
    recovered = [pair is not None and pair[0] >= REF_MIN and pair[0] - pair[1] >= REF_LEAD for pair in outer]
    edges = sum(recovered[i] or recovered[i + 1] for i in range(0, 8, 2))
    return sum(matching), strength, sum(recovered), edges


def propose(rgb, phase, *, check=None):
    import numpy as np

    from app.detectors.image import native_cfa

    if check is not None:
        check("compact native interpolation core")
    rates = native_cfa.stencil(rgb, cell=2 * STEP, minimum=4, half_step=True)
    height, width = (length - 10 for length in rates.shape[:2])
    result = dict.fromkeys(ARMS)
    ranking = {name: (0.,) for name in ARMS}

    def sample(offsets, channel):
        return np.stack([rates[y:y + height, x:x + width, channel] for y, x in offsets])

    def read(values, different):
        return [[float(a), float(b)] if np.isfinite(a) and np.isfinite(b) else None
                for a, b in zip(values, different, strict=True)]

    original, surroundings = sample(CORE, phase), sample(OUTER, phase)
    core_valid = np.isfinite(original).all(axis=0)
    for alternate in range(4):
        if alternate == phase:
            continue
        other, outside = sample(CORE, alternate), sample(OUTER, alternate)
        core = (other >= ALT_MIN) & (other - original >= ALT_LEAD)
        strength = np.where(core_valid & np.isfinite(other).all(axis=0),
                            np.where(core, other - original - ALT_LEAD, 0.).sum(axis=0) / 2, 0.)
        recovered = (surroundings >= REF_MIN) & (surroundings - outside >= REF_LEAD)
        outside_count = recovered.sum(axis=0)
        all_sides = (recovered[::2] | recovered[1::2]).all(axis=0)
        comparisons = {
            "core": np.where(core.sum(axis=0) >= 3, strength, 0.),
            "surrounded": np.where((core.sum(axis=0) >= 2) & (outside_count >= 6) & all_sides, strength * outside_count / 8, 0.),
        }
        for name, scores in comparisons.items():
            y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
            score = round(float(scores[y, x]), 8)
            key = score, -int(y), -int(x), -alternate
            if key <= ranking[name] or score <= 0:
                continue
            internal = read(original[:, y, x], other[:, y, x])
            external = read(surroundings[:, y, x], outside[:, y, x])
            if any(item is None for item in internal):
                continue
            left, top = 2 * STEP + int(x) * STEP, 2 * STEP + int(y) * STEP
            result[name] = {"region": [left, top, left + BOX, top + BOX], "alternate_phase": alternate,
                            "response": score, "core_cells": internal, "outer_cells": external}
            ranking[name] = key
        if check is not None:
            check("compact native interpolation core")
    return result
