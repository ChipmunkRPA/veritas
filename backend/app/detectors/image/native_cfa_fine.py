# SPDX-License-Identifier: Apache-2.0
"""Find a compact alternate camera interpolation phase at native resolution."""

import cv2
import numpy as np

from app.detectors.image import native_cfa

CELL = 16
STEP = CELL // 2
SIDES = (2, 3)
MINIMUM = 10
ALT_CLOSURE = .62
PHASE_LEAD = .25
POINT = .25


def propose(rgb, phase, *, check=None):
    """Test the same alternate phase in a majority of a small rectangle's cells."""
    if check is not None:
        check("small native interpolation region")
    rates = native_cfa.stencil(rgb, cell=CELL, minimum=MINIMUM, half_step=True)
    valid = np.isfinite(rates).all(axis=-1)
    strongest = None
    for alternate in range(4):
        if alternate == phase:
            continue
        change = rates[..., alternate] - rates[..., phase]
        supported = valid & (rates[..., alternate] >= ALT_CLOSURE) & (change >= PHASE_LEAD)
        margin = np.where(supported, change - PHASE_LEAD, 0.)
        for dy in (0, 1):
            for dx in (0, 1):
                strength = cv2.integral(margin[dy::2, dx::2])
                count = cv2.integral(supported[dy::2, dx::2].astype(np.uint8))
                for h in SIDES:
                    for w in SIDES:
                        total = h * w
                        cells = count[h:, w:] + count[:-h, :-w] - count[h:, :-w] - count[:-h, w:]
                        sums = strength[h:, w:] + strength[:-h, :-w] - strength[h:, :-w] - strength[:-h, w:]
                        score = np.where(3 * cells >= 2 * total, sums / np.sqrt(total), 0.)
                        y, x = np.unravel_index(int(score.argmax()), score.shape)
                        x0, y0 = x * CELL + dx * STEP, y * CELL + dy * STEP
                        rank = (float(score[y, x]), -total, -y0, -x0, -alternate)
                        if rank[0] > 0 and (strongest is None or rank > strongest[0]):
                            strongest = rank, {"region": [int(x0), int(y0), int(x0+w*CELL), int(y0+h*CELL)],
                                               "alternate_phase": alternate, "matching_cells": int(cells[y, x]),
                                               "compared_cells": total, "response": rank[0]}
    if check is not None:
        check("small native interpolation region")
    return strongest[1] if strongest is not None else None
