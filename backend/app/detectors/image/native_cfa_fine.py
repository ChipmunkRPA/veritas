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
EDGE_CELL = 8
EDGE_INTERIOR_MAX = .75
EDGE_EXTERIOR_MIN = .80
EDGE_CHANGE_MIN = .20


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


def refine_extent(rgb, phase, inner):
    """Extend an accepted region where native closure drops, then returns outside it."""
    x0, y0, x1, y1 = inner
    padding = 2 * EDGE_CELL
    height, width = rgb.shape[:2]
    if not (padding <= x0 < x1 <= width - padding and padding <= y0 < y1 <= height - padding):
        return None
    area = rgb[y0-padding:y1+padding, x0-padding:x1+padding]
    reference = native_cfa.stencil(area, cell=EDGE_CELL, minimum=4)[..., phase]
    strips = {
        "left": (reference[2:-2, 1], reference[2:-2, 0]),
        "top": (reference[1, 2:-2], reference[0, 2:-2]),
        "right": (reference[2:-2, -2], reference[2:-2, -1]),
        "bottom": (reference[-2, 2:-2], reference[-1, 2:-2]),
    }
    region = list(inner)
    edges = []
    for i, (side, (inside, outside)) in enumerate(strips.items()):
        supported = ((inside <= EDGE_INTERIOR_MAX) & (outside >= EDGE_EXTERIOR_MIN)
                     & (outside - inside >= EDGE_CHANGE_MIN))
        count = int(supported.sum())
        if 3 * count >= 2 * len(inside):
            region[i] += -EDGE_CELL if i < 2 else EDGE_CELL
            edges.append({"side": side, "supporting_pairs": count, "compared_pairs": len(inside)})
    if len(edges) < 2:
        return None
    return {"region": region, "inner_region": inner, "edges": edges}
