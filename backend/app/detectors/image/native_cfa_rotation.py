# SPDX-License-Identifier: Apache-2.0
"""Locate a camera interpolation pattern consistent with a small local rotation."""

from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image import native_cfa

SUPPORTED = (*native_cfa.SCOPE, 3)
GRIDS = ((0, 0), (0, 16), (16, 0), (16, 16))
LOSS_POINT = 2.4346873354224834
AXIS_VETO = .70
ANGLES = tuple(step / 4 for step in range(7, 25))
ROTATION_POINT = .08
ROTATION_MARGIN = 1.35


def inspect(rgb, *, check=None, include_candidate=False):
    if rgb.dtype != np.uint16 or rgb.shape != SUPPORTED:
        return {"status": "unsupported", "box": None, "readout": None}
    h, w = rgb.shape[:2]
    cell = native_cfa.CELL
    choices = []
    for dy, dx in GRIDS:
        if check is not None:
            check("native interpolation rotation")
        rows = (h - dy) // cell * cell
        columns = (w - dx) // cell * cell
        body = rgb[dy:dy + rows, dx:dx + columns]
        rates = native_cfa.stencil(body)
        learned = native_cfa.learn(rates)
        if learned is None:
            continue
        phase, expected, eligible = learned
        luma = body.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
        fine = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), .85))
        energy = (native_cfa.cell_sums(fine) / cell**2).astype(np.float32)
        nearby = cv2.GaussianBlur(cv2.medianBlur(energy, 3), (9, 9), 2.)
        proposal = native_cfa.best_region(np.where(energy >= np.maximum(nearby * .50, 1.),
                                                  expected - rates[..., phase] - .12, 0.), eligible)
        if proposal["box"]:
            x, y, x1, y1 = proposal["box"]
            choices.append((proposal["score"], [x + dx, y + dy, x1 + dx, y1 + dy], phase))
    if not choices:
        return {"status": "no_supported_region", "box": None, "readout": None}
    strength, box, phase = max(choices, key=lambda item: item[0])
    if strength <= LOSS_POINT:
        return {"status": "no_supported_region", "box": None, "readout": None}
    x, y, x1, y1 = box
    if max(map(min, interpolation_fit(rgb[y+8:y1-8, x+8:x1-8], phase))) > AXIS_VETO:
        return {"status": "ordinary_axis_interpolation", "box": None, "readout": None}
    if check is not None:
        check("native interpolation rotation")
    detail = rotated_pattern(rgb[y+6:y1-6, x+6:x1-6])
    return {"status": "local_rotation" if detail["selected"] else "no_rotation_support",
            "box": box if detail["selected"] or include_candidate else None, "readout": detail}


def interpolation_fit(tile, phase):
    from scipy.optimize import nnls

    axes = []
    for axis in (0, 1):
        channels = []
        for channel in (0, 2):
            plane = tile[..., channel].astype(np.float64)
            if axis:
                plane = plane.T
            base = (phase % 2 if axis else phase // 2) ^ (channel == 2)
            second = plane[2:] - 2 * plane[1:-1] + plane[:-2]
            first, second = second[base::2], second[1-base::2]
            n = min(len(first), len(second)) - 2
            best = 0.
            if n > 10:
                for anchor, other in ((first, second), (second, first)):
                    for offset in (0, 1):
                        left, right, target = (array[2:-2, 2:-2].ravel() for array in (
                            anchor[:n], anchor[1:n+1], other[offset:offset+n]))
                        norm = np.linalg.norm(target)
                        if norm < 150:
                            continue
                        model = np.stack((left, right), axis=1)
                        coefficients, _ = nnls(model, target)
                        fit = 1. - np.linalg.norm(target - model @ coefficients) / norm
                        best = max(best, float(fit))
            channels.append(best)
        axes.append(channels)
    return axes


def rotated_pattern(rgb):
    region = rgb.astype(np.float32)
    vertical = region[2:, 1:-1] - 2 * region[1:-1, 1:-1] + region[:-2, 1:-1]
    horizontal = region[1:-1, 2:] - 2 * region[1:-1, 1:-1] + region[1:-1, :-2]
    logged = np.log1p(np.abs(vertical) + np.abs(horizontal))
    detail = np.clip(logged - cv2.GaussianBlur(logged, (0, 0), 3.), -2., 2.)
    yy, xx = np.indices(detail.shape[:2], dtype=np.float64)
    halves = xx < xx.max() / 2
    red, blue = detail[..., 0], detail[..., 2]

    def at_angle(angle):
        rad = np.deg2rad(angle)
        a, b = np.cos(rad) * np.pi, np.sin(rad) * np.pi
        amplitudes = []
        for fx, fy in ((a,b),(-b,a)):
            carrier = np.exp(1j * (fx * xx + fy * yy))
            folds = []
            for selection in (halves, ~halves):
                ar = np.mean((red * carrier)[selection])
                ab = np.mean((blue * carrier)[selection])
                contra = ar - ab if np.real(ar * np.conj(-ab)) > 0 else 0j
                folds.append((contra, abs(ar) + abs(ab)))
            (left, strength_left), (right, strength_right) = folds
            agreement = np.real(left * np.conj(right)) / max(1e-7, abs(left) * abs(right))
            amplitudes.append(float(min(strength_left, strength_right) * max(0., agreement)))
        return min(amplitudes)

    zero = at_angle(0.)
    peak, angle = max((at_angle(a), a) for a in (*ANGLES, *(-a for a in ANGLES)))
    return {"zero": zero, "peak": peak, "angle": angle, "selected": peak >= ROTATION_POINT and peak > ROTATION_MARGIN * zero}
