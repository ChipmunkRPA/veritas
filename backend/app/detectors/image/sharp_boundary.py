"""Model-free inspection of abrupt rectangular boundaries in native RGB PNGs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DETECTOR_ID = "img.forensic.sharp_boundary"
SCHEMA = "sharp-boundary/1"
MIN_SIDE, MAX_SIDE = 512, 768
OFFSETS = (1 / 6, 1 / 4, 1 / 3)
CONTEXT_OFFSET = 3
REVIEW_POINT = 0.8292734622955322
MAX_STEP_RESPONSE = 3.0


@dataclass(frozen=True)
class Boundary:
    box: tuple[int, int, int, int]
    score: float
    sides: dict[str, float]
    contour: np.ndarray


def supported_size(width: int, height: int) -> bool:
    return (
        type(width) is int
        and type(height) is int
        and min(width, height) >= MIN_SIDE
        and max(width, height) <= MAX_SIDE
    )


def _step_response(pixels: np.ndarray) -> np.ndarray:
    steps = np.zeros(pixels.shape[:2], dtype=np.float32)
    steps[:, 1:] = np.abs(pixels[:, 1:] - pixels[:, :-1]).mean(axis=2)
    offset = CONTEXT_OFFSET
    nearby = (steps[:, : -2 * offset] + steps[:, 2 * offset :]) / 2
    response = np.zeros_like(steps)
    response[:, offset:-offset] = np.clip(
        np.log((steps[:, offset:-offset] + 3) / (nearby + 3)), 0, MAX_STEP_RESPONSE
    )
    return response


def measure(pixels: np.ndarray) -> Boundary:
    """Find the largest geometric mean of all four measured side responses."""
    height, width = pixels.shape[:2]
    offset = CONTEXT_OFFSET
    rgb = pixels.astype(np.float32)
    vertical_response = _step_response(rgb)
    horizontal_response = _step_response(rgb.transpose(1, 0, 2)).T
    vertical = np.pad(vertical_response.cumsum(0), ((1, 0), (0, 0)))
    horizontal = np.pad(horizontal_response.cumsum(1), ((0, 0), (1, 0)))
    best_score = -1.0
    best_box = (0, 0, 0, 0)
    for box_width in (round(width * fraction) for fraction in OFFSETS):
        for box_height in (round(height * fraction) for fraction in OFFSETS):
            left = np.arange(offset, width - offset - box_width)
            top = np.arange(offset, height - offset - box_height)
            right, bottom = left + box_width, top + box_height
            products = (vertical[np.ix_(bottom, left)] - vertical[np.ix_(top, left)]) / box_height
            products *= (vertical[np.ix_(bottom, right)] - vertical[np.ix_(top, right)]) / box_height
            products *= (horizontal[np.ix_(top, right)] - horizontal[np.ix_(top, left)]) / box_width
            products *= (horizontal[np.ix_(bottom, right)] - horizontal[np.ix_(bottom, left)]) / box_width
            row, col = np.unravel_index(int(products.argmax()), products.shape)
            score = float(max(0, products[row, col]) ** 0.25)
            if score > best_score:
                best_score = score
                best_box = int(left[col]), int(top[row]), int(right[col]), int(bottom[row])

    left, top, right, bottom = best_box
    box_width, box_height = right - left, bottom - top
    sides = {
        "left": float((vertical[bottom, left] - vertical[top, left]) / box_height),
        "top": float((horizontal[top, right] - horizontal[top, left]) / box_width),
        "right": float((vertical[bottom, right] - vertical[top, right]) / box_height),
        "bottom": float((horizontal[bottom, right] - horizontal[bottom, left]) / box_width),
    }
    # A step between neighboring pixels is indexed at its second pixel. The
    # right and bottom traces therefore sit at the half-open rectangle endpoints.
    contour = np.zeros((height, width), dtype=np.float32)
    contour[top:bottom, left] = vertical_response[top:bottom, left]
    contour[top:bottom, right] = vertical_response[top:bottom, right]
    contour[top, left:right] = np.maximum(contour[top, left:right], horizontal_response[top, left:right])
    contour[bottom, left:right] = horizontal_response[bottom, left:right]
    return Boundary(best_box, best_score, sides, contour)
