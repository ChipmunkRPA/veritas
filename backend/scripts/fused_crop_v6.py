"""Research-only crop inspection using the unchanged licensed inpainting runtime.

No registry, serving configuration, or whole-file confidence uses this method.
Coordinates are end-exclusive coordinates in the already-oriented source image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

SIDE = 512
MIN_SIDE = 320
CROP_SHARE = 0.6
INNER_MARGIN = 0.06
PIXEL_THRESHOLD = 0.5
MIN_IMAGE_GATE = 0.5


@dataclass(frozen=True)
class Crop:
    box: tuple[int, int, int, int]
    response: np.ndarray
    image_score: float
    rgb_sha256: str


def boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if min(width, height) < MIN_SIDE:
        return []
    w, h = math.ceil(CROP_SHARE * width), math.ceil(CROP_SHARE * height)
    return [(x, y, x + w, y + h) for y in (0, height - h) for x in (0, width - w)]


def validate(crops: list[Crop], width: int, height: int) -> None:
    if [c.box for c in crops] != boxes(width, height):
        raise ValueError("Missing, reordered or changed source crop geometry")
    for crop in crops:
        a = crop.response
        if (a.shape != (SIDE, SIDE) or a.dtype != np.float32 or not np.isfinite(a).all()
                or a.min() < 0 or a.max() > 1):
            raise ValueError("Invalid native crop map")
        if (isinstance(crop.image_score, bool) or not math.isfinite(crop.image_score)
                or not 0 <= crop.image_score <= 1):
            raise ValueError("Invalid uncalibrated crop response")
        if len(crop.rgb_sha256) != 64 or any(c not in "0123456789abcdef" for c in crop.rgb_sha256):
            raise ValueError("Invalid crop input commitment")


def predict(model, image: Image.Image) -> list[Crop]:
    import hashlib

    rgb = image.convert("RGB")
    out = []
    for box in boxes(*rgb.size):
        selected = rgb.crop(box)
        result = model.predict(selected)
        if result.source_size != selected.size:
            raise ValueError("Predictor returned different crop coordinates")
        out.append(Crop(box, result.raw_mask.copy(), result.image_score,
                        hashlib.sha256(np.asarray(selected).tobytes()).hexdigest()))
    validate(out, *rgb.size)
    return out


def stitch(crops: list[Crop], width: int, height: int, image_gate: float) -> np.ndarray:
    validate(crops, width, height)
    if (isinstance(image_gate, bool) or not math.isfinite(image_gate)
            or not MIN_IMAGE_GATE <= image_gate <= math.nextafter(1.0, math.inf)):
        raise ValueError("Invalid crop selection gate")
    output = np.zeros((height, width), dtype=np.float32)
    for crop in crops:
        if crop.image_score < image_gate:
            continue
        x0, y0, x1, y1 = crop.box
        projected = cv2.resize(crop.response, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        mx, my = math.ceil((x1 - x0) * INNER_MARGIN), math.ceil((y1 - y0) * INNER_MARGIN)
        if x0:
            projected[:, :mx] = 0
        if x1 < width:
            projected[:, -mx:] = 0
        if y0:
            projected[:my] = 0
        if y1 < height:
            projected[-my:, :] = 0
        output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], projected)
    return output


def complement(baseline: dict | None, extra: dict | None) -> tuple[dict | None, bool]:
    """A failed baseline is unassessed; an existing region is always preserved."""
    if baseline is None or baseline["predicted_positive_pixels"] or extra is None:
        return baseline, False
    if extra["predicted_positive_pixels"]:
        return extra, True
    return baseline, False


def choose_gate(originals: list[dict], displayed) -> dict:
    """Select only on authentic development controls, after actual display filtering.

    Evaluate each possible score transition. Cache a source's identical eligible
    crop sets, without assuming that component filtering is monotonic in area.
    """
    if len(originals) < 20 or len({r["source_id"] for r in originals}) != len(originals):
        raise ValueError("At least twenty distinct authentic source controls are required")
    if any(r["label"] != 0 or r["status"] != "ok" or r["baseline"] is None for r in originals):
        raise ValueError("Only complete authentic development controls may select the gate")
    budget = max(1, math.floor(0.025 * len(originals)))
    choices = {MIN_IMAGE_GATE}
    for row in originals:
        choices.update(math.nextafter(c.image_score, math.inf) for c in row["crops"]
                       if c.image_score >= MIN_IMAGE_GATE)
    cache = {}
    for gate in sorted(choices):
        additional = 0
        for row in originals:
            if row["baseline"]["predicted_positive_pixels"]:
                continue
            active = tuple(i for i, c in enumerate(row["crops"]) if c.image_score >= gate)
            key = (row["source_id"], active)
            if key not in cache:
                cache[key] = bool(active) and displayed(row, gate)
            additional += cache[key]
        if additional <= budget:
            return {"image_gate": gate, "additional_authentic_budget": budget,
                    "development_additional_flags": additional, "development_originals": len(originals),
                    "selection": "Smallest gate >=0.5 meeting the original-only displayed-region budget"}
    raise ValueError("No admissible gate, including the all-crops-withheld control")
