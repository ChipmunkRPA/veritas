"""Locate and compare an unscaled crop delivered as a lossy WebP."""

from collections import Counter

import cv2
import numpy as np

from app.detectors.image import reference_lossless_webp, reference_lossy_webp

SCHEMA = "chipmunk.image-reference-comparison/21"
MIN_SIDE = 384
MAX_INSET = 128
MIN_ANCHORS = 5
INTERPRETATION = ("Marked areas have local brightness differences within the matched crop after filtering small compression "
                  "variations. Pixels outside the crop were unassessed. Ordinary editing can cause these differences. "
                  "This comparison carries zero weight in the AI assessment.")


def _detail(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), .8) - cv2.GaussianBlur(gray, (0, 0), 3.)


def _native_offset(reference, submitted):
    h, w = submitted.shape[:2]
    dy, dx = reference.shape[0] - h, reference.shape[1] - w
    original, crop = _detail(reference), _detail(submitted)
    anchors = {}
    side = 48
    # Independent regions prevent a locally edited patch from choosing the mapping.
    for row in range(3):
        for column in range(3):
            y, x = (2 * row + 1) * h // 6 - side // 2, (2 * column + 1) * w // 6 - side // 2
            patch = crop[y:y+side, x:x+side]
            if patch.std() < 2.:
                continue
            search = original[y:y+dy+side, x:x+dx+side]
            scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (left, top) = cv2.minMaxLoc(scores)
            scores[top, left] = -1.
            if peak >= .86 and peak - float(scores.max()) >= .008:
                anchors[row, column] = left, top
    if not anchors:
        return None
    ranked = Counter(anchors.values()).most_common()
    offset, count = ranked[0]
    zones = [list(zone) for zone, value in anchors.items() if value == offset]
    if (count < MIN_ANCHORS or (len(ranked) > 1 and ranked[1][1] >= 2)
            or any(len({zone[axis] for zone in zones}) < 2 for axis in (0, 1))):
        return None
    left, top = offset
    return left, top, zones


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    rw, rh = earlier.size
    sw, sh = submitted.size
    if (earlier.size == submitted.size
            or any(not 0 <= a-b <= MAX_INSET or 5*b < 4*a for a, b in ((rw, sw), (rh, sh)))
            or not reference_lossless_webp.supported(earlier, earlier_bytes, png=True)
            or not reference_lossy_webp.supported(submitted, submitted_bytes, minimum_side=MIN_SIDE)):
        return reference_lossy_webp.unavailable()
    original, questioned = np.asarray(earlier), np.asarray(submitted)
    match = _native_offset(original, questioned)
    if match is None:
        return reference_lossy_webp.unavailable()
    x, y, zones = match
    result, mask = reference_lossy_webp.compare_pixels(original[y:y+sh, x:x+sw], questioned)
    if result["status"] != "comparable":
        return result, mask
    result["lossy_webp_crop"] = {
        "reference_crop": {"x": x, "y": y, "w": sw, "h": sh},
        "anchor_zones": zones,
        "framing": result.pop("lossy_webp"),
    }
    for region in result["regions"]:
        region["reference_region"] = {"x": region["x"] + x, "y": region["y"] + y, "w": region["w"], "h": region["h"]}
    return result, mask
