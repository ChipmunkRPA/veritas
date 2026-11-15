"""Same-framing comparison of a lossy WebP with an earlier lossless RGB copy."""

from __future__ import annotations

import hashlib
from collections import Counter

import cv2
import numpy as np

from app.detectors.image import reference_compare, reference_lossless_webp

SCHEMA = "chipmunk.image-reference-comparison/19"
UNASSESSED = "webp_lossy_delivery_unassessed"
TILE = 32
METHOD = {
    "name": "low_frequency_luminance_difference",
    "sigma_px": 2.5,
    "brightness_weights": [0.299, 0.587, 0.114],
    "maximum_matching_tile_residual": 4.,
    "region_residual": 6.,
    "minimum_region_area_fraction": .001,
}
LIMITATION = "This comparison needs matching framing with enough detailed areas that agree after allowing for WebP compression."
INTERPRETATION = ("Marked areas have local brightness differences after filtering small compression variations. Ordinary editing can "
                  "cause these differences. The comparison carries zero weight in the AI assessment.")


def webp_size(data: bytes) -> tuple[int, int] | None:
    """Recognize a simple opaque VP8 key picture and check its native size."""
    if (len(data) < 31 or data[:4] != b"RIFF" or data[8:16] != b"WEBPVP8 "
            or int.from_bytes(data[4:8], "little") != len(data) - 8 or data[23:26] != b"\x9d\x01\x2a"):
        return None
    size = int.from_bytes(data[16:20], "little")
    tag = int.from_bytes(data[20:23], "little")
    padding = size % 2
    version, partition = (tag >> 1) & 7, tag >> 5
    if (size < 11 or 20 + size + padding != len(data) or (padding and data[-1] != 0)
            or tag & 1 or version > 3 or not tag & 16 or not 0 < partition < size - 10):
        return None
    w, h = (int.from_bytes(data[index:index + 2], "little") for index in (26, 28))
    if not 512 <= min(w, h) <= max(w, h) <= 1024:
        return None
    return w, h


def supported(image, body):
    return (image.format == "WEBP" and image.mode == "RGB" and getattr(image, "n_frames", 1) == 1
            and webp_size(body) == image.size and image.getexif().get(274, 1) == 1)


def _smooth(rgb):
    return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), METHOD["sigma_px"])


def _blocks(layer):
    rows, columns = (size // TILE for size in layer.shape[:2])
    return layer[:rows*TILE, :columns*TILE].reshape(rows, TILE, columns, TILE, *layer.shape[2:]).swapaxes(1, 2)


def _framing(reference, difference):
    blocks = _blocks(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY))
    textured = np.ptp(blocks, axis=(-2, -1)) >= 24
    errors = _blocks(difference).mean(axis=(-2, -1))
    agreed = textured & (errors <= METHOD["maximum_matching_tile_residual"])
    rows, columns = agreed.shape
    zones = Counter((int(y) * 3 // rows, int(x) * 3 // columns) for y, x in np.argwhere(agreed))
    supported_zones = [list(zone) for zone, count in sorted(zones.items()) if count >= 2]
    matched, total = int(agreed.sum()), int(textured.sum())
    if (matched < 24 or 20 * matched < 13 * total or len(supported_zones) < 6
            or any(len({p[axis] for p in supported_zones}) != 3 for axis in (0, 1))):
        return None
    return {"textured_tiles": total, "matching_tiles": matched,
            "tile_side_px": TILE, "supported_zones": supported_zones,
            "median_tile_residual": round(float(np.median(errors[textured])), 6),
            "earlier_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest()}


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    inaccessible = {"status": UNASSESSED, "regions": [], "affects_overall_confidence": False}
    if (not reference_lossless_webp.supported(earlier, earlier_bytes, png=True) or not supported(submitted, submitted_bytes)
            or earlier.size != submitted.size):
        return inaccessible, None
    reference, questioned = np.asarray(earlier), np.asarray(submitted)
    shift, response = reference_compare._alignment(reference, questioned)
    if response < reference_compare.MIN_REGISTRATION_RESPONSE or max(map(abs, shift)) > reference_compare.MAX_NATIVE_SHIFT:
        return inaccessible, None
    delta = _smooth(questioned) - _smooth(reference)
    offset = np.median(delta[::8, ::8], axis=(0, 1))
    weighted = np.asarray(METHOD["brightness_weights"], np.float32)
    difference = np.abs((delta - offset) @ weighted)
    proof = _framing(reference, difference)
    changed = difference > METHOD["region_residual"]
    background = 1. - float(changed.mean())
    if proof is None or background < reference_compare.MIN_BACKGROUND_FRACTION:
        return inaccessible, None
    opened = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    regions, mask, omitted = reference_compare._regions(opened, round(changed.size * METHOD["minimum_region_area_fraction"]))
    return {"status": "comparable", "regions": regions, "additional_regions": omitted, "lossy_webp": proof,
            "affects_overall_confidence": False}, mask
