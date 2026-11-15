"""Compare a whole-pixel earlier crop after qualified WebP downsizing."""

import numpy as np

from app.detectors.image import (
    reference_crop_resize,
    reference_lossless_webp,
    reference_lossy_webp,
    reference_lossy_webp_crop,
    reference_lossy_webp_resize,
)

SCHEMA = "chipmunk.image-reference-comparison/22"
INTERPRETATION = ("Marked areas show brightness differences after matching the crop and size of the earlier copy. "
                  "The areas outside the matched crop were unassessed. Ordinary editing can cause these differences. "
                  "This comparison carries zero weight in the AI assessment.")


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    rw, rh = earlier.size
    sw, sh = submitted.size
    resized = reference_lossy_webp_resize
    if (not 512 <= min(rw, rh) <= max(rw, rh) <= 1024
            or not all(round(a * .8 * resized.MIN_SCALE) <= b < a for a, b in ((rw, sw), (rh, sh)))
            or not reference_lossless_webp.supported(earlier, earlier_bytes, png=True)
            or not reference_lossy_webp.supported(submitted, submitted_bytes, minimum_side=resized.MIN_SIDE)):
        return reference_lossy_webp.unavailable()
    bounds = reference_crop_resize.measure(np.asarray(earlier), np.asarray(submitted))
    if bounds is None:
        return reference_lossy_webp.unavailable()
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    if ((w, h) == (rw, rh)
            or any(a - b > reference_lossy_webp_crop.MAX_INSET or 5 * b < 4 * a for a, b in ((rw, w), (rh, h)))
            or not all(round(a * resized.MIN_SCALE) <= b <= round(a * resized.MAX_SCALE) for a, b in ((w, sw), (h, sh)))
            or abs(sw * h - sh * w) > max(w, h)):
        return reference_lossy_webp.unavailable()
    with earlier.crop((x, y, x+w, y+h)) as crop:
        result, mask = resized.compare_resized(crop, submitted)
    if result["status"] != "comparable":
        return result, mask
    result["lossy_webp_crop_resize"] = {"reference_crop": bounds, "projection": result.pop("lossy_webp_resize")}
    for region in result["regions"]:
        region["reference_region"]["x"] += x
        region["reference_region"]["y"] += y
    return result, mask
