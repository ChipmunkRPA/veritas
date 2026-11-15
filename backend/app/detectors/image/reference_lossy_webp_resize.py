"""Confirm shared framing after downsampling an earlier lossless image."""

from __future__ import annotations

import hashlib

import numpy as np

from app.detectors.image import (
    reference_compare,
    reference_lossless_webp,
    reference_lossy_webp,
    reference_resize,
)

SCHEMA = "chipmunk.image-reference-comparison/20"
MIN_SIDE = 384
MIN_SCALE = .72
MAX_SCALE = .95
LIMITATION = ("The earlier copy needs matching framing, proportional downsampling and enough shared detail "
              "after allowing for WebP compression.")
INTERPRETATION = ("Marked areas have local brightness differences that persisted across resize filters. The mask is in the "
                  "submitted image's coordinates. The crops show decoded pixels from each native file. Ordinary editing can "
                  "cause these differences. This comparison carries zero weight in the AI assessment.")


def compare(earlier, submitted, earlier_bytes, submitted_bytes):
    rw, rh = earlier.size
    sw, sh = submitted.size
    # One pixel of dimension rounding is allowed. Rescaling the two axes independently is outside scope.
    if (not 512 <= min(rw, rh) <= max(rw, rh) <= 1024
            or not all(round(a * MIN_SCALE) <= b <= round(a * MAX_SCALE) for a, b in ((rw, sw), (rh, sh)))
            or abs(sw * rh - sh * rw) > max(rw, rh)
            or not reference_lossless_webp.supported(earlier, earlier_bytes, png=True)
            or not reference_lossy_webp.supported(submitted, submitted_bytes, minimum_side=MIN_SIDE)):
        return reference_lossy_webp.unavailable()
    votes = np.zeros((sh, sw), np.uint8)
    questioned = np.asarray(submitted)
    supporters = []
    for kernel in reference_resize.KERNELS:
        with earlier.resize(submitted.size, kernel) as projected:
            result, mask = reference_lossy_webp.compare_pixels(np.asarray(projected), questioned)
            if result["status"] != "comparable":
                continue
            supporters.append({"kernel": kernel.name.lower(), **result["lossy_webp"]})
            if mask is not None:
                votes += mask
    if len(supporters) < reference_resize.MIN_VOTES:
        return reference_lossy_webp.unavailable()
    fraction = reference_lossy_webp.METHOD["minimum_region_area_fraction"]
    regions, mask, omitted = reference_compare._regions(votes >= reference_resize.MIN_VOTES, round(votes.size * fraction))
    for region in regions:
        region["reference_region"] = reference_resize.native_crop(region, earlier.size, submitted.size)
    return {"status": "comparable", "regions": regions, "additional_regions": omitted,
            "affects_overall_confidence": False, "lossy_webp_resize": {
                "submitted_over_reference_xy": [[sw, rw], [sh, rh]],
                "earlier_rgb_sha256": hashlib.sha256(earlier.tobytes()).hexdigest(),
                "supported_kernels": supporters,
            }}, mask
