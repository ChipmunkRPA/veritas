"""Compare visible samples on a corroborated, same-framing PNG pair."""
from __future__ import annotations

import hashlib

import numpy as np

from app.detectors.image import reference_compare
from app.detectors.image import reference_png_exact_resize as exact

SCHEMA = "chipmunk.image-reference-comparison/17"
MAX_PIXELS = 4_000_000
UNASSESSED = "png_opacity_unassessed"
INTERPRETATION = (
    "This comparison measures changes to opacity and visible color samples in the supplied PNGs. "
    "Crops are shown on white. Ordinary editing can produce these differences."
)
UNASSESSED_INTERPRETATION = (
    "Local opacity comparison requires the same framing, matching color metadata and supported 8-bit PNGs. "
    "The opacity comparison did not measure this pair."
)


def supported_png(source, body):
    return (source.format == "PNG" and (source.mode, body[24:26]) in {("RGB", b"\x08\x02"), ("RGBA", b"\x08\x06")}
            and source.get_format_mimetype() == "image/png"
            and (source.mode == "RGBA" or "transparency" not in source.info))


def on_white(rgba):
    alpha = rgba[:, :, 3:].astype(np.uint16)
    return ((rgba[:, :, :3].astype(np.uint16) * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)


def compare(earlier, submitted):
    if earlier.shape != submitted.shape or min(submitted.shape[:2]) < 160 or submitted[:, :, 3].size > MAX_PIXELS:
        return None, None
    earlier_alpha, submitted_alpha = earlier[:, :, 3], submitted[:, :, 3]
    opacity = earlier_alpha != submitted_alpha
    # Select visible textured witnesses. Hidden RGB in an alpha-zero pixel
    # neither establishes alignment nor contributes to the change mask.
    display = on_white(submitted)
    tiles = [(row, col, bounds) for row, col, bounds in exact._textured_tiles(display)
             if (submitted_alpha[bounds] == 255).all()]
    visible_color = np.any(earlier[:, :, :3] != submitted[:, :, :3], axis=2) & (
        (earlier_alpha > 0) | (submitted_alpha > 0))
    changed = opacity | visible_color
    witnesses = [(row, col) for row, col, bounds in tiles if not changed[bounds].any()]
    rows, columns = np.flatnonzero(changed.any(axis=1)), np.flatnonzero(changed.any(axis=0))
    if (not exact._spread(witnesses) or 1 - changed.mean() < exact.MIN_GLOBAL_EQUAL
            or (rows.size and (columns[-1] + 1 - columns[0]) * (rows[-1] + 1 - rows[0])
                > exact.MAX_MISMATCH_BOX * changed.size)):
        return None, None
    regions, mask, omitted = reference_compare._regions(changed, max(48, round(changed.size * .0004)))
    return {
        "status": "comparable", "regions": regions, "additional_regions": omitted,
        "affects_overall_confidence": False,
        "png_opacity": {"opacity_changed_pixels": int(opacity.sum()), "matched_pixels": int((~changed).sum()),
                        "matched_tiles": [list(cell) for cell in witnesses],
                        "earlier_rgba_sha256": hashlib.sha256(earlier.tobytes()).hexdigest(),
                        "submitted_rgba_sha256": hashlib.sha256(submitted.tobytes()).hexdigest()},
    }, mask
