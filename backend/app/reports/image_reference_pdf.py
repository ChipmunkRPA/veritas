"""Render a separate image comparison from the exact evidence returned to the caller."""

from __future__ import annotations

import pymupdf

from app.detectors.image import (
    reference_exact_jpeg,
    reference_png_orientation,
    reference_png_oriented_crop,
    reference_png_oriented_resize,
    reference_png_screenshot,
    reference_same_jpeg,
    reference_shifted_jpeg,
    reference_structure_jpeg,
)
from app.evidence.image_reference import CROP_RESIZE_SCHEMA, EXACT_PNG_CROP_RESIZE_SCHEMA, RESIZE_SCHEMA
from app.evidence.image_reference_delivery_envelope import SCHEMA as JPEG_PNG_ENVELOPE_SCHEMA

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN = 42
RIGHT = PAGE_WIDTH - MARGIN
INK = (0.11, 0.18, 0.14)
MUTED = (0.31, 0.37, 0.33)
RULE = (0.81, 0.85, 0.82)
RESIZE_UNASSESSED = {
    "unsupported_resize_geometry": "Comparison unassessed. Image dimensions are outside the proportional resize limits.",
    "same_framing_unresolved": "Comparison unassessed. Matching framing could not be established after resizing.",
    "resize_delivery_unresolved": "Comparison unassessed. Too few resize comparisons agreed on the image delivery.",
}


def _text(page, x, y, text, *, size=11, font="helv", color=INK):
    page.insert_text((x, y), text, fontsize=size, fontname=font, color=color)


def _page(document, title, subtitle):
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _text(page, MARGIN, 52, "CHIPMUNK VERIFY", size=11, font="hebo")
    page.draw_line((MARGIN, 65), (RIGHT, 65), color=RULE)
    _text(page, MARGIN, 100, title, size=25, font="hebo")
    _text(page, MARGIN, 121, subtitle, size=11, color=MUTED)
    page.draw_line((MARGIN, 744), (RIGHT, 744), color=RULE)
    _text(page, MARGIN, 763, "Supplemental comparison | Pixel differences only", size=9, color=MUTED)
    _text(page, RIGHT - 33, 763, f"Page {len(document)}", size=9, color=MUTED)
    return page


def _area(region):
    return f"x {region['x']}, y {region['y']}, {region['w']} x {region['h']} px"


def _image(page, bounds, asset, assets, overlay=None):
    page.draw_rect(bounds, color=RULE, fill=(0.96, 0.97, 0.96))
    page.insert_image(bounds, stream=assets[asset["file"]], keep_proportion=True)
    if overlay:
        page.insert_image(bounds, stream=assets[overlay["file"]], keep_proportion=True)


def _summary(document, comparison, assets):
    regions = comparison["regions"]
    if comparison["status"] != "comparable":
        status = RESIZE_UNASSESSED.get(comparison["status"],
                                      "Comparison unassessed. The inputs did not meet the alignment or delivery checks.")
    elif regions:
        count = len(regions)
        status = f"{count} difference {'region' if count == 1 else 'regions'} retained."
    else:
        status = "Comparable images. No region met the difference check's selection rule."
    page = _page(document, "Earlier-copy image comparison", status)
    page.insert_textbox(
        (MARGIN, 141, RIGHT, 194), comparison["interpretation"], fontsize=11, fontname="helv", color=INK,
    )
    for y, label, source in ((207, "Earlier copy", comparison["reference"]),
                             (252, "Submitted image", comparison["submitted"])):
        _text(page, MARGIN, y, f"{label} | {source['width']} x {source['height']} px", font="hebo")
        _text(page, MARGIN, y + 16, f"SHA-256 {source['sha256']}", size=9, font="cour", color=MUTED)
    schema = comparison["schema"]
    if schema == reference_exact_jpeg.SCHEMA:
        proof = comparison["alignment"]
        _text(page, MARGIN, 302, "Same framing supported by exact JPEG block matches.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"{proof['exact_tiles']:,} of {proof['full_tiles']:,} full blocks matched after JPEG encoding.",
              size=10, color=MUTED)
        _text(page, MARGIN, 334, f"{proof['anchor_count']:,} unique detailed blocks supported alignment.", size=10, color=MUTED)
    if schema == reference_same_jpeg.SCHEMA:
        proof = comparison["alignment"]
        _text(page, MARGIN, 302, "Matching JPEG blocks support the same framing.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"{proof['exact_tiles']:,} of {proof['full_tiles']:,} complete decoded blocks match exactly.",
              size=10, color=MUTED)
        _text(page, MARGIN, 334, f"{proof['anchor_count']:,} unique detailed blocks support the alignment.", size=10, color=MUTED)
    if schema == reference_structure_jpeg.SCHEMA:
        proof = comparison["same_framing_structure"]
        aligned, detailed = len(proof["aligned_tiles"]), proof["detailed_tiles"]
        _text(page, MARGIN, 302, "Matching detail supports the same framing.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"{aligned} of {detailed} detailed areas matched across the two images.", size=10, color=MUTED)
        _text(page, MARGIN, 334, "Highlights require local color and structure changes under both tone fits.", size=10, color=MUTED)
    elif schema != reference_same_jpeg.SCHEMA and (alignment := comparison.get("alignment")) and "submitted_overlap" in alignment:
        label = "area compared" if schema == reference_shifted_jpeg.SCHEMA else "overlap"
        _text(page, MARGIN, 302, f"Earlier copy {label}: {_area(alignment['reference_overlap'])}", size=10)
        _text(page, MARGIN, 318, f"Submitted image {label}: {_area(alignment['submitted_overlap'])}", size=10)
        if schema == reference_shifted_jpeg.SCHEMA:
            count, total = alignment["exact_tiles"], alignment["full_tiles"]
            _text(page, MARGIN, 334, f"{count:,} of {total:,} complete JPEG blocks matched after whole-pixel alignment.", size=10)
            _text(page, MARGIN, 350, "Margins outside these bounds were unassessed.", size=10, color=MUTED)
        else:
            _text(page, MARGIN, 334, "Areas outside these bounds were unassessed.", size=10, color=MUTED)
    if schema == reference_png_orientation.SCHEMA:
        proof = comparison["exact_orientation"]
        label = reference_png_orientation.LABELS[proof["transform"]]
        _text(page, MARGIN, 302, f"Earlier copy aligned by {label}.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"Exact match on {proof['matched_pixels']:,} pixels and {len(proof['matched_tiles'])} textured tiles.",
              size=10, color=MUTED)
        _text(page, MARGIN, 334, "Each crop shows the pixels and orientation in its own source.", size=10, color=MUTED)
    if schema == reference_png_oriented_crop.SCHEMA:
        proof = comparison["exact_oriented_crop"]
        label = reference_png_orientation.LABELS[proof["transform"]]
        _text(page, MARGIN, 302, f"Earlier crop matched using {label}.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"Earlier copy overlap: {_area(proof['native_crop'])}", size=10)
        _text(page, MARGIN, 334, f"Exact match on {proof['matched_pixels']:,} pixels and {len(proof['matched_tiles'])} textured tiles.",
              size=10, color=MUTED)
        _text(page, MARGIN, 350, "Pixels outside this earlier area were unassessed.", size=10, color=MUTED)
    if schema == reference_png_oriented_resize.SCHEMA:
        proof = comparison["exact_oriented_resize"]
        label = reference_png_orientation.LABELS[proof["transform"]]
        _text(page, MARGIN, 302, f"Earlier crop matched after {label} and resizing.", size=10, font="hebo")
        _text(page, MARGIN, 318, f"Earlier copy overlap: {_area(proof['native_crop'])}", size=10)
        _text(page, MARGIN, 334, f"Exact match on {proof['matched_pixels']:,} pixels and {len(proof['matched_tiles'])} textured tiles.",
              size=10, color=MUTED)
        _text(page, MARGIN, 350, "Pixels outside this earlier area were unassessed.", size=10, color=MUTED)
    if schema == reference_png_screenshot.SCHEMA:
        proof = comparison["embedded_native_photo"]
        count = proof["exact_pixels"]
        _text(page, MARGIN, 294, "Earlier photo matched at native scale within the screenshot.", size=10, font="hebo")
        _text(page, MARGIN, 310, f"Earlier copy area: {_area(proof['reference_bounds'])}", size=10)
        _text(page, MARGIN, 326, f"Screenshot photo area: {_area(proof['submitted_bounds'])}", size=10)
        _text(page, MARGIN, 342, f"{count:,} exact matching pixels. Areas outside these bounds were unassessed.",
              size=10, color=MUTED)
        _text(page, MARGIN, 357, "Local edits or interface overlays can produce differences. AI origin remains unassessed.",
              size=10, color=MUTED)
    if schema in (RESIZE_SCHEMA, CROP_RESIZE_SCHEMA, EXACT_PNG_CROP_RESIZE_SCHEMA, JPEG_PNG_ENVELOPE_SCHEMA) and comparison["status"] == "comparable":
        crop = comparison.get("reference_crop_bounds")
        if crop:
            _text(page, MARGIN, 302, f"Earlier copy area used: {_area(crop)}", size=10, font="hebo")
            _text(page, MARGIN, 318, "The earlier area was resized to the submitted pixel grid for comparison.", size=10)
            _text(page, MARGIN, 334, "Pixels outside the earlier area were unassessed.", size=10, color=MUTED)
        else:
            _text(page, MARGIN, 302, "Same framing at different pixel dimensions", size=10, font="hebo")
            _text(page, MARGIN, 318, "Differences were measured on the submitted pixel grid after resizing the earlier copy.", size=10)
            _text(page, MARGIN, 334, "The displayed crops come directly from their respective original files.", size=10, color=MUTED)
        if schema == EXACT_PNG_CROP_RESIZE_SCHEMA:
            proof = comparison["exact_projection"]
            tiles = len(proof["matched_tiles"])
            kernels = ", ".join(kernel.replace("opencv_area", "area") for kernel in comparison["supported_resize_kernels"])
            _text(page, MARGIN, 350, f"Exact match on {proof['matched_pixels']:,} pixels and {tiles} textured tiles ({kernels}).",
                  size=10, color=MUTED)
        elif schema == JPEG_PNG_ENVELOPE_SCHEMA:
            _text(page, MARGIN, 350, "Compared with the expected range for resize, tone and compression.", size=10, color=MUTED)
        elif regions:
            _text(page, MARGIN, 350, "Retained pixels were flagged by at least two tested resizes.", size=10, color=MUTED)
    if mask := comparison.get("mask"):
        mask_top = 381 if schema == reference_png_screenshot.SCHEMA else 366
        _text(page, MARGIN, mask_top, "Difference mask in submitted image coordinates", font="hebo")
        _text(page, MARGIN, mask_top + 16, "White marks retained difference pixels.", size=10, color=MUTED)
        _image(page, pymupdf.Rect(MARGIN, mask_top + 31, RIGHT, 693), mask, assets)
        _text(page, MARGIN, 711, f"Mask SHA-256 {mask['sha256']}", size=9, font="cour", color=MUTED)


def render(comparison: dict, assets: dict[str, bytes]) -> bytes:
    """Render native source crops and mask without mutating the saved analysis."""
    with pymupdf.open() as document:
        document.set_metadata({"title": "Earlier-copy image comparison", "author": "Chipmunk Verify"})
        _summary(document, comparison, assets)
        for index, region in enumerate(comparison["regions"], 1):
            page = _page(document, f"Difference region {index}", "Coordinates use each image's oriented native pixels.")
            overlay = region.get("submitted_overlay")
            for x, right, label, bounds, asset, hash_y in (
                (MARGIN, 296, "Earlier copy", region.get("reference_region", region), region["reference_crop"], 585),
                (316, RIGHT, "Submitted image", region, region["submitted_crop"], 625),
            ):
                _text(page, x, 153, label, font="hebo")
                _text(page, x, 170, _area(bounds), size=10, color=MUTED)
                _image(page, pymupdf.Rect(x, 186, right, 353 if overlay else 496), asset, assets)
                _text(page, MARGIN, hash_y, f"{label} crop SHA-256", size=10, font="hebo")
                _text(page, MARGIN, hash_y + 16, asset["sha256"], size=9, font="cour", color=MUTED)
            if overlay:
                _text(page, 316, 379, "Submitted image with highlighted differences", size=10, font="hebo")
                _text(page, 316, 395, "Amber marks the selected pixels in this crop.", size=10, color=MUTED)
                _image(page, pymupdf.Rect(316, 407, RIGHT, 558), region["submitted_crop"], assets, overlay=overlay)
                _text(page, MARGIN, 379, f"Retained difference pixels: {region['selected_pixels']}", size=10)
                _text(page, MARGIN, 395, f"/regions/{index - 1}/submitted_overlay", size=9, font="cour", color=MUTED)
                _text(page, MARGIN, 665, "Difference overlay SHA-256", size=10, font="hebo")
                _text(page, MARGIN, 681, overlay["sha256"], size=9, font="cour", color=MUTED)
            else:
                _text(page, MARGIN, 670, f"Retained difference pixels: {region['selected_pixels']}", size=10, color=MUTED)
        return document.tobytes(deflate=True, garbage=4)
