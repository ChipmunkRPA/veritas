"""Render a separate image comparison from the exact evidence returned to the caller."""

from __future__ import annotations

import pymupdf

from app.evidence.image_reference import RESIZE_SCHEMA

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


def _image(page, bounds, asset, assets):
    page.draw_rect(bounds, color=RULE, fill=(0.96, 0.97, 0.96))
    page.insert_image(bounds, stream=assets[asset["file"]], keep_proportion=True)


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
    if (alignment := comparison.get("alignment")) and "submitted_overlap" in alignment:
        _text(page, MARGIN, 302, f"Earlier copy overlap: {_area(alignment['reference_overlap'])}", size=10)
        _text(page, MARGIN, 318, f"Submitted image overlap: {_area(alignment['submitted_overlap'])}", size=10)
        _text(page, MARGIN, 334, "Areas outside these bounds were unassessed.", size=10, color=MUTED)
    if comparison["schema"] == RESIZE_SCHEMA and comparison["status"] == "comparable":
        _text(page, MARGIN, 302, "Same framing at different pixel dimensions", size=10, font="hebo")
        _text(page, MARGIN, 318, "Differences were measured on the submitted pixel grid after resizing the earlier copy.", size=10)
        _text(page, MARGIN, 334, "The displayed crops come directly from their respective original files.", size=10, color=MUTED)
        if regions:
            _text(page, MARGIN, 350, "Retained pixels were flagged by at least two tested resizes.", size=10, color=MUTED)
    if mask := comparison.get("mask"):
        _text(page, MARGIN, 366, "Difference mask in submitted image coordinates", font="hebo")
        _text(page, MARGIN, 382, "White marks retained difference pixels.", size=10, color=MUTED)
        _image(page, pymupdf.Rect(MARGIN, 397, RIGHT, 693), mask, assets)
        _text(page, MARGIN, 711, f"Mask SHA-256 {mask['sha256']}", size=9, font="cour", color=MUTED)


def render(comparison: dict, assets: dict[str, bytes]) -> bytes:
    """Render native source crops and mask without mutating the saved analysis."""
    with pymupdf.open() as document:
        document.set_metadata({"title": "Earlier-copy image comparison", "author": "Chipmunk Verify"})
        _summary(document, comparison, assets)
        for index, region in enumerate(comparison["regions"], 1):
            page = _page(document, f"Difference region {index}", "Coordinates use each image's oriented native pixels.")
            for x, right, label, bounds, asset, hash_y in (
                (MARGIN, 296, "Earlier copy", region.get("reference_region", region), region["reference_crop"], 537),
                (316, RIGHT, "Submitted image", region, region["submitted_crop"], 582),
            ):
                _text(page, x, 153, label, font="hebo")
                _text(page, x, 170, _area(bounds), size=10, color=MUTED)
                _image(page, pymupdf.Rect(x, 186, right, 496), asset, assets)
                _text(page, MARGIN, hash_y, f"{label} crop SHA-256", size=10, font="hebo")
                _text(page, MARGIN, hash_y + 16, asset["sha256"], size=9, font="cour", color=MUTED)
            _text(page, MARGIN, 632, f"Retained difference pixels: {region['selected_pixels']}", size=10, color=MUTED)
        return document.tobytes(deflate=True, garbage=4)
