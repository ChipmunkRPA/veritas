"""Locate color stored in the fully transparent interior of an uploaded PNG."""
from __future__ import annotations

import io
from hashlib import sha256

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.engine.exhibits import ExhibitBudgetExceeded
from app.engine.media import refuse_truncated_images
from app.evidence import image_transparency_binding as binding

KEY = binding.KEY
CREATOR = KEY
TITLE = "Color stored under PNG transparency"
VISIBLE = "Crop as displayed on white"
HIDDEN = "Stored colors revealed in the same crop"
LOCATION_TITLE = "PNG transparency crop location"
SCOPE = ("Checks single-frame 8-bit RGBA PNGs up to 4,096 pixels per side and 4,194,304 pixels total. "
         "In tiles up to 128 × 128, the rule counts neighboring pixels that differ by at least 16 levels in any "
         "color channel. Both pixels must be fully transparent and outside the three-pixel border near any visible color. "
         "A tile qualifies at 32 pairs. Only the tile with the most qualifying pairs is shown.")
LIMITATION = ("Ordinary cutouts and editing tools can preserve colors under transparency. A highlighted area carries "
              "no weight in the AI assessment. Smooth or low-contrast hidden content may go undetected.")
SUMMARY = ["Width", "Height", "Fully transparent pixels", "Tiles with at least 32 qualifying pairs"]
LOCATION = ["Source x", "Source y", "Crop width", "Crop height", "Qualifying neighboring pairs"]


def measure(data):
    # IHDR identifies the exact color type and sample depth before decompression.
    if (len(data) > 24 * 1024 * 1024 or data[:8] != b"\x89PNG\r\n\x1a\n"
            or data[12:16] != b"IHDR" or data[24:26] != b"\x08\x06"):
        return None
    refuse_truncated_images()
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as source:
            if (source.width * source.height > binding.MAX_PIXELS or max(source.size) > binding.MAX_EDGE or source.mode != "RGBA"
                    or source.get_format_mimetype() == "image/apng"):
                return None
            with ImageOps.exif_transpose(source) as oriented:
                pixels = np.asarray(oriented)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None

    height, width = pixels.shape[:2]
    transparent = (pixels[:, :, 3] == 0).astype(np.uint8)
    count = int(transparent.sum())
    if not count:
        return (width, height, 0, 0), None
    # Ignore the three-pixel band around visible or partially visible pixels to
    # reduce ordinary color bleed at antialiased edges.
    interior = cv2.erode(transparent, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    tiles, best = 0, None
    for y in range(0, height, binding.TILE):
        for x in range(0, width, binding.TILE):
            selected = interior[y:y + binding.TILE, x:x + binding.TILE]
            if selected.sum() <= binding.MIN_PAIRS // 2:
                continue
            tile = pixels[y:y + binding.TILE, x:x + binding.TILE]
            colors = tile[:, :, :3].astype(np.int16)
            across = (np.abs(colors[:, 1:] - colors[:, :-1]) >= 16).any(axis=2) & (selected[:, 1:] & selected[:, :-1])
            down = (np.abs(colors[1:] - colors[:-1]) >= 16).any(axis=2) & (selected[1:] & selected[:-1])
            pairs = int(across.sum() + down.sum())
            if pairs >= binding.MIN_PAIRS:
                tiles += 1
                if best is None or pairs > best[0][-1]:
                    crop_height, crop_width = selected.shape
                    best = (x, y, crop_width, crop_height, pairs), tile, selected
    return (width, height, count, tiles), best


def crop_pixels(tile, interior):
    source = Image.fromarray(tile)
    shown = np.asarray(Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB"))
    revealed = np.full(tile.shape[:2] + (3,), 210, dtype=np.uint8)
    revealed[interior] = tile[:, :, :3][interior]
    return shown, revealed


def verify(data, record, read_asset):
    measured = measure(data)
    if measured is None or measured[0] != tuple(record[k] for k in binding.COUNTS):
        raise ValueError("PNG transparency coverage differs from the original")
    loc = record["location"]
    if loc is None:
        return False
    if measured[1] is None:
        raise ValueError("The original has no qualifying PNG transparency crop")
    (x, y, w, h, pairs), tile, interior = measured[1]
    if (x, y, w, h, pairs) != (*(loc["bounds"][k] for k in ("x", "y", "w", "h")), loc["qualifying_pairs"]):
        raise ValueError("PNG transparency location differs from the original")
    for role, match in zip(binding.ROLES, crop_pixels(tile, interior), strict=True):
        digest = loc[role]["sha256"]
        raw = read_asset(digest)
        if len(raw) > binding.MAX_ASSET_BYTES or sha256(raw).hexdigest() != digest:
            raise ValueError("PNG transparency crop differs from its declared bytes or size limit")
        with Image.open(io.BytesIO(raw), formats=["PNG"]) as image:
            if image.mode != "RGB" or image.size != (w, h) or not np.array_equal(np.asarray(image), match):
                raise ValueError("PNG transparency crop differs from the source rendering")
    return True


def add_exhibits(data, writer):
    measured = measure(data)
    if measured is None:
        return []
    summary, best = measured
    width, height, count, tiles = summary
    caption = (f"Checked {width * height:,} pixels. {count:,} are fully transparent. "
               f"{tiles:,} {'tile' if tiles == 1 else 'tiles'} met the color-variation rule. {LIMITATION}")
    ids = set()
    location = None
    if best is not None:
        location, tile, interior = best
        x, y, w, h, pairs = location
        coords = f"Source x {x}, y {y}, {w} × {h} px. {pairs:,} qualifying neighboring pairs."
        shown, revealed = crop_pixels(tile, interior)
        try:
            for title, pixels, note in ((VISIBLE, shown, "Composited on white."),
                                        (HIDDEN, revealed, "Only the fully transparent interior is revealed. Other pixels are gray.")):
                output = io.BytesIO()
                Image.fromarray(pixels).save(output, format="PNG")
                ids.add(writer.image(output.getvalue(), role="comparison", media_type="image/png",
                                     title=title, caption=f"{coords} {note}", media_ref="original"))
        except ExhibitBudgetExceeded:
            ids.clear()
            caption += " The image exhibit budget prevented saving the crop pair."
    ids.add(writer.table(SUMMARY, [list(summary)], title=TITLE, caption=f"{caption} {SCOPE}"))
    if len(ids) == 3 and location is not None:
        ids.add(writer.table(LOCATION, [list(location)], title=LOCATION_TITLE))
    return ids


def crops(record):
    location = record.get("location")
    return (location["visible_crop"], location["revealed_crop"]) if location else ()


def from_result(result):
    exhibits = {item["title"]: item for item in result.get("exhibits") or [] if item.get("created_by") == CREATOR}
    table = exhibits.get(TITLE)
    digest = (result.get("input") or {}).get("sha256")
    if not digest or not table or table.get("columns") != SUMMARY:
        return None
    (width, height, transparent, tiles), = table["rows"]
    crops = {}
    if (location := exhibits.get(LOCATION_TITLE)) and location.get("columns") == LOCATION:
        (x, y, w, h, pairs), = location["rows"]
        for role, title in (("visible_crop", VISIBLE), ("revealed_crop", HIDDEN)):
            item = exhibits[title]
            crops[role] = {"exhibit_id": item["id"], "sha256": item["sha256"], "title": title, "caption": item["caption"],
                           "url": f"/api/v2/exhibits/{item['id']}/payload"}
        crops["bounds"] = dict(zip(("x", "y", "w", "h"), map(int, (x, y, w, h)), strict=True))
        crops["qualifying_pairs"] = int(pairs)
    return {"schema": binding.SCHEMA, "source_sha256": digest, "width": int(width), "height": int(height),
            "fully_transparent_pixels": int(transparent), "qualifying_tiles": int(tiles), "location": crops or None,
            "scope": SCOPE, "limitation": LIMITATION, "authorship_confidence_contribution": 0}
