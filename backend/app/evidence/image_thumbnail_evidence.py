"""Publish registered embedded-preview differences as source-bound image exhibits."""

from __future__ import annotations

import io
import math
import struct

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.engine.exhibits import ExhibitBudgetExceeded
from app.provenance import image_thumbnail_localization as localization

# Keep this in the render stage namespace. On resume the shared writer starts
# with empty counters even when the earlier detector exhibits are reused.
CREATOR = "render"
TABLE = "Embedded preview difference locations"
MAIN = "Main image crop for preview difference"
PREVIEW = "Embedded preview crop for difference"
FOOTPRINT = "Measured preview difference footprint"
REGIONS = "Main image preview differences"
LEGACY_COLUMNS = ["Region", "Main image x, y, width, height", "Embedded preview x, y, width, height", "Marked preview pixels",
                  "Main crop display width, height", "Preview crop display width, height"]
COLUMNS = [*LEGACY_COLUMNS, "Main and preview orientation"]
FOOTPRINT_COLUMNS = [*COLUMNS, "Aligned preview grid x, y, width, height"]
FOOTPRINT_NOTE = "White cells mark measured differences on the aligned preview grid."
INTERPRETATION = ("The file’s embedded preview and main image differ in these areas. The comparison uses the preview’s "
                  "lower resolution to locate native crops. Main coordinates follow the displayed orientation. Preview "
                  "coordinates refer to the stored preview pixels. Ordinary editing and image processing can also produce these "
                  "differences. The preview’s origin is unknown. This evidence carries zero AI authorship weight.")
MAX_REGIONS = 3
DISPLAY_EDGE = 384
BOX_KEYS = ("x", "y", "w", "h")


def _preview_bounds(region, measurement):
    x, y, w, h = (region["embedded_thumbnail"][key] for key in BOX_KEYS)
    left, top, right, bottom = measurement["trim"]
    shifts = [variant["shift_and_response"] for variant in measurement["support"]]
    x0 = max(left, math.floor(min(x - dx for dx, _, _ in shifts)))
    y0 = max(top, math.floor(min(y - dy for _, dy, _ in shifts)))
    x1 = min(right, math.ceil(max(x + w - 1 - dx for dx, _, _ in shifts)) + 1)
    y1 = min(bottom, math.ceil(max(y + h - 1 - dy for _, dy, _ in shifts)) + 1)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _stored_preview_bounds(region, measurement):
    box = _preview_bounds(region, measurement)
    orientation = measurement.get("preview_orientation", 0)
    if not orientation:
        return box
    width, height = measurement["preview_raw_size"]
    x, y, w, h = (box[key] for key in BOX_KEYS)
    if orientation in localization.SWAPPED_AXES:
        x, y, w, h = y, x, h, w
    if orientation in (2, 3, 7, 8):
        x = width - x - w
    if orientation in (3, 4, 6, 7):
        y = height - y - h
    return {"x": x, "y": y, "w": w, "h": h}


def _coordinates(box):
    return ", ".join(str(box[key]) for key in BOX_KEYS)


def _crop(image, box):
    x, y, w, h = (box[key] for key in BOX_KEYS)
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    reduced = max(crop.size) > DISPLAY_EDGE
    if reduced:
        crop.thumbnail((DISPLAY_EDGE, DISPLAY_EDGE), Image.Resampling.LANCZOS)
    with io.BytesIO() as output:
        crop.save(output, format="PNG")
        return output.getvalue(), crop.size


def _caption(box, display_size):
    x, y, w, h = (box[key] for key in BOX_KEYS)
    shown = f" Crop preview shown at {display_size[0]} × {display_size[1]} pixels." if display_size != (w, h) else ""
    return f"x {x}, y {y}, {w} × {h} pixels." + shown


def _footprint(region, trim, labels, stats):
    x, y, w, h = (region["embedded_thumbnail"][key] for key in BOX_KEYS)
    x, y = x - trim[0], y - trim[1]
    shape = np.array((x, y, w, h, region["selected_thumbnail_pixels"]))
    matches = np.flatnonzero(np.all(stats[1:] == shape, axis=1)) + 1
    if len(matches) != 1:
        return None
    mask = (labels[y:y + h, x:x + w] == matches[0]).astype(np.uint8) * 255
    with io.BytesIO() as output:
        Image.fromarray(mask).convert("RGB").save(output, format="PNG")
        return dict(zip(BOX_KEYS, (x, y, w, h), strict=True)), output.getvalue()


def add_exhibits(data, writer):
    """Keep local evidence separate from metadata anomalies and score inputs."""
    try:
        measurement, masks = localization.measure(data, minimum_component_fraction=localization.CONFIRMATION_COMPONENT_FRACTION)
        if not measurement.get("regions"):
            return []
        _, labels, stats, _ = cv2.connectedComponentsWithStats(masks[0].astype(np.uint8), connectivity=8)
        with Image.open(io.BytesIO(data)) as source:
            embedded = localization._pillow_thumbnail(source, source.getexif())
            if embedded is None:
                return []
            main = ImageOps.exif_transpose(source) if measurement.get("main_orientation") else source
            with Image.open(io.BytesIO(embedded)) as preview:
                prepared = []
                for region in measurement["regions"][:MAX_REGIONS]:
                    native = region["submitted"]
                    thumbnail = _stored_preview_bounds(region, measurement)
                    footprint = _footprint(region, measurement["trim"], labels, stats)
                    prepared.append((native, thumbnail, region["selected_thumbnail_pixels"], _crop(main, native), _crop(preview, thumbnail), footprint))
    except (OSError, ValueError, TypeError, SyntaxError, KeyError, IndexError, OverflowError, struct.error,
            Image.DecompressionBombError, cv2.error):
        # An unreadable preview must not change the existing metadata finding or score.
        return []

    exhibit_ids, rows, regions = [], [], []
    for n, (main, thumb, marked, main_crop, thumb_crop, footprint) in enumerate(prepared, 1):
        try:
            main_id = writer.image(main_crop[0], role="comparison", media_type="image/png", title=f"{MAIN} {n}",
                                   caption=_caption(main, main_crop[1]), media_ref="original")
            thumb_id = writer.image(thumb_crop[0], role="comparison", media_type="image/png", title=f"{PREVIEW} {n}",
                                    caption=_caption(thumb, thumb_crop[1]), media_ref="original")
        except ExhibitBudgetExceeded:
            break
        exhibit_ids.extend((main_id, thumb_id))
        footprint_location = ""
        if footprint is not None:
            box, pixels = footprint
            try:
                exhibit_ids.append(writer.image(pixels, role="comparison", media_type="image/png", title=f"{FOOTPRINT} {n}",
                                                caption=f"{marked} marked cells. {FOOTPRINT_NOTE} {_caption(box, (box['w'], box['h']))}",
                                                media_ref="original"))
                footprint_location = _coordinates(box)
            except ExhibitBudgetExceeded:
                pass
        rows.append([str(n), _coordinates(main), _coordinates(thumb), str(marked),
                     ", ".join(map(str, main_crop[1])), ", ".join(map(str, thumb_crop[1])),
                     f"{measurement.get('main_orientation', 1)}, {measurement.get('preview_orientation', 0)}", footprint_location])
        regions.append({"region": {"kind": "bbox", "bbox": main}, "label": f"Preview difference {n}"})
    if rows:
        exhibit_ids.append(writer.table(FOOTPRINT_COLUMNS, rows, title=TABLE, caption=INTERPRETATION))
        width, height = measurement["native_size"]
        exhibit_ids.append(writer.regions(regions, title=REGIONS, caption=INTERPRETATION,
                                          media_ref="original", coordinate_space={"width": width, "height": height, "scale_to_original": 1.0}))
    return exhibit_ids


def is_preview_exhibit(item):
    title = str(item.get("title", ""))
    return item.get("created_by") == CREATOR and (title in {TABLE, REGIONS} or title.startswith((f"{MAIN} ", f"{PREVIEW} ", f"{FOOTPRINT} ")))


def from_result(result):
    """Project the same immutable coordinate and crop exhibits into a JSON export."""
    exhibits = [item for item in result.get("exhibits") or [] if item.get("created_by") == CREATOR]
    table = next((item for item in exhibits if item.get("title") == TABLE and item.get("kind") == "table"
                  and item.get("columns") in (FOOTPRINT_COLUMNS, COLUMNS, LEGACY_COLUMNS)), None)
    if not table:
        return None
    crops = {item["title"]: item for item in exhibits if item.get("kind") == "image" and item.get("role") == "comparison"}
    rows, orientations = [], set()
    try:
        for region, main, thumb, marked, main_size, thumb_size, *extra in table["rows"]:
            if len(extra) != len(table["columns"]) - len(LEGACY_COLUMNS):
                return None
            main_tag, preview_alignment = map(int, extra[0].split(",")) if extra else (1, 0)
            if (main_tag not in (1, *localization.TRANSFORMS)
                    or (preview_alignment != 0 and (main_tag == 1 or preview_alignment != main_tag))):
                return None
            orientations.add((main_tag, preview_alignment))
            photos = {key: crops[f"{name} {region}"] for key, name in (("main_image", MAIN), ("embedded_preview", PREVIEW))}
            boxes = {key: dict(zip(BOX_KEYS, map(int, data.split(",")), strict=True))
                     for key, data in (("main_image", main), ("embedded_preview", thumb))}
            assets = {}
            for (key, exhibit), size in zip(photos.items(), (main_size, thumb_size), strict=True):
                width, height = map(int, size.split(","))
                assets[key] = {"exhibit_id": exhibit["id"], "sha256": exhibit["sha256"],
                               "url": f"/api/v2/exhibits/{exhibit['id']}/payload", "display_width": width, "display_height": height,
                               "display_reduced": (width, height) != (boxes[key]["w"], boxes[key]["h"])}
            row = {"region": int(region), **boxes, "marked_preview_pixels": int(marked), "crops": assets}
            if len(extra) == 2 and extra[1]:
                x, y, width, height = map(int, extra[1].split(","))
                footprint = crops[f"{FOOTPRINT} {region}"]
                if min(x, y) < 0 or min(width, height) <= 0 or not 0 < int(marked) <= width * height:
                    return None
                row["footprint_grid"] = {"coordinate_space": "aligned-preview-grid", "x": x, "y": y, "w": width, "h": height}
                assets["measured_footprint"] = {"exhibit_id": footprint["id"], "sha256": footprint["sha256"],
                                                "url": f"/api/v2/exhibits/{footprint['id']}/payload",
                                                "display_width": width, "display_height": height, "display_reduced": False}
            rows.append(row)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    if len(orientations) != 1:
        return None
    main_orientation, preview_orientation = next(iter(orientations))
    return {"source_sha256": (result.get("input") or {}).get("sha256"), "coordinate_exhibit_id": table["id"],
            "main_coordinate_space": "exif-oriented-main-image", "preview_coordinate_space": "stored-embedded-preview",
            "main_exif_orientation": main_orientation, "preview_alignment_orientation": preview_orientation,
            "regions": rows, "interpretation": table.get("caption", INTERPRETATION), "authorship_confidence_contribution": 0} if rows else None
