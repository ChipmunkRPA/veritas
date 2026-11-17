# SPDX-License-Identifier: Apache-2.0
"""Locate visible changes within the same picture position in retained PDF saves."""

from __future__ import annotations

from math import isfinite

from app.evidence import document_pdf_picture_binding as binding
from app.evidence.document_pdf_history import scan_saves
from app.evidence.document_pdf_history_binding import _sha

MAX_PAINT = 4096
MAX_NATIVE_PIXELS = 4_194_304


def unavailable(size, reason):
    return {"schema": binding.SCHEMA, "affects_overall_confidence": False, "source_bytes": size,
            "status": "unavailable", "reason": reason, "saves": [], "comparisons": []}


def _key(bounds):
    return tuple(round(value, 4) for value in bounds)


def _intersects(area, other):
    return (max(area[0], other[0]) < min(area[2], other[2])
            and max(area[1], other[1]) < min(area[3], other[3]))


def _eligible(page, info, paint, annotations):
    box = _key(info["bbox"])
    left, top, right, bottom = box
    a, b, c, d, _, _ = info["transform"]
    if (not all(isfinite(value) for value in (*box, a, b, c, d))
            or a <= 0 or d <= 0 or abs(b) > .001 or abs(c) > .001
            or not (0 <= left < right <= page.cropbox.width and 0 <= top < bottom <= page.cropbox.height)
            or info.get("has-mask") or info["width"] * info["height"] > MAX_NATIVE_PIXELS):
        return False
    order = [i for i, (kind, rect) in enumerate(paint) if kind == "fill-image" and _key(rect) == box]
    if len(order) != 1:
        return False
    return not (any(_intersects(box, rect) for kind, rect in paint[order[0]+1:] if kind != "ignore-text")
                or any(_intersects(box, rect) for rect in annotations))


def _pictures(page):
    paint = page.get_bboxlog()
    if len(paint) > MAX_PAINT or sum(kind == "fill-image" for kind, _ in paint) > binding.MAX_PICTURES:
        return None
    images = page.get_image_info()
    if len(images) > binding.MAX_PICTURES:
        return None
    annotations = [tuple(item.rect) for item in (*page.annots(), *page.widgets())]
    result = {}
    for image in images:
        box = _key(image["bbox"])
        eligible = _eligible(page, image, paint, annotations)
        result[box] = eligible and box not in result
    return result


def _render(page, scale, pixels):
    import numpy as np
    import pymupdf

    clip = pymupdf.Rect([value / scale for value in pixels])
    rendered = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, colorspace=pymupdf.csRGB,
                               annots=False, alpha=False)
    if tuple(rendered.irect) != tuple(pixels):
        return None
    return np.frombuffer(rendered.samples, dtype=np.uint8).reshape(rendered.height, rendered.width, 3)


def _regions(original, submitted, metadata):
    import cv2
    import numpy as np

    from app.detectors.image.reference_compare import compare
    from app.detectors.image.reference_exact_jpeg import alignment_proof

    if np.array_equal(original, submitted):
        return [], 0
    proof = alignment_proof(original, submitted)
    if proof is None:
        return None, 0
    result, mask = compare(original, submitted)
    if result["status"] != "comparable" or mask is None:
        return None, 0
    if not result["regions"]:
        return [], 0
    x0, y0, _, _ = binding.raster_box(metadata["picture_box"], metadata["render_scale"])
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # Use the same component order as the pixel comparison. Numbering is fixed by the mask's scan order.
    by_stats = {tuple(int(x) for x in values): label for label, values in enumerate(stats[1:], 1)}
    rows = []
    for region in result["regions"]:
        left, top, width, height, area = (region[field] for field in ("x", "y", "w", "h", "selected_pixels"))
        label = by_stats[(left, top, width, height, area)]
        component = labels == label
        pixels = [x0 + left, y0 + top, x0 + left + width, y0 + top + height]
        digest = _sha(component[top:top+height, left:left+width].astype(np.uint8).tobytes())
        rows.append((metadata | {"region_pixels": pixels, "marked_pixels": area, "mask_sha256": digest,
                                 "matching_tiles": proof["exact_tiles"], "tested_tiles": proof["full_tiles"]}, component))
    return rows, result["additional_regions"]


def _plates(submitted, region, component, index, assets):
    from PIL import Image

    from app.evidence.image_embedded_pair_pixels import encode, overlay

    box = binding.raster_box(region["picture_box"], region["render_scale"])
    area = region["region_pixels"]
    current = Image.fromarray(submitted)
    crop = current.crop((area[0] - box[0], area[1] - box[1], area[2] - box[0], area[3] - box[1]))
    for role, image, pixels in (("crop", crop, area), ("overlay", overlay(current, component), box)):
        payload = encode(image)
        assets[f"picture{index}_{role}"] = (payload, {"media_type": "image/png", "sha256": _sha(payload),
                                                      "width": image.width, "height": image.height, "role": role,
                                                      "page": region["current_page"], "source_pixels": pixels})


def _compare_page(earlier, current, coverage, comparison, assets):
    from app.detectors.image.reference_compare import MIN_SIDE

    older, newer = _pictures(earlier), _pictures(current)
    if older is None or newer is None:
        coverage["skipped"] = 1
        return
    if not older and not newer:
        return
    rotation = current.rotation
    if earlier.rotation != rotation or _key(earlier.cropbox) != _key(current.cropbox):
        coverage["skipped"] = len(older.keys() | newer.keys())
        return
    earlier.set_rotation(0)
    current.set_rotation(0)
    try:
        for box in sorted(older.keys() | newer.keys()):
            coverage["skipped"] += 1
            if not older.get(box) or not newer.get(box):
                continue
            scale = 2 if max(box[2] - box[0], box[3] - box[1]) * 2 <= binding.MAX_EDGE else 1
            pixels = binding.raster_box(box, scale)
            if not all(MIN_SIDE <= pixels[i+2] - pixels[i] <= binding.MAX_EDGE for i in (0, 1)):
                continue
            old, new = (_render(page, scale, pixels) for page in (earlier, current))
            if old is None or new is None:
                continue
            metadata = {"earlier_page": coverage["earlier"], "current_page": coverage["current"],
                        "page_xref": current.xref, "page_rotation": rotation, "picture_box": list(box),
                        "render_scale": scale}
            regions, remaining = _regions(old, new, metadata)
            if regions is None:
                continue
            coverage["skipped"] -= 1
            coverage["compared"] += 1
            comparison["changed_regions"] += len(regions) + remaining
            for region, component in regions[:max(0, binding.MAX_CHANGES - len(comparison["changes"]))]:
                comparison["changes"].append(region)
                if assets is not None:
                    _plates(new, region, component, len(comparison["changes"]), assets)
    finally:
        earlier.set_rotation(rotation)
        current.set_rotation(rotation)


def inspect(data, *, render_assets=False):
    assets = {} if render_assets else None

    def compare(earlier, current, _):
        old, new = ({pdf.page_xref(index): index for index in range(len(pdf))} for pdf in (earlier, current))
        if len(old) != len(earlier) or len(new) != len(current):
            raise ValueError("Page identities are ambiguous in a retained save")
        comparison = {"earlier_version": earlier.version_count, "pages": [], "changed_regions": 0, "changes": [],
                      "unpaired_pages": [sorted(old[key] + 1 for key in old.keys() - new.keys()),
                                         sorted(new[key] + 1 for key in new.keys() - old.keys())]}
        for key in sorted(old.keys() & new.keys(), key=new.__getitem__):
            coverage = {"earlier": old[key] + 1, "current": new[key] + 1, "compared": 0, "skipped": 0}
            _compare_page(earlier[old[key]], current[new[key]], coverage, comparison, assets)
            comparison["pages"].append(coverage)
        return comparison

    record = scan_saves(data, unavailable(len(data), None), compare, binding.changed, binding.incomplete)
    return record, assets


def previews(data, record):
    if record["status"] != "changed":
        return
    measured, assets = inspect(data, render_assets=True)
    if measured != record:
        raise ValueError("Saved picture measurements differ from the submitted PDF")
    for role, (payload, asset) in assets.items():
        yield role, payload, asset
