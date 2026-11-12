# SPDX-License-Identifier: Apache-2.0
"""Conservatively localize differences between a JPEG and its embedded preview."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.reference_compare import _regions, _tone
from app.provenance.exiftool import _pillow_thumbnail

MIN_SIDE, MAX_SIDE = 64, 512
MAX_MAIN_PIXELS = 24_000_000
MAX_ASPECT_DELTA = .028
MAX_SHIFT = 1.25
MIN_PHASE_RESPONSE = .42
MIN_BACKGROUND = .86
MAX_SELECTED_FRACTION = .25
PILOT_COMPONENT_FRACTION = .0012
CONFIRMATION_COMPONENT_FRACTION = .004
KERNELS = (Image.Resampling.LANCZOS, Image.Resampling.BOX)
ROTATIONS = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}


def _trim(image):
    """Find dark, uniform camera padding on the thumbnail canvas."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    top, bottom, left, right = 0, h, 0, w
    while top < h // 4 and gray[top].mean() < 16 and gray[top].std() < 4:
        top += 1
    while bottom > h - h // 4 and gray[bottom - 1].mean() < 16 and gray[bottom - 1].std() < 4:
        bottom -= 1
    while left < w // 4 and gray[top:bottom, left].mean() < 16 and gray[top:bottom, left].std() < 4:
        left += 1
    while right > w - w // 4 and gray[top:bottom, right - 1].mean() < 16 and gray[top:bottom, right - 1].std() < 4:
        right -= 1
    return image[top:bottom, left:right], (left, top, right, bottom)


def _preview(blob, native_size, operation=0):
    with Image.open(io.BytesIO(blob)) as image:
        if image.format != "JPEG" or min(image.size) < MIN_SIDE or max(image.size) > MAX_SIDE:
            return None
        rgb = image.convert("RGB")
        image = np.asarray(rgb.transpose(ROTATIONS[operation]) if operation else rgb)
    native_aspect = native_size[0] / native_size[1]
    if abs((image.shape[1] / image.shape[0]) / native_aspect - 1) <= MAX_ASPECT_DELTA:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)
    image, trim = _trim(image)
    if min(image.shape[:2]) < MIN_SIDE or abs((image.shape[1] / image.shape[0]) / native_aspect - 1) > MAX_ASPECT_DELTA:
        return None
    return image, trim


def _align(before, after):
    gray_before, gray_after = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) for image in (before, after))
    if min(gray_before.std(), gray_after.std()) < 8.:
        return None
    h, w = gray_before.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(gray_before, gray_after, window)
    if not np.isfinite((dx, dy, response)).all() or response < MIN_PHASE_RESPONSE or max(abs(dx), abs(dy)) > MAX_SHIFT:
        return None
    transform = np.array(((1, 0, dx), (0, 1, dy)), dtype=np.float32)
    shifted = cv2.warpAffine(before, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return shifted, (float(dx), float(dy), float(response))


def _differences(before, after):
    # Adjust color globally in the majority of pixels, then test for a local residual.
    before, after = (cv2.GaussianBlur(image.astype(np.float32), (0, 0), .7) for image in (before, after))
    for index in range(3):
        gain, offset = _tone(before[:, :, index], after[:, :, index])
        before[:, :, index] = np.clip(before[:, :, index] * gain + offset, 0, 255)
    lab_before, lab_after = (cv2.cvtColor(image / 255., cv2.COLOR_RGB2LAB) for image in (before, after))
    diff = lab_after - lab_before
    luma = np.abs(diff[:, :, 0]) * 2.55
    chroma = np.hypot(diff[:, :, 1], diff[:, :, 2]) * 1.65
    detail = np.abs(cv2.Laplacian(lab_after[:, :, 0], cv2.CV_32F))
    return luma, chroma, detail


def _over(residual, detail, minimum):
    center = np.median(residual[3:-3, 3:-3])
    spread = np.median(np.abs(residual[3:-3, 3:-3] - center)) * 1.4826
    point = max(minimum, float(center + 5 * max(1.1, spread)))
    return residual > point + .25 * detail, point


def _mask(selected, minimum_component_fraction):
    selected[:3], selected[-3:], selected[:, :3], selected[:, -3:] = False, False, False, False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    selected = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    if selected.mean() > MAX_SELECTED_FRACTION:
        return [], np.zeros(selected.shape, dtype=bool)
    regions, mask, _ = _regions(selected, max(12, round(selected.size * minimum_component_fraction)))
    return regions, mask


def _mapped(regions, trim, thumbnail_size, native_size):
    left, top = trim[:2]
    tw, th = thumbnail_size
    nw, nh = native_size
    for region in regions:
        x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
        sx, sy = x * nw // tw, y * nh // th
        right = ((x + w) * nw + tw - 1) // tw
        bottom = ((y + h) * nh + th - 1) // th
        yield {"embedded_thumbnail": {"x": x + left, "y": y + top, "w": w, "h": h},
               "submitted": {"x": sx, "y": sy, "w": right - sx, "h": bottom - sy},
               "selected_thumbnail_pixels": region["selected_pixels"]}


def _projected(main, before, trim, native_size, minimum_component_fraction):
    h, w = before.shape[:2]
    supported, controls, support = [], [], []
    for kernel in KERNELS:
        after = np.asarray(main.resize((w, h), kernel))
        alignment = _align(before, after)
        if alignment is None:
            return {"status": "alignment_unresolved"}, None
        registered, shift = alignment
        luma, chroma, detail = _differences(registered, after)
        background = float(np.mean(luma[3:-3, 3:-3] <= 12.))
        if background < MIN_BACKGROUND:
            return {"status": "insufficient_common_background"}, None
        bright, luma_point = _over(luma, detail, 9.)
        color, chroma_point = _over(chroma, detail, 10.)
        supported.append(bright | color)
        controls.append(bright)
        support.append({"shift_and_response": shift, "background": background, "points": [luma_point, chroma_point]})
    regions, mask = _mask(supported[0] & supported[1], minimum_component_fraction)
    luma_regions, luma_mask = _mask(controls[0] & controls[1], minimum_component_fraction)
    return {"status": "comparable", "native_size": native_size, "trim": trim, "thumbnail_size": (w, h),
            "regions": list(_mapped(regions, trim, (w, h), native_size)),
            "luma_regions": list(_mapped(luma_regions, trim, (w, h), native_size)), "support": support,
            "confidence_weight": 0}, (mask, luma_mask)


def measure(data: bytes, *, minimum_component_fraction: float = PILOT_COMPONENT_FRACTION):
    """Measure a compact difference only after both thumbnail renderings agree."""
    with Image.open(io.BytesIO(data)) as original:
        raw = original.size
        exif = original.getexif()
        orientation = exif.get(274, 1)
        if (original.format != "JPEG" or raw[0] * raw[1] > MAX_MAIN_PIXELS or min(raw) < 160
                or orientation not in (1, *ROTATIONS)):
            return {"status": "unsupported_carrier"}, None
        blob = _pillow_thumbnail(original, exif)
        if blob is None:
            return {"status": "no_retained_thumbnail"}, None
        native_size = raw[::-1] if orientation in (6, 8) else raw
        if orientation == 1:
            proof = _preview(blob, native_size)
            if proof is None:
                return {"status": "unsupported_thumbnail"}, None
            before, trim = proof
            h, w = before.shape[:2]
            original.draft("RGB", (w * 3, h * 3))
            main = original.convert("RGB")
            return _projected(main, before, trim, native_size, minimum_component_fraction)

        with Image.open(io.BytesIO(blob)) as preview:
            preview_size = preview.size
        proofs = [(operation, proof) for operation in (0, orientation)
                  if (proof := _preview(blob, native_size, operation)) is not None]
        if not proofs:
            return {"status": "unsupported_thumbnail"}, None
        edge = max(preview_size)
        original.draft("RGB", tuple(round(3 * edge * n / max(raw)) for n in raw))
        main = original.convert("RGB").transpose(ROTATIONS[orientation])

    matches = []
    for operation, (before, trim) in proofs:
        result, masks = _projected(main, before, trim, native_size, minimum_component_fraction)
        if masks is not None:
            matches.append(({**result, "main_orientation": orientation, "preview_orientation": operation,
                             "preview_raw_size": preview_size}, masks))
    if len(matches) != 1:
        return {"status": "ambiguous_preview_orientation" if matches else "alignment_unresolved"}, None
    return matches[0]
