"""Measure local pixel changes against an independently supplied aligned image."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

MAX_PIXELS = 16_777_216
MIN_SIDE = 64
MAX_REGISTRATION_SIDE = 512
MAX_NATIVE_SHIFT = .75
MIN_REGISTRATION_RESPONSE = .20
MIN_BACKGROUND_FRACTION = .80
MAX_REGIONS = 5
PRIOR_JPEG_QUALITIES = (80, 88, 96)
_INVERSE_ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_270,
}


def _alignment(original: np.ndarray, submitted: np.ndarray) -> tuple[list[float], float]:
    height, width = original.shape[:2]
    scale = min(1., MAX_REGISTRATION_SIDE / max(height, width))
    size = (max(16, round(width * scale)), max(16, round(height * scale)))

    def grayscale(rgb):
        reduced = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY).astype(np.float32)

    left, right = grayscale(original), grayscale(submitted)
    if min(float(left.std()), float(right.std())) < 1:
        return [0., 0.], 0.
    shift, response = cv2.phaseCorrelate(left, right, cv2.createHanningWindow(size, cv2.CV_32F))
    return [float(shift[0] * width / size[0]), float(shift[1] * height / size[1])], float(response)


def _tone(reference: np.ndarray, examined: np.ndarray) -> tuple[float, float]:
    x, y = (image[::8, ::8].ravel().astype(np.float64) for image in (reference, examined))
    keep = np.ones(x.size, dtype=bool)
    gain = 1.
    offset = 0.
    for _ in range(3):
        a, b = x[keep], y[keep]
        centered = a - a.mean()
        variance = float(centered @ centered)
        gain = float(np.clip(centered @ (b - b.mean()) / variance, .55, 1.8)) if variance > len(a) else 1.
        offset = float(np.median(b - gain * a))
        residual = np.abs(y - (x * gain + offset))
        keep = residual <= max(2., 2.5 * float(np.median(residual)))
    return gain, offset


def _regions(candidates: np.ndarray, area_floor: int) -> tuple[list[dict], np.ndarray, int]:
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    keep = sorted((i for i, row in enumerate(stats) if i and row[4] >= area_floor), key=lambda i: (-stats[i, 4], i))
    mask = np.zeros(candidates.shape, bool)
    regions = []
    for component in keep[:MAX_REGIONS]:
        x, y, w, h, area = (int(value) for value in stats[component])
        mask |= labels == component
        regions.append({"x": x, "y": y, "w": w, "h": h, "selected_pixels": area})
    return regions, mask, max(0, len(keep) - MAX_REGIONS)


def compare(original: np.ndarray, submitted: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Compare without moving source pixels. Refuse pairs lacking measurable alignment."""
    if original.ndim != 3 or submitted.ndim != 3 or original.shape[2] != 3 or submitted.shape[2] != 3:
        raise ValueError("Both inputs must be RGB images")
    if original.dtype != np.uint8 or submitted.dtype != np.uint8:
        raise ValueError("Both inputs must use 8-bit RGB")
    if original.shape != submitted.shape:
        return {"status": "different_dimensions", "regions": []}, None
    height, width = original.shape[:2]
    if min(height, width) < MIN_SIDE or height * width > MAX_PIXELS:
        return {"status": "unsupported_dimensions", "regions": []}, None
    if np.array_equal(original, submitted):
        return {"status": "comparable", "background_fraction": 1., "regions": []}, np.zeros((height, width), bool)

    shift, response = _alignment(original, submitted)
    if response < MIN_REGISTRATION_RESPONSE or max(map(abs, shift)) > MAX_NATIVE_SHIFT:
        return {"status": "unaligned", "estimated_native_shift": shift, "registration_response": response, "regions": []}, None

    difference = np.zeros((height, width), dtype=np.float32)
    for channel in range(3):
        before = cv2.GaussianBlur(original[:, :, channel].astype(np.float32), (0, 0), 1.25)
        after = cv2.GaussianBlur(submitted[:, :, channel].astype(np.float32), (0, 0), 1.25)
        gain, offset = _tone(before, after)
        expected = np.clip(before * gain + offset, 0, 255)
        np.maximum(difference, np.abs(after - expected), out=difference)
    background = float(np.mean(difference <= 8.))
    result = {"status": "comparable", "estimated_native_shift": shift, "registration_response": response,
              "background_fraction": background, "regions": []}
    if background < MIN_BACKGROUND_FRACTION:
        return {**result, "status": "insufficient_common_background"}, None

    threshold = max(5., 2.5 * float(np.percentile(difference, 65)))
    candidates = (difference > threshold).astype(np.uint8)
    opened = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    connected = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    regions, mask, omitted = _regions(connected, max(48, round(width * height * .0004)))
    return {**result, "regions": regions, "residual_review_point": threshold, "additional_regions": omitted}, mask


def _codec(original: np.ndarray, carrier: Image.Image) -> tuple[np.ndarray, bool]:
    """Apply the questioned JPEG's quantization and sampling to the reference."""
    if carrier.format != "JPEG" or carrier.mode not in {"RGB", "L"} or not carrier.quantization:
        return original, False
    sampling = JpegImagePlugin.get_sampling(carrier)
    if carrier.mode == "RGB" and sampling not in {0, 1, 2}:
        return original, False
    orientation = int(carrier.getexif().get(274, 1) or 1)
    exif = Image.Exif()
    exif[274] = orientation
    image = Image.fromarray(original)
    if orientation in _INVERSE_ORIENTATION:
        image = image.transpose(_INVERSE_ORIENTATION[orientation])
    if carrier.mode == "L":
        image = image.convert("L")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", qtables=carrier.quantization, subsampling=max(0, sampling), exif=exif)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return np.array(ImageOps.exif_transpose(encoded).convert("RGB")), True


def compare_delivered(original: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict, np.ndarray | None]:
    """Account for whole-image tone and supported JPEG delivery before local comparison."""
    if (original.shape != submitted.shape or min(original.shape[:2]) < MIN_SIDE
            or original.shape[0] * original.shape[1] > MAX_PIXELS or np.array_equal(original, submitted)):
        return compare(original, submitted)
    expected, codec_matched = _codec(original, carrier)
    if np.array_equal(expected, submitted):
        result, mask = compare(expected, submitted)
        return {**result, "jpeg_delivery_modeled": codec_matched, "global_tone_adjusted": False}, mask

    corrected = original.copy()
    adjusted = False
    for channel in range(3):
        before = cv2.GaussianBlur(expected[:, :, channel].astype(np.float32), (0, 0), 1.8)
        after = cv2.GaussianBlur(submitted[:, :, channel].astype(np.float32), (0, 0), 1.8)
        gain, offset = _tone(before, after)
        if abs(gain - 1) > .035 or abs(offset) > 3.:
            corrected[:, :, channel] = np.clip(np.rint(original[:, :, channel] * gain + offset), 0, 255).astype(np.uint8)
            adjusted = True
    if adjusted:
        expected, codec_matched = _codec(corrected, carrier)
    result, mask = compare(expected, submitted)
    return {**result, "jpeg_delivery_modeled": codec_matched, "global_tone_adjusted": adjusted}, mask


def _prior_jpeg(original: np.ndarray, quality: int) -> np.ndarray:
    with io.BytesIO() as buffer:
        Image.fromarray(original).save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return np.array(encoded.convert("RGB"))


def compare_across_deliveries(original: np.ndarray, submitted: np.ndarray, carrier: Image.Image) -> tuple[dict, np.ndarray | None]:
    """Keep local residuals also supported under plausible earlier recompressions."""
    result, mask = compare_delivered(original, submitted, carrier)
    if mask is None or not mask.any():
        return result, mask
    support = np.zeros(mask.shape, np.uint8)
    comparable = []
    for quality in PRIOR_JPEG_QUALITIES:
        _, alternate = compare_delivered(_prior_jpeg(original, quality), submitted, carrier)
        if alternate is not None:
            comparable.append(quality)
            support += alternate
    if len(comparable) < 2:
        return {**result, "status": "uncertain_delivery", "regions": []}, None
    regions, confirmed, omitted = _regions(mask & (support >= 2), max(48, round(mask.size * .0015)))
    return {**result, "regions": regions, "additional_regions": omitted, "prior_jpeg_qualities_compared": comparable}, confirmed
