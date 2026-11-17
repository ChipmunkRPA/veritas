"""Compare an earlier 8-bit HEIF photograph with a JPEG export on its delivered grid."""

from __future__ import annotations

import io
from math import sqrt

import cv2
import numpy as np
from PIL import ImageCms, ImageOps, JpegImagePlugin

from app.detectors.image import reference_perspective as perspective

SCHEMA = "chipmunk.image-reference-comparison/26"
UNASSESSED = "heif_to_jpeg_pair_unassessed"
MAX_PIXELS = 16_000_000
MIN_SIDE = 320
MAX_SUBMITTED_SIDE = 1536
MAX_PROFILE_BYTES = 1024 * 1024
INTERPRETATION = (
    "The two photographs share detail across the compared area. The highlighted regions differ after matching "
    "framing and display colors. Editing and resampling can produce differences. This comparison carries zero "
    "weight in the AI assessment."
)
SCOPE = (
    "Use an earlier single-frame 8-bit HEIC photo and a JPEG export of the same photo. The comparison requires "
    "matching detail across the submitted image. The earlier photo can have up to 16 megapixels."
)


def supported_source(image):
    color = image.info.get("nclx_profile") or {}
    profile = image.info.get("icc_profile")
    return (
        image.format == "HEIF" and image.mode == "RGB" and image.info.get("bit_depth") == 8
        and getattr(image, "n_frames", 1) == 1
        and min(image.size) >= MIN_SIDE and image.width * image.height <= MAX_PIXELS
        and (bool(profile) or (color.get("color_primaries", 2) in {1, 2}
                               and color.get("transfer_characteristics", 2) in {1, 2, 13}))
    )


def supported_target(image):
    return (image.format == "JPEG" and image.mode == "RGB" and min(image.size) >= MIN_SIDE
            and max(image.size) <= MAX_SUBMITTED_SIDE and JpegImagePlugin.get_sampling(image) == 2
            and bool(image.quantization) and image.quantization.get(0, [255])[0] <= 8)


def display_pixels(image):
    """Return sRGB view pixels at the same native coordinates that the worker uses."""
    profile = image.info.get("icc_profile")
    with ImageOps.exif_transpose(image) as oriented, oriented.convert("RGB") as rgb:
        if not profile:
            return np.array(rgb)
        if len(profile) > MAX_PROFILE_BYTES:
            raise ValueError("The image's color profile exceeds the comparison limit")
        try:
            source = ImageCms.getOpenProfile(io.BytesIO(profile))
            with ImageCms.profileToProfile(rgb, source, ImageCms.createProfile("sRGB"), outputMode="RGB") as srgb:
                return np.array(srgb)
        except ImageCms.PyCMSError as error:
            raise ValueError("The embedded color profile could not be converted to sRGB") from error


def _resize_to(image, scale):
    if scale >= 1:
        return image
    height, width = image.shape[:2]
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _to_native(image, grid):
    """Map resized pixel centers to the oriented source's pixel centers."""
    x, y = np.array(image.shape[:2][::-1]) / grid.shape[:2][::-1]
    return np.array(((x, 0., (x - 1) / 2), (0., y, (y - 1) / 2), (0., 0., 1.)))


def measure(reference, submitted):
    if min(submitted.shape[:2]) < MIN_SIDE or max(submitted.shape[:2]) > MAX_SUBMITTED_SIDE:
        return None
    goal = min(perspective.MAX_SIDE, max(submitted.shape[:2]))
    earlier_grid = _resize_to(reference, goal * 1.25 / max(reference.shape[:2]))
    submitted_grid = _resize_to(submitted, goal / max(submitted.shape[:2]))
    fitted = perspective._alignment(perspective._gray(earlier_grid), perspective._gray(submitted_grid),
                                    area_range=(.34, 2.8))
    if fitted is None:
        return None
    matrix, count, error = fitted
    native = _to_native(reference, earlier_grid) @ matrix @ np.linalg.inv(_to_native(submitted, submitted_grid))
    native /= native[2, 2]
    h, w = submitted.shape[:2]
    boundary = perspective._project(native, perspective._bounds(w, h)).astype(np.float32)
    sample_scale = sqrt(cv2.contourArea(boundary) / (w * h))
    if not .65 <= sample_scale <= 10:
        return None
    sampled_reference = _resize_to(reference, 1 / sample_scale)
    sample_matrix = np.linalg.inv(_to_native(reference, sampled_reference)) @ native
    result = perspective.measure_aligned(sampled_reference, submitted, sample_matrix, count, error,
                                         native_matrix=native)
    if result is not None:
        result.evidence.update({"earlier_registration_size": list(earlier_grid.shape[1::-1]),
                                "submitted_registration_size": list(submitted_grid.shape[1::-1]),
                                "earlier_sampling_size": list(sampled_reference.shape[1::-1]),
                                "crop_color_space": "sRGB"})
    return result
