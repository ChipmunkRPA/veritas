"""Measure and replay JPEG color differences after diagonal alignment."""

import cv2

from app.detectors.image import jpeg_turned_copy
from app.evidence.image_copy_similarity_binding import (
    DIAGONAL_TRANSFORMS,
    JPEG_DIAGONAL_MIN_PARALLEL_SUM,
    ROLES,
    jpeg_diagonal_displacement,
    jpeg_diagonal_size,
)
from app.evidence.image_jpeg_offgrid_source import encoded_profile


def supported(data, rgb):
    return jpeg_diagonal_size(rgb.shape[1], rgb.shape[0]) and encoded_profile(data) == 75


def review(data, rgb):
    if not supported(data, rgb):
        return {"status": "unsupported_profile", "measurement": None}, [], None
    status, candidates = jpeg_turned_copy.analyse_diagonal(rgb)
    if not candidates:
        return {"status": status, "measurement": None}, [], None
    match, = candidates
    transform = DIAGONAL_TRANSFORMS[match["turns"] - 1]
    if jpeg_diagonal_displacement(match["pair"], transform) < JPEG_DIAGONAL_MIN_PARALLEL_SUM:
        return {"status": "same_axis", "measurement": None}, [], None
    return {"status": status, "measurement": match["measurement"]}, [match["pair"]], transform


def verify(data, rgb, pair, measurement):
    transform = pair["transform"]
    if transform not in DIAGONAL_TRANSFORMS or not supported(data, rgb):
        raise ValueError("Diagonal JPEG evidence requires a supported upright JPEG")
    boxes = [tuple(pair["regions"][role][axis] for axis in "xywh") for role in ROLES]
    first, second = boxes
    if (min(first[2:]) < jpeg_turned_copy.MIN_CROP_SIDE or first[2:] != second[2:][::-1]
            or jpeg_diagonal_displacement(boxes, transform) < JPEG_DIAGONAL_MIN_PARALLEL_SUM):
        raise ValueError("Diagonal JPEG evidence has incompatible crop dimensions or relative position")
    try:
        actual = jpeg_turned_copy.diagonal_witness(rgb, boxes, DIAGONAL_TRANSFORMS.index(transform) + 1)
    except cv2.error as error:
        raise ValueError("The source JPEG crops could not be aligned") from error
    if actual is None or actual != measurement:
        raise ValueError("Diagonal JPEG measurements differ from the original crops")
