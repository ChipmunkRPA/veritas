"""Admit tested JPEG profiles and verify color measurements on their source pixels."""

import cv2

from app.detectors.image import jpeg_turned_copy
from app.evidence import image_jpeg_offgrid_source
from app.evidence.image_copy_similarity_binding import PNG_COPY_TURNS, ROLES


def review(data, rgb):
    if not image_jpeg_offgrid_source.supported(data, include_quality_85=False):
        return {"status": "unsupported_profile", "measurement": None}, [], 0
    status, candidates = jpeg_turned_copy.analyse(rgb)
    if not candidates:
        return {"status": status, "measurement": None}, [], 0
    match, = candidates
    return {"status": status, "measurement": match["measurement"]}, [match["pair"]], match["turns"]


def verify(data, rgb, pair, measurement):
    if not image_jpeg_offgrid_source.supported(data, include_quality_85=False):
        raise ValueError("Rotated-copy evidence requires a supported encoded JPEG")
    boxes = [tuple(pair["regions"][role][key] for key in "xywh") for role in ROLES]
    turns = PNG_COPY_TURNS.get(pair["transform"])
    if turns not in {1, 2}:
        raise ValueError("Rotated JPEG evidence requires a quarter turn or half turn")
    try:
        actual = jpeg_turned_copy.witness(rgb, boxes, turns)
    except cv2.error as error:
        raise ValueError("The original JPEG crops could not be aligned for comparison") from error
    if actual is None or actual != measurement:
        raise ValueError("Rotated JPEG color measurements differ from the submitted image")
