"""Admit tested JPEG profiles and verify color measurements on their source pixels."""

import cv2

from app.detectors.image import jpeg_turned_copy
from app.evidence import image_jpeg_offgrid_source
from app.evidence.image_copy_similarity_binding import PNG_COPY_TURNS, ROLES


def _supported(data, *, expanded, offgrid_status, native):
    profile = image_jpeg_offgrid_source.encoded_profile(data, native_rotation=native)
    if native:
        return profile in {90, 95} and offgrid_status is None
    if offgrid_status == "unsupported_profile":
        return expanded and profile == 95
    return offgrid_status == "no_pair" and (profile in {75, 90} or expanded and profile == 85)


def review(data, rgb, *, expanded=False, offgrid_status="no_pair", native=False):
    if not _supported(data, expanded=expanded, offgrid_status=offgrid_status, native=native):
        return {"status": "unsupported_profile", "measurement": None}, [], 0
    if native:
        from app.detectors.image import jpeg_native_turned_copy

        status, candidates = jpeg_native_turned_copy.analyse(rgb)
    else:
        status, candidates = jpeg_turned_copy.analyse(rgb)
    if not candidates:
        return {"status": status, "measurement": None}, [], 0
    match, = candidates
    return {"status": status, "measurement": match["measurement"]}, [match["pair"]], match["turns"]


def verify(data, rgb, pair, measurement, *, expanded=False, offgrid_status="no_pair", native=False):
    if not _supported(data, expanded=expanded, offgrid_status=offgrid_status, native=native):
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
