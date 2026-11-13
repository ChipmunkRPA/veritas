"""Check the submitted JPEG profile and reproduce its saved patch measurement."""

from __future__ import annotations

import hashlib
import io

import cv2
from PIL import Image, JpegImagePlugin

from app.detectors.image import jpeg_offgrid_copy
from app.evidence.image_copy_similarity_binding import OFFGRID_MIN_SIDE, ROLES, V8, supported_size

_QUANTIZATION_SHA256 = frozenset({
    "db12807ffe7ea7d208d803a4ad9d156e941af36754c009d3ec5de5aae4112903",  # Standard JPEG quality 75.
    "4a947d5d753cb42d30d364a580030dd557d37404f5e12e09378359b0e9e397ac",  # Standard JPEG quality 90.
})
_QUALITY_85_SHA256 = "25bb3739664e47c1cd8aadf2ff486d477738b962bc84a097ae7bab5d88084a12"
_QUALITY_85_MIN_SIDE = 56


def _minimum_crop_side(data: bytes) -> int | None:
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
        if (image.mode != "RGB" or JpegImagePlugin.get_sampling(image) != 2 or image.getexif().get(274, 1) != 1
                or not supported_size(*image.size, version=V8) or set(image.quantization) != {0, 1}
                or tuple(component[3] for component in image.layer) != (0, 1, 1)
                or any(not 0 < level <= 255 for table in image.quantization.values() for level in table)):
            return None
        tables = bytes(image.quantization[0]) + bytes(image.quantization[1])
    digest = hashlib.sha256(tables).hexdigest()
    if digest == _QUALITY_85_SHA256:
        return _QUALITY_85_MIN_SIDE
    return OFFGRID_MIN_SIDE if digest in _QUANTIZATION_SHA256 else None


def supported(data: bytes) -> bool:
    return _minimum_crop_side(data) is not None


def review(data, rgb):
    minimum_side = _minimum_crop_side(data)
    if minimum_side is None:
        return {"status": "unsupported_profile", "measurement": None}, []
    status, candidates = jpeg_offgrid_copy.analyse(rgb)
    if not candidates:
        return {"status": status, "measurement": None}, []
    item = candidates[0]
    if min(item["pair"][0][2:]) < minimum_side:
        return {"status": "no_pair", "measurement": None}, []
    return {"status": status, "measurement": item["measurement"]}, [item["pair"]]


def verify(data, rgb, pair, measurement) -> None:
    minimum_side = _minimum_crop_side(data)
    if minimum_side is None:
        raise ValueError("Between-block JPEG evidence has an unsupported encoded image")
    boxes = [tuple(pair["regions"][role][key] for key in ("x", "y", "w", "h")) for role in ROLES]
    if any(min(box[2:]) < minimum_side for box in boxes):
        raise ValueError("Between-block JPEG crops are smaller than the supported footprint for this compression profile")
    try:
        actual = jpeg_offgrid_copy.witness(rgb, boxes)
    except cv2.error as error:
        raise ValueError("The original between-block JPEG pixels could not be compared") from error
    if actual != measurement:
        raise ValueError("Between-block JPEG evidence differs from the original crop and surrounding pixels")
