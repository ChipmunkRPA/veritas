"""Read back the encoded JPEG brightness blocks shown in saved copy evidence."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, JpegImagePlugin

from app.detectors.image.jpeg_dq import read_coefficients
from app.evidence.image_copy_similarity_binding import ROLES, V8, box_bounds, supported_size


def verify(data, pairs) -> int:
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
        if (image.mode != "RGB" or JpegImagePlugin.get_sampling(image) != 2 or image.getexif().get(274, 1) != 1
                or not supported_size(*image.size, version=V8) or max(image.size) > 1024):
            raise ValueError("JPEG copy evidence has an unsupported encoded image")
        size = image.size
    try:
        coefficients = read_coefficients(data)
    except ImportError as error:
        raise ValueError("The local JPEG coefficient reader is required to verify matching JPEG blocks") from error
    if coefficients is None or (coefficients.width, coefficients.height) != size:
        raise ValueError("The original JPEG brightness blocks could not be read")
    matched = 0
    for pair in pairs:
        tiles = []
        for role in ROLES:
            left, top, right, bottom = box_bounds(pair["regions"][role], *size)
            if any(point % 8 for point in (left, top, right, bottom)):
                raise ValueError("JPEG copy evidence does not follow the encoded block grid")
            tiles.append(coefficients.y[top // 8:bottom // 8, left // 8:right // 8])
        if not np.array_equal(*tiles):
            raise ValueError("JPEG copy evidence refers to different encoded brightness blocks")
        matched += tiles[0].shape[0] * tiles[0].shape[1]
    return matched
