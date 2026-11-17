# SPDX-License-Identifier: Apache-2.0
"""Decode an HEIC still and the single thumbnail linked to its primary image."""

from __future__ import annotations

import io
from contextlib import contextmanager

MAX_BYTES = 60 * 1024 * 1024
MIMES = frozenset({"image/heic", "image/heif"})
_HDR = {"content_light_level", "mastering_display_colour_volume"}


def _color_space(info):
    profile = info.get("nclx_profile") or {}
    return (info.get("icc_profile"), profile.get("color_primaries", 1), profile.get("transfer_characteristics", 13))


@contextmanager
def read_pair(data: bytes):
    """Yield a reason and a decoded pair, with identical color interpretation."""
    status, pair = _decode(data)
    if pair is None:
        yield status, None
    else:
        main, thumbnail = pair
        with main, thumbnail:
            yield status, pair


def _decode(data):
    from pillow_heif import open_heif

    if len(data) > MAX_BYTES or data[4:8] != b"ftyp":
        return "unsupported_carrier", None
    container = open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
    if container.mimetype not in MIMES or len(container) != 1:
        return "unsupported_carrier", None
    primary = container[container.primary_index]
    width, height = primary.size
    info = primary.info
    if (primary.mode != "RGB" or info.get("bit_depth") != 8 or min(width, height) < 160
            or width * height > 24_000_000 or _HDR.intersection(info)
            or info.get("pixel_aspect_ratio") not in (None, (1, 1))):
        return "unsupported_carrier", None
    thumbnails = info.get("thumbnails", ())
    if not thumbnails:
        return "no_retained_thumbnail", None
    if len(thumbnails) != 1 or not hasattr(primary, "get_thumbnail"):
        return "unsupported_thumbnail", None
    thumbnail = primary.get_thumbnail(0)
    tw, th = thumbnail.size
    color = _color_space(info)
    if (thumbnail.mode != "RGB" or thumbnail.info.get("bit_depth") != 8
            or not 64 <= min(tw, th) <= max(tw, th) <= 512 or not (tw < width and th < height)
            or color != _color_space(thumbnail.info) or color[-1] in (16, 18)):
        return "unsupported_thumbnail", None
    main = primary.to_pillow()
    try:
        # The decoder has already applied the container's transform and
        # normalized the Exif orientation, just as in the serving image path.
        if main.getexif().get(274, 1) != 1:
            main.close()
            return "unsupported_carrier", None
        preview = thumbnail.to_pillow()
    except Exception:
        main.close()
        raise
    return "decoded", (main, preview)
