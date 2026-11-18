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
def read_pair(data: bytes, *, allow_10bit_sdr: bool = False):
    """Yield a reason and a decoded pair, with identical color interpretation."""
    status, pair = _decode(data, allow_10bit_sdr=allow_10bit_sdr)
    if pair is None:
        yield status, None
    else:
        main, thumbnail = pair
        with main, thumbnail:
            yield status, pair


def _decode(data, *, allow_10bit_sdr):
    from pillow_heif import open_heif

    if len(data) > MAX_BYTES or data[4:8] != b"ftyp":
        return "unsupported_carrier", None
    # Use the same rendered RGB as the upload viewer. The original bit depth
    # remains available in each item's metadata for admission and source replay.
    container = open_heif(io.BytesIO(data), convert_hdr_to_8bit=allow_10bit_sdr)
    if container.mimetype not in MIMES or len(container) != 1:
        return "unsupported_carrier", None
    primary = container[container.primary_index]
    width, height = primary.size
    info = primary.info
    depth = info.get("bit_depth")
    color = _color_space(info)
    if (primary.mode != "RGB" or depth not in ((8, 10) if allow_10bit_sdr else (8,)) or min(width, height) < 160
            or width * height > 24_000_000 or _HDR.intersection(info)
            or (depth == 10 and (color[-1] in (16, 18) or info.get("aux")))
            or info.get("pixel_aspect_ratio") not in (None, (1, 1))):
        return "unsupported_carrier", None
    thumbnails = info.get("thumbnails", ())
    if not thumbnails:
        return "no_retained_thumbnail", None
    if len(thumbnails) != 1 or not hasattr(primary, "get_thumbnail"):
        return "unsupported_thumbnail", None
    thumbnail = primary.get_thumbnail(0)
    tw, th = thumbnail.size
    preview_depths = (8, 10) if depth == 10 else (8,)
    if (thumbnail.mode != "RGB" or thumbnail.info.get("bit_depth") not in preview_depths
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
