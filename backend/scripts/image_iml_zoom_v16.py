"""Fixed magnified-crop image experiment; inspection maps, never AI confidence."""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter

GRID, SHARE, MARGIN = 256, 0.6, 0.06


def require(value, reason):
    if not value:
        raise ValueError(reason)


def resize(values, height, width):
    """Canonical float64 half-pixel bilinear sampling, with clamped edges."""
    require(
        values.ndim == 2 and min(values.shape) > 0 and height > 0 and width > 0 and np.isfinite(values).all(),
        "Finite nonempty scalar map required",
    )
    h, w = values.shape
    y = np.clip((np.arange(height, dtype=np.float64) + 0.5) * h / height - 0.5, 0, h - 1)
    x = np.clip((np.arange(width, dtype=np.float64) + 0.5) * w / width - 0.5, 0, w - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    v = values.astype(np.float64)
    return (1 - fy) * ((1 - fx) * v[y0[:, None], x0] + fx * v[y0[:, None], x1]) + fy * (
        (1 - fx) * v[y1[:, None], x0] + fx * v[y1[:, None], x1]
    )


def encode(values):
    require(
        values.ndim == 2
        and values.dtype == np.float32
        and np.isfinite(values).all()
        and 0 <= values.min() <= values.max() <= 1,
        "Complete native float32 response required",
    )
    return np.rint(np.clip(resize(values, GRID, GRID), 0, 1) * 255).astype(np.uint8)


def boxes(width, height):
    require(
        type(width) is type(height) is int and min(width, height) >= 32 and max(width, height) <= 1024,
        "Complete32..1024 native source dimensions required",
    )
    w, h = math.ceil(width * SHARE), math.ceil(height * SHARE)
    return [(x, y, x + w, y + h) for y in (0, height - h) for x in (0, width - w)]


def zoom(rgb, box):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "Native uint8 RGB required")
    x0, y0, x1, y1 = box
    require(
        0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0], "Crop must stay inside stored source"
    )
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    size = (1024, max(1, round(h * 1024 / w))) if w >= h else (max(1, round(w * 1024 / h)), 1024)
    return np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.BILINEAR)).copy()


def ordinary(rgb):
    """Fixed ordinary central-quarter blur/tone operation, with16-pixel feather."""
    h, w = rgb.shape[:2]
    require(
        rgb.dtype == np.uint8 and rgb.shape == (h, w, 3) and min(h, w) >= 32, "Native RGB source required"
    )
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(2)), dtype=np.float64)
    target = np.clip(blurred * 1.08, 0, 255)
    yy, xx = np.mgrid[:h, :w]
    distance = np.minimum.reduce(
        [xx - w // 4 + 0.5, 3 * w // 4 - xx - 0.5, yy - h // 4 + 0.5, 3 * h // 4 - yy - 0.5]
    )
    alpha = np.clip(distance / 16, 0, 1)[..., None]
    return np.rint(rgb.astype(np.float64) * (1 - alpha) + target * alpha).astype(np.uint8)


def image_bytes(rgb):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return output.getvalue()


def reconstruct(grids, width, height):
    require(
        grids.dtype == np.uint8 and grids.shape == (5, GRID, GRID),
        "All five lossless recorded response grids required",
    )
    full = resize(grids[0] / 255.0, height, width)
    total, count = np.zeros((height, width), np.float64), np.zeros((height, width), np.int16)
    for grid, box in zip(grids[1:], boxes(width, height), strict=True):
        x0, y0, x1, y1 = box
        projected = resize(grid / 255.0, y1 - y0, x1 - x0)
        keep = np.ones(projected.shape, bool)
        mx, my = math.ceil((x1 - x0) * MARGIN), math.ceil((y1 - y0) * MARGIN)
        if x0:
            keep[:, :mx] = False
        if x1 < width:
            keep[:, -mx:] = False
        if y0:
            keep[:my] = False
        if y1 < height:
            keep[-my:] = False
        total[y0:y1, x0:x1] += projected * keep
        count[y0:y1, x0:x1] += keep
    require((count > 0).all(), "Every source pixel needs actual tile support")
    return {"whole": full, "zoom": 0.5 * full + 0.5 * total / count}


def fraction(response):
    values = np.rint(np.clip(response, 0, 1) * 255) / 255.0
    return int(np.count_nonzero(values >= 0.5)) / values.size


def point(rows):
    grouped = {}
    for row in rows:
        require(
            row["role"] == "point"
            and row["kind"] in ("original", "ordinary")
            and row["status"] == "ok"
            and type(row["label"]) is int
            and row["label"] == 0,
            "Only complete genuine point controls allowed",
        )
        grouped.setdefault(row["source_id"], []).append(row)
    require(
        len(grouped) == 24
        and all({r["kind"] for r in part} == {"original", "ordinary"} for part in grouped.values()),
        "All24 genuine sources and both operations required",
    )
    require(
        len({r["format"] for r in rows}) == 1 and rows[0]["format"] in ("native", "jpeg75"),
        "One known input format per genuine point required",
    )
    result = {}
    for arm in ("whole", "zoom"):
        values = [
            max(
                r["responses"][arm]["active_fraction"]
                if r["responses"][arm]["ungated"]["predicted_positive_pixels"]
                else 0.0
                for r in part
            )
            for part in grouped.values()
        ]
        cut = max(0.001, math.nextafter(sorted(values, reverse=True)[1], math.inf))
        result[arm] = {
            "minimum_active_fraction": cut,
            "source_groups": 24,
            "genuine_group_budget": 1,
            "groups_flagged": sum(v >= cut for v in values),
            "uses_ai_or_diagnostic": False,
        }
    return result
