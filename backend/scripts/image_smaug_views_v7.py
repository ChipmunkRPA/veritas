"""Research-only three-view preprocessing with explicit oriented-image coordinates."""
from __future__ import annotations

import math

from PIL import Image, ImageOps


def views(image, count=3):
    if count not in (1, 3):
        raise ValueError('Only the publisher single/three-view modes are defined')
    upright = ImageOps.exif_transpose(image).convert('RGB')
    width, height = upright.size
    if min(upright.size) > 576:
        factor = 576 / min(upright.size)
        resized_size = (round(width * factor), round(height * factor))
        resampling = Image.Resampling.LANCZOS
    elif min(upright.size) < 528:
        factor = 528 / min(upright.size)
        resized_size = (math.ceil(width * factor), math.ceil(height * factor))
        resampling = Image.Resampling.BICUBIC
    else:
        resized_size, resampling = upright.size, None
    resized = upright if resampling is None else upright.resize(resized_size, resampling)
    rw, rh = resized.size
    cx, cy = (rw - 528) // 2, (rh - 528) // 2
    starts = ([(cx, cy)] if count == 1 else
              [(x, cy) for x in (0, cx, rw - 528)] if rw >= rh else
              [(cx, y) for y in (0, cy, rh - 528)])
    crops, traces = [], []
    for x, y in starts:
        bounds = (x, y, x + 528, y + 528)
        crops.append(resized.crop(bounds))
        traces.append({'coordinate_space': 'EXIF-oriented original RGB pixel-edge coordinates',
            'oriented_original_size': [width, height], 'resized_size': [rw, rh],
            'resized_crop_xyxy': list(bounds),
            'source_edge_bounds_rational': [[x * width, rw], [y * height, rh],
                                           [(x + 528) * width, rw], [(y + 528) * height, rh]],
            'resampling': None if resampling is None else resampling.name})
    return crops, traces
