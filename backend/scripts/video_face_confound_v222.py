"""Model-free measurements of appearance-window geometry and photometric context.

The caller supplies already source-bound RGB frames and tracked face metadata.
This module does not decode media, invoke a model, or make a review decision.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ROLES = ("before", "during", "after")
CHANNELS = ("luma", "blue_chroma", "red_chroma")
MIN_ROLE_FRAMES = 3
MIN_FACE_PIXELS = 64
MIN_CONTEXT_PIXELS = 256


class Unavailable(ValueError):
    """Inputs do not support a measurement, not evidence of any outcome."""


def require(condition, reason):
    if not condition:
        raise Unavailable(reason)


def geometry(landmarks):
    """Remove translation, scale and in-plane rotation using the two eye points."""
    eyes = landmarks[1] - landmarks[0]
    scale = float(np.linalg.norm(eyes))
    require(np.isfinite(scale) and scale > 1e-6, "degenerate_eye_landmarks")
    x = eyes / scale
    y = np.array([-x[1], x[0]])
    return (landmarks - np.mean(landmarks[:2], axis=0)) @ np.stack((x, y), axis=1) / scale


def measure_frame(row):
    require(isinstance(row, Mapping), "invalid_frame_record")
    try:
        rgb = np.asarray(row.get("rgb"))
    except (TypeError, ValueError):
        raise Unavailable("invalid_rgb_shape_or_type") from None
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[0] > 0 and rgb.shape[1] > 0
            and np.issubdtype(rgb.dtype, np.number) and not np.issubdtype(rgb.dtype, np.complexfloating),
            "invalid_rgb_shape_or_type")
    require(np.isfinite(rgb).all() and np.min(rgb) >= 0 and np.max(rgb) <= 255, "rgb_out_of_range")
    height, width = rgb.shape[:2]
    try:
        marks = np.asarray(row.get("landmarks"), dtype=np.float64)
        box = row.get("bbox")
        require(isinstance(box, Mapping), "invalid_bbox")
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except (TypeError, ValueError, KeyError, OverflowError):
        raise Unavailable("invalid_geometry_input") from None
    require(marks.shape == (5, 2) and np.isfinite(marks).all() and np.isfinite([x, y, w, h]).all(),
            "invalid_geometry_input")
    require(w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height,
            "bbox_out_of_range")
    require(((marks[:, 0] >= x) & (marks[:, 0] <= x + w) & (marks[:, 1] >= y)
             & (marks[:, 1] <= y + h)).all(), "landmarks_out_of_bbox")
    shape = geometry(marks)
    yy, xx = np.ogrid[:height, :width]
    # The face is a box, not a segmentation. Exclude a half-box margin from context.
    face = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    context = ~((xx >= x - w / 2) & (xx < x + 1.5 * w) & (yy >= y - h / 2) & (yy < y + 1.5 * h))
    face_n, context_n = int(face.sum()), int(context.sum())
    require(face_n >= MIN_FACE_PIXELS, "insufficient_face_pixels")
    require(context_n >= MIN_CONTEXT_PIXELS, "insufficient_nonface_context")
    pixels = rgb.astype(np.float64)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    # Full-range YCbCr-like linear combinations of supplied RGB code values.
    luma = .299 * red + .587 * green + .114 * blue
    transformed = np.stack((luma, .564 * (blue - luma), .713 * (red - luma)), axis=-1)
    return {"shape": shape, "face": np.median(transformed[face], axis=0),
            "context": np.median(transformed[context], axis=0), "face_pixels": face_n,
            "context_pixels": context_n, "frame_shape": (height, width)}


def _vector(values):
    return {channel: float(value) for channel, value in zip(CHANNELS, values, strict=True)}


def _compare(first, second):
    face, context = second["face"] - first["face"], second["context"] - first["context"]
    face_norm, context_norm = float(np.linalg.norm(face)), float(np.linalg.norm(context))
    cosine = float(np.clip(np.dot(face, context) / (face_norm * context_norm), -1, 1)) if face_norm and context_norm else None
    return {"geometry_rms_eye_units": float(np.sqrt(np.mean(np.sum((second["shape"] - first["shape"]) ** 2, axis=1)))),
            "face_shift": _vector(face), "nonface_shift": _vector(context),
            "face_minus_nonface_shift": _vector(face - context), "face_nonface_shift_cosine": cosine}


def measure(roles):
    """Return descriptive within-track role comparisons or an explicit unavailable reason.

    Each role contains ordered, unique mappings with ``frame_index``, RGB array,
    bbox {x,y,w,h}, and five image-coordinate landmarks. Input frames are already
    selected and source-bound by the caller. Positive shift means second minus
    first. The non-face mask is not motion-registered; no cue is a verdict.
    """
    try:
        require(isinstance(roles, Mapping) and set(roles) == set(ROLES), "missing_role_context")
        require(all(isinstance(roles[r], Sequence) and not isinstance(roles[r], (str, bytes))
                    and len(roles[r]) >= MIN_ROLE_FRAMES for r in ROLES), "insufficient_role_frames")
        all_rows = [row for role in ROLES for row in roles[role]]
        require(all(isinstance(row, Mapping) and type(row.get("frame_index")) is int and row["frame_index"] >= 0
                    for row in all_rows), "invalid_frame_index")
        indices = [row["frame_index"] for row in all_rows]
        require(indices == sorted(set(indices)), "overlapping_or_unordered_role_frames")
        measured = {role: [measure_frame(row) for row in roles[role]] for role in ROLES}
        require(len({item["frame_shape"] for group in measured.values() for item in group}) == 1,
                "inconsistent_frame_size")
        pooled = {role: {key: np.median(np.stack([item[key] for item in measured[role]]), axis=0)
                         for key in ("shape", "face", "context")} for role in ROLES}
        return {"status": "measured", "role_frame_counts": {r: len(measured[r]) for r in ROLES},
                "minimum_face_pixels": min(item["face_pixels"] for group in measured.values() for item in group),
                "minimum_nonface_pixels": min(item["context_pixels"] for group in measured.values() for item in group),
                "comparisons": {"before_to_during": _compare(pooled["before"], pooled["during"]),
                                "after_to_during": _compare(pooled["after"], pooled["during"]),
                                "before_to_after": _compare(pooled["before"], pooled["after"])},
                "interpretation": "Descriptive only: geometry, light, scene and processing can all affect these measurements; no edit or origin decision."}
    except Unavailable as error:
        return {"status": "unavailable", "reason": str(error)}
