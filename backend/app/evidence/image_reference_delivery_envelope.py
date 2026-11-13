"""Render the scoped earlier JPEG to PNG comparison after existing checks abstain."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.image import reference_jpeg_png_envelope as detector
from app.evidence import image_reference

SCHEMA = detector.SCHEMA
INTERPRETATION = (
    "Highlighted pixels differ beyond the tested resize, tone and compression variations. "
    "Ordinary processing can still produce differences. This comparison gives no evidence of AI authorship."
)


def supports(earlier: Image.Image, submitted: Image.Image) -> bool:
    """Admit the upright standard color JPEG and single-frame RGB PNG tested locally."""
    return (submitted.format == "PNG" and submitted.mode == "RGB" and submitted.n_frames == 1
            and image_reference._color_jpeg420(earlier) and earlier.quantization == image_reference._jpeg_tables(95)
            and [component[3] for component in earlier.layer] == [0, 1, 1]
            and earlier.getexif().get(274, 1) == submitted.getexif().get(274, 1) == 1
            and max(earlier.size) <= image_reference.CROP_RESIZE_MAX_REFERENCE_SIDE
            and max(submitted.size) <= image_reference.CROP_RESIZE_MAX_SUBMITTED_SIDE)


def compare(earlier: Image.Image, submitted: Image.Image) -> tuple[dict | None, np.ndarray | None]:
    if not supports(earlier, submitted):
        return None, None
    return detector.compare(np.asarray(earlier), np.asarray(submitted), submitted,
                            accept_crop=image_reference.accepted_crop_resize)


def inspect(earlier_path: Path, submitted_path: Path, evidence_dir: Path) -> dict | None:
    """Build the exact source and asset bindings for this comparison."""
    with Image.open(earlier_path) as earlier, Image.open(submitted_path) as submitted:
        result, mask = compare(earlier, submitted)
        if result is None:
            return None
        sources = {}
        for key, path, image in (("reference", earlier_path, earlier), ("submitted", submitted_path, submitted)):
            with path.open("rb") as stream:
                sources[key] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                "width": image.width, "height": image.height}
        output = {**result, **sources, "schema": SCHEMA, "coordinate_space": "exif-oriented-submitted",
                  "interpretation": INTERPRETATION}
        image_reference._write_evidence(evidence_dir, output, np.asarray(earlier), np.asarray(submitted), mask)
        return output
