"""Local reference comparison and source-coordinate evidence for two images."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

from app.detectors.image import (
    reference_alignment,
    reference_compare,
    reference_crop_resize,
    reference_exact_jpeg,
    reference_png_exact_resize,
    reference_png_orientation,
    reference_png_oriented_crop,
    reference_png_oriented_resize,
    reference_png_screenshot,
    reference_resize,
    reference_shifted_jpeg,
    reference_structure_jpeg,
)

MAX_INPUT_BYTES = 60 * 1024 * 1024
RESIZE_SCHEMA = "chipmunk.image-reference-comparison/3"
CROP_RESIZE_SCHEMA = "chipmunk.image-reference-comparison/4"
EXACT_PNG_CROP_RESIZE_SCHEMA = "chipmunk.image-reference-comparison/5"
DIFFERENCE_MARK = (255, 173, 46, 144)
INTERPRETATION = ("Regions show differences that survived the comparison settings. Ordinary processing can produce "
                  "differences. The earlier file's authenticity and the cause of changes require separate evidence.")
CROP_RESIZE_MAX_REFERENCE_SIDE = 768
CROP_RESIZE_MAX_SUBMITTED_SIDE = 845
CROP_RESIZE_MIN_SCALE = .72
CROP_RESIZE_MAX_SCALE = 1.17


@lru_cache(maxsize=4)
def _jpeg_tables(quality: int) -> dict:
    with io.BytesIO() as buffer:
        Image.new("RGB", (16, 16)).save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as expected:
            return expected.quantization


def _color_jpeg420(image: Image.Image) -> bool:
    return (image.format == "JPEG" and image.mode == "RGB" and JpegImagePlugin.get_sampling(image) == 2
            and bool(image.quantization))


def supported_crop_resize(carrier: Image.Image, reference_size: tuple[int, int], *, earlier: Image.Image | None = None) -> bool:
    """Admit the verified quantization and orientation profiles for cropped JPEGs."""
    if (max(reference_size) > CROP_RESIZE_MAX_REFERENCE_SIDE or max(carrier.size) > CROP_RESIZE_MAX_SUBMITTED_SIDE
            or not _color_jpeg420(carrier)):
        return False
    if carrier.quantization == _jpeg_tables(75):
        return True
    return (carrier.quantization == _jpeg_tables(90) and earlier is not None and _color_jpeg420(earlier)
            and carrier.getexif().get(274, 1) == earlier.getexif().get(274, 1) == 1
            and earlier.quantization in (_jpeg_tables(82), _jpeg_tables(95)))


def accepted_crop_resize(result: dict, submitted_size: tuple[int, int]) -> bool:
    if result["status"] != "comparable":
        return False
    crop = result["reference_crop_bounds"]
    return all(round(crop[side] * CROP_RESIZE_MIN_SCALE) <= size <= round(crop[side] * CROP_RESIZE_MAX_SCALE)
               for side, size in zip(("w", "h"), submitted_size, strict=True))


def _bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("An image exceeds the 60 MiB input limit")
    return data


def _png(path: Path, pixels: np.ndarray) -> dict:
    Image.fromarray(pixels).save(path, format="PNG")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": path.name, "sha256": digest, "width": pixels.shape[1], "height": pixels.shape[0]}


def _write_evidence(directory, comparison, reference, submitted, mask):
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Evidence directory already exists: {directory}")
    with tempfile.TemporaryDirectory(prefix=".image-comparison-", dir=directory.parent) as workspace:
        staged = Path(workspace) / "evidence"
        staged.mkdir()
        if mask is not None and mask.any():
            comparison["mask"] = _png(staged / "difference-mask.png", mask.astype(np.uint8) * 255)
            for i, region in enumerate(comparison["regions"], 1):
                for label, pixels, bounds in (("reference", reference, region.get("reference_region", region)),
                                              ("submitted", submitted, region)):
                    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
                    region[f"{label}_crop"] = _png(staged / f"{label}-{i:02d}.png", pixels[y:y+h, x:x+w])
                x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[mask[y:y+h, x:x+w]] = DIFFERENCE_MARK
                region["submitted_overlay"] = _png(staged / f"submitted-overlay-{i:02d}.png", overlay)
        (staged / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Evidence directory already exists: {directory}")
        staged.rename(directory)


def inspect(reference_path: Path, submitted_path: Path, *, evidence_dir: Path | None = None,
            allow_integer_alignment: bool = False, allow_proportional_resize: bool = False,
            allow_oriented_crop: bool = False, allow_oriented_resize: bool = False,
            allow_embedded_photo: bool = False, allow_exact_jpeg_identity: bool = False,
            allow_shifted_jpeg: bool = False, allow_structure_jpeg: bool = False) -> dict:
    """Use an earlier reference supplied by the reviewer. This measures pixel differences."""
    before, after = _bytes(reference_path), _bytes(submitted_path)
    with Image.open(io.BytesIO(before)) as original, Image.open(io.BytesIO(after)) as questioned:
        if max(image.width * image.height for image in (original, questioned)) > reference_compare.MAX_PIXELS:
            raise ValueError("Reference comparison supports images up to 16 megapixels")
        with ImageOps.exif_transpose(original) as oriented_before, ImageOps.exif_transpose(questioned) as oriented_after:
            reference = np.array(oriented_before.convert("RGB"))
            submitted = np.array(oriented_after.convert("RGB"))
        schema = None
        if (allow_integer_alignment or allow_proportional_resize or allow_oriented_crop or allow_oriented_resize or allow_embedded_photo
                ) and not np.array_equal(reference, submitted):
            result, mask = (reference_png_orientation.compare(reference, submitted, questioned)
                            if allow_proportional_resize else (None, None))
            if result is not None:
                schema = reference_png_orientation.SCHEMA
            else:
                result, mask = reference_alignment.compare(reference, submitted, questioned)
            if allow_proportional_resize and result["status"] == "integer_alignment_unresolved":
                if reference.shape != submitted.shape:
                    result, mask = reference_resize.compare(reference, submitted, questioned)
                    schema = RESIZE_SCHEMA
                if result["status"] != "comparable" and supported_crop_resize(questioned, reference.shape[1::-1], earlier=original):
                    candidate, selection = reference_crop_resize.compare(reference, submitted, questioned)
                    if accepted_crop_resize(candidate, submitted.shape[1::-1]):
                        result, mask, schema = candidate, selection, CROP_RESIZE_SCHEMA
                if result["status"] != "comparable":
                    exact, selection = reference_png_exact_resize.compare(reference, submitted, questioned)
                    if exact is not None:
                        result, mask, schema = exact, selection, EXACT_PNG_CROP_RESIZE_SCHEMA
            if allow_oriented_crop and result["status"] != "comparable":
                oriented_crop, selection = reference_png_oriented_crop.compare(reference, submitted, questioned)
                if oriented_crop is not None:
                    result, mask, schema = oriented_crop, selection, reference_png_oriented_crop.SCHEMA
            if allow_oriented_resize and result["status"] != "comparable":
                oriented_resize, selection = reference_png_oriented_resize.compare(reference, submitted, questioned)
                if oriented_resize is not None:
                    result, mask, schema = oriented_resize, selection, reference_png_oriented_resize.SCHEMA
            if allow_embedded_photo and result["status"] != "comparable":
                embedded_photo, selection = reference_png_screenshot.compare(reference, submitted, questioned)
                if embedded_photo is not None:
                    result, mask, schema = embedded_photo, selection, reference_png_screenshot.SCHEMA
        else:
            result, mask = reference_compare.compare_across_deliveries(reference, submitted, questioned)
        if allow_exact_jpeg_identity and result["status"] == "integer_alignment_unresolved":
            exact, selection = reference_exact_jpeg.compare(original, questioned, before, after)
            if exact["status"] == "comparable":
                result, mask, schema = exact, selection, reference_exact_jpeg.SCHEMA
        if allow_shifted_jpeg and result["status"] != "comparable":
            shifted, selection = reference_shifted_jpeg.compare(original, questioned, before, after, result)
            if shifted is not None:
                result, mask, schema = shifted, selection, reference_shifted_jpeg.SCHEMA
        if allow_structure_jpeg and result["status"] != "comparable":
            structured, selection = reference_structure_jpeg.compare(original, questioned, before, after)
            if structured is not None:
                result, mask, schema = structured, selection, reference_structure_jpeg.SCHEMA
    aligned = "alignment" in result
    output = {"schema": schema or f"chipmunk.image-reference-comparison/{2 if aligned else 1}",
              "coordinate_space": "exif-oriented-submitted" if aligned or schema else "exif-oriented-original",
              "reference": {"sha256": hashlib.sha256(before).hexdigest(), "width": reference.shape[1], "height": reference.shape[0]},
              "submitted": {"sha256": hashlib.sha256(after).hexdigest(), "width": submitted.shape[1], "height": submitted.shape[0]},
              **result, "interpretation": (reference_structure_jpeg.INTERPRETATION if schema == reference_structure_jpeg.SCHEMA
                                            else INTERPRETATION)}
    if evidence_dir is not None:
        _write_evidence(evidence_dir, output, reference, submitted, mask)
    return output
