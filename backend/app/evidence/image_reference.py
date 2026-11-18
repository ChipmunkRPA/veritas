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
    reference_heif,
    reference_jpeg_detail_loss,
    reference_lossless_webp,
    reference_lossy_webp,
    reference_lossy_webp_crop,
    reference_lossy_webp_crop_resize,
    reference_lossy_webp_resize,
    reference_native_jpeg,
    reference_perspective,
    reference_png_exact_resize,
    reference_png_opacity,
    reference_png_orientation,
    reference_png_oriented_crop,
    reference_png_oriented_resize,
    reference_png_screenshot,
    reference_resize,
    reference_same_jpeg,
    reference_scaled_screenshot,
    reference_shifted_jpeg,
    reference_structure_jpeg,
    reference_webp_color,
)
from app.engine.media import refuse_truncated_images

MAX_INPUT_BYTES = 60 * 1024 * 1024
RESIZE_SCHEMA = "chipmunk.image-reference-comparison/3"
CROP_RESIZE_SCHEMA = "chipmunk.image-reference-comparison/4"
EXACT_PNG_CROP_RESIZE_SCHEMA = "chipmunk.image-reference-comparison/5"
DIFFERENCE_MARK = (255, 173, 46, 144)
INTERPRETATION = ("Regions show differences that survived the comparison settings. Ordinary processing can produce "
                  "differences. The earlier file's authenticity and the cause of changes require separate evidence.")
CROP_RESIZE_MAX_REFERENCE_SIDE = 768
CROP_RESIZE_MAX_SUBMITTED_SIDE = 845
PERSPECTIVE_POINT = 18.
HEIF_POINT = 18.
CROP_RESIZE_MIN_SCALE = .72
CROP_RESIZE_MAX_SCALE = 1.17
LOSSY_WEBP_SCHEMAS = frozenset((reference_lossy_webp.SCHEMA, reference_lossy_webp_resize.SCHEMA,
                                reference_lossy_webp_crop.SCHEMA, reference_lossy_webp_crop_resize.SCHEMA))


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


def _write_evidence(directory, comparison, reference, submitted, mask, *, coverage=None):
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Evidence directory already exists: {directory}")
    with tempfile.TemporaryDirectory(prefix=".image-comparison-", dir=directory.parent) as workspace:
        staged = Path(workspace) / "evidence"
        staged.mkdir()
        if coverage is not None:
            comparison["coverage_mask"] = _png(staged / "compared-area.png", coverage.astype(np.uint8) * 255)
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


def _opacity_pair(original, questioned, before, after):
    if not (original.format == questioned.format == "PNG"
            and any("A" in image.getbands() or "transparency" in image.info for image in (original, questioned))):
        return None
    unassessed = {"status": reference_png_opacity.UNASSESSED, "regions": [], "affects_overall_confidence": False}
    if (max(image.width * image.height for image in (original, questioned)) > reference_png_opacity.MAX_PIXELS
            or not reference_png_opacity.supported_png(original, before)
            or not reference_png_opacity.supported_png(questioned, after)
            or any(original.info.get(key) != questioned.info.get(key)
                   for key in ("icc_profile", "gamma", "srgb", "chromaticity"))):
        return unassessed, None, None, None
    pixels = []
    for image in (original, questioned):
        with ImageOps.exif_transpose(image) as oriented, oriented.convert("RGBA") as rgba:
            pixels.append(np.array(rgba))
    if all((rgba[:, :, 3] == 255).all() for rgba in pixels):
        return None
    result, mask = reference_png_opacity.compare(*pixels)
    if result is None:
        return unassessed, None, None, None
    return result, mask, *(reference_png_opacity.on_white(rgba) for rgba in pixels)


def _oriented_rgb(image):
    with ImageOps.exif_transpose(image) as oriented, oriented.convert("RGB") as rgb:
        return np.array(rgb)


def _perspective_check(earlier, questioned, reference, submitted):
    if not (_color_jpeg420(earlier) and earlier.quantization == _jpeg_tables(90)
            and _color_jpeg420(questioned) and questioned.quantization in (_jpeg_tables(75), _jpeg_tables(90))):
        return None, None, None
    measurement = reference_perspective.measure(reference, submitted)
    reading, mask = reference_perspective.compare(measurement, reference.shape[1::-1], PERSPECTIVE_POINT)
    return reading, mask, measurement.coverage if reading is not None else None


def _heif_comparison(earlier, questioned, before, after, evidence_dir):
    reference = reference_heif.display_pixels(earlier)
    submitted = reference_heif.display_pixels(questioned)
    measurement = None
    if reference_heif.supported_source(earlier) and reference_heif.supported_target(questioned):
        measurement = reference_heif.measure(reference, submitted)
    reading, mask = reference_perspective.compare(measurement, reference.shape[1::-1], HEIF_POINT)
    comparison = {
        "schema": reference_heif.SCHEMA, "coordinate_space": "exif-oriented-submitted",
        "reference": {"format": "HEIC", "sha256": hashlib.sha256(before).hexdigest(),
                      "width": reference.shape[1], "height": reference.shape[0]},
        "submitted": {"format": questioned.format, "sha256": hashlib.sha256(after).hexdigest(),
                      "width": submitted.shape[1], "height": submitted.shape[0]},
        "affects_overall_confidence": False,
        **(reading if reading is not None else {"status": reference_heif.UNASSESSED, "regions": []}),
        "interpretation": reference_heif.INTERPRETATION if reading is not None else reference_heif.SCOPE,
    }
    if evidence_dir is not None:
        _write_evidence(evidence_dir, comparison, reference, submitted, mask,
                        coverage=measurement.coverage if reading is not None else None)
    return comparison


def inspect(reference_path: Path, submitted_path: Path, *, evidence_dir: Path | None = None,
            allow_integer_alignment: bool = False, allow_proportional_resize: bool = False,
            allow_oriented_crop: bool = False, allow_oriented_resize: bool = False,
            allow_embedded_photo: bool = False, allow_scaled_screenshot: bool = False, allow_exact_jpeg_identity: bool = False,
            allow_shifted_jpeg: bool = False, allow_structure_jpeg: bool = False,
            allow_same_jpeg: bool = False, allow_jpeg_detail_loss: bool = False,
            allow_jpeg_detail_loss_extra_qualities: bool = False, allow_png_opacity: bool = False,
            allow_lossy_webp: bool = False, allow_lossy_webp_crop: bool = False,
            allow_lossy_webp_crop_resize: bool = False, allow_lossy_webp_color: bool = False,
            allow_lossy_webp_earlier_jpeg: bool = False, allow_perspective: bool = False,
            allow_earlier_heif: bool = False, allow_native_jpeg: bool = False) -> dict:
    """Use an earlier reference supplied by the reviewer. This measures pixel differences."""
    refuse_truncated_images()
    before, after = _bytes(reference_path), _bytes(submitted_path)
    coverage = None
    with Image.open(io.BytesIO(before)) as original, Image.open(io.BytesIO(after)) as questioned:
        if max(image.width * image.height for image in (original, questioned)) > reference_compare.MAX_PIXELS:
            raise ValueError("Reference comparison supports images up to 16 megapixels")
        if allow_earlier_heif and original.format == "HEIF":
            return _heif_comparison(original, questioned, before, after, evidence_dir)
        schema = None
        webp = "WEBP" in {original.format, questioned.format}
        opacity = _opacity_pair(original, questioned, before, after) if allow_png_opacity and not webp else None
        if opacity is not None:
            result, mask, earlier_preview, submitted_preview = opacity
            schema = reference_png_opacity.SCHEMA
            reference, submitted = ((earlier_preview, submitted_preview) if earlier_preview is not None
                                    else (_oriented_rgb(original), _oriented_rgb(questioned)))
        else:
            reference, submitted = _oriented_rgb(original), _oriented_rgb(questioned)
        if (allow_lossy_webp and allow_lossy_webp_earlier_jpeg and original.format == "JPEG"
                and reference_lossy_webp.supported(questioned, after, minimum_side=reference_lossy_webp_resize.MIN_SIDE)):
            schema = reference_lossy_webp.EARLIER_JPEG_SCHEMA
            result, mask = reference_lossy_webp.compare_earlier_jpeg(original, questioned, before, after)
        elif (allow_lossy_webp and allow_proportional_resize and original.size != questioned.size
                and reference_lossy_webp.supported(questioned, after, minimum_side=reference_lossy_webp_resize.MIN_SIDE)):
            schema = reference_lossy_webp_resize.SCHEMA
            result, mask = reference_lossy_webp_resize.compare(original, questioned, before, after)
            if allow_lossy_webp_crop and result["status"] != "comparable":
                cropped, crop_mask = reference_lossy_webp_crop.compare(original, questioned, before, after)
                if cropped["status"] == "comparable":
                    result, mask, schema = cropped, crop_mask, reference_lossy_webp_crop.SCHEMA
            if allow_lossy_webp_crop_resize and result["status"] != "comparable":
                combined, combined_mask = reference_lossy_webp_crop_resize.compare(original, questioned, before, after)
                if combined["status"] == "comparable":
                    result, mask, schema = combined, combined_mask, reference_lossy_webp_crop_resize.SCHEMA
        elif allow_lossy_webp and reference_lossy_webp.supported(questioned, after):
            schema = reference_lossy_webp.SCHEMA
            result, mask = reference_lossy_webp.compare(original, questioned, before, after)
        elif webp:
            schema = reference_lossless_webp.SCHEMA
            result = {"status": reference_lossless_webp.UNASSESSED, "regions": []}
            mask = None
            if (reference_lossless_webp.supported(original, before, png=True)
                    and reference_lossless_webp.supported(questioned, after, png=True)):
                result, mask = reference_lossless_webp.compare(reference, submitted)
            result["affects_overall_confidence"] = False
        elif opacity is None and (allow_integer_alignment or allow_proportional_resize or allow_oriented_crop
                                or allow_oriented_resize or allow_embedded_photo or allow_scaled_screenshot
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
            if allow_scaled_screenshot and result["status"] != "comparable":
                scaled_photo, selection, display_reference = reference_scaled_screenshot.compare(
                    reference, submitted, original, questioned)
                if scaled_photo is not None:
                    result, mask, schema = scaled_photo, selection, reference_scaled_screenshot.SCHEMA
                    reference = display_reference
                    area = result["scaled_screenshot"]["compared_bounds"]
                    x, y, w, h = (area[key] for key in ("x", "y", "w", "h"))
                    coverage = np.zeros(submitted.shape[:2], dtype=bool)
                    coverage[y:y+h, x:x+w] = True
        elif opacity is None:
            result, mask = reference_compare.compare_across_deliveries(reference, submitted, questioned)
        if not webp and allow_exact_jpeg_identity and result["status"] == "integer_alignment_unresolved":
            exact, selection = reference_exact_jpeg.compare(original, questioned, before, after)
            if exact["status"] == "comparable":
                result, mask, schema = exact, selection, reference_exact_jpeg.SCHEMA
        if not webp and allow_shifted_jpeg and result["status"] != "comparable":
            shifted, selection = reference_shifted_jpeg.compare(original, questioned, before, after, result)
            if shifted is not None:
                result, mask, schema = shifted, selection, reference_shifted_jpeg.SCHEMA
        if not webp and allow_structure_jpeg and result["status"] != "comparable":
            structured, selection = reference_structure_jpeg.compare(original, questioned, before, after)
            if structured is not None:
                result, mask, schema = structured, selection, reference_structure_jpeg.SCHEMA
        if not webp and allow_same_jpeg and result["status"] != "comparable":
            exact, selection = reference_same_jpeg.compare(original, questioned, before, after)
            if exact is not None:
                result, mask, schema = exact, selection, reference_same_jpeg.SCHEMA
        if not webp and allow_jpeg_detail_loss and not result["regions"] and schema != reference_scaled_screenshot.SCHEMA:
            detailed, selection = reference_jpeg_detail_loss.compare(original, questioned, before, after)
            if detailed is not None:
                result, mask, schema = detailed, selection, reference_jpeg_detail_loss.SCHEMA
            elif allow_jpeg_detail_loss_extra_qualities:
                for quality in reference_jpeg_detail_loss.EXTRA_QUALITIES:
                    detailed, selection = reference_jpeg_detail_loss.compare(original, questioned, before, after,
                                                                             alternate_quality=quality)
                    if detailed is not None:
                        detailed["alignment"]["jpeg_quality"] = quality
                        result, mask, schema = detailed, selection, reference_jpeg_detail_loss.EXTENDED_SCHEMA
                        break
        if allow_perspective and result["status"] != "comparable":
            perspective, selection, coverage = _perspective_check(original, questioned, reference, submitted)
            if perspective is not None:
                result, mask, schema = perspective, selection, reference_perspective.SCHEMA
        if allow_native_jpeg and not result["regions"] and original.format == questioned.format == "JPEG":
            native_difference, selection = reference_native_jpeg.compare(original, questioned, before, after)
            if native_difference is not None:
                result, mask, schema, coverage = native_difference, selection, reference_native_jpeg.SCHEMA, None
    aligned = "alignment" in result
    interpretation = {reference_structure_jpeg.SCHEMA: reference_structure_jpeg.INTERPRETATION,
                      reference_same_jpeg.SCHEMA: reference_same_jpeg.INTERPRETATION,
                      reference_native_jpeg.SCHEMA: reference_native_jpeg.INTERPRETATION,
                      reference_perspective.SCHEMA: reference_perspective.INTERPRETATION,
                      reference_scaled_screenshot.SCHEMA: reference_scaled_screenshot.INTERPRETATION,
                      reference_lossless_webp.SCHEMA: (reference_lossless_webp.INTERPRETATION
                                                       if result["status"] == "comparable" else reference_lossless_webp.LIMITATION),
                      reference_lossy_webp.SCHEMA: (reference_lossy_webp.INTERPRETATION
                                                   if result["status"] == "comparable" else reference_lossy_webp.LIMITATION),
                      reference_lossy_webp.EARLIER_JPEG_SCHEMA: (reference_lossy_webp.EARLIER_JPEG_INTERPRETATION
                                                               if result["status"] == "comparable" else reference_lossy_webp.EARLIER_JPEG_LIMITATION),
                      reference_lossy_webp_resize.SCHEMA: (reference_lossy_webp_resize.INTERPRETATION
                                                          if result["status"] == "comparable" else reference_lossy_webp_resize.LIMITATION),
                      reference_lossy_webp_crop.SCHEMA: reference_lossy_webp_crop.INTERPRETATION,
                      reference_lossy_webp_crop_resize.SCHEMA: reference_lossy_webp_crop_resize.INTERPRETATION,
                      reference_png_opacity.SCHEMA: (reference_png_opacity.INTERPRETATION if result["status"] == "comparable"
                                                    else reference_png_opacity.UNASSESSED_INTERPRETATION),
                      **dict.fromkeys(reference_jpeg_detail_loss.SCHEMAS, reference_jpeg_detail_loss.INTERPRETATION)}
    output = {"schema": schema or f"chipmunk.image-reference-comparison/{2 if aligned else 1}",
              "coordinate_space": "exif-oriented-submitted" if aligned or schema else "exif-oriented-original",
              "reference": {"sha256": hashlib.sha256(before).hexdigest(), "width": reference.shape[1], "height": reference.shape[0]},
              "submitted": {"sha256": hashlib.sha256(after).hexdigest(), "width": submitted.shape[1], "height": submitted.shape[0]},
              **result, "interpretation": interpretation.get(schema, INTERPRETATION)}
    if schema in {reference_same_jpeg.SCHEMA, *reference_jpeg_detail_loss.SCHEMAS}:
        output["affects_overall_confidence"] = False
        output["comparison_options"] = {"exact_jpeg_identity": allow_exact_jpeg_identity,
                                        "shifted_jpeg": allow_shifted_jpeg, "structure_jpeg": allow_structure_jpeg}
        if schema in reference_jpeg_detail_loss.SCHEMAS:
            output["comparison_options"]["same_jpeg"] = allow_same_jpeg
    if schema == reference_native_jpeg.SCHEMA:
        output["affects_overall_confidence"] = False
        output["comparison_options"] = {"exact_jpeg_identity": allow_exact_jpeg_identity,
                                        "shifted_jpeg": allow_shifted_jpeg, "structure_jpeg": allow_structure_jpeg,
                                        "same_jpeg": allow_same_jpeg, "jpeg_detail_loss": allow_jpeg_detail_loss,
                                        "jpeg_detail_loss_extra_qualities": allow_jpeg_detail_loss_extra_qualities,
                                        "perspective": allow_perspective}
    if allow_lossy_webp_color and schema in LOSSY_WEBP_SCHEMAS and result["status"] == "comparable":
        proof, regions, color_mask, omitted = reference_webp_color.compare(reference, submitted, result)
        output.update(schema=reference_webp_color.SCHEMA, comparison_schema=schema, color_review=proof,
                      regions=[*result["regions"], *regions], additional_regions=result["additional_regions"] + omitted,
                      interpretation=("This comparison measures local brightness and color differences after matching "
                                      "the two images. Ordinary editing can produce either difference. The comparison "
                                      "carries zero weight in the AI assessment."))
        mask = mask | color_mask
    if evidence_dir is not None:
        _write_evidence(evidence_dir, output, reference, submitted, mask, coverage=coverage)
    return output
