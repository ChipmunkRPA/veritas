"""Replay the browser export and reject tampering even after rehashing an asset."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from app.evidence.image_reference import inspect
from scripts.review_image_reference_export import review
from tests.test_image_reference_compare import _scene


def _write(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def saved_comparison(tmp_path):
    source = _scene()
    edited = source[11:241, 16:372].copy()
    edited[65:142, 118:207] = (210, 52, 119)
    reference, submitted = tmp_path / "earlier.png", tmp_path / "submitted.jpg"
    Image.fromarray(source).save(reference)
    Image.fromarray(edited).save(submitted, quality=75, subsampling=2)
    evidence = tmp_path / "evidence"
    comparison = inspect(reference, submitted, evidence_dir=evidence, allow_integer_alignment=True)
    assert comparison["alignment"]["reference_minus_submitted"] == {"x": 16, "y": 11}
    document = {"comparison": comparison, "pngs": {
        png.name: base64.b64encode(png.read_bytes()).decode("ascii") for png in evidence.glob("*.png")}}
    path = tmp_path / "export.json"
    _write(path, document)
    return path, reference, submitted, document


def test_replays_both_native_crop_spaces_and_binary_mask(saved_comparison):
    path, reference, submitted, document = saved_comparison
    assert review(path, reference, submitted) == {"reported_status": "comparable", "source_hashes_verified": 2,
        "crops_verified": 2, "mask_pixels_counted": sum(region["selected_pixels"] for region in document["comparison"]["regions"])}


def test_refuses_wrong_source_and_shifted_bounds(saved_comparison):
    path, reference, submitted, document = saved_comparison
    with pytest.raises(ValueError, match="Original source SHA-256"):
        review(path, submitted, reference)
    document["comparison"]["regions"][0]["reference_region"]["x"] += 1
    _write(path, document)
    with pytest.raises(ValueError, match="Region coordinates conflict"):
        review(path, reference, submitted)


@pytest.mark.parametrize(("field", "reason"), [
    ("reference_crop", "Evidence crop pixels differ"),
    ("mask", "Mask pixels fall outside"),
    ("submitted_overlay", "Difference overlay pixels differ"),
])
def test_rehashed_altered_png_still_fails(saved_comparison, field, reason):
    path, reference, submitted, document = saved_comparison
    comparison = document["comparison"]
    asset = comparison["mask"] if field == "mask" else comparison["regions"][0][field]
    filename = asset["file"]
    with Image.open(io.BytesIO(base64.b64decode(document["pngs"][filename]))) as image:
        if field == "mask":
            image.putpixel((0, 0), 255)
        elif field == "submitted_overlay":
            image.putpixel((0, 0), (255, 173, 46, 255))
        else:
            image.putpixel((0, 0), (0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
    content = buffer.getvalue()
    document["pngs"][filename] = base64.b64encode(content).decode("ascii")
    asset["sha256"] = hashlib.sha256(content).hexdigest()
    _write(path, document)
    with pytest.raises(ValueError, match=reason):
        review(path, reference, submitted)


def test_reports_measured_empty_and_unassessed_separately(tmp_path):
    original, other = tmp_path / "original.png", tmp_path / "other.png"
    Image.fromarray(_scene()).save(original)
    Image.fromarray(np.rot90(_scene())).save(other)
    path = tmp_path / "empty.json"
    for source, status in ((original, "comparable"), (other, "integer_alignment_unresolved")):
        _write(path, {"comparison": inspect(original, source, allow_integer_alignment=True), "pngs": {}})
        result = review(path, original, source)
        assert result == {"reported_status": status, "source_hashes_verified": 2, "crops_verified": 0, "mask_pixels_counted": 0}


@pytest.mark.parametrize("extension", ["png", "webp"])
@pytest.mark.parametrize("edited", [False, True])
def test_replays_width_and_height_resize_with_native_crops(tmp_path, extension, edited):
    from app.evidence.image_reference import AXIS_RESIZE_SCHEMA

    rng = np.random.default_rng(587)
    pixels = rng.integers(40, 220, (576, 512, 3), dtype=np.uint8)
    reference = tmp_path / "earlier.png"
    submitted = tmp_path / f"submitted.{extension}"
    Image.fromarray(pixels).save(reference)
    if edited:
        pixels[210:258, 180:241] = (130, 32, 170)
    Image.fromarray(pixels).resize((576, 512), Image.Resampling.BICUBIC).save(submitted, lossless=True)
    assert inspect(reference, submitted, allow_proportional_resize=True)["status"] != "comparable"
    evidence = tmp_path / "evidence"
    result = inspect(reference, submitted, evidence_dir=evidence, allow_proportional_resize=True, allow_axis_resize=True)
    assert result["schema"] == AXIS_RESIZE_SCHEMA and result["affects_overall_confidence"] is False
    proof = result["axis_resize"]
    assert proof["submitted_over_reference_xy"] == [[576, 512], [512, 576]]
    assert proof["basis"] == ("exact_distributed_texture" if edited else "exact_full_grid")
    export = tmp_path / "export.json"
    bundle = {"comparison": result, "pngs": {png.name: base64.b64encode(png.read_bytes()).decode("ascii")
                                             for png in evidence.glob("*.png")}}
    _write(export, bundle)
    assertion = review(export, reference, submitted)
    assert assertion["source_hashes_verified"] == 2 and (assertion["crops_verified"] > 0) is edited
    proof["projected_rgb_sha256"] = "0" * 64
    _write(export, bundle)
    with pytest.raises(ValueError, match="measurements differ from the supplied sources"):
        review(export, reference, submitted)


@pytest.mark.parametrize(("quality", "writes", "edited"), [(75, 1, True), (90, 2, True), (75, 2, False), (90, 1, False)])
def test_replays_stretched_jpeg_delivery_with_both_native_sources(tmp_path, quality, writes, edited):
    from app.detectors.image import reference_axis_jpeg_resize as axis_jpeg

    pixels = np.random.default_rng(647).integers(40, 220, (576, 512, 3), dtype=np.uint8)
    reference, submitted = tmp_path / "earlier.png", tmp_path / "submitted.jpeg"
    Image.fromarray(pixels).save(reference)
    if edited:
        pixels[210:258, 180:241] = (130, 32, 170)
    Image.fromarray(pixels).resize((555, 512), Image.Resampling.BICUBIC).save(submitted, quality=quality, subsampling=2)
    if writes == 2:
        with Image.open(submitted) as image:
            rendered = image.convert("RGB")
        rendered.save(submitted, quality=quality, subsampling=2)

    ordinary = inspect(reference, submitted, allow_proportional_resize=True, allow_axis_resize=True)
    assert ordinary["status"] != "comparable"
    evidence = tmp_path / "evidence"
    result = inspect(reference, submitted, evidence_dir=evidence, allow_proportional_resize=True, allow_axis_jpeg_resize=True)
    assert result["schema"] == axis_jpeg.SCHEMA and result["affects_overall_confidence"] is False
    proof = result["axis_jpeg_resize"]
    assert proof["basis"] == (axis_jpeg.LOCAL_DIFFERENCE if edited else axis_jpeg.FULL_GRID)
    assert proof["jpeg_quality"] == quality and proof["submitted_over_reference_xy"] == [[555, 512], [512, 576]]
    assert [(row["resize_kernel"], row["jpeg_writes"]) for row in proof["projection"]] == [("bicubic", writes)]
    bundle = {"comparison": result, "pngs": {png.name: base64.b64encode(png.read_bytes()).decode("ascii")
                                             for png in evidence.glob("*.png")}}
    export = tmp_path / "export.json"
    _write(export, bundle)
    verified = review(export, reference, submitted)
    assert verified["source_hashes_verified"] == 2 and (verified["crops_verified"] > 0) is edited
    proof["projection"][0]["jpeg_writes"] = 3 - writes
    _write(export, bundle)
    with pytest.raises(ValueError, match="JPEG resize measurements differ"):
        review(export, reference, submitted)


def test_replays_stretched_jpeg_with_no_retained_local_region(tmp_path):
    from app.detectors.image import reference_axis_jpeg_resize as axis_jpeg

    pixels = np.random.default_rng(647).integers(40, 220, (576, 512, 3), dtype=np.uint8)
    reference, submitted = tmp_path / "earlier.png", tmp_path / "submitted.jpg"
    Image.fromarray(pixels).save(reference)
    pixels[236, 190] += 8
    Image.fromarray(pixels).resize((555, 512), Image.Resampling.BICUBIC).save(submitted, quality=75, subsampling=2)
    result = inspect(reference, submitted, allow_axis_jpeg_resize=True)
    assert result["axis_jpeg_resize"]["basis"] == axis_jpeg.LOCAL_DIFFERENCE
    assert result["regions"] == [] and "mask" not in result
    export = tmp_path / "export.json"
    _write(export, {"comparison": result, "pngs": {}})
    assert review(export, reference, submitted) == {"reported_status": "comparable", "source_hashes_verified": 2,
        "crops_verified": 0, "mask_pixels_counted": 0}


@pytest.mark.parametrize("variant,status", [("resize", "comparable"), ("aspect", "unsupported_resize_geometry"),
                                              ("rotated", "same_framing_unresolved")])
def test_resize_reviewer_recomputes_empty_comparison_status(tmp_path, variant, status):
    source = Image.fromarray(_scene())
    original, submitted = tmp_path / "original.png", tmp_path / "submitted.png"
    source.save(original)
    if variant == "rotated":
        source = source.rotate(2, Image.Resampling.BICUBIC)
    source.resize((315, 248 if variant == "aspect" else 210), Image.Resampling.BICUBIC).save(submitted)
    comparison = inspect(original, submitted, allow_proportional_resize=True)
    assert comparison["schema"] == "chipmunk.image-reference-comparison/3"
    assert comparison["status"] == status and not comparison["regions"]
    path = tmp_path / "export.json"
    _write(path, {"comparison": comparison, "pngs": {}})
    assert review(path, original, submitted)["reported_status"] == status
    if variant == "resize":
        comparison["submitted_over_reference_xy"][0][0] += 1
    else:
        comparison["status"] = "comparable"
    _write(path, {"comparison": comparison, "pngs": {}})
    with pytest.raises(ValueError, match="differs from recomputed status, framing or regions"):
        review(path, original, submitted)


def test_refuses_duplicate_json_fields(saved_comparison):
    path, reference, submitted, _ = saved_comparison
    path.write_text(path.read_text().replace('{"comparison":', '{"pngs": {}, "comparison":', 1))
    with pytest.raises(ValueError, match="duplicate fields"):
        review(path, reference, submitted)


def test_refuses_claimed_alignment_without_recorded_support(saved_comparison):
    path, reference, submitted, document = saved_comparison
    alignment = document["comparison"]["alignment"]
    alignment.update(matching_features=0, supporting_tiles=[])
    _write(path, document)
    with pytest.raises(ValueError, match="insufficient recorded support"):
        review(path, reference, submitted)


def test_refuses_unshifted_regions_on_identical_sources(saved_comparison):
    path, reference, _, document = saved_comparison
    comparison = document["comparison"]
    comparison.update(schema="chipmunk.image-reference-comparison/1", coordinate_space="exif-oriented-original",
                      submitted=comparison["reference"])
    comparison.pop("alignment")
    _write(path, document)
    with pytest.raises(ValueError, match="unshifted browser comparison conflicts"):
        review(path, reference, reference)
