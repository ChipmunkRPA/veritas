"""Presentation-only regression checks for the separate resized-JPEG comparison."""

from __future__ import annotations

import hashlib
import io

import pymupdf
import pytest
from PIL import Image

from app.reports.image_reference_pdf import render


def _png(size, color):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _comparison(*, quality=75, writes=(1,), edited=True):
    source = {"sha256": "a" * 64, "width": 512, "height": 576}
    submitted = {"sha256": "b" * 64, "width": 576, "height": 512}
    proof = {"basis": "reproduced_jpeg_background_with_local_sample_difference" if edited else
             "exact_full_jpeg_pixel_reproduction", "jpeg_quality": quality,
             "submitted_over_reference_xy": [[576, 512], [512, 576]],
             "projection": [{"resize_kernel": kernel, "jpeg_writes": count, "body_sha256": "c" * 64,
                             "rgb_sha256": "d" * 64} for kernel, count in zip(("bicubic", "opencv_area"), writes, strict=False)]}
    result = {"schema": "chipmunk.image-reference-comparison/30", "status": "comparable",
              "coordinate_space": "exif-oriented-submitted", "reference": source, "submitted": submitted,
              "axis_jpeg_resize": proof, "affects_overall_confidence": False, "regions": [],
              "interpretation": "Two supplied copies were compared. Ordinary editing and JPEG encoding can cause differences."}
    if not edited:
        proof["matched_pixels"] = 576 * 512
        return result, {}
    result["alignment"] = {"method": "native_jpeg_sample_difference", "full_tiles": 1152,
                           "exact_tiles": 1100, "anchor_count": 40,
                           "submitted_overlap": {"x": 0, "y": 0, "w": 576, "h": 512},
                           "reference_overlap": {"x": 0, "y": 0, "w": 576, "h": 512}}
    assets = {"reference.png": _png((90, 70), (128, 64, 32)), "submitted.png": _png((80, 60), (32, 96, 144)),
              "overlay.png": _png((80, 60), (255, 173, 46)), "mask.png": _png((576, 512), (0, 0, 0))}

    def artifact(name, width, height):
        return {"file": name, "width": width, "height": height, "sha256": hashlib.sha256(assets[name]).hexdigest()}

    result["regions"] = [{"x": 118, "y": 65, "w": 80, "h": 60, "selected_pixels": 500,
                           "reference_region": {"x": 134, "y": 76, "w": 90, "h": 70},
                           "reference_crop": artifact("reference.png", 90, 70),
                           "submitted_crop": artifact("submitted.png", 80, 60),
                           "submitted_overlay": artifact("overlay.png", 80, 60)}]
    result["mask"] = artifact("mask.png", 576, 512)
    return result, assets


@pytest.mark.parametrize("quality", [75, 90])
@pytest.mark.parametrize("writes", [(1,), (2,), (1, 2)])
def test_exact_jpeg_reproduction_is_distinct_from_an_empty_local_check(quality, writes):
    comparison, assets = _comparison(quality=quality, writes=writes, edited=False)
    with pymupdf.open(stream=render(comparison, assets), filetype="pdf") as document:
        assert len(document) == 1
        text = document[0].get_text()
        assert "All pixels match after resizing and JPEG encoding." in text
        assert "Every decoded RGB pixel matches the reproduced JPEG." in text
        assert f"JPEG quality {quality}" in text
        assert "Zero AI confidence weight. AI origin is undetermined." in text
        assert "No region met" not in text and "Projected JPEG grid" not in text
        assert "Highlights show" not in text and "No local difference" not in text
        assert "Width 112.5% of earlier copy. Height 88.9%." in text
        assert "1 or 2 encodes" in text if writes == (1, 2) else f"{writes[0]} encode" in text
        assert document[0].get_images() == []


@pytest.mark.parametrize("quality", [75, 90])
def test_local_jpeg_differences_keep_both_native_crop_clocks_and_images(quality):
    comparison, assets = _comparison(quality=quality, writes=(1, 2))
    with pymupdf.open(stream=render(comparison, assets), filetype="pdf") as document:
        assert len(document) == 2
        summary, region = (page.get_text() for page in document)
        assert "1 difference region retained." in summary
        assert "On the projected JPEG grid, 1,100 of 1,152 decoded blocks match." in summary
        assert "Matching resize filters include bicubic, area." in summary
        assert "1 or 2 encodes" in summary
        assert f"JPEG quality {quality}" in summary
        assert "Zero AI confidence weight. AI origin is undetermined." in summary
        assert "Earlier copy overlap" not in summary and "Submitted image overlap" not in summary
        assert "JPEG sample difference 1" in region
        assert "Crops use native source pixels. Differences were measured on the reproduced JPEG grid." in region
        assert "x 134, y 76, 90 x 70 px" in region
        assert "x 118, y 65, 80 x 60 px" in region
        assert comparison["reference"]["sha256"] in summary and comparison["submitted"]["sha256"] in summary
        embedded = []
        for image in document[1].get_images(full=True):
            with Image.open(io.BytesIO(document.extract_image(image[0])["image"])) as decoded:
                embedded.append((decoded.size, decoded.convert("RGB").tobytes()))
        for filename in ("reference.png", "submitted.png"):
            with Image.open(io.BytesIO(assets[filename])) as original:
                assert (original.size, original.convert("RGB").tobytes()) in embedded
        for page in document:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        assert 40 <= x0 < x1 <= 572 and 20 <= y0 < y1 <= 775


def test_supported_jpeg_samples_with_no_retained_region_stay_distinct_from_full_reproduction():
    comparison, _ = _comparison()
    comparison["regions"] = []
    del comparison["mask"]
    with pymupdf.open(stream=render(comparison, {}), filetype="pdf") as document:
        assert len(document) == 1
        text = " ".join(document[0].get_text().split())
        assert "No region met the difference check's selection rule." in text
        assert "No local difference met the retention rule." in text
        assert "Small changes may still be present." in text
        assert "All pixels match" not in text and "Highlights show" not in text
        assert "Zero AI confidence weight. AI origin is undetermined." in text
        assert document[0].get_images() == []
