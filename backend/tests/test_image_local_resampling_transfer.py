from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts import image_local_resampling_transfer as study


def authored_image():
    rng = np.random.default_rng(22)
    y, x = np.indices((480, 480))
    noise = rng.normal(0, 25, x.shape)
    gray = 126 + noise + 24 * np.sin(y / 23) * np.cos(x / 19) + 3.5 * cv2.GaussianBlur(noise, (0, 0), 2)
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def test_integral_frequency_projection_uses_native_coordinates_for_both_axes():
    image = authored_image()[:256, :320]
    terms, _ = study.coefficient_grids(image)
    fields, _ = study.residual_fields(image)
    frequency = study.FREQUENCIES[3]
    expected_box = np.asarray([[3, 2, 7, 5]])
    y, x = np.indices((48, 64))
    for axis, coordinates in enumerate((x + 48, y + 32)):
        direct = (fields[axis][32:80, 48:112] * np.exp(-2j * np.pi * coordinates * frequency)).sum()
        measured = study.panel.sums(terms[axis], expected_box)[0, 3]
        assert measured == pytest.approx(direct, abs=2e-4)


@pytest.mark.parametrize("factor,frequency", [((6, 5), 1 / 6), ((7, 5), 2 / 7)])
def test_authored_resampling_localizes_but_whole_image_and_native_copy_do_not(factor, frequency):
    image = authored_image()
    variants = study.transformed_variants(image, (144, 144, 336, 336), factor)
    for fmt in ("png", "jpeg75"):
        for role in study.AUTHORED:
            pixels = variants[role]
            decoded = study._encoded(pixels, fmt)[1]
            truth = np.any(pixels != image, axis=2)
            proposal = study.response(decoded)["quadrant_agreement"]
            assert proposal["frequency_cycles_per_pixel"] == pytest.approx(frequency)
            mask = study.panel.selection(proposal, study.POINT_FLOOR, truth.shape)
            assert study.panel.useful(study.panel.scoring.measure(mask, truth))
        for role in study.EXTRA_CONTROLS:
            decoded = study._encoded(variants[role], fmt)[1]
            assert study.response(decoded)["quadrant_agreement"]["score"] < study.POINT_FLOOR


def test_flat_images_are_unassessed_and_jpeg_harmonics_are_excluded():
    assert study.response(np.full((256, 256, 3), 128, dtype=np.uint8)) == dict.fromkeys(study.ARMS)
    assert np.min(abs(study.FREQUENCIES[:, None] - np.array([1, 2, 3, 4]) / 8)) >= .019
