"""Authored source geometry and invalid-input refusals; no quality examples."""

from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

from scripts.image_imdet_native_v17 import Native, fixture, preprocess


@pytest.mark.parametrize("size", [(31, 77), (88, 47), (256, 256), (515, 389), (389, 515), (1025, 129)])
def test_crop_geometry_stays_in_original_source(size):
    tensor, record = preprocess(fixture(*size))
    w, h = size
    a, b, c, d = record["inspected_source_xyxy"]
    assert 0 <= a < c <= w and 0 <= b < d <= h
    assert tensor.shape == (1, 3, 256, 256) and tensor.dtype == np.float32
    ow, oh = record["resized_size"]
    left, top, right, bottom = record["resized_crop_xyxy"]
    nominal = Fraction((right - left) * w, ow) * Fraction((bottom - top) * h, oh) / (w * h)
    assert record["source_area_fraction"] == float(nominal)
    assert "Neither is an edit mask" in record["coordinates_mean"]


def test_wide_crop_cannot_claim_complete_image_coverage():
    _, record = preprocess(fixture(1025, 129))
    assert record["source_area_fraction"] < 0.13
    assert record["inspected_source_xyxy"][0] > 400
    assert record["inspected_source_xyxy"][2] < 600


def test_source_and_input_hash_bind_values_not_just_dimensions():
    image = fixture(37, 41)
    _, a = preprocess(image)
    values = np.asarray(image).copy()
    values[20, 18] ^= 255
    _, b = preprocess(Image.fromarray(values))
    assert a["source_size"] == b["source_size"]
    assert a["source_rgb_sha256"] != b["source_rgb_sha256"]
    assert a["input_float32_sha256"] != b["input_float32_sha256"]


@pytest.mark.parametrize("size", [(31, 77), (88, 47), (515, 389), (1025, 129)])
def test_pixels_outside_conservative_resampling_support_do_not_enter_tensor(size):
    image = fixture(*size)
    before, record = preprocess(image)
    values = np.asarray(image).copy()
    a, b, c, d = record["conservative_resampling_support_xyxy"]
    outside = np.ones(values.shape[:2], bool)
    outside[b:d, a:c] = False
    values[outside] ^= 255
    after, changed = preprocess(Image.fromarray(values))
    assert np.array_equal(before, after)
    assert record["source_rgb_sha256"] != changed["source_rgb_sha256"]


@pytest.mark.parametrize("value", ["path.png", Image.new("RGB", (0, 12))])
def test_missing_or_empty_source_cannot_be_assessed(value):
    with pytest.raises(ValueError):
        preprocess(value)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "range", "mode"])
def test_invalid_tensor_never_enters_native_model(fault):
    model = object.__new__(Native)
    model.sessions = {"canonical": None}
    model.attempted_forwards = model.completed_forwards = 0
    tensor = np.zeros((1, 3, 256, 256), np.float32)
    if fault == "shape":
        tensor = tensor[:, :, :10, :]
    elif fault == "dtype":
        tensor = tensor.astype(np.float64)
    elif fault == "nan":
        tensor[0, 0, 0, 0] = np.nan
    elif fault == "range":
        tensor[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError):
        model.predict(tensor, "unknown" if fault == "mode" else "canonical")
    assert model.attempted_forwards == model.completed_forwards == 0
