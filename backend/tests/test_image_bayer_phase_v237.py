import io

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts import image_bayer_phase_v237 as study


def authored_bayer():
    rng = np.random.default_rng(237)
    mosaic = np.clip(np.rint(128 + rng.normal(0, 11, (512, 512))), 16, 239).astype(np.uint8)
    return cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB)


def jpeg(pixels, quality):
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=quality, subsampling=2)
    with Image.open(encoded) as decoded:
        return np.asarray(decoded.convert("RGB"))


def test_phase_profile_matches_hand_computed_directional_second_differences():
    tile = np.tile(np.array([[80, 82], [83, 88]], dtype=np.uint8), (32, 32))
    expected = np.log1p([[4, 4, 10, 10], [6, 12, 6, 12]])
    expected -= expected.mean(axis=1, keepdims=True)
    assert np.array_equal(study.phase_profile(tile), expected.ravel())
    assert study.phase_profile(np.zeros((64, 64), dtype=np.uint8)) is None


def test_scores_use_only_opposite_fold_reference(monkeypatch):
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    y, x = np.indices((256, 256))
    pixels[:, :, 1] = (y // 64 + x // 64) % 2
    vector = np.array([1, -1, -1, 1] * 2, dtype=np.float64)
    monkeypatch.setattr(study, "phase_profile", lambda tile: vector * (1 + int(tile[0, 0])))
    result = study.analyze_bayer_phase(pixels)
    assert result.status == "scored"
    assert result.available_tiles == (8, 8)
    assert result.held_out_support == (1, 1)
    assert np.array_equal(result.scores, np.ones((4, 4)))


def test_hard_composite_localizes_a_phase_change_in_a_minority_of_tiles():
    pixels = authored_bayer()
    original = study.analyze_bayer_phase(pixels)
    assert original.status == "scored"
    edited = pixels.copy()
    rng = np.random.default_rng(92)
    replacement = np.clip(np.rint(128 + rng.normal(0, 11, (128, 128, 3))), 16, 239).astype(np.uint8)
    edited[128:256, 192:320] = replacement
    result = study.analyze_bayer_phase(edited)
    assert result.status == "scored"
    assert result.source_size == (512, 512)
    assert result.scores.shape == (8, 8)
    touched = np.zeros((8, 8), dtype=bool)
    touched[2:4, 3:5] = True
    assert result.scores[touched].min() > result.scores[~touched].max()
    assert np.allclose(result.scores[~touched], original.scores[~touched], atol=0.02, rtol=0)


def test_uniform_phase_absence_and_conflicting_folds_abstain(monkeypatch):
    rng = np.random.default_rng(24)
    absent = np.clip(np.rint(128 + rng.normal(0, 11, (512, 512, 3))), 16, 239).astype(np.uint8)
    result = study.analyze_bayer_phase(absent)
    assert result.status == "phase_not_supported" and result.scores is None

    vector = np.array([1, -1, -1, 1] * 2, dtype=np.float64)
    y, x = np.indices((256, 256))
    absent = absent[:256, :256].copy()
    absent[:, :, 1] = (y // 64 + x // 64) % 2
    monkeypatch.setattr(study, "phase_profile", lambda tile: vector * (1 if tile[0, 0] else -1))
    result = study.analyze_bayer_phase(absent)
    assert result.status == "phase_not_supported" and result.held_out_support == (0, 0)
    assert result.scores is None


def test_crop_sharpen_and_missing_tiles_preserve_source_geometry():
    pixels = authored_bayer()
    cropped = pixels[1:-3, 3:-2]
    sharpened = cv2.addWeighted(pixels, 1.3, cv2.GaussianBlur(pixels, (3, 3), 0), -0.3, 0)
    for view in (cropped, sharpened):
        result = study.analyze_bayer_phase(view)
        assert result.status == "scored"
        assert result.source_size == view.shape[:2][::-1]
        assert result.scores.shape == tuple(size // 64 for size in view.shape[:2])
        assert np.nanmax(result.scores) < 0.12
    pixels[64:128, 128:192] = 255
    result = study.analyze_bayer_phase(pixels)
    assert result.status == "scored"
    assert np.argwhere(np.isnan(result.scores)).tolist() == [[1, 2]]


@pytest.mark.parametrize("operation", ["jpeg75", "jpeg90", "rescale"])
def test_ordinary_global_processing_is_not_a_regional_alarm(operation):
    pixels = authored_bayer()
    if operation.startswith("jpeg"):
        view = jpeg(pixels, quality=int(operation[4:]))
    else:
        view = cv2.resize(pixels, (384, 384), interpolation=cv2.INTER_AREA)
    result = study.analyze_bayer_phase(view)
    if result.status == "scored":
        assert np.nanmax(result.scores) < 0.12
    else:
        assert result.status == "phase_not_supported" and result.scores is None


def test_small_or_invalid_input_cannot_report_a_clear_result():
    small = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = study.analyze_bayer_phase(small)
    assert result.status == "insufficient_tiles" and result.scores is None
    with pytest.raises(ValueError, match="Native uint8 RGB"):
        study.analyze_bayer_phase(small.astype(np.float32))
