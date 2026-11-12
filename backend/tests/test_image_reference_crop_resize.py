import io

import numpy as np
import pytest
from PIL import Image

from app.detectors.image import reference_crop_resize, reference_resize
from app.evidence.image_reference import accepted_crop_resize, supported_crop_resize
from scripts.image_reference_crop_resize_transfer import cases
from tests.test_image_reference_compare import _scene


@pytest.mark.parametrize("format", ("PNG", "JPEG"))
def test_corroborated_crop_preserves_exact_native_coordinates_and_rejects_fractional_shift(format):
    original = _scene()
    edited = original.copy()
    edited[68:146, 130:220] = (210, 52, 119)
    views = dict((name, (image, truth, bounds)) for name, image, truth, bounds in cases(
        1, {"original": original, "edited": edited}, (130, 68, 220, 146), np.rot90(original)))
    for name in ("feathered", "untouched_crop", "fractional", "rotation"):
        pixels, truth, expected = views[name]
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format, quality=75, subsampling=2)
        with Image.open(stream) as carrier:
            result, mask = reference_crop_resize.compare(original, np.asarray(carrier), carrier)
        if expected is None:
            assert result == {"status": "crop_resize_unresolved", "regions": []} and mask is None
            continue
        assert result["status"] == "comparable"
        assert {key: result["reference_crop_bounds"][key] for key in expected} == expected
        if name == "untouched_crop":
            assert not mask.any() and not result["regions"]
        else:
            overlap = int((mask & truth).sum())
            assert overlap / truth.sum() >= .50
            assert overlap / (mask | truth).sum() >= .30
            for region in result["regions"]:
                native = reference_resize.native_crop(region, (expected["w"], expected["h"]), pixels.shape[1::-1])
                assert region["reference_region"] == {**native, "x": native["x"] + expected["x"], "y": native["y"] + expected["y"]}


def test_serving_caps_reject_larger_sources_and_unvalidated_scale():
    for width in (845, 846):
        stream = io.BytesIO()
        Image.new("RGB", (width, 300)).save(stream, "JPEG", quality=75, subsampling=2)
        with Image.open(stream) as carrier:
            assert supported_crop_resize(carrier, (768, 600)) == (width == 845)
            assert not supported_crop_resize(carrier, (769, 600))
    result = {"status": "comparable", "reference_crop_bounds": {"w": 400, "h": 300}}
    assert accepted_crop_resize(result, (288, 216))
    assert accepted_crop_resize(result, (468, 351))
    assert not accepted_crop_resize(result, (287, 216))
    assert not accepted_crop_resize(result, (468, 352))
