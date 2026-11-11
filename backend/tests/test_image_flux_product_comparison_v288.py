import numpy as np
import pytest
from PIL import Image

from scripts import image_flux_product_comparison_v288 as compare


def test_all_frozen_views_appear_once_in_three_batches_per_format():
    batches = [compare.tasks(index) for index in range(compare.PHASES)]
    assert [len(batch) for batch in batches] == [10, 10, 10]
    assert [item for batch in batches for item in batch] == [
        (number, kind) for number in range(6) for kind in compare.KINDS
    ]
    assert len(compare.QUALITIES) * len(compare.ARMS) * sum(len(batch) for batch in batches) == 120


def test_current_png_uses_strict_area_gate_and_jpeg_uses_image_response():
    raw = np.full((384, 384), 0.70, dtype=np.float32)
    raw.flat[:7992] = 0.71
    assert compare.current_gate(
        raw, {"image_score": None, "image_gate": False, "pixel_threshold": 0.70}, 0
    ) == (False, 0.70)
    raw.flat[7992] = 0.71
    assert compare.current_gate(
        raw, {"image_score": None, "image_gate": True, "pixel_threshold": 0.70}, 0
    ) == (True, 0.70)
    with pytest.raises(ValueError, match="Producer gate differs"):
        compare.current_gate(raw, {"image_score": None, "image_gate": False, "pixel_threshold": 0.70}, 0)
    jpeg = np.zeros((512, 512), dtype=np.float32)
    assert compare.current_gate(
        jpeg, {"image_score": 0.50, "image_gate": True, "pixel_threshold": 0.50}, 75
    ) == (True, 0.50)
    assert compare.current_gate(
        jpeg, {"image_score": 0.49, "image_gate": False, "pixel_threshold": 0.50}, 75
    ) == (False, 0.50)


@pytest.mark.parametrize("quality", compare.QUALITIES)
def test_production_and_frozen_masks_match_without_granting_controls_ai_truth(quality):
    pixels, selection = compare.photo.flux.fixture("interior")
    payload, image = compare.previous.encode(Image.fromarray(pixels), quality)
    footprint = selection > 0
    current = np.full(compare.response_shape("baseline", quality, image), 0.1, dtype=np.float32)
    current[110:250, 110:250] = 0.95
    candidate = np.full(compare.response_shape("candidate", quality, image), 0.1, dtype=np.float32)
    candidate[75:150, 75:150] = 0.95
    details = {
        "image_score": 0.55 if quality else None,
        "image_gate": True,
        "pixel_threshold": 0.5 if quality else 0.7,
    }
    positive = compare.measure(
        quality, current, details, candidate, payload, np.asarray(image), "ai", footprint
    )
    ordinary = compare.measure(
        quality, current, details, candidate, payload, np.asarray(image), "tone", footprint
    )
    assert positive["current"]["ai"]["reference_positive_pixels"] == int(footprint.sum())
    assert ordinary["current"]["ai"]["reference_positive_pixels"] == 0
    assert ordinary["current"]["ordinary"]["reference_positive_pixels"] == int(footprint.sum())
    assert (
        positive["shared"]["ai"]["predicted_positive_pixels"]
        <= positive["sensitive"]["ai"]["predicted_positive_pixels"]
    )
