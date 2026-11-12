import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.detectors.image.adapters import regional
from app.evidence.image_eligibility import validate_flip_consistency


def test_repeat_pass_retains_exact_original_component_and_rank(monkeypatch):
    monkeypatch.setattr(regional.settings, "environment", "dev")
    monkeypatch.setattr(regional.settings, "image_region_flip_review", True)
    shape = (512, 768)
    pixels = np.random.default_rng(45).integers(30, 220, (*shape, 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    monkeypatch.setattr(regional, "original_bytes", lambda _: output.getvalue())

    original = np.zeros(shape, np.float32)
    original[20:80, 30:100] = .99
    original[120:180, 190:250] = .96
    supported = original.copy()
    supported[:100] = 0
    calls = []

    def predict(image):
        calls.append(np.asarray(image).copy())
        raw = original if len(calls) == 1 else supported[:, ::-1].copy()
        return SimpleNamespace(raw_mask=raw, candidate_eligibility={"eligible": True}, binding={},
                               geometry={"fit_size": [768, 512], "source_rgb_sha256": "1" * 64,
                                         "input_float32_sha256": "2" * 64})

    model = SimpleNamespace(obj=SimpleNamespace(predict=predict), revision="fixture")
    ctx = SimpleNamespace(lease={regional.RegionalDetector.MODEL: model}, deadline=SimpleNamespace(check=lambda _: None))
    filtered = regional.RegionalDetector().localise(ctx, pixels)
    np.testing.assert_array_equal(calls[1], pixels[:, ::-1])
    assert [region["display_id"] for region in filtered.regions] == ["R2"]
    repeat = filtered.resolution_policy["flip_consistency"]
    assert validate_flip_consistency(repeat, 2, 768, 512) == 1
    assert repeat["candidates"][0]["shared_pixels"] == 0
    assert repeat["candidates"][1]["bounds"] == [190, 120, 60, 60]
    assert repeat["candidates"][1]["shared_pixels"] == 3600

    monkeypatch.setattr(regional.settings, "image_region_flip_review", False)
    calls.clear()
    unfiltered = regional.RegionalDetector().localise(ctx, pixels)
    assert len(calls) == 1
    assert filtered.regions == unfiltered.regions[1:]
    assert filtered.region_stats == unfiltered.region_stats[1:]
    assert filtered.integrity_score == unfiltered.integrity_score
    np.testing.assert_array_equal(filtered.anomaly_map, unfiltered.anomaly_map)
    repeat["candidates"][1]["shared_pixels"] = 1
    with pytest.raises(ValueError, match="contradicts"):
        validate_flip_consistency(repeat, 2, 768, 512)
