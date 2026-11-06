"""Coordinate, source-integrity and original-only selection contracts for crop research."""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.fused_crop_v6 import Crop, boxes, choose_gate, complement, predict, stitch, validate


def crops(width=667, height=401, scores=(0.9, 0.1, 0.1, 0.1)):
    return [Crop(box, np.ones((512, 512), dtype=np.float32), score, "a" * 64)
            for box, score in zip(boxes(width, height), scores, strict=True)]


def test_native_coordinates_and_internal_edges_are_distinct_from_source_edges():
    regions = crops()
    assert regions[0].box == (0, 0, 401, 241)
    assert regions[-1].box == (266, 160, 667, 401)
    mask = stitch(regions, 667, 401, 0.5)
    # First crop is the only admitted one: native source edges survive, seams do not.
    assert mask[0, 0] == 1
    assert mask[225, 375] == 1
    assert mask[226, 375] == 0
    assert mask[100, 376] == 0
    assert mask[300, 500] == 0
    assert np.all(regions[0].response == 1), "Projection must not mutate original response arrays"


@pytest.mark.parametrize("change", ["omit", "reorder", "out_of_bounds", "nan", "float64", "wrong_shape", "score"])
def test_partial_or_corrupted_crop_evidence_fails_closed(change):
    regions = crops()
    first = regions[0]
    if change == "omit":
        regions.pop()
    elif change == "reorder":
        regions.reverse()
    elif change == "out_of_bounds":
        regions[0] = Crop((-1, 0, 401, 241), first.response, first.image_score, first.rgb_sha256)
    elif change == "nan":
        first.response[0, 0] = np.nan
    elif change == "float64":
        regions[0] = Crop(first.box, first.response.astype(np.float64), first.image_score, first.rgb_sha256)
    elif change == "wrong_shape":
        regions[0] = Crop(first.box, first.response[:256], first.image_score, first.rgb_sha256)
    else:
        regions[0] = Crop(first.box, first.response, True, first.rgb_sha256)
    with pytest.raises(ValueError):
        stitch(regions, 667, 401, 0.5)


@pytest.mark.parametrize("gate", [0.499, True, math.nan, math.inf, 1.01])
def test_invalid_gate_cannot_admit_crops(gate):
    with pytest.raises(ValueError):
        stitch(crops(), 667, 401, gate)


def test_prediction_commits_exact_native_crop_pixels_without_changing_input():
    pixels = np.random.default_rng(11).integers(0, 256, (401, 667, 3), dtype=np.uint8)
    original = pixels.copy()
    seen = []

    class Model:
        def predict(self, image):
            seen.append(np.asarray(image).copy())
            return SimpleNamespace(source_size=image.size, image_score=0.6,
                                   raw_mask=np.zeros((512, 512), dtype=np.float32))

    result = predict(Model(), Image.fromarray(pixels))
    assert np.array_equal(original, pixels)
    for crop, actual in zip(result, seen, strict=True):
        x0, y0, x1, y1 = crop.box
        assert np.array_equal(actual, original[y0:y1, x0:x1])
        assert crop.rgb_sha256 == hashlib.sha256(actual.tobytes()).hexdigest()


def test_small_sources_remain_unassessed_instead_of_manufacturing_a_map():
    assert boxes(319, 1024) == []
    validate([], 319, 1024)
    assert not stitch([], 319, 1024, 0.5).any()


def test_existing_region_and_failed_baseline_are_preserved():
    baseline = {"predicted_positive_pixels": 10, "intersection": 0}
    extra = {"predicted_positive_pixels": 20, "intersection": 20}
    assert complement(baseline, extra) == (baseline, False)
    assert complement(None, extra) == (None, False)
    assert complement({"predicted_positive_pixels": 0}, None) == ({"predicted_positive_pixels": 0}, False)
    assert complement({"predicted_positive_pixels": 0}, extra) == (extra, True)


def controls():
    return [{"source_id": str(i), "label": 0, "status": "ok", "baseline": {"predicted_positive_pixels": 0},
             "crops": crops(scores=(0.8 if i < 2 else 0.3, 0.1, 0.1, 0.1))} for i in range(20)]


def test_gate_selection_handles_ties_without_using_edited_controls():
    originals = controls()
    rule = choose_gate(originals, lambda row, gate: any(c.image_score >= gate for c in row["crops"]))
    assert rule["image_gate"] == math.nextafter(0.8, math.inf)
    assert rule["development_additional_flags"] == 0
    originals[0]["label"] = 1
    with pytest.raises(ValueError, match="authentic"):
        choose_gate(originals, lambda *_: False)


@pytest.mark.parametrize("change", ["failed", "missing", "duplicate", "too_few"])
def test_gate_selection_refuses_incomplete_or_duplicated_controls(change):
    originals = controls()
    if change == "failed":
        originals[0]["status"] = "failed"
    elif change == "missing":
        originals[0]["baseline"] = None
    elif change == "duplicate":
        originals[0]["source_id"] = originals[1]["source_id"]
    else:
        originals.pop()
    with pytest.raises(ValueError):
        choose_gate(originals, lambda *_: False)
