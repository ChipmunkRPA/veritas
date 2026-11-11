from copy import deepcopy

import pytest

from scripts import video_frame_batch_qualification_v160 as qualification


def receipt(batch):
    n = qualification.SAMPLE_COUNT
    return {
        "device": "mps",
        "dtype": "float32",
        "model_revisions": {model: model for model in qualification.MODEL_IDS},
        "frames": [f"frame:{i}" for i in range(n)],
        "probabilities": [0.25 + i / 100 for i in range(n)],
        "aggregate": 0.32,
        "evidence": [{"times": [i / 25 for i in range(n)]}],
        "members": {
            model: {
                "batches": qualification.expected_batches(batch),
                "logits": [-1.0 + i / 100 for i in range(n)],
                "crop_sha256": [f"crop:{i}" for i in range(n)],
            }
            for model in qualification.MODEL_IDS
        },
    }


def test_parity_requires_the_actual_partial_tail_and_all_batch_one_forwards():
    reference, candidate = receipt(8), receipt(1)
    assert qualification.expected_batches(8) == [8, 7]
    assert qualification.compare(reference, candidate)["parity_within_frozen_tolerances"]
    candidate["members"][qualification.MODEL_IDS[0]]["batches"][-2:] = [2]
    with pytest.raises(ValueError, match="complete member forwards"):
        qualification.compare(reference, candidate)


def test_changes_in_evidence_or_crops_cannot_pass_with_identical_scores():
    reference, candidate = receipt(8), receipt(1)
    candidate["members"][qualification.MODEL_IDS[0]]["crop_sha256"].reverse()
    candidate["evidence"][0]["times"].reverse()
    observed = qualification.compare(reference, candidate)
    assert observed["maximum_absolute_probability_error"] == 0
    assert not observed["same_actual_float32_crops"]
    assert not observed["same_evidence_scope_and_times"]
    assert not observed["parity_within_frozen_tolerances"]


def test_tiny_error_crossing_the_raw_boundary_cannot_pass():
    reference, candidate = receipt(8), receipt(1)
    reference["probabilities"][0] = 0.5
    candidate["probabilities"][0] = 0.499999
    observed = qualification.compare(reference, candidate)
    assert observed["maximum_absolute_probability_error"] < qualification.PROBABILITY_ATOL
    assert not observed["same_raw_boundary_decisions"]
    assert not observed["parity_within_frozen_tolerances"]
    another_device = deepcopy(candidate)
    another_device["device"] = "cpu"
    with pytest.raises(ValueError, match="Same-device reference"):
        qualification.compare(reference, another_device)
