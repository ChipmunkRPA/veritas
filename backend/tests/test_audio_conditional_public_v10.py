import copy
import hashlib
import json

import pytest

from scripts.audio_conditional_v10 import METHOD
from scripts.export_audio_conditional_v10 import COUNTS, FIELDS, ORIGINAL, review


def fixture():
    rows = [
        {
            "id": f"{role}:{i}",
            "role": role,
            "sha256": "a" * 64,
            "pcm_sha256": "b" * 64,
            "status": "ok",
            "quality": {"flatness": 0.3, "floor": 0.8},
            "cached_probe_quality": None,
        }
        for role, count in COUNTS.items()
        for i in range(count)
    ]
    lock = dict.fromkeys(FIELDS)
    lock.update(ORIGINAL)
    lock.update(
        schema="chipmunk.audio-conditional-lock/10",
        role_counts=COUNTS,
        fitting_started=False,
        new_neural_forwards=False,
        deployment_eligible=False,
        method=copy.deepcopy(METHOD),
        implementation_sha256={},
    )
    body = {"schema": "chipmunk.audio-conditional-public-freeze/10", "lock": lock}
    source = {
        "variants": {"development": [{k: r[k] for k in ("id", "role", "sha256", "pcm_sha256")} for r in rows]}
    }
    return body, rows, source


def encoded(body, rows):
    header = {
        "original_receipt_sha256": ORIGINAL["original_receipt_sha256"],
        "original_responses_sha256": ORIGINAL["original_responses_sha256"],
    }
    raw = "".join(json.dumps(r) + "\n" for r in (header, *rows)).encode()
    body["lock"]["quality_sha256"] = hashlib.sha256(raw).hexdigest()
    return raw


def test_complete_quality_source_matrix_has_model_free_review():
    body, rows, source = fixture()
    result = review(body, encoded(body, rows), source)
    assert result["passed"] and result["inputs"] == 4248


@pytest.mark.parametrize(
    "value", [None, {"flatness": float("nan"), "floor": 0.2}, {"flatness": 1.1, "floor": 0.2}]
)
def test_original_quality_cannot_be_missing_nonfinite_or_outside_its_box(value):
    body, rows, source = fixture()
    rows[0]["quality"] = value
    with pytest.raises(ValueError, match="bounded original/probe"):
        review(body, encoded(body, rows), source)


def test_role_change_and_unsupported_prospective_claim_are_rejected():
    body, rows, source = fixture()
    rows[0]["role"] = "confirmation"
    with pytest.raises(ValueError, match="Every corrected"):
        review(body, encoded(body, rows), source)
    body, rows, source = fixture()
    body["lock"]["fitting_started"] = True
    with pytest.raises(ValueError, match="Prospective freeze"):
        review(body, encoded(body, rows), source)


def test_random_early_stopping_cannot_enter_the_fixed_experiment():
    body, rows, source = fixture()
    body["lock"]["method"]["parameters"]["early_stopping"] = "auto"
    with pytest.raises(ValueError, match="no-random-holdout"):
        review(body, encoded(body, rows), source)
