import copy

import pytest

from scripts.audio_full_representation_v11 import METHOD
from scripts.export_audio_full_representation_v11 import (
    COUNTS,
    FIELDS,
    ORIGINAL,
    REFERENCE_SHA,
    SOURCE_SHA,
    review,
)


def fixture():
    lock = dict.fromkeys(FIELDS)
    lock.update(ORIGINAL)
    lock.update(
        schema="chipmunk.audio-full-representation-lock/11",
        method=copy.deepcopy(METHOD),
        implementation_sha256={},
        runtime={"cpu": True},
        weights=[{"sha256": "a" * 64}],
        available_cached_probes=424,
        role_counts=dict(COUNTS),
        serving_bundle={"version": "unchanged"},
        fitting_started=False,
        new_neural_forwards=False,
        deployment_eligible=False,
    )
    body = {
        "schema": "chipmunk.audio-full-representation-public-freeze/11",
        "source_capsule_sha256": SOURCE_SHA,
        "reference_freeze_sha256": REFERENCE_SHA,
        "lock": lock,
    }
    source = {"variants": {"development": [{"role": role} for role, n in COUNTS.items() for _ in range(n)]}}
    reference = {
        "lock": {
            k: copy.deepcopy(lock[k])
            for k in ("weights", "runtime", "serving_bundle", "serving_calibration_sha256")
        }
    }
    return body, source, reference


def test_public_preflight_retains_complete_roles_without_fitting():
    body, source, reference = fixture()
    result = review(body, source, reference)
    assert result["passed"] and result["inputs"] == 4248
    assert result["fitting_started"] is False and result["fresh_confirmation_started"] is False


@pytest.mark.parametrize("alteration", ["model", "objective", "role", "path", "fitting"])
def test_public_preflight_rejects_changed_checkpoints_roles_or_prospective_claims(alteration):
    body, source, reference = fixture()
    if alteration == "model":
        body["lock"]["weights"] = []
    elif alteration == "objective":
        body["lock"]["method"]["residual_ridge"] = 0.001
    elif alteration == "role":
        source["variants"]["development"][0]["role"] = "confirmation"
    elif alteration == "path":
        body["extra"] = "/Users/example/private-source.wav"
    else:
        body["lock"]["fitting_started"] = True
    with pytest.raises(ValueError):
        review(body, source, reference)
