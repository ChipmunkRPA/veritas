import copy
from pathlib import Path

import pytest

from scripts.audio_layer_response_v12 import METHOD
from scripts.export_audio_layer_v12 import (
    BODY_FIELDS,
    COUNTS,
    EXTRA_FILES,
    FIELDS,
    ORIGINAL,
    PREVIOUS,
    REFERENCE_SHA,
    SOURCE_SHA,
    load,
    native_ids,
    review,
)


def fixture():
    root = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
    source = load(root / "audio-v9-identity-plan.json.gz")
    reference = load(root / "audio-v9-execution-freeze.json")
    lock = dict.fromkeys(FIELDS)
    lock.update(ORIGINAL)
    lock.update(
        schema="chipmunk.audio-layer-lock/12",
        method=copy.deepcopy(METHOD),
        implementation_sha256={},
        previous_control=copy.deepcopy(PREVIOUS),
        role_counts=dict(COUNTS),
        planned=4248,
        original_input_bytes_verified=4248,
        expected_unassessed=62,
        available_prior_probe_references=450,
        native_ids=native_ids(source["variants"]["development"]),
        fitting_started=False,
        new_neural_forwards=False,
        deployment_eligible=False,
    )
    for k in ("weights", "runtime", "serving_bundle"):
        lock[k] = copy.deepcopy(reference["lock"][k])
    lock["implementation_sha256"] = {
        **reference["lock"]["implementation_sha256"],
        **{name: "a" * 64 for name in EXTRA_FILES},
    }
    body = dict.fromkeys(BODY_FIELDS)
    body.update(
        schema="chipmunk.audio-layer-public-freeze/12",
        private_lock_sha256="a" * 64,
        source_capsule_sha256=SOURCE_SHA,
        reference_freeze_sha256=REFERENCE_SHA,
        lock=lock,
    )
    return body, source, reference


def test_public_metadata_retains_every_original_role_and_native_cell():
    body, source, reference = fixture()
    result = review(body, source, reference)
    assert result["passed"] and result["inputs"] == 4248 and result["native_cells"] == 99
    assert result["new_neural_forwards"] is False


@pytest.mark.parametrize(
    "kind", ["layer", "checkpoint", "missing", "native", "source", "private", "fitting", "code"]
)
def test_public_metadata_refuses_unsupported_or_altered_commitments(kind):
    body, source, reference = fixture()
    lock = body["lock"]
    if kind == "layer":
        lock["method"]["layer_ids"] = [1, 2, 3]
    elif kind == "checkpoint":
        lock["weights"] = {}
    elif kind == "missing":
        lock["expected_unassessed"] = 0
    elif kind == "native":
        lock["native_ids"].pop()
    elif kind == "source":
        lock["original_receipt_sha256"] = "b" * 64
    elif kind == "private":
        body["path"] = "/Users/example/private.wav"
    elif kind == "code":
        lock["implementation_sha256"].pop("backend/scripts/audio_layer_features_v12.py")
    else:
        lock["fitting_started"] = True
    with pytest.raises(ValueError):
        review(body, source, reference)
