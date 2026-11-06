import copy
import gzip
import json
from pathlib import Path

import pytest

from scripts.review_audio_identity_execution_v9 import FIELDS, export, review


def inputs():
    root = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
    plan = json.loads(gzip.decompress((root / "audio-v9-identity-plan.json.gz").read_bytes()))
    original = json.loads(gzip.decompress((root / "audio-v8-source-plan.json.gz").read_bytes()))
    cells = {}
    for row in plan["variants"]["development"]:
        if row["role"] == "fit":
            key = (row["label"], row["condition"])
            cells[key] = min(cells.get(key, row["id"]), row["id"])
    lock = {
        **copy.deepcopy(original["lock"]),
        "schema": "chipmunk.audio-identity-execution-lock/9",
        "source_plan_sha256": plan["source_plan_sha256"],
        "variants_sha256": plan["variants_sha256"]["development"],
        "reserved_confirmation_sha256": plan["variants_sha256"]["confirmation"],
        "cache_sha256": "68673bfb01c3e5b80609299c4455f3e436d723312f328536bacf95fbd2a7e6f4",
        "cache_old_manifest_sha256": original["lock"]["variants_sha256"],
        "cache_eligible_inputs": 1846,
        "planned": 4248,
        "development_sources": 161,
        "native_cache_ids": sorted(cells.values()),
    }
    lock["implementation_sha256"].update({f"extra{i}.py": "a" * 64 for i in range(10)})
    body = {
        "schema": "chipmunk.audio-identity-public-execution/9",
        "private_lock_sha256": "a" * 64,
        "corrected_capsule_sha256": "b" * 64,
        "lock": {k: lock[k] for k in FIELDS},
        "new_neural_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
    }
    return body, plan, original


def test_public_execution_preserves_unchanged_method_and_corrected_denominators():
    body, plan, original = inputs()
    result = review(body, plan, original)
    assert result["passed"] and result["planned"] == 4248 and result["reserved"] == 1080


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", {}),
        ("weights", {}),
        ("runtime", {}),
        ("serving_bundle", {}),
        ("development_sources", 158),
        ("native_cache_ids", []),
        ("deployment_eligible", True),
    ],
)
def test_altered_execution_cannot_pass_metadata_review(key, value):
    body, plan, original = inputs()
    body["lock"][key] = value
    with pytest.raises(ValueError):
        review(body, plan, original)


def test_export_does_not_copy_private_paths_transcripts_or_source_ids(tmp_path):
    body, plan, original = inputs()
    private = {
        **body["lock"],
        "source_plan_path": "/private/source",
        "cache_path": "/private/cache",
        "transcript": "private transcript",
        "provider_id": "private provider",
    }
    paths = [tmp_path / name for name in ("private.json", "plan.json", "original.json")]
    for path, data in zip(paths, (private, plan, original), strict=True):
        path.write_text(json.dumps(data))
    output = tmp_path / "public.json"
    export(*paths, output)
    text = output.read_text()
    assert all(value not in text for value in ("/private/", "private transcript", "private provider"))
    assert review(json.loads(text), plan, original)["passed"]
